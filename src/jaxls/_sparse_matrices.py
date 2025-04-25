from __future__ import annotations

from typing import Hashable

import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class SparseBlockRow:
    """A sparse block-row. Each block-row contains:

    - A set of N blocks with shape (rows, cols_i), for i=1...N.
        - Every block has an equal number of rows.
        - Blocks can have different numbers of columns. We store these in
          the `block_num_cols` attribute.
        - Concatenated values are stored in `blocks_concat`.
    - An initial column index for each block in the block row.
        - We store these in `start_cols`.

    A `num_block_rows` leading axis will often be prepended to all contained
    arrays. In this case, the `SparseBlockRow` structure represents multiple
    sequential block rows. Each resulting block-row has the same block count
    and widths; they would otherwise not be stackable. However, their sparsity
    patterns may vary due to different values held in `start_cols`.
    """

    num_cols: jdc.Static[int]
    """Total width of the block-row, including columns with zero values."""
    start_cols: tuple[jax.Array, ...]
    """Column indices of the start of each block. Shape in tuple should be
    `([num_block_rows],)`."""
    block_num_cols: jdc.Static[tuple[int, ...]]
    """# of columns for each block in the block-row."""
    blocks_concat: jax.Array
    """Blocks of matrix, concatenated along the column axis. Shape in tuple
    should be `([num_block_rows,] rows, cols)`."""

    def treedef(self) -> Hashable:
        return tuple(block.shape for block in self.blocks_concat)

    def to_dense(self) -> jax.Array:
        """Convert block-row or batched block-rows to dense representation."""
        if self.blocks_concat.ndim == 3:
            # Batched block-rows.
            (num_block_rows, num_rows, _) = self.blocks_concat.shape
            return jax.vmap(SparseBlockRow.to_dense)(self).reshape(
                (num_block_rows * num_rows, self.num_cols)
            )

        assert self.blocks_concat.ndim == 2
        num_rows, num_cols_concat = self.blocks_concat.shape
        out = jnp.zeros((num_rows, self.num_cols))

        start_concat_col = 0
        for start_col, block_width in zip(self.start_cols, self.block_num_cols):
            end_concat_col = start_concat_col + block_width
            out = jax.lax.dynamic_update_slice(
                out,
                update=self.blocks_concat[:, start_concat_col:end_concat_col],
                start_indices=(0, start_col),
            )
            start_concat_col = end_concat_col

        assert start_concat_col == num_cols_concat
        return out


@jdc.pytree_dataclass
class BlockRowSparseMatrix:
    block_rows: tuple[SparseBlockRow, ...]
    """Batched block-rows, ordered. Each element in the tuple has a leading
    axis, which represents consecutive block-rows."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""

    def multiply(self, target: jax.Array) -> jax.Array:
        """Sparse-dense multiplication."""
        assert target.ndim == 1

        out_slices = []
        for block_row in self.block_rows:
            # Do matrix multiplies for all blocks in block-row.
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            # Get slices corresponding to nonzero terms in block-row.
            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            target_slice_parts = list[jax.Array]()
            for start_cols, width in zip(
                block_row.start_cols, block_row.block_num_cols
            ):
                assert start_cols.shape == (n_block,)
                assert isinstance(width, int)
                slice_part = jax.vmap(
                    lambda start_col: jax.lax.dynamic_slice_in_dim(
                        target, start_index=start_col, slice_size=width, axis=0
                    )
                )(start_cols)
                assert slice_part.shape == (n_block, width)
                target_slice_parts.append(slice_part)

            # Concatenate slices to form target slice.
            target_slice = jnp.concatenate(target_slice_parts, axis=1)
            assert target_slice.shape == (n_block, block_nz_cols)

            # Multiply block-rows with target slice.
            out_slices.append(
                jnp.einsum(
                    "bij,bj->bi", block_row.blocks_concat, target_slice
                ).flatten()
            )

        result = jnp.concatenate(out_slices, axis=0)
        return result

    def compute_column_norms(self) -> jax.Array:
        """Compute column norms."""
        squared_sum = jnp.zeros(self.shape[1])
        for block_row in self.block_rows:
            # Get slices corresponding to nonzero terms in block-row.
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            block_row_squared_sum = jnp.sum(block_row.blocks_concat**2, axis=-2)
            assert block_row_squared_sum.shape == (n_block, block_nz_cols)
            block_nz_col_idx = 0
            for start_cols, width in zip(
                block_row.start_cols, block_row.block_num_cols
            ):
                # Add squared terms to the correct column in the result.
                assert start_cols.shape == (block_row.blocks_concat.shape[0],)
                assert isinstance(width, int)
                assert squared_sum.shape == (self.shape[1],)
                assert block_row_squared_sum.shape == (n_block, block_nz_cols)
                squared_sum = squared_sum.at[
                    start_cols[:, None] + jnp.arange(width)[None, :]
                ].add(
                    block_row_squared_sum[
                        :, block_nz_col_idx : block_nz_col_idx + width
                    ]
                )
                block_nz_col_idx += width
        return jnp.sqrt(squared_sum)

    def scale_columns(self, scales: jax.Array) -> BlockRowSparseMatrix:
        """Scale columns of the matrix by a vector."""
        assert scales.ndim == 1
        assert scales.shape[0] == self.shape[1]
        scaled_block_rows = list[SparseBlockRow]()
        for block_row in self.block_rows:
            # Do matrix multiplies for all blocks in block-row.
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            # Get slices corresponding to nonzero terms in block-row.
            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            scale_slice_parts = list[jax.Array]()
            for start_cols, width in zip(
                block_row.start_cols, block_row.block_num_cols
            ):
                assert start_cols.shape == (n_block,)
                assert isinstance(width, int)
                slice_part = jax.vmap(
                    lambda start_col: jax.lax.dynamic_slice_in_dim(
                        scales, start_index=start_col, slice_size=width, axis=0
                    )
                )(start_cols)
                assert slice_part.shape == (n_block, width)
                scale_slice_parts.append(slice_part)

            # Concatenate slices to form target slice.
            scale_slice = jnp.concatenate(scale_slice_parts, axis=1)
            assert scale_slice.shape == (n_block, block_nz_cols)

            # Scale and append updated block row.
            with jdc.copy_and_mutate(block_row) as scaled_block_row:
                scaled_block_row.blocks_concat = (
                    block_row.blocks_concat * scale_slice[:, None, :]
                )
            scaled_block_rows.append(scaled_block_row)

        # Create new matrix with scaled block rows.
        return BlockRowSparseMatrix(tuple(scaled_block_rows), shape=self.shape)

    def to_dense(self) -> jax.Array:
        """Convert to a dense matrix."""
        out = jnp.concatenate(
            [block_row.to_dense() for block_row in self.block_rows],
            axis=0,
        )
        assert out.shape == self.shape
        return out


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    indices: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""
    indptr: jax.Array
    """Index of start to each row. Shape should be `(*, num_rows)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""


@jdc.pytree_dataclass
class SparseCsrMatrix:
    """Data structure for sparse CSR matrices."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCsrCoordinates
    """Indices describing non-zero entries."""

    def as_jax_bcsr(self) -> jax.experimental.sparse.BCSR:
        return jax.experimental.sparse.BCSR(
            args=(self.values, self.coords.indices, self.coords.indptr),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )


@jdc.pytree_dataclass
class SparseCooCoordinates:
    rows: jax.Array
    """Row indices of non-zero entries. Shape should be `(*, N)`."""
    cols: jax.Array
    """Column indices of non-zero entries. Shape should be `(*, N)`."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""


@jdc.pytree_dataclass
class SparseCooMatrix:
    """Sparse matrix in COO form."""

    values: jax.Array
    """Non-zero matrix values. Shape should be `(*, N)`."""
    coords: SparseCooCoordinates
    """Indices describing non-zero entries."""

    def as_jax_bcoo(self) -> jax.experimental.sparse.BCOO:
        return jax.experimental.sparse.BCOO(
            args=(
                self.values,
                jnp.stack([self.coords.rows, self.coords.cols], axis=-1),
            ),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )
