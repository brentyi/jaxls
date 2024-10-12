from __future__ import annotations

from typing import Hashable

import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class MatrixBlockRow:
    start_row: jax.Array
    """Row indices of the start of each block. Shape should be `(num_blocks,)`."""
    start_cols: tuple[jax.Array, ...]
    """Column indices of the start of each block."""
    block_widths: jdc.Static[tuple[int, ...]]
    """Width of each block in the block-row."""
    blocks_concat: jax.Array
    """Blocks of matrix, concatenated along the column axis. Shape in tuple should be `(num_blocks, rows, cols)`."""

    def treedef(self) -> Hashable:
        return tuple(block.shape for block in self.blocks_concat)


@jdc.pytree_dataclass
class BlockRowSparseMatrix:
    block_rows: tuple[MatrixBlockRow, ...]
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
            assert len(block_row.start_cols) == len(block_row.block_widths)
            target_slice_parts = list[jax.Array]()
            for start_cols, width in zip(block_row.start_cols, block_row.block_widths):
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
