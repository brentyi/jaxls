from __future__ import annotations

import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class MatrixBlock:
    start_row: jax.Array
    start_col: jax.Array
    values: jax.Array


@jdc.pytree_dataclass
class BlockSparseMatrix:
    blocks: dict[tuple[int, int], MatrixBlock]
    """Map from block shape to block (values, start row, start col)."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""

    def transpose(self) -> BlockSparseMatrix:
        new_blocks = {}
        for block_shape, block in self.blocks.items():
            new_block = MatrixBlock(
                start_row=block.start_col,
                start_col=block.start_row,
                values=jnp.swapaxes(block.values, -1, -2),
            )
            new_blocks[block_shape[::-1]] = new_block
        return BlockSparseMatrix(new_blocks, (self.shape[1], self.shape[0]))

    def multiply(self, target: jax.Array) -> jax.Array:
        result = jnp.zeros(self.shape[0])
        for block_shape, block in self.blocks.items():
            start_row, start_col = block.start_row, block.start_col
            assert len(start_row.shape) == 1
            assert len(start_col.shape) == 1
            values = block.values
            assert values.shape == (len(start_row), *block_shape)

            def multiply_one_block(col, vals) -> jax.Array:
                target_slice = jax.lax.dynamic_slice_in_dim(
                    target, col, block_shape[1], axis=0
                )
                return jnp.einsum("ij,j->i", vals, target_slice)

            update_indices = start_row[:, None] + jnp.arange(block_shape[0])[None, :]
            result = result.at[update_indices].add(
                jax.vmap(multiply_one_block)(start_col, values)
            )
        return result

    def todense(self) -> jax.Array:
        result = jnp.zeros(self.shape)
        for block_shape, block in self.blocks.items():
            start_row, start_col = block.start_row, block.start_col
            assert len(start_row.shape) == 1
            assert len(start_col.shape) == 1
            values = block.values
            assert values.shape == (len(start_row), *block_shape)

            row_indices = start_row[:, None] + jnp.arange(block_shape[0])[None, :]
            col_indices = start_col[:, None] + jnp.arange(block_shape[1])[None, :]
            result = result.at[row_indices, col_indices].set(values)

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
