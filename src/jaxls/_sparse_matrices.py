from __future__ import annotations

import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class BatchedBlock:
    start_row: jax.Array
    """Starting row index for each block. Shape should be `(num_blocks,)`."""
    start_col: jax.Array
    """Starting column index for each block. Shape should be `(num_blocks,)`."""
    values: jax.Array
    """Values of blocks. Shape should be `(num_blocks, blow_rows, block_cols)`."""


@jdc.pytree_dataclass
class BatchedBlockSparseMatrix:
    """A block-sparse matrix data structure, where blocks of the same shape are batched."""

    blocks: dict[tuple[int, int], BatchedBlock]
    """Map from block shape to block (values, start row, start col)."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""

    def get_ATA_diagonals(self) -> jax.Array:
        output = jnp.zeros(self.shape[1])
        for block_shape, batched_block in self.blocks.items():
            assert batched_block.values.shape == (
                len(batched_block.start_row),
                *block_shape,
            )
            output = output.at[
                batched_block.start_col[:, None] + jnp.arange(block_shape[1])[None, :]
            ].add(jnp.sum(batched_block.values**2, axis=1))
        return output

    def transpose(self) -> BatchedBlockSparseMatrix:
        new_blocks = {}
        for block_shape, block in self.blocks.items():
            new_block = BatchedBlock(
                start_row=block.start_col,
                start_col=block.start_row,
                values=jnp.swapaxes(block.values, -1, -2),
            )
            new_blocks[block_shape[::-1]] = new_block
        return BatchedBlockSparseMatrix(new_blocks, (self.shape[1], self.shape[0]))

    def __matmul__(self, target: jax.Array) -> jax.Array:
        result = jnp.zeros(self.shape[0])
        for block_shape, batched_block in self.blocks.items():
            start_row, start_col = batched_block.start_row, batched_block.start_col
            assert len(start_row.shape) == 1
            assert len(start_col.shape) == 1
            values = batched_block.values
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


# @jdc.pytree_dataclass
# class SparseCooMatrix:
#     """Sparse matrix in COO form."""
#
#     values: jax.Array
#     """Non-zero matrix values. Shape should be `(*, N)`."""
#     coords: SparseCooCoordinates
#     """Indices describing non-zero entries."""
#
#     def as_jax_bcoo(self) -> jax.experimental.sparse.BCOO:
#         return jax.experimental.sparse.BCOO(
#             args=(
#                 self.values,
#                 jnp.stack([self.coords.rows, self.coords.cols], axis=-1),
#             ),
#             shape=self.coords.shape,
#             indices_sorted=True,
#             unique_indices=True,
#         )
