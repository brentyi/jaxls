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
    """Column indices of the start of each block. Shape in tuple should be `(num_blocks,)`."""
    blocks: tuple[jax.Array, ...]
    """Blocks of matrix. Shape in tuple should be `(num_blocks, rows, cols)`."""

    def treedef(self) -> Hashable:
        return tuple(block.shape for block in self.blocks)


@jdc.pytree_dataclass
class BlockRowSparseMatrix:
    block_rows: tuple[MatrixBlockRow, ...]
    """Batched blocks. Each element in the tuple has a leading axis, which
    represents consecutive block rows."""
    shape: jdc.Static[tuple[int, int]]
    """Shape of matrix."""

    def multiply(self, target: jax.Array) -> jax.Array:
        """Sparse-dense multiplication."""
        assert target.ndim == 1

        def multiply_one_block_row(
            start_cols: tuple[jax.Array, ...], blocks: tuple[jax.Array, ...]
        ) -> jax.Array:
            vecs = list[jax.Array]()
            for start_col, block in zip(start_cols, blocks):
                vecs.append(
                    jnp.einsum(
                        "ij,j->i",
                        block,
                        jax.lax.dynamic_slice_in_dim(target, start_col, block.shape[1]),
                    )
                )
            return jax.tree.reduce(jnp.add, vecs)

        out_slices = []
        for block_row in self.block_rows:
            # Do matrix multiplies for all blocks in block-row.
            vecs = jax.vmap(multiply_one_block_row)(
                block_row.start_cols, block_row.blocks
            )
            proto_block = block_row.blocks[0]
            assert proto_block.ndim == 3  # (batch, rows, cols)
            assert vecs.shape == (proto_block.shape[0], proto_block.shape[1])
            out_slices.append(vecs.flatten())
            assert block_row.start_row.shape == (vecs.shape[0],)

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
