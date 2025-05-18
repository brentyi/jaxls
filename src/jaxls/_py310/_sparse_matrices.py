from __future__ import annotations
from typing import Any


import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp


@jdc.pytree_dataclass
class SparseBlockRow:
    num_cols: jdc.Static[Any]
    start_cols: Any
    block_num_cols: jdc.Static[Any]
    blocks_concat: Any

    def treedef(self) -> Any:
        return tuple(block.shape for block in self.blocks_concat)

    def to_dense(self) -> Any:
        if self.blocks_concat.ndim == 3:
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
    block_rows: Any
    shape: jdc.Static[Any]

    def multiply(self, target: Any) -> Any:
        assert target.ndim == 1

        out_slices = []
        for block_row in self.block_rows:
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            target_slice_parts = list()
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

            target_slice = jnp.concatenate(target_slice_parts, axis=1)
            assert target_slice.shape == (n_block, block_nz_cols)

            out_slices.append(
                jnp.einsum(
                    "bij,bj->bi", block_row.blocks_concat, target_slice
                ).flatten()
            )

        result = jnp.concatenate(out_slices, axis=0)
        return result

    def compute_column_norms(self) -> Any:
        squared_sum = jnp.zeros(self.shape[1])
        for block_row in self.block_rows:
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            block_row_squared_sum = jnp.sum(block_row.blocks_concat**2, axis=-2)
            assert block_row_squared_sum.shape == (n_block, block_nz_cols)
            block_nz_col_idx = 0
            for start_cols, width in zip(
                block_row.start_cols, block_row.block_num_cols
            ):
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

    def scale_columns(self, scales: Any) -> Any:
        assert scales.ndim == 1
        assert scales.shape[0] == self.shape[1]
        scaled_block_rows = list()
        for block_row in self.block_rows:
            (n_block, block_rows, block_nz_cols) = block_row.blocks_concat.shape
            del block_rows

            assert len(block_row.start_cols) == len(block_row.block_num_cols)
            scale_slice_parts = list()
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

            scale_slice = jnp.concatenate(scale_slice_parts, axis=1)
            assert scale_slice.shape == (n_block, block_nz_cols)

            with jdc.copy_and_mutate(block_row) as scaled_block_row:
                scaled_block_row.blocks_concat = (
                    block_row.blocks_concat * scale_slice[:, None, :]
                )
            scaled_block_rows.append(scaled_block_row)

        return BlockRowSparseMatrix(tuple(scaled_block_rows), shape=self.shape)

    def to_dense(self) -> Any:
        out = jnp.concatenate(
            [block_row.to_dense() for block_row in self.block_rows],
            axis=0,
        )
        assert out.shape == self.shape
        return out


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    indices: Any
    indptr: Any
    shape: jdc.Static[Any]


@jdc.pytree_dataclass
class SparseCsrMatrix:
    values: Any
    coords: Any

    def as_jax_bcsr(self) -> Any:
        return jax.experimental.sparse.BCSR(
            args=(self.values, self.coords.indices, self.coords.indptr),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )


@jdc.pytree_dataclass
class SparseCooCoordinates:
    rows: Any
    cols: Any
    shape: jdc.Static[Any]


@jdc.pytree_dataclass
class SparseCooMatrix:
    values: Any
    coords: Any

    def as_jax_bcoo(self) -> Any:
        return jax.experimental.sparse.BCOO(
            args=(
                self.values,
                jnp.stack([self.coords.rows, self.coords.cols], axis=-1),
            ),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )
