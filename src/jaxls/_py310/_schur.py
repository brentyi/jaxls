from __future__ import annotations

import dataclasses
from typing import Any
from typing_extensions import assert_never

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp

from .utils import _batched_gram, _batched_matmul, _batched_outer_last


class _TracedVariableIdsError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class _TypeInfo:
    count: Any
    dim: Any
    orig_start: Any
    start: Any


@jdc.pytree_dataclass
class _Slot:
    type_index: jdc.Static[Any]
    is_kept: jdc.Static[Any]
    col_start: jdc.Static[Any]
    index: Any


@jdc.pytree_dataclass
class _Combo:
    kept_type_index: jdc.Static[Any]
    elim_type_index: jdc.Static[Any]
    entries: jdc.Static[Any]
    kept_index: Any
    elim_index: Any


@jdc.pytree_dataclass
class _CrossPairs:
    combo_a: jdc.Static[Any]
    combo_b: jdc.Static[Any]
    obs_a: Any
    obs_b: Any
    block_id: Any
    block_rows: Any
    block_cols: Any
    num_blocks: jdc.Static[Any]


@jdc.pytree_dataclass
class EliminationPlan:
    kept_types: jdc.Static[Any]
    elim_types: jdc.Static[Any]
    group_slots: Any
    combos: Any
    pairs: Any
    reduced_dim: jdc.Static[Any]
    tangent_dim: jdc.Static[Any]


def infer_eliminate(
    problem: Any,
) -> Any:

    slots_per_group = [
        {
            var_type: ids.shape[-1]
            for var_type, ids in cost.sorted_ids_from_var_type.items()
        }
        for cost in problem._stacked_costs
    ]

    def total_elim_slots_ok(candidate: Any) -> Any:
        return all(
            sum(slots.get(var_type, 0) for var_type in candidate) <= 1
            for slots in slots_per_group
        )

    sizes = {
        var_type: (int(ids.shape[0]) * var_type.tangent_dim, int(ids.shape[0]))
        for var_type, ids in problem._sorted_ids_from_var_type.items()
    }

    candidates = sorted(sizes, key=lambda t: sizes[t], reverse=True)

    chosen = list()
    for var_type in candidates:
        if len(chosen) + 1 == len(candidates):
            break
        if total_elim_slots_ok(chosen + [var_type]):
            chosen.append(var_type)

    eliminated_dim = sum(sizes[t][0] for t in chosen)
    if eliminated_dim * 2 < problem._tangent_dim:
        return ()
    return tuple(chosen)


def build_elimination_plan(
    problem: Any,
    eliminate: Any,
) -> Any:

    elim_set = list()
    for var_type in eliminate:
        if var_type not in elim_set:
            elim_set.append(var_type)

    for var_type in elim_set:
        if var_type not in problem._sorted_ids_from_var_type:
            raise ValueError(
                f"eliminate={var_type.__name__} was requested, but no variables "
                "of this type exist in the problem."
            )

    def _concrete(array: Any) -> Any:
        try:
            return onp.asarray(array)
        except Exception as e:
            raise _TracedVariableIdsError(
                "Variable elimination requires concrete variable IDs; the "
                "problem appears to be traced inside jax.jit."
            ) from e

    kept_types = list()
    elim_types = list()
    kept_pos = dict()
    elim_pos = dict()
    reduced_dim = 0
    for var_type, ids in problem._tangent_ordering.ordered_dict_items(
        problem._sorted_ids_from_var_type
    ):
        (count,) = ids.shape
        if var_type in elim_set:
            elim_pos[var_type] = len(elim_types)
            elim_types.append(
                _TypeInfo(
                    count=count,
                    dim=var_type.tangent_dim,
                    orig_start=problem._tangent_start_from_var_type[var_type],
                    start=0,
                )
            )
        else:
            kept_pos[var_type] = len(kept_types)
            kept_types.append(
                _TypeInfo(
                    count=count,
                    dim=var_type.tangent_dim,
                    orig_start=problem._tangent_start_from_var_type[var_type],
                    start=reduced_dim,
                )
            )
            reduced_dim += count * var_type.tangent_dim

    if len(kept_types) == 0:
        raise ValueError(
            "All variable types were marked for elimination; at least one "
            "type must be kept to form the reduced system."
        )

    group_slots = list()
    slot_indices_onp = list()
    for cost in problem._stacked_costs:
        slots = list()
        indices_onp = list()
        col = 0
        num_elim_slots = 0
        for var_type, ids in problem._tangent_ordering.ordered_dict_items(
            cost.sorted_ids_from_var_type
        ):
            ids_onp = _concrete(ids)
            sorted_ids = _concrete(problem._sorted_ids_from_var_type[var_type])
            assert ids_onp.ndim == 2
            for k in range(ids_onp.shape[-1]):
                global_index = onp.searchsorted(sorted_ids, ids_onp[:, k]).astype(
                    onp.int32
                )
                is_kept = var_type not in elim_set
                if not is_kept:
                    num_elim_slots += 1
                slots.append(
                    _Slot(
                        type_index=kept_pos[var_type]
                        if is_kept
                        else elim_pos[var_type],
                        is_kept=is_kept,
                        col_start=col,
                        index=jnp.asarray(global_index),
                    )
                )
                indices_onp.append(global_index)
                col += var_type.tangent_dim
        if num_elim_slots >= 2:
            elim_names = ", ".join(
                t.__name__ for t in cost.sorted_ids_from_var_type if t in elim_set
            )
            raise ValueError(
                f"Cost '{cost._get_name()}' couples multiple eliminated "
                f"variables ({elim_names}); the eliminated block would not be "
                "block-diagonal. Eliminate fewer variable types."
            )
        group_slots.append(tuple(slots))
        slot_indices_onp.append(indices_onp)

    combo_entries = dict()
    for g, slots in enumerate(group_slots):
        elim_positions = [p for p, s in enumerate(slots) if not s.is_kept]
        if len(elim_positions) == 0:
            continue
        (e_pos,) = elim_positions
        for p, s in enumerate(slots):
            if s.is_kept:
                key = (s.type_index, slots[e_pos].type_index)
                combo_entries.setdefault(key, []).append((g, p, e_pos))

    combos = list()
    combo_kept_onp = list()
    combo_elim_onp = list()
    for (kt_i, et_i), entries in sorted(combo_entries.items()):
        kept_idx = onp.concatenate(
            [slot_indices_onp[g][p] for g, p, _ in entries], axis=0
        )
        elim_idx = onp.concatenate(
            [slot_indices_onp[g][e] for g, _, e in entries], axis=0
        )
        combos.append(
            _Combo(
                kept_type_index=kt_i,
                elim_type_index=et_i,
                entries=tuple(entries),
                kept_index=jnp.asarray(kept_idx),
                elim_index=jnp.asarray(elim_idx),
            )
        )
        combo_kept_onp.append(kept_idx)
        combo_elim_onp.append(elim_idx)

    pairs = list()
    for ci_a in range(len(combos)):
        for ci_b in range(ci_a, len(combos)):
            if combos[ci_a].elim_type_index != combos[ci_b].elim_type_index:
                continue
            num_elim = elim_types[combos[ci_a].elim_type_index].count
            pair_a, pair_b = _matching_pairs(
                combo_elim_onp[ci_a], combo_elim_onp[ci_b], num_elim
            )
            if ci_a == ci_b:
                keep = pair_a < pair_b
                pair_a, pair_b = pair_a[keep], pair_b[keep]
            if pair_a.shape[0] == 0:
                continue
            block_rows_all = combo_kept_onp[ci_a][pair_a]
            block_cols_all = combo_kept_onp[ci_b][pair_b]
            num_cols_type = kept_types[combos[ci_b].kept_type_index].count
            keys = block_rows_all.astype(onp.int64) * num_cols_type + block_cols_all
            unique_keys, block_id = onp.unique(keys, return_inverse=True)

            order = onp.argsort(block_id, kind="stable")
            pair_a, pair_b, block_id = pair_a[order], pair_b[order], block_id[order]
            pairs.append(
                _CrossPairs(
                    combo_a=ci_a,
                    combo_b=ci_b,
                    obs_a=jnp.asarray(pair_a.astype(onp.int32)),
                    obs_b=jnp.asarray(pair_b.astype(onp.int32)),
                    block_id=jnp.asarray(block_id.astype(onp.int32)),
                    block_rows=jnp.asarray(
                        (unique_keys // num_cols_type).astype(onp.int32)
                    ),
                    block_cols=jnp.asarray(
                        (unique_keys % num_cols_type).astype(onp.int32)
                    ),
                    num_blocks=int(unique_keys.shape[0]),
                )
            )

    return EliminationPlan(
        kept_types=tuple(kept_types),
        elim_types=tuple(elim_types),
        group_slots=tuple(group_slots),
        combos=tuple(combos),
        pairs=tuple(pairs),
        reduced_dim=reduced_dim,
        tangent_dim=problem._tangent_dim,
    )


def _matching_pairs(elim_a: Any, elim_b: Any, num_elim: Any) -> Any:
    order_a = onp.argsort(elim_a, kind="stable")
    order_b = onp.argsort(elim_b, kind="stable")
    counts_a = onp.bincount(elim_a, minlength=num_elim)
    counts_b = onp.bincount(elim_b, minlength=num_elim)
    starts_a = onp.cumsum(counts_a) - counts_a
    starts_b = onp.cumsum(counts_b) - counts_b
    pairs_per_value = counts_a * counts_b
    total = int(pairs_per_value.sum())
    value_of_pair = onp.repeat(onp.arange(num_elim), pairs_per_value)
    offset = onp.arange(total) - onp.repeat(
        onp.cumsum(pairs_per_value) - pairs_per_value, pairs_per_value
    )
    a_local = offset // counts_b[value_of_pair]
    b_local = offset % counts_b[value_of_pair]
    return (
        order_a[starts_a[value_of_pair] + a_local],
        order_b[starts_b[value_of_pair] + b_local],
    )


@jdc.pytree_dataclass
class SchurFactors:
    plan: Any
    v_blocks: Any
    cross_blocks: Any
    keep_diag: Any
    hcc_dense: Any
    keep_jacs: Any
    b_keep: Any
    b_elim: Any


def prepare_schur(
    plan: Any,
    A_blocksparse: Any,
    ATb: Any,
    linear_solver: Any,
) -> Any:
    blocks_by_group = tuple(
        block_row.blocks_concat for block_row in A_blocksparse.block_rows
    )
    assert len(blocks_by_group) == len(plan.group_slots)
    dtype = ATb.dtype
    need_dense = linear_solver == "dense_cholesky"

    def slot_jac(g: Any, slot: Any) -> Any:
        info = (plan.kept_types if slot.is_kept else plan.elim_types)[slot.type_index]
        return blocks_by_group[g][:, :, slot.col_start : slot.col_start + info.dim]

    v_blocks = [
        jnp.zeros((info.count, info.dim, info.dim), dtype=dtype)
        for info in plan.elim_types
    ]
    keep_diag = [
        jnp.zeros((info.count, info.dim, info.dim), dtype=dtype)
        for info in plan.kept_types
    ]
    for g, slots in enumerate(plan.group_slots):
        for slot in slots:
            J = slot_jac(g, slot)
            gram = _batched_gram(J, J)
            target = keep_diag if slot.is_kept else v_blocks
            target[slot.type_index] = target[slot.type_index] + jax.ops.segment_sum(
                gram, slot.index, num_segments=target[slot.type_index].shape[0]
            )

    cross_blocks = list()
    for combo in plan.combos:
        parts = [
            _batched_gram(
                slot_jac(g, plan.group_slots[g][kp]),
                slot_jac(g, plan.group_slots[g][ep]),
            )
            for g, kp, ep in combo.entries
        ]
        cross_blocks.append(
            parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        )

    b_keep = jnp.concatenate(
        [
            ATb[info.orig_start : info.orig_start + info.count * info.dim]
            for info in plan.kept_types
        ]
    )
    b_elim = tuple(
        ATb[info.orig_start : info.orig_start + info.count * info.dim].reshape(
            (info.count, info.dim)
        )
        for info in plan.elim_types
    )

    hcc_dense = None
    if need_dense:
        hcc_dense = jnp.zeros((plan.reduced_dim, plan.reduced_dim), dtype=dtype)
        for kt_i, info in enumerate(plan.kept_types):
            rows = (
                info.start
                + jnp.arange(info.count)[:, None] * info.dim
                + jnp.arange(info.dim)[None, :]
            )
            hcc_dense = hcc_dense.at[rows[:, :, None], rows[:, None, :]].add(
                keep_diag[kt_i]
            )
        for g, slots in enumerate(plan.group_slots):
            kept_slots = [s for s in slots if s.is_kept]
            for a in range(len(kept_slots)):
                for b in range(a + 1, len(kept_slots)):
                    slot_a, slot_b = kept_slots[a], kept_slots[b]
                    blk = _batched_gram(slot_jac(g, slot_a), slot_jac(g, slot_b))
                    rows = _block_rows(plan.kept_types[slot_a.type_index], slot_a.index)
                    cols = _block_rows(plan.kept_types[slot_b.type_index], slot_b.index)
                    hcc_dense = hcc_dense.at[rows[:, :, None], cols[:, None, :]].add(
                        blk
                    )
                    hcc_dense = hcc_dense.at[cols[:, :, None], rows[:, None, :]].add(
                        jnp.swapaxes(blk, 1, 2)
                    )

    keep_jacs = list()
    if linear_solver == "conjugate_gradient":
        for g, slots in enumerate(plan.group_slots):
            kept = [slot_jac(g, s) for s in slots if s.is_kept]
            if len(kept) == 0:
                keep_jacs.append(None)
            else:
                keep_jacs.append(
                    kept[0] if len(kept) == 1 else jnp.concatenate(kept, axis=2)
                )
    else:
        keep_jacs = [None] * len(plan.group_slots)

    return SchurFactors(
        plan=plan,
        v_blocks=tuple(v_blocks),
        cross_blocks=tuple(cross_blocks),
        keep_diag=tuple(keep_diag),
        hcc_dense=hcc_dense,
        keep_jacs=tuple(keep_jacs),
        b_keep=b_keep,
        b_elim=b_elim,
    )


def _block_rows(info: Any, index: Any) -> Any:
    return info.start + index[:, None] * info.dim + jnp.arange(info.dim)[None, :]


def _type_slice(x: Any, info: Any) -> Any:
    return x[info.start : info.start + info.count * info.dim].reshape(
        (info.count, info.dim)
    )


def _invert_psd_blocks(blocks: Any) -> Any:
    dim = blocks.shape[-1]
    if dim == 1:
        return 1.0 / blocks
    if dim == 2:
        a = blocks[..., 0, 0]
        b = blocks[..., 0, 1]
        c = blocks[..., 1, 0]
        d = blocks[..., 1, 1]
        det = a * d - b * c
        inv = jnp.stack(
            [
                jnp.stack([d, -b], axis=-1),
                jnp.stack([-c, a], axis=-1),
            ],
            axis=-2,
        )
        return inv / det[..., None, None]
    if dim == 3:
        m = blocks
        c00 = m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1]
        c01 = m[..., 1, 2] * m[..., 2, 0] - m[..., 1, 0] * m[..., 2, 2]
        c02 = m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]
        c10 = m[..., 0, 2] * m[..., 2, 1] - m[..., 0, 1] * m[..., 2, 2]
        c11 = m[..., 0, 0] * m[..., 2, 2] - m[..., 0, 2] * m[..., 2, 0]
        c12 = m[..., 0, 1] * m[..., 2, 0] - m[..., 0, 0] * m[..., 2, 1]
        c20 = m[..., 0, 1] * m[..., 1, 2] - m[..., 0, 2] * m[..., 1, 1]
        c21 = m[..., 0, 2] * m[..., 1, 0] - m[..., 0, 0] * m[..., 1, 2]
        c22 = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
        det = m[..., 0, 0] * c00 + m[..., 0, 1] * c01 + m[..., 0, 2] * c02
        cof = jnp.stack(
            [
                jnp.stack([c00, c10, c20], axis=-1),
                jnp.stack([c01, c11, c21], axis=-1),
                jnp.stack([c02, c12, c22], axis=-1),
            ],
            axis=-2,
        )
        return cof / det[..., None, None]
    return jnp.linalg.inv(blocks)


def _damped_vinv(factors: Any, lambd: Any) -> Any:
    out = list()
    damp = jnp.maximum(lambd, 1e-10)
    for v in factors.v_blocks:
        dim = v.shape[-1]
        out.append(_invert_psd_blocks(v + damp * jnp.eye(dim, dtype=v.dtype)))
    return out


def _reduced_rhs(factors: Any, vinv: Any) -> Any:
    plan = factors.plan
    b_red = factors.b_keep
    u = [
        jnp.einsum("nef,nf->ne", vinv_e, b_e)
        for vinv_e, b_e in zip(vinv, factors.b_elim)
    ]
    for combo, cross in zip(plan.combos, factors.cross_blocks):
        info = plan.kept_types[combo.kept_type_index]
        w = jnp.einsum("mte,me->mt", cross, u[combo.elim_type_index][combo.elim_index])
        contrib = jax.ops.segment_sum(w, combo.kept_index, num_segments=info.count)
        b_red = b_red.at[info.start : info.start + info.count * info.dim].add(
            -contrib.flatten()
        )
    return b_red


def _wvwt_terms(factors: Any, vinv: Any) -> Any:
    plan = factors.plan
    y_blocks = list()
    diag = [
        jnp.zeros((info.count, info.dim, info.dim), dtype=factors.b_keep.dtype)
        for info in plan.kept_types
    ]
    for combo, cross in zip(plan.combos, factors.cross_blocks):
        y = _batched_matmul(cross, vinv[combo.elim_type_index][combo.elim_index])
        y_blocks.append(y)
        kt_i = combo.kept_type_index
        diag[kt_i] = diag[kt_i] + jax.ops.segment_sum(
            _batched_outer_last(y, cross),
            combo.kept_index,
            num_segments=plan.kept_types[kt_i].count,
        )
    return y_blocks, diag


def _assemble_dense_S(factors: Any, lambd: Any, vinv: Any) -> Any:
    plan = factors.plan
    assert factors.hcc_dense is not None
    diag_idx = jnp.arange(plan.reduced_dim)
    S = factors.hcc_dense.at[diag_idx, diag_idx].add(lambd)

    y_blocks, wvwt_diag = _wvwt_terms(factors, vinv)
    for pair in plan.pairs:
        combo_a = plan.combos[pair.combo_a]
        combo_b = plan.combos[pair.combo_b]
        info_a = plan.kept_types[combo_a.kept_type_index]
        info_b = plan.kept_types[combo_b.kept_type_index]
        blk = _batched_outer_last(
            y_blocks[pair.combo_a][pair.obs_a],
            factors.cross_blocks[pair.combo_b][pair.obs_b],
        )
        summed = jax.ops.segment_sum(
            blk,
            pair.block_id,
            num_segments=pair.num_blocks,
            indices_are_sorted=True,
        )
        rows = _block_rows(info_a, pair.block_rows)
        cols = _block_rows(info_b, pair.block_cols)

        S = S.at[rows[:, :, None], cols[:, None, :]].add(-summed)
        S = S.at[cols[:, :, None], rows[:, None, :]].add(-jnp.swapaxes(summed, 1, 2))
    for kt_i, info in enumerate(plan.kept_types):
        rows = (
            info.start
            + jnp.arange(info.count)[:, None] * info.dim
            + jnp.arange(info.dim)[None, :]
        )
        S = S.at[rows[:, :, None], rows[:, None, :]].add(-wvwt_diag[kt_i])
    return S


def _solve_spd_scaled(S: Any, b: Any) -> Any:
    diag = jnp.diagonal(S)

    scale = jnp.sqrt(jnp.maximum(jnp.abs(diag), jnp.finfo(S.dtype).tiny))
    S_scaled = S / (scale[:, None] * scale[None, :])
    eps = jnp.finfo(S.dtype).eps
    floor = eps * (2e4 if S.dtype == jnp.float32 else 4.0)
    diag_idx = jnp.arange(S.shape[0])
    S_scaled = S_scaled.at[diag_idx, diag_idx].add(floor)
    factor = jax.scipy.linalg.cho_factor(S_scaled)
    return jax.scipy.linalg.cho_solve(factor, b / scale) / scale


def _back_substitute(factors: Any, vinv: Any, dc: Any) -> Any:
    plan = factors.plan
    wt_dc = [
        jnp.zeros((info.count, info.dim), dtype=dc.dtype) for info in plan.elim_types
    ]
    for combo, cross in zip(plan.combos, factors.cross_blocks):
        info = plan.kept_types[combo.kept_type_index]
        dc_obs = _type_slice(dc, info)[combo.kept_index]
        et_i = combo.elim_type_index
        wt_dc[et_i] = wt_dc[et_i] + jax.ops.segment_sum(
            jnp.einsum("mte,mt->me", cross, dc_obs),
            combo.elim_index,
            num_segments=plan.elim_types[et_i].count,
        )

    delta = jnp.zeros(plan.tangent_dim, dtype=dc.dtype)
    for info in plan.kept_types:
        delta = delta.at[info.orig_start : info.orig_start + info.count * info.dim].set(
            dc[info.start : info.start + info.count * info.dim]
        )
    for et_i, info in enumerate(plan.elim_types):
        dl = jnp.einsum("nef,nf->ne", vinv[et_i], factors.b_elim[et_i] - wt_dc[et_i])
        delta = delta.at[info.orig_start : info.orig_start + info.count * info.dim].set(
            dl.flatten()
        )
    return delta


def solve_schur_dense(factors: Any, lambd: Any) -> Any:
    vinv = _damped_vinv(factors, lambd)
    b_red = _reduced_rhs(factors, vinv)
    S = _assemble_dense_S(factors, lambd, vinv)
    dc = _solve_spd_scaled(S, b_red)
    return _back_substitute(factors, vinv, dc)


def solve_schur_cg(
    factors: Any,
    lambd: Any,
    cg_config: Any,
    prev_state: Any,
) -> Any:
    from ._solvers import _ConjugateGradientState

    plan = factors.plan

    lam_eff = lambd + 1e-5
    vinv = _damped_vinv(factors, lam_eff)
    b_red = _reduced_rhs(factors, vinv)

    if cg_config.preconditioner is None:
        precondition = lambda x: x
    else:
        _, wvwt_diag = _wvwt_terms(factors, vinv)
        diag_blocks = [
            factors.keep_diag[kt_i]
            + (lam_eff + 1e-6) * jnp.eye(info.dim, dtype=b_red.dtype)
            - wvwt_diag[kt_i]
            for kt_i, info in enumerate(plan.kept_types)
        ]
        if cg_config.preconditioner == "block_jacobi":
            precond_inv = [jnp.linalg.inv(block) for block in diag_blocks]

            def precondition(x: Any) -> Any:
                parts = list()
                for kt_i, info in enumerate(plan.kept_types):
                    xs = _type_slice(x, info)
                    parts.append(
                        jnp.einsum("nij,nj->ni", precond_inv[kt_i], xs).flatten()
                    )
                return jnp.concatenate(parts)
        elif cg_config.preconditioner == "point_jacobi":
            inv_diag = [
                1.0 / jnp.diagonal(block, axis1=1, axis2=2) for block in diag_blocks
            ]

            def precondition(x: Any) -> Any:
                parts = list()
                for kt_i, info in enumerate(plan.kept_types):
                    parts.append((_type_slice(x, info) * inv_diag[kt_i]).flatten())
                return jnp.concatenate(parts)
        else:
            assert_never(cg_config.preconditioner)

    def matvec(x: Any) -> Any:
        out = lam_eff * x

        for g, slots in enumerate(plan.group_slots):
            keep_jac = factors.keep_jacs[g]
            if keep_jac is None:
                continue
            kept_slots = [s for s in slots if s.is_kept]
            gathered = list()
            for slot in kept_slots:
                info = plan.kept_types[slot.type_index]
                gathered.append(_type_slice(x, info)[slot.index])
            x_cat = (
                gathered[0] if len(gathered) == 1 else jnp.concatenate(gathered, axis=1)
            )
            residual_space = jnp.einsum("crk,ck->cr", keep_jac, x_cat)
            back = jnp.einsum("crk,cr->ck", keep_jac, residual_space)
            col = 0
            for slot in kept_slots:
                info = plan.kept_types[slot.type_index]
                contrib = jax.ops.segment_sum(
                    back[:, col : col + info.dim],
                    slot.index,
                    num_segments=info.count,
                )
                out = out.at[info.start : info.start + info.count * info.dim].add(
                    contrib.flatten()
                )
                col += info.dim

        wt_x = [
            jnp.zeros((info.count, info.dim), dtype=x.dtype) for info in plan.elim_types
        ]
        for combo, cross in zip(plan.combos, factors.cross_blocks):
            info = plan.kept_types[combo.kept_type_index]
            xs = _type_slice(x, info)[combo.kept_index]
            et_i = combo.elim_type_index
            wt_x[et_i] = wt_x[et_i] + jax.ops.segment_sum(
                jnp.einsum("mte,mt->me", cross, xs),
                combo.elim_index,
                num_segments=plan.elim_types[et_i].count,
            )
        z = [jnp.einsum("nef,nf->ne", vinv_e, wt_e) for vinv_e, wt_e in zip(vinv, wt_x)]
        for combo, cross in zip(plan.combos, factors.cross_blocks):
            info = plan.kept_types[combo.kept_type_index]
            back = jnp.einsum(
                "mte,me->mt", cross, z[combo.elim_type_index][combo.elim_index]
            )
            contrib = jax.ops.segment_sum(
                back, combo.kept_index, num_segments=info.count
            )
            out = out.at[info.start : info.start + info.count * info.dim].add(
                -contrib.flatten()
            )
        return out

    b_norm = jnp.linalg.norm(b_red)
    eta = jnp.minimum(
        cg_config.eisenstat_walker_gamma
        * (b_norm / (prev_state.ATb_norm_prev + 1e-7))
        ** cg_config.eisenstat_walker_alpha,
        cg_config.tolerance_max,
    )
    eta = jnp.maximum(cg_config.tolerance_min, jnp.minimum(eta, prev_state.eta))

    dc, _ = jax.scipy.sparse.linalg.cg(
        A=matvec,
        b=b_red,
        x0=jnp.zeros_like(b_red),
        maxiter=plan.reduced_dim,
        tol=eta,
        M=precondition,
    )
    delta = _back_substitute(factors, vinv, dc)
    return delta, _ConjugateGradientState(ATb_norm_prev=b_norm, eta=eta)
