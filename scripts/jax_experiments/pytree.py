import time
from typing import Dict

import jax
import jax.numpy as jnp

import jaxfg

A = {
    "a": [jnp.eye(4), jnp.eye(4)],
    "b": [jnp.eye(4), jnp.eye(2)],
}
x = {
    "a": [jnp.ones(4), jnp.ones(4)],
    "b": [jnp.ones(4), jnp.ones(2)],
}

print(jax.tree_multimap(lambda a, x: a @ x, A, x))

N = 100


class Key:
    def op(self, a: jnp.ndarray) -> jnp.ndarray:
        return a + 5

    def __lt__(self, other):
        return hash(self) < hash(other)


# @jax.jit
# def op(tree: Dict[Key, jnp.ndarray]):
#     out = {}
#     for k, v in tree.items():
#         out[k] = k.op(v)
#     return out
#
#
# tree = {Key(): jnp.eye(4) for n in range(N)}
# with jaxfg.utils.stopwatch("manual build"):
#     op(tree)
#
# with jaxfg.utils.stopwatch("manual loop"):
#     for _ in range(5):
#         op(tree)


@jax.jit
def op(tree: Dict[Key, jnp.ndarray]):
    # stacked = jnp.stack(jax.tree_flatten(tree)[0], axis=0)
    stacked = tree

    def f(carry, x):
        # assert False, carry
        carry = carry
        y = jax.tree_map(lambda i: i + 5, x)
        return carry, y

    # init = 0
    init = {k: 0 for k in tree}
    out = jax.lax.scan(f, init, stacked)
    # out = jax.vmap(Key().op)(stacked)
    return out


tree = {str(n): jnp.eye(4) for n in range(N)}

# with jaxfg.utils.stopwatch("non-manual build"):
#     op({Key(): jnp.eye(4)})
#
# with jaxfg.utils.stopwatch("non-manual build"):
#     op({Key(): jnp.eye(4)})
#
with jaxfg.utils.stopwatch("non-manual build"):
    op(tree)

with jaxfg.utils.stopwatch("non-manual loop"):
    for _ in range(5):
        op(tree)

tree = {str(n): jnp.eye(4) for n in range(N)}
with jaxfg.utils.stopwatch("non-manual loop"):
    for _ in range(5):
        op(tree)
with jaxfg.utils.stopwatch("non-manual loop"):
    for _ in range(5):
        op(tree)
tree = {str(n): jnp.eye(8) + 10 for n in range(N)}
with jaxfg.utils.stopwatch("non-manual loop"):
    for _ in range(5):
        op(tree)
with jaxfg.utils.stopwatch("non-manual loop"):
    for _ in range(5):
        op(tree)
