import dataclasses
from typing import NamedTuple

import jax
from jax import numpy as jnp


class Ray(NamedTuple):
    origin: jnp.ndarray
    direction: jnp.ndarray


@jax.jit
def get_unit_x() -> Ray:
    return Ray(origin=jnp.zeros(2), direction=jnp.array([1.0, 0.0]))


print(get_unit_x())
