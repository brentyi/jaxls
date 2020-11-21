import time

import jax
import jax.numpy as jnp

# import numpy as jnp
from jax import jit

# jit = lambda x: x


N = 1000
m = 40
n = 40


def x_vectorized():
    A = jnp.ones((N, m, n))
    x = jnp.ones((n, 1))
    return A @ x


def x_loop():
    A = jnp.ones((N, m, n))
    x = jnp.ones((n, 1))

    b = jnp.zeros((N, m, 1))
    for i, Ai in enumerate(A):
        b.at[i].set(Ai @ x)
    return b


def x_scan():
    A = jnp.ones((N, m, n))
    x = jnp.ones((n, 1))

    def f(carry, x):
        y = A[carry] @ x
        return carry + 1, y

    return jax.lax.scan(f=f, init=0, xs=A)[1]


def benchmark(**kwargs):
    for name, func in kwargs.items():
        print()
        print()
        print(name)
        print("=======")

        start_time = time.time()
        func()
        print("Dynamic runtime: ", time.time() - start_time)

        start_time = time.time()
        func = jit(func)
        func()
        print("JIT compile time: ", time.time() - start_time)

        start_time = time.time()
        func()
        print("JIT runtime: ", time.time() - start_time)


jnp.zeros(5)
benchmark(
    x_vectorized=x_vectorized,
    x_loop=x_loop,
    x_scan=x_scan,
)


# def x_vectorized():
#     N = 10000000
#     a = jnp.ones(N)
#     b = jnp.ones(N)
#     return a + b
#
#
# def x_slow():
#     N = 10000000
#     a = jnp.ones(N)
#     b = jnp.ones(N)
#
#     c = a + b
#     return c
#
#
# iters = 100
#
# start_time = time.time()
# for _ in range(iters):
#     x_slow()
# print("x_slow time delta: ", time.time() - start_time)
#
#
# start_time = time.time()
# for _ in range(iters):
#     x_vectorized()
# print("x_vectorized time delta: ", time.time() - start_time)
#
#
# x_slow_jit = jit(x_slow)
# x_slow_jit()
# start_time = time.time()
# for _ in range(iters):
#     x_slow_jit()
# print("x_slow jit time delta: ", time.time() - start_time)
#
#
# x_vectorized_jit = jit(x_vectorized)
# x_vectorized_jit()
# start_time = time.time()
# for _ in range(iters):
#     x_vectorized_jit()
# print("x_vectorized jit time delta: ", time.time() - start_time)
