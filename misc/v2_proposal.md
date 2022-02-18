# `jaxfg` API v2 Proposal

Status: Draft. Not everything makes sense yet!

Last update: September 28, 2021.

## Preliminaries

For the purposes of this doc, we'll be working with the factor graph defined by
our [simple pose graph optimization example](../scripts/pose_graph_simple.py).
It has two SE(2) variables with costs induced by three factors:

```
    ┌────────┐             ┌────────┐
    │ Pose 0 ├───Between───┤ Pose 1 │
    └───┬────┘             └────┬───┘
        │                       │
        │                       │
      Prior                   Prior
```

All code snippets below will assume the following imports:

```python
from typing import List, Type

import jaxlie
import jax_dataclasses as jdc
from jax import numpy as jnp

import jaxfg
```

## Summary of current interface

We'll start off by highlighting the current interface. The general workflow for
defining and solving factor graphs is as follows:

1. **Create variables.**

   Each variable object represents something that we want to solve for.

   The variable objects themselves don't hold any values, but can be used as a
   key for accessing values from a VariableAssignments object. (see below)

   ```python
   pose_variables: List[jaxfg.geometry.SE2Variable] = [
       jaxfg.geometry.SE2Variable(),
       jaxfg.geometry.SE2Variable(),
   ]
   ```

2. **Create factors.** Each defines a conditional probability distribution over
   some subset of our variables.

   ```python
   factors: List[jaxfg.core.FactorBase] = [
       jaxfg.geometry.PriorFactor.make(
           variable=pose_variables[0],
           mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
           noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
       ),
       jaxfg.geometry.PriorFactor.make(
           variable=pose_variables[1],
           mu=jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0),
           noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
       ),
       jaxfg.geometry.BetweenFactor.make(
           variable_T_world_a=pose_variables[0],
           variable_T_world_b=pose_variables[1],
           T_a_b=jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0),
           noise_model=jaxfg.noises.DiagonalGaussian(jnp.ones(3)),
       ),
   ]
   ```

3. **Create a "stacked" factor graph.** (this is the only kind of factor graph)

   We loop through the factors and preprocess them to enable vectorization of
   computations. If we have 1000 PriorFactor objects, we stack all of the
   associated values and perform a batched operation that computes all 1000
   residuals.

   ```python
   graph = jaxfg.core.StackedFactorGraph.make(factors)
   ```

4. **Solve.**

   We first create an initial assignments object, which you can think of as a
   (variable => value) mapping. These initial values will be used by our
   nonlinear optimizer.

   We just use each variables' default values here -- SE(2) identity -- but for
   bigger problems bad initializations => no convergence when we run our
   nonlinear optimizer.

   The `graph.solve()` function then runs an optimizer (Gauss-Newton by default)
   to produce a set of solution assignments.

   ```python
   initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables)
   solution_assignments = graph.solve(initial_assignments)
   ```

5. **Read solutions.**

   Finally, using the original variable objects as keys, we can pull variable
   values out of our assignments object.

   To grab and print a single variable value at a time:

   ```python
   print("First pose (jaxlie.SE2 object):")
   print(solution_assignments.get_value(pose_variables[0]))

   print("Second pose (jaxlie.SE2 object):")
   print(solution_assignments.get_value(pose_variables[1]))

   ```

   It can also be useful to grab all values of a single type at once:

   ```python
   print("All poses (still a jaxlie.SE2 object, but underlying parameters are stacked):")
   print(solution_assignments.get_stacked_value(jaxfg.geometry.SE2Variable))
   ```

## Goals

We've been using the current jaxfg interface in various projects and it's been
super powerful as-is! However, we identified a few limitations that need to be
addressed:

1. **Dynamic updates to factor graphs are slow.** A core limitation that we’ve
   run into is with building “dynamic” graphs using our factor graph
   optimization infrastructure. What we mean by this: computations for graphs
   are implemented as JIT-compiled functions, which require fixed sequences of
   operations on arrays with fixed shapes. This makes it tough to dynamically
   add or remove variables and factors without triggering a new JIT compilation.

2. **The update interface for stacked graphs is a bit opaque.** Updating a
   `StackedFactorGraph` object directly is generally far more efficient than
   creating new factors and re-stacking. Implementing `jdc.copy_and_mutate()`
   helped with this enormously, but the interface is somewhat error-prone and
   generally not super intuitive.

3. **Marginalization.** We have an experimental implementation for computing
   marginal covariance matrices, but it currently doesn't scale well.

The purpose of the v2 interface will primarily be to address points (1) and (2);
resolving (3) we believe is an orthogonal design and engineering effort that's
agnostic to the proposed changes.

## Overview of changes

Toward the goals above, we're proposing the following changes for the jaxfg v2
API:

- Rather than using order-agnostic hashable objects as variable identifiers,
  switch to an enumerated/indexed approach, where variables are identified by
  `(type: Type[jaxfg.core.VariableBase], index: int)` pairs. The reasoning for
  this is that it would reduce edges on the graph to simple integer indices,
  which could be changed dynamically without recompilation.
- Support solves when a partial set of the factors or variables are disabled.
- We should elevate the concepts of stacked factors to the API level, rather
  than leaving

1. With the new interface, we would be begin by creating an empty factor graph:

   ```python
   graph = jaxfg.core.FactorGraph()
   ```

2. Variables can then be allocated by type, without creating any new objects:

   ```python
   num_poses: int = 2
   graph = graph.add_variables(jaxfg.geometry.SE2Variable, count=num_poses)
   ```

3. Factors would be first manually stacked, and then added to the graph:

   **Draft; may not make total sense.**

   ```python
   stacked_prior_factor = jaxfg.stacked_factor_from_duplicates(
       jaxfg.geometry.PriorFactor(...),
       count=10,
   )
   stacked_between_factor = jaxfg.stack_factors(
       [
           jaxfg.geometry.BetweenFactor(...),
           jaxfg.geometry.BetweenFactor(...),
       ]
   )

   graph = (
       jaxfg.core.FactorGraph()
       .add_stacked_factor(stacked_prior_factor)
       .add_stacked_factor(stacked_between_factor)
   )
   ```

4. Finally, we can set the edges in the graph. In contrast to the current
   interface, this reduces under-the-hood to a sequence of JIT-compilable array
   operations:

   **Draft; may not make total sense.**

   ```python
    graph = (
       graph.assign_edges(
           prior_factor,
           jnp.array([0]),  # Variable 0 index
       )
       .assign_edges(
           between_factor,
           jnp.arange(num_poses),  # Variable 0 index
           jnp.arange(1, num_poses + 1),  # Variable 1 index
       )
    )
   ```

Written more succinctly, we would set up the simple pose graph optimization as
follows:

**Draft; may not make total sense. This is similar to but not consistent with
what is written above.**

```python
# Build a factor graph. This is done by starting with an empty graph, and then
# progressively adding variable nodes, factor nodes, and finally edges between variables
# and factors.
num_poses: int = 2

graph = (
    jaxfg.core.FactorGraph()
    .add_variables(jaxfg.geometry.SE2Variable, count=num_poses)
    .add_factors(
        prior_factor,
        count=1,
    )
    .add_factors(
        between_factor,
        count=(num_poses - 1),
    )
    .assign_edges(
        prior_factor,
        jnp.array([0]),  # Variable 0 index
    )
    .assign_edges(
        between_factor,
        jnp.arange(num_poses),  # Variable 0 index
        jnp.arange(1, num_poses + 1),  # Variable 1 index
    )
)
```

---

**Alternative prototype. Draft; likely doesn't make total sense.**

```python
# First, we preallocate a factor graph with the correct number of factors and variables.
graph = jaxfg.core.PreallocatedFactorGraph(
    variable_counts={
        jaxfg.geometry.SE3Variable: 5,
    },
    factor_counts={
        jaxfg.geometry.PriorFactor: 1,
        jaxfg.geometry.BetweenFactor: 4,
    },
)

# Then, we connect variables to their nodes
graph = graph.link_nodes(
    jaxfg.geometry.PriorFactor,
    jnp.array([0]),  # Variable 0 index
)
graph = graph.link_nodes(
    jaxfg.geometry.PriorFactor,
    jnp.arange(4),  # Variable 0 index
    jnp.arange(4) + 1,  # Variable 1 index
)


# Update factor parameters as needed...
with jdc.copy_and_mutate(graph) as graph:
    graph.factor_from_type(
        jaxfg.geometry.PriorFactor
    ).mu = some_batched_transformation

    graph.factor_from_type(
        jaxfg.geometry.BetweenFactor
    ).T_a_b = some_batched_transformation

# MAP inference
solution: jaxfg.core.VariableAssignments = graph.compute_map_estimate()

# Read variables
print("All variables:", solution.get_stacked_value(jaxfg.geometry.SE3Variable))
print("Pose 0:", solution.get_value(jaxfg.geometry.SE3Variable, 0))
```
