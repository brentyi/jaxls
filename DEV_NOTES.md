# Development Notes

## Commands

```bash
# Type checking.
uv run --extra dev --extra examples pyright ./src ./examples ./tests

# Linting and formatting.
uv run --extra dev --extra examples ruff check --fix .
uv run --extra dev --extra examples ruff format .

# Run tests.
uv run --extra dev pytest tests/

# Transpile for Python 3.10/3.11 compatibility.
uv run --extra dev ./transpile_py310.py
```

## Style Guidelines

### Use `jdc.copy_and_mutate` instead of `jdc.replace`

```python
# Bad - **kwargs in replace() can't be type-checked.
new_obj = jdc.replace(obj, field=value)

# Good - preserves types.
with jdc.copy_and_mutate(obj) as new_obj:
    new_obj.field = value
```

### Avoid truthy/falsey checks

```python
# Bad.
if my_list:
if my_value:

# Good.
if len(my_list) > 0:
if my_value is not None:
if my_value != 0:
```

### JIT compatibility

All code paths must work with traced arrays. Array values cannot affect control flow.

```python
# Bad - breaks tracing.
if array_value > 0:
    return x
else:
    return y

# Good - use jax.lax.cond or jnp.where.
return jnp.where(array_value > 0, x, y)
```
