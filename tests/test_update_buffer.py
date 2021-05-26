import dataclasses

import numpy as onp
import pytest

import jaxfg


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass(
    # frozen=True will do nothing for registered dataclasses
    frozen=True
)
class Foo:
    array: jaxfg.hints.Array


@jaxfg.utils.register_dataclass_pytree
@dataclasses.dataclass
class Bar:
    child: Foo
    array: jaxfg.hints.Array
    array_unchanged: jaxfg.hints.Array


def test_replace_context():
    obj = Bar(
        child=Foo(array=onp.zeros(3)), array=onp.ones(3), array_unchanged=onp.ones(3)
    )

    # Registered dataclasses are generally immutable!
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)

    # But we can use a context that copies a dataclass and temporarily makes the copy
    # mutable:
    with jaxfg.utils.replace_context(obj) as obj:
        # Updates can then very easily be applied!
        obj.array = onp.zeros(3)
        obj.child.array = onp.ones(3)

        # Shapes can be validated...
        with pytest.raises(AssertionError):
            obj.child.array = onp.ones(1)

        # As well as dtypes
        with pytest.raises(AssertionError):
            obj.child.array = onp.ones(3, dtype=onp.int32)

    # Outside of the replace context, the copied object becomes immutable:
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.array = onp.zeros(3)

    onp.testing.assert_allclose(obj.array, onp.zeros(3))
    onp.testing.assert_allclose(obj.array_unchanged, onp.ones(3))
    onp.testing.assert_allclose(obj.child.array, onp.ones(3))
