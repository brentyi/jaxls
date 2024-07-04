from typing import cast

import jax
import jaxlie
from jaxfg2 import Factor, SO3Var, Var, VarValues

x = SO3Var(0)
print(VarValues.from_defaults((x,))[x])
factor = Factor.make(lambda vals, x: vals[x].log(), args=(x,))
