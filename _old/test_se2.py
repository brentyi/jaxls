import jaxfg
import numpy as onp
from jax import numpy as jnp
from scipy.linalg import expm, logm

for _ in range(5):
    tangent = onp.random.randn(3)
    exp = jaxfg.utils.SE2.exp(tangent)
    assert exp.shape == (3, 3)
    log_exp = jaxfg.utils.SE2.log(exp)
    assert log_exp.shape == (3,)
    onp.testing.assert_allclose(tangent, log_exp)

#
# print(recovered)
# print(recovered[0])
