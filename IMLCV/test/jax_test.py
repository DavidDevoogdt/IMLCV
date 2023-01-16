from functools import partial
from jax import jit, grad, vmap
import matplotlib.pyplot as plt
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def leg(x, n):
    if n == 0:
        return 1.0
    if n == 1:
        return x

    return ((2 * n - 1) * x * leg(x, n - 1) - (n - 1) * leg(x, n - 2)) / n


print(grad(lambda x: leg(x, 2))(3.0))


plt.figure()
x = jnp.arange(-1.0, 1.0, 0.0001)
for n in range(6):

    plt.plot(x, vmap(grad(lambda x: leg(x, n)))(x))

plt.show()

print("done")
