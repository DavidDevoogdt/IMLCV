from parsl import python_app

from configs.vsc_stevin import config
from functools import partial, wraps

config()


def unpack(num):
    def _unpack(fun):
        @wraps(fun)
        def inner(*args, **kwargs):
            fut = fun(*args, *kwargs)

            @python_app
            def __unpack(fut, ind):
                return fut[ind]

            return (*[__unpack(fut, i) for i in range(num)],)

        return inner

    return _unpack


@unpack(num=2)
@python_app
def app_random():
    import random
    import time

    time.sleep(2.0)

    return random.random(), random.random()


results_x = []
results_y = []
for i in range(0, 10):
    x, y = app_random()
    results_x.append(x)
    results_y.append(y)


@python_app
def sum(*x):
    out = 0
    for i in x:
        out += i
    return out


print(sum(*results_x).result())
print(sum(*results_y).result())
