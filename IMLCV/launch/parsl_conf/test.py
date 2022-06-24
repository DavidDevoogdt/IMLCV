from IMLCV.launch.parsl_conf.config import config
from parsl import python_app

config()


# Map function that returns double the input integer
@python_app
def app_random():
    import random
    return random.random()


results = []
for i in range(0, 10):
    x = app_random()
    results.append(x)

for r in results:
    print(r.result())
