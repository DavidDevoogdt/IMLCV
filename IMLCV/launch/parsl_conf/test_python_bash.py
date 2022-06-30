# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but


from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from IMLCV.launch.parsl_conf.config import config


@bash_app_python(executors=['threads'])
def myfunc(i, str):
    print(f"got i={i} str={str}")
    return 3


if __name__ == "__main__":
    config()

    fut = myfunc(7, str=20, stdout='test.out', stderr='test.stderr')

    a = fut.result()

    print(a)
