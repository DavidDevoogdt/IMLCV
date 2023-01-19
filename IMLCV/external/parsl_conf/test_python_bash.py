# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but


from configs.bash_app_python import bash_app_python
from configs.vsc_stevin import config


@bash_app_python()
def myfunc(i, str):
    print(f"got i={i} str={str}")
    return 3


if __name__ == "__main__":
    config(cluster="slaking")

    fut = myfunc(7, str=20, stdout="test.out", stderr="test.stderr")

    a = fut.result()

    print(a)
