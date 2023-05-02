from configs.bash_app_python import bash_app_python
from configs.config_general import config


@bash_app_python
def _f(i, inputs=[], outputs=[]):
    from time import sleep

    print(f"i: {i}")

    sleep(2)

    return i


def test_parallel(tmp_path):
    config(env="local", path_internal=tmp_path)

    n = 5

    futs = [_f(i, execution_folder=tmp_path) for i in range(n)]

    res = [f.result() for f in futs]

    assert res == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    test_parallel("tmp")
