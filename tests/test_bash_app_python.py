from __future__ import annotations

from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import config
from mpi4py import MPI


def test_parallel(tmp_path):
    config(env="local", path_internal=tmp_path)

    @bash_app_python
    def _f(i, inputs=[], outputs=[]):
        from time import sleep

        print(f"i: {i}")

        sleep(2)

        return i

    n = 5

    futs = [_f(i, execution_folder=tmp_path) for i in range(n)]

    res = [f.result() for f in futs]

    assert res == [0, 1, 2, 3, 4]


@bash_app_python(precommand="mpirun -n 4")
def _f_MPI(i, inputs=[], outputs=[]):
    from time import sleep

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # num_ranks = comm.Get_size()

    print(f"i: {i}")

    sleep(2)

    return i, rank


def test_parallel_MPI(tmp_path):
    config(env="local", path_internal=tmp_path)

    i_enum = 5

    futs = [_f_MPI(i, execution_folder=tmp_path) for i in range(i_enum)]

    res = [list(zip(*f.result())) for f in futs]
    for i_enum, (i, r) in enumerate(res):
        for ir in i:
            assert ir == i_enum

        for r_enum, r in enumerate(sorted(r)):
            assert r == r_enum


if __name__ == "__main__":
    test_parallel_MPI("tmp")
