import jax.numpy as jnp
import pytest

from IMLCV.base.CV import CvFlow, SystemParams
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.CV import dihedral


def f_test_parallel(i):
    from time import sleep

    print(f"i: {i}")

    sleep(2)

    return 2 * i


def test_parallel(tmp_path, config_test):
    n = 4

    futs = [
        bash_app_python(
            f_test_parallel,
            auto_log=True,
        )(i, execution_folder=tmp_path)
        for i in range(n)
    ]

    res = [f.result() for f in futs]

    assert res == [0, 2, 4, 6]


def f_test_py_env(sp):
    d_flow: CvFlow = dihedral((0, 1, 2, 3))

    return d_flow.compute_cv_flow(sp, None)[0]


@pytest.mark.skip(reason="MPI not installed")
def test_py_env(tmp_path, config_test):
    n = 4
    sp = SystemParams(
        coordinates=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        cell=None,
    )

    futs = [
        bash_app_python(
            f_test_py_env,
            auto_log=True,
        )(sp, execution_folder=tmp_path)
        for i in range(n)
    ]

    for f in futs:
        assert jnp.allclose(f.result().cv, 0.95531662)


def _f_MPI(i):
    from time import sleep

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # num_ranks = comm.Get_size()

    print(f"i: {i}")

    sleep(2)

    return i, rank


@pytest.mark.skip(reason="MPI not installed")
def test_parallel_MPI(tmp_path, config_test):
    i_enum = 5

    futs = [
        bash_app_python(
            _f_MPI,
            precommand="mpirun -n 4",
            uses_mpi=True,
            auto_log=True,
        )(i, execution_folder=tmp_path)
        for i in range(i_enum)
    ]

    res = [list(zip(*f.result())) for f in futs]
    for i_enum, (i, r) in enumerate(res):
        for ir in i:
            assert ir == i_enum

        for r_enum, r in enumerate(sorted(r)):
            assert r == r_enum


if __name__ == "__main__":
    test_py_env("tmp")
