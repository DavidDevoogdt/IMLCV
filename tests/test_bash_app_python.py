import jax.numpy as jnp
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import SystemParams
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import config
from IMLCV.implementations.CV import dihedral
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


def test_py_env(tmp_path):
    config(env="local", path_internal=tmp_path)

    @bash_app_python
    def _f(sp, inputs=[], outputs=[]):
        d_flow: CvFlow = dihedral([0, 1, 2, 3])

        return d_flow.compute_cv_flow(sp, None)

    n = 5
    sp = SystemParams(
        coordinates=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        cell=None,
    )

    futs = [_f(sp, execution_folder=tmp_path) for i in range(n)]

    for f in futs:
        assert jnp.allclose(f.result().cv, 0.95531662)


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
    test_py_env("tmp")
