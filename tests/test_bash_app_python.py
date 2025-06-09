import jax.numpy as jnp
import pytest  # type:ignore

from IMLCV.base.CV import CvTrans, SystemParams
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.CV import dihedral

# def f_test_parallel(i, inputs=[]):
#     from time import sleep

#     with open(inputs[0], "r") as p:
#         i0 = int(p.read())

#     with open(inputs[1], "r") as p:
#         i1 = int(p.read())


#     # print(f"i: {i=} {i0=} {inputs=}")

#     sleep(2)

#     return i0 + i1 - 2*i


# n = 4

# futs = []

# for i in range(n):
#     exec_folder = Path(".") / "futs" / f"fut_{i}"

#     exec_folder.mkdir(exist_ok=True, parents=True)

#     f1 = exec_folder / "file1.txt"

#     with open(f1, "w") as p:
#         p.write(f"{i}")

#     f2 = exec_folder / "file2.txt"

#     with open(f2, "w") as p:
#         p.write(f"{3*i}")

#     futs.append(
#         bash_app_python(
#             f_test_parallel,
#             pickle_extension="p",
#             pass_files=True,
#             executors=Executors.threadpool,
#             # auto_log=True,
#         )(
#             i,
#             execution_folder=Path(".") / "futs" / f"fut_{i}",
#             inputs=[f1,f2],
#         )
#     )

# res = [f.result() for f in futs]

# assert res == [0, 2, 4, 6]


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
            execution_folder=tmp_path,
        )(i)
        for i in range(n)
    ]

    res = [f.result() for f in futs]

    assert res == [0, 2, 4, 6]


def f_test_py_env(sp):
    d_flow: CvTrans = dihedral(jnp.array([0, 1, 2, 3]))

    return d_flow.compute_cv(sp, None)[0]


@pytest.mark.skip(reason="MPI not installed")
def test_py_env(tmp_path, config_test):
    n = 4
    sp = SystemParams(
        coordinates=jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        cell=None,
    )

    futs = [bash_app_python(f_test_py_env, auto_log=True, execution_folder=tmp_path)(sp) for i in range(n)]

    for f in futs:
        assert jnp.allclose(f.result().cv, 0.95531662)


def _f_MPI(i):
    from time import sleep

    from mpi4py import MPI  # type:ignore

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"i: {i}")

    sleep(2)

    return i, rank


# @pytest.mark.skip(reason="MPI not installed")
# def test_parallel_MPI(tmp_path, config_test):
#     i_enum = 5

#     futs = [
#         bash_app_python(
#             _f_MPI,
#             # precommand="mpirun -n 4",
#             uses_mpi=True,
#             auto_log=True,
#         )(i, execution_folder=tmp_path)
#         for i in range(i_enum)
#     ]

#     res = [list(zip(*f.result())) for f in futs]
#     for i_enum, (i, r) in enumerate(res):
#         for ir in i:
#             assert ir == i_enum

#         for r_enum, r in enumerate(sorted(r)):
#             assert r == r_enum


# if __name__ == "__main__":
#     test_py_env("tmp")
