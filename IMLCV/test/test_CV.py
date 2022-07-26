import jax.numpy as jnp
import numpy as np
import pytest
from IMLCV.base.bias import BiasF, HarmonicBias
from IMLCV.base.CV import CV, CVUtils, Metric, SystemParams


def test_harmonic():
    # cv0 = CV(CVUtils.dihedral,
    #          numbers=[4, 6, 8, 14],
    #          metric=Metric([True], jnp.array([0, 2 * np.pi])))
    # cv1 = CV(CVUtils.dihedral,
    #          numbers=[6, 8, 14, 16],
    #          metric=Metric([True], jnp.array([0, 2 * np.pi])))

    # cvs = CombineCV([cv0, cv1])  # combine

    cvs = CV(
        f=[CVUtils.dihedral(numbers=[4, 6, 8, 14]),
           CVUtils.dihedral(numbers=[6, 8, 14, 16])],
        metric=Metric(
            periodicities=[True, True],
            bounding_box=[[0, 2 * np.pi],
                          [0, 2 * np.pi]])
    )

    bias = HarmonicBias(cvs, q0=np.array([np.pi, -np.pi]), k=1.0)

    x = np.random.rand(2)

    a1, _ = bias.compute(np.array([np.pi, np.pi]) + x)
    a2, _ = bias.compute(np.array([-np.pi, -np.pi] + x))
    a3, _ = bias.compute(np.array([np.pi, -np.pi]) + x)
    a4, _ = bias.compute(np.array([-np.pi, np.pi] + x))
    a5, _ = bias.compute(np.array([np.pi, np.pi]) + x.T)

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5


def test_virial():
    # virial for volume based CV is V*I(3)

    metric = Metric(periodicities=[False], bounding_box=[0, 4])
    cv0 = CV(f=CVUtils.Volume(), metric=metric)
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    bias = BiasF(cvs=cv0, f=lambda x: x)  # simply take volume as lambda

    vol, _, vir = bias.compute_coor(SystemParams(
        coordinates=None, cell=cell), vir=True)
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


if __name__ == "__main__":
    test_harmonic()
    test_virial()
