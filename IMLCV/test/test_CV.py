import numpy as np
import pytest

from IMLCV.base.bias import BiasF, GridBias, HarmonicBias
from IMLCV.base.CV import CV, CvFlow, Metric, SystemParams


def test_harmonic():
    # cv0 = CV(CVUtils.dihedral,
    #          numbers=[4, 6, 8, 14],
    #          metric=Metric([True], jnp.array([0, 2 * np.pi])))
    # cv1 = CV(CVUtils.dihedral,
    #          numbers=[6, 8, 14, 16],
    #          metric=Metric([True], jnp.array([0, 2 * np.pi])))

    # cvs = CombineCV([cv0, cv1])  # combine

    cvs = CV(
        f=[
            CVUtils.dihedral(numbers=[4, 6, 8, 14]),
            CVUtils.dihedral(numbers=[6, 8, 14, 16]),
        ],
        metric=Metric(
            periodicities=[True, True], bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]]
        ),
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

    vol, _, vir = bias.compute_coor(SystemParams(coordinates=None, cell=cell), vir=True)
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


def test_grid_bias():

    # bounds = [[0, 3], [0, 3]]
    n = [4, 6]

    cv = CV(
        CvFlow(func=lambda x: x.coordinates),
        Metric(
            periodicities=[False, False],
            bounding_box=np.array([[-2, 2], [1, 5]]),
        ),
    )

    bins = [
        np.linspace(a, b, ni, endpoint=True, dtype=np.double)
        for ni, (a, b) in zip(n, cv.metric.bounding_box)
    ]

    def f(x, y):
        return x**3 + y

    # reevaluation of thermolib histo
    bin_centers1, bin_centers2 = 0.5 * (bins[0][:-1] + bins[0][1:]), 0.5 * (
        bins[1][:-1] + bins[1][1:]
    )
    xc, yc = np.meshgrid(bin_centers1, bin_centers2, indexing="ij")
    xcf = np.reshape(xc, (-1))
    ycf = np.reshape(yc, (-1))
    val = np.array([f(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)

    # print(f"xc:\n{xc}\nyc:{yc}\nval:\n{ val}")

    bias = GridBias(cvs=cv, vals=val)

    def c(x, y):
        return bias.compute(cvs=np.array([x, y]))[0]

    # test along grid centers

    val2 = np.array([c(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)
    assert np.allclose(val, val2)


if __name__ == "__main__":
    # test_harmonic()
    # test_virial()
    test_grid_bias()
