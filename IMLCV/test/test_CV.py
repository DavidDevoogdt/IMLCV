import numpy as np
from IMLCV.base.CV import *
from IMLCV.base.bias import BiasF, HarmonicBias
import pytest


def test_harmonic():
    cv0 = CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], metric=Metric([True], jnp.array([0, 2 * np.pi])))
    cv1 = CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], metric=Metric([True], jnp.array([0, 2 * np.pi])))

    cvs = CombineCV([cv0, cv1])  #combine

    bias = HarmonicBias(cvs, q0=np.array([np.pi, -np.pi]), k=1.0)

    x = np.random.rand(2)

    a1 = bias._compute(np.array([np.pi, np.pi]) + x)
    a2 = bias._compute(np.array([-np.pi, -np.pi] + x))
    a3 = bias._compute(np.array([np.pi, -np.pi]) + x)
    a4 = bias._compute(np.array([-np.pi, np.pi] + x))
    a5 = bias._compute(np.array([np.pi, np.pi]) + x.T)
    a6 = bias._compute(np.array([np.pi, np.pi]) - x)

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5
    assert pytest.approx(a1, abs=1e-5) == a6


def test_split_combine():
    cv0 = CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], metric=Metric([True], jnp.array([0, 2 * np.pi])))
    cv1 = CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], metric=Metric([True], jnp.array([0, 2 * np.pi])))

    cvs = CombineCV([cv0, cv1])  #combine
    [cv0b, cv1b] = cvs.split_cv()  #split again in components

    coordinates = np.random.random((20, 3))

    [cv0_cv, cv0_grad, _] = cv0.compute(coordinates=coordinates, cell=None, jac_p=True)
    [cv0b_cv, cv0b_grad, _] = cv0b.compute(coordinates=coordinates, cell=None, jac_p=True)

    #check whether combine and split are each others inverses
    assert pytest.approx(cv0_cv) == cv0b_cv
    assert pytest.approx(cv0_grad) == cv0b_grad
    #check whether CV is same as inputed function f
    assert pytest.approx(CVUtils.dihedral(coordinates, cell=None, numbers=[4, 6, 8, 14])) == cv0_cv[0]


def test_virial():
    #virial for volume based CV is V*I(3)
    cv0 = CV(CVUtils.Volume)
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    bias = BiasF(cvs=cv0, f=(lambda x: x))  #simply take volume as lambda
    vol, _, vir = bias.compute_coor(coordinates=None, cell=cell, vir=vir)
    assert pytest.approx(vir, abs=1e-5) == vol * np.eye(3)


if __name__ == "__main__":
    test_harmonic()
    test_split_combine()
    test_virial()