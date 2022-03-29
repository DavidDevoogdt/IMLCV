import numpy as np
from IMLCV.base.CV import *
from IMLCV.base.bias import BiasF
import pytest


def test_split_combine():
    cv0 = CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=2.0 * np.pi)
    cv1 = CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=2.0 * np.pi)

    cvs = CombineCV([cv0, cv1])  #combine
    [cv0b, cv1b] = cvs.split_cv()  #split again in components

    coordinates = np.random.random((20, 3))

    [cv0_cv, cv0_grad, _] = cv0.compute(coordinates=coordinates, cell=None, jac_p=True)
    [cv0b_cv, cv0b_grad, _] = cv0b.compute(coordinates=coordinates, cell=None, jac_p=True)

    #check whether combine and split are each others inverses
    assert pytest.approx(cv0_cv, cv0b_cv)
    assert pytest.approx(np.sum((cv0_grad - cv0b_grad)**2)**(1 / 2), 0)
    #check whether CV is same as inputed function f
    assert pytest.approx(CVUtils.dihedral(coordinates, cell=None, numbers=[4, 6, 8, 14]), cv0_cv)


def test_virial():
    #virial for volume based CV is V*I(3)
    cv0 = CV(CVUtils.Volume)
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    bias = BiasF(cvs=cv0, f=(lambda x: x))  #simply take volume as lambda
    vol = bias.compute_coor(coordinates=None, cell=cell, vir=vir)
    assert pytest.approx(np.linalg.norm(vir - vol * np.eye(3), 2), 0)


if __name__ == "__main__":
    test_split_combine()
    test_virial()