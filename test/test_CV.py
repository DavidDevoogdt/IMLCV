import numpy as onp
from IMLCV.base.CV import *
import pytest


def test_split_combine():
    cv0 = CV(CVUtils.dihedral, numbers=[4, 6, 8, 14])
    cv1 = CV(CVUtils.dihedral, numbers=[6, 8, 14, 16])

    cvs = CombineCV([cv0, cv1])  #combine
    [cv0b, cv1b] = cvs.split_cv()  #split again in components

    coordinates = onp.random.random((20, 3))

    [cv0_cv, cv0_grad, _] = cv0.compute(coordinates=coordinates, cell=None, grad=True)
    [cv0b_cv, cv0b_grad, _] = cv0b.compute(coordinates=coordinates, cell=None, grad=True)

    #check whether combine and split are each others inverses
    assert pytest.approx(cv0_cv, cv0b_cv)
    assert pytest.approx(np.linalg.norm(cv0_grad - cv0b_grad, 2), 0)
    #check whether CV is same as inputed function f
    assert pytest.approx(CVUtils.dihedral(coordinates, cell=None, numbers=[4, 6, 8, 14]), cv0_cv)


def test_virial():
    cv0 = CV(CVUtils.Volume)
    cell = onp.random.random((3, 3))
    vol, _, vir = cv0.compute(coordinates=None, cell=cell, vir=True)
    assert pytest.approx(np.linalg.norm(vir - vol * np.eye(3), 2), 0)


if __name__ == "__main__":
    test_split_combine()
    test_virial()