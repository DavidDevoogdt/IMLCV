from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.scheme import Scheme

from yaff.log import log
import os

from IMLCV.test.common import ala_yaff

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep():

    yaffmd = ala_yaff()

    cvd = CVDiscovery()

    scheme = Scheme(md=yaffmd, cvd=cvd)
    scheme.run(2, 1e3, 3e4)


if __name__ == "__main__":
    test_ala_dipep()
