from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.scheme import Scheme

from yaff.log import log
import os

from IMLCV.test.common import ala_yaff

from yaff.test.common import get_alaninedipeptide_amber99ff

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep():

    yaffmd = ala_yaff(write=20)

    cvd = CVDiscovery()

    scheme = Scheme(md=yaffmd, cvd=cvd)

    load = True

    if load:
        scheme.load_round(round=0)
        scheme.calc_obs()
    else:
        scheme.run(2, 1e5, 1e5)


if __name__ == "__main__":
    test_ala_dipep()
