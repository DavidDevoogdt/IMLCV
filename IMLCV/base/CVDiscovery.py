from IMLCV.base.CV import CV


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self) -> None:
        pass

    def compute(self, data) -> CV:
        raise NotImplementedError


if __name__ == '__main__':
    from IMLCV.test.test_scheme import test_cv_discovery
    test_cv_discovery()
