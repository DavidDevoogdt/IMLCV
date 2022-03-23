from .CV import CV


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self) -> None:
        pass

    def compute(self, data) -> CV:
        NotImplementedError
