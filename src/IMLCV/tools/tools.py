from __future__ import annotations

from typing import Generic
from typing import TypeVar

T = TypeVar("T")  # Declare type variable


class HashableArrayWrapper(Generic[T]):
    """see https://github.com/google/jax/issues/4572"""

    def __init__(self, val: T):
        self.val = val

    def __getattribute__(self, prop):
        if prop == "val" or prop == "__hash__" or prop == "__eq__":
            return super().__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        return self.val == other
