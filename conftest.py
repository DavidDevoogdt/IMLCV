from __future__ import annotations

import os

import jax
import pytest

if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


jax.config.update("jax_enable_x64", True)

# cpu based
jax.config.update("jax_platform_name", "cpu")
