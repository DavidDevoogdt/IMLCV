"""
    Dummy conftest.py for IMLCV.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
# import pytest
import os

import jax
import pytest
from IMLCV.configs.config_general import config


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def config_test(tmpdir, local_ref_threads=4):
    config(env="local", path_internal=tmpdir, local_ref_threads=local_ref_threads)
