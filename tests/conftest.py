"""
    Dummy conftest.py for dst.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import logging
import shutil

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def setup_tmp_directories(request):
    path = request.param
    if not path.exists():
        path.mkdir(exist_ok=False, parents=True)
    logger.info("Set up directories for parser testing")
    def remove_parser_outputs():
        shutil.rmtree(path)
        logger.info("Removed parser outputs after testing")
    request.addfinalizer(remove_parser_outputs)



