from pathlib import Path
from shutil import rmtree
import pytest

@pytest.fixture(scope="function")
def make_temp_dir():
    temp_dir = Path("temp_dir")
    if temp_dir.exists():
        rmtree(temp_dir)
    Path.mkdir(temp_dir)
    yield
    rmtree(temp_dir)

