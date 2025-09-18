# tests/conftest.py
import math
import pytest

M_SUN = 1.98847e30  # kg
RTOL = 5e-6
ATOL = 5e-12

@pytest.fixture(scope="session")
def Msun():
    return M_SUN

@pytest.fixture(scope="session")
def mass_30Msun(Msun):
    return 30.0 * Msun
