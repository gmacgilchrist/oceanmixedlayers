import numpy as np
import pytest

from oceanmixedlayers.heat_content import heat_content as _heat_content


def test_heat_content_constant_CT():
    N = 300
    thck = np.ones(N)  # 1 m layers
    z_c = -(np.arange(N) + 0.5)  # centers at -0.5, -1.5, ...
    CT = np.ones(N) * 10.0  # 10 deg C everywhere
    depth = -200.0

    rho = 1025.0
    cp = 3991.86795711963

    hc = _heat_content(CT, z_c, thck, depth).HC
    expected = rho * cp * (10.0 - 0.0) * abs(depth)
    assert np.isfinite(hc)
    assert np.allclose(hc, expected, rtol=1e-12, atol=1e-9)

    # Check reference temperature effect
    T_ref = 2.0
    hc_ref = _heat_content(CT, z_c, thck, depth, T_ref=T_ref).HC
    expected_ref = rho * cp * (10.0 - T_ref) * abs(depth)
    assert np.allclose(hc_ref, expected_ref, rtol=1e-12, atol=1e-9)


def test_heat_content_partial_cell_linear_CT():
    # Five 10 m layers: centers at -5, -15, -25, -35, -45
    thck = np.ones(5) * 10.0
    z_c = -np.array([5, 15, 25, 35, 45], dtype=float)
    CT = np.array([1, 2, 3, 4, 5], dtype=float)
    depth = -25.0  # cuts halfway into the third layer

    rho = 1025.0
    cp = 3991.86795711963

    hc = _heat_content(CT, z_c, thck, depth).HC
    # Expected: 1*10 + 2*10 + 3*5 = 45 degC·m
    expected = rho * cp * 45.0
    assert np.isfinite(hc)
    assert np.allclose(hc, expected, rtol=1e-12, atol=1e-9)
