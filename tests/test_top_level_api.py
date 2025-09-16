import numpy as np
import pytest

from oceanmixedlayers import oceanmixedlayers


@pytest.fixture(scope="module")
def linear_column(idealized_columns):
    return idealized_columns["linear"]


def test_threshold_api(linear_column):
    depth, idx = oceanmixedlayers.threshold(-linear_column.zc, linear_column.prho, delta=0.03, ref=10.0)
    np.testing.assert_allclose(depth, np.array([21.45235602]), atol=1e-6)
    np.testing.assert_array_equal(idx, np.array([215]))


def test_gradient_api(linear_column):
    depth, idx = oceanmixedlayers.gradient(-linear_column.zc, -linear_column.prho, critical_gradient=1e-5)
    np.testing.assert_allclose(depth, np.array([0.15]), atol=1e-6)
    np.testing.assert_array_equal(idx, np.array([1]))


def test_linearfit_api(linear_column):
    depth, idx = oceanmixedlayers.linearfit(-linear_column.zc, -linear_column.prho, error_tolerance=1e-10)
    np.testing.assert_array_equal(idx, np.array([5]))
    np.testing.assert_allclose(depth, np.array([0.55]), atol=1e-8)


def test_holtetalley_api(linear_column):
    result = oceanmixedlayers.holtetalley(linear_column.pc / 1.0e4, linear_column.S, linear_column.T, linear_column.prho)[2]
    np.testing.assert_allclose(result, np.array([10.10231894]), atol=1e-6)


def test_mld_pe_anomaly_api(linear_column):
    depth = oceanmixedlayers.mld_pe_anomaly(linear_column.zc, linear_column.dz, linear_column.prho, energy=10.0)
    depth_newton = oceanmixedlayers.mld_pe_anomaly(
        linear_column.zc, linear_column.dz, linear_column.prho, energy=10.0, iteration="Newton"
    )
    np.testing.assert_allclose(depth, np.array([16.7015625]), atol=1e-6)
    np.testing.assert_allclose(depth_newton, np.array([16.70154982]), atol=1e-6)


def test_mld_delta_pe_api(linear_column):
    depth = oceanmixedlayers.mld_delta_pe(
        linear_column.pc / 1.0e4,
        linear_column.dp / 1.0e4,
        linear_column.T,
        linear_column.S,
        energy=10.0,
    )
    np.testing.assert_allclose(depth, np.array([16.70390614]), atol=1e-6)


def test_pe_anomaly_api(linear_column):
    pe = oceanmixedlayers.pe_anomaly(
        linear_column.zc,
        linear_column.dz,
        linear_column.prho,
        depth=np.array([-50.0]),
    )
    np.testing.assert_allclose(pe, np.array([266.8824665115634]), atol=1e-6)
