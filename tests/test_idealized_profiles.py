import numpy as np
import pytest

from oceanmixedlayers import oceanmixedlayers


@pytest.mark.parametrize(
    "column_key, expected_depth, expected_index",
    [
        ("linear", 21.45235602, 215),
        ("two_layer", 49.85, 498),
        ("shelf", 79.95, 799),
    ],
)
def test_threshold_gradient_linearfit_match_notebooks(idealized_columns, column_key, expected_depth, expected_index):
    col = idealized_columns[column_key]
    if column_key == "linear":
        depth, idx = col.threshold()
    elif column_key == "two_layer":
        depth, idx = col.gradient()
    else:
        depth, idx = col.linearfit()
    assert depth.shape == (1,)
    assert idx.shape == (1,)
    np.testing.assert_allclose(depth[0], expected_depth, rtol=0, atol=1e-6)
    assert int(idx[0]) == expected_index


def test_holtetalley_matches_notebook_values(idealized_columns):
    expected = {
        "linear": 10.10231894,
        "two_layer": 48.92842264,
        "shelf": 78.32012462,
    }
    for key, col in idealized_columns.items():
        result = col.holtetalley()[2]
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], expected[key], rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "key, expected_bisection, expected_newton",
    [
        ("linear", 16.7015625, 16.70154982),
        ("two_layer", 50.03044205, 50.03044228),
        ("shelf", 85.17050781, 85.17048589),
    ],
)
def test_mld_pe_anomaly_iterations(idealized_columns, key, expected_bisection, expected_newton):
    col = idealized_columns[key]
    bisection = col.mld_pe_anomaly(iteration="Bisection")
    newton = col.mld_pe_anomaly(iteration="Newton")
    np.testing.assert_allclose(bisection, np.array([expected_bisection]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(newton, np.array([expected_newton]), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("linear", 16.70390614),
        ("two_layer", 50.04950324),
        ("shelf", 85.14624031),
    ],
)
def test_mld_delta_pe_matches_notebook(idealized_columns, key, expected):
    col = idealized_columns[key]
    result = col.mld_delta_pe()
    np.testing.assert_allclose(result, np.array([expected]), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "depth, expected_pe",
    [(-20.0, 17.163012048997334), (-50.0, 266.8824665115634)],
)
def test_pe_anomaly_energy_profile(idealized_columns, depth, expected_pe):
    col = idealized_columns["linear"]
    energy = col.pe_anomaly(Dpt=depth)[0]
    np.testing.assert_allclose(energy, expected_pe, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "depth, expected",
    [(-20.0, 1.6284825331069531e09), (-50.0, 4.0405188478470254e09)],
)
def test_heat_content_interface(depth, expected, idealized_columns):
    col = idealized_columns["linear"]
    result = oceanmixedlayers.heat_content(col.zc, col.dz, col.T, depth=depth)
    np.testing.assert_allclose(result, np.array([expected]), rtol=0, atol=1e-6)
