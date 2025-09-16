import numpy as np


def test_argo_threshold(argo_column):
    depth, idx = argo_column.threshold()
    np.testing.assert_allclose(depth, np.array([62.28435503]), atol=1e-6)
    np.testing.assert_array_equal(idx, np.array([9]))


def test_argo_gradient(argo_column):
    depth, idx = argo_column.gradient()
    np.testing.assert_allclose(depth, np.array([7.39941589]), atol=1e-6)
    np.testing.assert_array_equal(idx, np.array([1]))


def test_argo_linearfit(argo_column):
    depth, idx = argo_column.linearfit()
    np.testing.assert_allclose(depth, np.array([86.64067142]), atol=1e-6)
    np.testing.assert_array_equal(idx, np.array([12]))


def test_argo_holtetalley(argo_column):
    result = argo_column.holtetalley()[2]
    np.testing.assert_allclose(result, np.array([62.56534527]), atol=1e-6)


def test_argo_energy_methods(argo_column):
    pe = argo_column.mld_pe_anomaly()
    delta = argo_column.mld_delta_pe()
    np.testing.assert_allclose(pe, np.array([33.91029853]), atol=1e-6)
    np.testing.assert_allclose(delta, np.array([33.88470738]), atol=1e-6)
