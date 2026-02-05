import numpy as np

from oceanmixedlayers.mixed_layer_mean import mixed_layer_mean as _mixed_layer_mean


def test_mlm_constant_profile_any_depth():
    N = 50
    thck = np.ones(N)
    z_c = -(np.arange(N) + 0.5)
    tracer = np.ones(N) * 7.5
    depth = -120.0

    mlm = _mixed_layer_mean(tracer, z_c, thck, depth)
    assert np.isfinite(mlm.ML)
    assert np.allclose(mlm.ML, tracer[0])
    assert np.allclose(mlm.DZ_ML, abs(depth))


def test_mlm_partial_cell():
    # Five 10 m layers: centers at -5, -15, -25, -35, -45
    thck = np.ones(5) * 10.0
    z_c = -np.array([5, 15, 25, 35, 45], dtype=float)
    tracer = np.array([1, 3, 5, 7, 9], dtype=float)
    depth = -25.0  # cuts halfway into the third layer

    mlm = _mixed_layer_mean(tracer, z_c, thck, depth)

    # Expected thickness-weighted mean: (1*10 + 3*10 + 5*5) / 25 = 2.6
    expected_mean = (1 * 10 + 3 * 10 + 5 * 5) / 25.0
    assert np.isfinite(mlm.ML)
    assert np.allclose(mlm.ML, expected_mean, rtol=1e-12, atol=1e-9)
    assert np.allclose(mlm.DZ_ML, 25.0)


def test_mlm_broadcast_multicolumn():
    thck = np.ones(3) * 10.0
    z_c = -np.array([5, 15, 25], dtype=float)[:, None, None]
    tracer = np.dstack(
        [
            np.array([1, 2, 3], dtype=float),  # col 0
            np.array([3, 3, 3], dtype=float),  # col 1
        ]
    )
    depth = np.array([[-15.0, -30.0]])  # broadcast to x,y

    mlm = _mixed_layer_mean(tracer, z_c, thck[:, None, None], depth)

    # Column 0 depth -15: (1*10 + 2*5)/15 = 4/3
    expected0 = (1 * 10 + 2 * 5) / 15.0
    # Column 1 depth -30: full column mean = 3
    expected1 = 3.0

    assert np.allclose(mlm.ML[..., 0], expected0)
    assert np.allclose(mlm.ML[..., 1], expected1)
    assert np.allclose(mlm.DZ_ML[..., 0], 15.0)
    assert np.allclose(mlm.DZ_ML[..., 1], 30.0)


def test_mlm_zero_depth():
    thck = np.ones(3) * 10.0
    z_c = -(np.arange(3) * 10 + 5)
    tracer = np.arange(3, dtype=float)

    mlm = _mixed_layer_mean(tracer, z_c, thck, 0.0)
    assert np.allclose(mlm.ML, 0.0)
    assert np.allclose(mlm.DZ_ML, 0.0)
