import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import uncertainty as U              


def test_variance_map_known_values():
    r = np.zeros((2, 3, 4, 3), np.uint8)
    r[1] = 255
    assert np.allclose(U.variance_map(r), 0.5)
    same = np.repeat(np.random.default_rng(0).integers(0, 256, (1, 5, 6, 3), dtype=np.uint8), 4, axis=0)
    assert np.allclose(U.variance_map(same), 0.0)


def test_variance_map_matches_definition():
    rng = np.random.default_rng(1)
    r = rng.integers(0, 256, (4, 7, 9, 3), dtype=np.uint8)
    x = r.astype(np.float64) / 255
    k = x.shape[0]
    mean = x.sum(axis=0) / k
    ref = (((x - mean) ** 2).sum(axis=0) / (k - 1)).mean(axis=-1)
    assert np.allclose(U.variance_map(r), ref)
    assert np.allclose(U.variance_map(r[::-1]), ref)
    assert np.allclose(U.variance_map(x + 0.1), ref)


def test_bias_map():
    rng = np.random.default_rng(2)
    x = rng.random((4, 6, 6, 3))
    ref = x.mean(axis=0)
    assert np.allclose(U.bias2_map(x, ref), 0.0)
    assert (U.bias2_map(x, rng.random((6, 6, 3))) >= 0).all()


def test_tiles_cover_the_centred_crop():
    a = np.arange(10 * 13).reshape(10, 13).astype(float)
    t = U.tile_view(a, 4)
    assert t.shape == (6, 4, 4)
    y0, x0, rows, cols = U.tile_grid(10, 13, 4)
    assert (y0, x0, rows, cols) == (1, 0, 2, 3)
    assert np.array_equal(t[4], a[y0 + 4:y0 + 8, x0 + 4:x0 + 8])


def test_strata_equal_size():
    u = np.random.default_rng(3).random(480)
    s, edges = U.assign_strata(u)
    assert all((s == k).sum() == 96 for k in range(1, 6))
    assert (np.diff(edges[:, 0]) > 0).all() and (edges[:-1, 1] < edges[1:, 0]).all()
    s2, _ = U.assign_strata(np.random.default_rng(4).random(576))
    sizes = np.array([(s2 == k).sum() for k in range(1, 6)])
    assert sizes.sum() == 576 and sizes.max() - sizes.min() <= 1


def _scores(rng, n_tiles, n_metric=13):
    ctrl = rng.normal(size=(n_tiles, n_metric))
    model = ctrl + rng.normal(0.2, 1.0, size=(n_tiles, n_metric))
    return ctrl, model


def _overall_and_strata(n_tiles, seed):
    rng = np.random.default_rng(seed)
    ctrl, model = _scores(rng, n_tiles)
    scale = rng.random(13) + 0.5
    stratum, _ = U.assign_strata(rng.random(n_tiles))
    own, others = 5, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
    z_all = (model.mean(axis=0) - ctrl.mean(axis=0)) / scale
    S_all = U.self_score(z_all, own, others)
    m_k, sizes = U.stratum_means(model, stratum)
    c_k, _ = U.stratum_means(ctrl, stratum)
    S_k = U.self_score((m_k - c_k) / scale, own, others)
    return S_all, S_k, sizes


def test_overall_is_mean_of_strata():
    S_all, S_k, sizes = _overall_and_strata(1400, 5)
    assert np.isclose(S_all, S_k.mean(), rtol=0, atol=1e-12)


def test_overall_is_weighted_mean_when_sizes_differ():
    S_all, S_k, sizes = _overall_and_strata(576, 6)
    assert np.isclose(S_all, (S_k * sizes).sum() / sizes.sum(), rtol=0, atol=1e-12)
