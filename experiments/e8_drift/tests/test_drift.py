import os
import sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import drift as DR              
import drift_analysis as DA              


def test_distance_is_zero_for_identical_images():
    ident = dict(l1=0.0, mse=0.0, psnr=120.0, ssim=1.0, ms_ssim=1.0, lpips_vgg=0.0, lpips_alex=0.0, dists=0.0, flip=0.0,
                 cvvdp_image=10.0, dreamsim=0.0, fsim=1.0, vif=1.0)
    for m, v in ident.items():
        assert float(DA.dist(m, v)) == 0.0, m
    assert float(DA.dist("psnr", 35.0)) == 25.0
    assert float(DA.dist("ssim", 0.9)) == pytest.approx(0.1)
    assert float(DA.dist("cvvdp_image", 9.25)) == pytest.approx(0.75)


def test_other_sets_follow_the_families():
    names = lambda ix: {DA.IMG[i] for i in ix}              
    assert names(DA.others("lpips_vgg", True)) == set(DA.IMG) - {"lpips_vgg", "lpips_alex"}
    assert names(DA.others("lpips_vgg")) == set(DA.IMG) - {"lpips_vgg", "lpips_alex", "psnr"}
    assert names(DA.others("l1")) == set(DA.IMG) - {"l1", "mse", "psnr"}
    assert names(DA.others("dists", True)) == set(DA.IMG) - {"dists"}


def test_colour_match_recovers_an_affine_change():
    rng = np.random.default_rng(0)
    ref = rng.integers(30, 220, (40, 50, 3)).astype(np.uint8)
    A = np.array([[0.9, 0.05, 0.0], [0.0, 0.85, 0.05], [0.02, 0.0, 0.95]])
    x = np.clip(ref.astype(float) / 255 @ A + np.array([0.03, -0.02, 0.01]), 0, 1)
    x8 = np.round(x * 255).astype(np.uint8)
    out = DR.colour_match(x8, ref, "affine")
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 2
    g = np.round(np.clip(ref.astype(float) / 255 * np.array([0.8, 0.9, 1.0]) + 0.05, 0, 1) * 255).astype(np.uint8)
    out = DR.colour_match(g, ref, "gain")
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 2
    assert np.abs(DR.colour_match(ref, ref, "affine").astype(int) - ref.astype(int)).max() == 0


def test_partial_spearman():
    rng = np.random.default_rng(1)
    c = rng.normal(size=400)
    x, y = c + 0.3 * rng.normal(size=400), c + 0.3 * rng.normal(size=400)
    assert DA.rho(x, y) > 0.8
    assert abs(DA.partial_spearman(x, y, c)) < 0.15
    y2 = x + 0.1 * rng.normal(size=400)
    assert DA.partial_spearman(x, y2, c) > 0.8


def test_self_terms_definition():
    models = ["lpips_vgg__pure", "l1__pure"]
    z = np.zeros((2, len(DA.IMG)))
    z[0, DA.IMG.index("lpips_vgg")] = 3.0
    z[0, DA.IMG.index("lpips_alex")] = 2.0
    z[0, DA.IMG.index("psnr")] = -1.1
    z[1, DA.IMG.index("l1")] = 1.0
    own, ref, S = DA.self_terms(z, models)
    assert own.tolist() == [3.0, 1.0]
    assert ref[0] == pytest.approx(-0.1) and ref[1] == 0.0
    assert S[0] == pytest.approx(3.1) and S[1] == 1.0


def test_new_losses_are_registered_without_touching_the_old_ones():
    from gh import metrics as M
    before = dict(M.LOSSES)
    import metrics_ext as X
    X.install()
    for k, v in before.items():
        assert M.LOSSES[k] == v
    assert set(X.NEW_LOSSES) == {"charbonnier", "lpips_squeeze", "haarpsi", "vif", "dreamsim"}
    assert M.FAMILY["lpips_squeeze"] == "lpips" and M.FAMILY["charbonnier"] == "pixel" and M.FAMILY["haarpsi"] == "haarpsi"
    assert float(M.loss_form("haarpsi", 1.0)) == 0.0 and float(M.loss_form("vif", 1.0)) == 0.0
    assert M.loss_form("lpips_vgg", 0.25) == 0.25
