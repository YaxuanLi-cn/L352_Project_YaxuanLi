import os
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))                 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
import analysis as A              
import uncertainty as U              
PURE = ["l1", "l2", "dssim", "msssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp"]
MIXED = ["l2", "msssim", "lpips_vgg", "lpips_alex", "dists", "flip", "cvvdp"]
@pytest.fixture(scope="module")


def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot")
    rng = np.random.default_rng(0)
    scenes = [f"gsqa/s{i}" for i in range(4)] + [f"mip360/m{i}" for i in range(4)]
    models = ["default__ctrl"] + [f"{l}__pure" for l in PURE] + [f"{l}__mixed" for l in MIXED]
    frames = []
    nview, ntile = 10, 8
    for sc in scenes:
        for split in ("test", "train"):
            u = rng.random(nview * ntile)
            q, _ = U.assign_strata(u)
            view = np.repeat(np.arange(nview), ntile)
            tile = np.tile(np.arange(ntile), nview)
            for sd in ([0, 1, 2] if sc == scenes[0] else [0]):
                for m in models:
                    own = None if m == "default__ctrl" else A.own_metric(m)
                    for met in A.IMG:
                        v = rng.normal(size=nview * ntile) + (0.5 * u if met == own else 0.0)
                        frames.append(pd.DataFrame(dict(scene=sc, split=split, view=view, tile=tile, model=m, seed=sd,
                                                        metric=met, value=v, U=u, B=rng.random(len(u)),
                                                        G=rng.random(len(u)), quintile=q)))
    pd.concat(frames, ignore_index=True).to_parquet(os.path.join(out, "tiles_long.parquet"), index=False)
    d = A.Data(str(out))
    delta = A.improvements(d)
    scale = A.column_scale(d, delta)
    return d, delta, scale, str(out)


def test_overall_equals_mean_of_strata(data):
    d, delta, scale, _ = data
    for split in ("test", "train"):
        for group in ("pure", "mixed"):
            for mode in ("paper", "families", "nondefault"):
                O, R, S = A.per_scene_group(d, delta, scale, split, group, mode)
                assert O.shape == (8, 7)
                for arr in (O, R, S):
                    assert np.allclose(arr[:, 5], arr[:, :5].mean(axis=1), atol=1e-12)
                    assert np.allclose(arr[:, 6], arr[:, :2].mean(axis=1), atol=1e-12)                 
                assert np.allclose(S, O - R, atol=1e-12)


def test_scale_definition(data):
    d, delta, scale, _ = data
    rows = [delta[(sc, "test")][m][0, 5] for sc in d.scenes for m in d.loss_models]
    assert len(rows) == 8 * 16
    assert np.allclose(scale, np.std(np.stack(rows), axis=0, ddof=1))


def test_all_analyses_run(data):
    d, delta, scale, out = data
    rng = np.random.default_rng(0)
    a1, a2, verdict = A.a1_a2(d, delta, scale, rng)
    assert set(a1.stratum) == set(A.STRATA + ["overall"])
    assert set(a2.stratum) == set(A.STRATA + ["overall", "Q1+Q2"])
    assert ("test", "pure", "a") in verdict and ("test", "mixed", "b") in verdict
    assert len(A.per_scene_table(d, delta, scale)) == 2 * 2 * 8 * 6
    sd = A.seed_sd(d, delta, scale)
    assert len(sd) == 2 * 2 * 6 and sd.sd_over_seeds.notna().all()
    curves, shares, summ = A.a3_concentration(d, scale, rng)
    assert np.allclose([c[-1] for c in curves.S], 1.0)
    reg = A.a5_regression(d, scale, rng)
    assert set(reg.variable) == {"U", "B", "G"}
    a4, thr = A.a4_agreement(d, out, rng)
    assert set(a4.source) == {"loss"}              
    neg, perm = A.negative_control(d, scale, rng, n_perm=5)
    assert "Q5-Q1" in set(neg.stratum)
    assert len(perm) == 4 and (perm.n_perm == 5).all()
