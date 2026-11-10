import numpy as np
from imuhar import model
from imuhar.augment import _rot, augment
from imuhar.signal import FS, WIN, expand, moving_mean, prepare, resample, viterbi, window_load, windows


def test_resample_recovers_sine():
    t = np.sort(np.random.default_rng(0).uniform(0, 10, 2000))
    x = np.sin(2 * np.pi * 0.5 * t)[:, None]
    r = resample(x, t)
    assert abs(len(r) - 10 * FS) <= 2
    assert np.abs(r[: 9 * FS, 0] - np.sin(2 * np.pi * 0.5 * np.arange(9 * FS) / FS)).max() < 0.02


def test_moving_mean_constant_and_edges():
    x = np.ones((300, 3)) * 2.0
    assert np.allclose(moving_mean(x), 2.0)


def test_prepare_splits_gravity():
    n = 600
    g = np.tile([0, 0, 1.0], (n, 1))
    walk = 0.3 * np.sin(2 * np.pi * 2 * np.arange(n) / FS)[:, None] * np.array([1, 0, 0])
    sig = prepare(g + walk, np.zeros((n, 3)))
    assert sig.shape == (n, 9)
    assert np.allclose(sig[100:500, 5], 1.0, atol=0.02)          # gravity z
    assert np.abs(sig[100:500, 0]).max() > 0.25                   # user acc kept


def test_windows_shape():
    assert windows(np.zeros((1000, 9))).shape == (14, WIN, 9)
    assert windows(np.zeros((50, 9))).shape[0] == 0


def test_rotation_invariant_channels():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, WIN, 9)); x[..., 3:6] = rng.normal(size=(5, WIN, 3)) + [0, 0, 1]
    R = _rot(rng, 5, 60.0); xr = x.copy()
    for c in range(0, 9, 3):
        xr[..., c : c + 3] = np.einsum("bij,blj->bli", R, x[..., c : c + 3])
    assert np.allclose(expand(x, "inv")[..., :4], expand(xr, "inv")[..., :4], atol=1e-5)


def test_augment_keeps_shape_and_changes_data():
    x = np.random.default_rng(0).normal(size=(8, WIN, 9)).astype(np.float32)
    a = augment(x, np.random.default_rng(1))
    assert a.shape == x.shape and not np.allclose(a, x)


def test_window_load_orders_intensity():
    t = np.arange(WIN) / FS
    calm = (0.02 * np.random.default_rng(0).normal(size=(1, WIN, 3)))
    run = np.zeros((1, WIN, 3)); run[0, :, 2] = 2.5 * np.sin(2 * np.pi * 2.7 * t)
    a, b = window_load(calm), window_load(run)
    assert b["load_gps"][0] > 10 * a["load_gps"][0] and b["peak_g"][0] > 2
    assert abs(b["rhythm_cpm"][0] - 2.7 * 60) < 15


def test_viterbi_removes_blips_but_keeps_real_changes():
    lp = np.log(np.full((40, 6), 0.02)); lp[:20, 2] = np.log(0.9); lp[20:, 3] = np.log(0.9)
    lp[10, 2] = np.log(0.2); lp[10, 4] = np.log(0.7)               # single noisy window
    p = viterbi(lp, 0.9)
    assert (p[:20] == 2).all() and (p[20:] == 3).all()


def test_np_forward_shape():
    import jax
    p = model.init(jax.random.PRNGKey(0), 7)
    assert model.np_forward(p, np.zeros((3, WIN, 7), np.float32)).shape == (3, 6)
