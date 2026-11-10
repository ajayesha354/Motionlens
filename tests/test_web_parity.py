"""The website re-implements preprocessing, the CNN, smoothing and load metrics in JavaScript.
These tests run that JavaScript under Node and compare it with the Python reference."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax

from imuhar import model
from imuhar.signal import expand, prepare, viterbi, window_load, windows

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
CSV = "samples/demo_session.csv"


def run_node(*args):
    out = subprocess.run(["node", "tests/parity_node.js", CSV, *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def python_logits(tag, gyro: bool, df=None):
    df = pd.read_csv(CSV) if df is None else df
    acc = df[["ax", "ay", "az"]].values
    sig = prepare(acc, df[["gx", "gy", "gz"]].values if gyro else None)
    z = np.load(f"models/cnn_{tag}.npz")
    W = expand(windows(sig), str(z["variant"]))
    params = [{"w": z[f"w{i}"], "b": z[f"b{i}"]} for i in range(6)]
    return model.np_forward(params, (W - z["mean"]) / z["std"]), sig, float(z["T"]), float(z["stay"])


@pytest.mark.parametrize("tag,gyro", [("full", True), ("acc", False)])
def test_logits_match(tag, gyro):
    js = run_node("gyro" if gyro else "acc")
    ref, sig, _, _ = python_logits(tag, gyro)
    lg = np.array(js["lg"])
    assert lg.shape == ref.shape
    assert np.abs(lg - ref).max() < 5e-3
    assert (lg.argmax(1) == ref.argmax(1)).mean() > 0.995


def test_load_metrics_match():
    js = run_node("gyro")
    _, sig, _, _ = python_logits("full", True)
    ref = window_load(windows(sig[:, :3]))
    ld = js["ld"]
    assert np.allclose([l["load"] for l in ld], ref["load_gps"], rtol=2e-3, atol=2e-3)
    assert np.allclose([l["peak"] for l in ld], ref["peak_g"], rtol=2e-3, atol=2e-3)
    jr = np.array([np.nan if l["rhythm"] is None else l["rhythm"] for l in ld], float)
    both = np.isfinite(jr) & np.isfinite(ref["rhythm_cpm"])
    assert both.mean() > 0.5 and np.allclose(jr[both], ref["rhythm_cpm"][both], atol=1.0)


def test_viterbi_matches():
    js = run_node("gyro", "0", "auto", "smooth")
    ref, _, T, stay = python_logits("full", True)
    assert (viterbi(log_softmax(ref / T, axis=1), stay) == np.array(js["sm"])).mean() > 0.99


def test_resampling_path_agrees():
    """Same data at 100 Hz with a time column in milliseconds must give nearly the same answers."""
    df = pd.read_csv(CSV)
    t = df.time_s.values
    t2 = np.arange(0, t[-1], 0.01)
    up = pd.DataFrame({c: np.interp(t2, t, df[c].values) for c in ["ax", "ay", "az", "gx", "gy", "gz"]})
    up.insert(0, "timestamp_ms", np.round(t2 * 1000).astype(int))
    up.to_csv("/tmp/demo_100hz_ms.csv", index=False)
    out = subprocess.run(["node", "tests/parity_node.js", "/tmp/demo_100hz_ms.csv", "gyro"], capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    base = run_node("gyro")
    n = min(len(js["raw"]), len(base["raw"]))
    assert (np.array(js["raw"][:n]) == np.array(base["raw"][:n])).mean() > 0.9
    assert any("Resampled" in w for w in js["warn"])


def test_units_autodetected():
    df = pd.read_csv(CSV)
    df[["ax", "ay", "az"]] *= 9.80665                               # m/s^2
    df[["gx", "gy", "gz"]] *= 180 / np.pi                           # deg/s
    df.to_csv("/tmp/demo_si.csv", index=False)
    out = subprocess.run(["node", "tests/parity_node.js", "/tmp/demo_si.csv", "gyro"], capture_output=True, text=True, check=True).stdout
    js = json.loads(out); base = run_node("gyro")
    assert np.abs(np.array(js["lg"]) - np.array(base["lg"])).max() < 0.05
