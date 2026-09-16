"""Gate D (closed loop): is it the readout?

Readout-invariance falsification adapted to the closed-loop captures. For every
(mode, seed) f1 capture the SAME recorded spike counts are decoded several ways
and compared against two targets:

- vref(t): the analytic task profile (0.5 -> 0.8 -> 0.5 @ 40 Hz).
- cmd(t) : the command the loop actually issued (cmd_series), i.e. the causal
           edge "spikes -> command" that Gate A measured with TE.

Decoders:
- canon     : the loop's own decoded command (cmd_series) that drove the plant.
- meanpd    : predict the run-mean of the target everywhere - the honest floor.
- ridgeX0   : best L2-regularized LINEAR readout of the INSTANTANEOUS per-tick
              spike counts (interleaved splits). Reference variant only.
- ridgeWF   : WALK-FORWARD causal readout: features are the strictly-past spike
              counts (lags 1..L ticks; L=48 = the hub's 1.2 s window), trained
              chronologically on the past and evaluated on each 100-tick test
              window across both phase transitions. This is the fair upper
              bound for any fixed causal linear decode in the closed loop.

Fits use nested train/val/test with the Woodbury (kernel) identity
(cost O(n_train^3)); lambda is selected on a chronological validation block
(the last fifth of the training window).

Decisive stats = carried_vref and carried_cmd, each = mean-over-windows of
(local-mean RMSE - walk-forward ridge RMSE), pooled per mode. The
dead-substrate null (poisson) is expected to sit at the mean-predictor floor.
Positive carries for the live substrates would then falsify the
"readout-channel artifact" hypothesis for Gate A/B; absent does NOT falsify the
full-loop behavioral battery (which measures the whole plant+inaar loop, not
the instantaneous spike stream).

Closed-loop caveat (documented): all non-poisson modes see the plant state
through the encoder, so spikes always encode some state even when the decode is
dead (zero) or uncoupled (random). Gate D therefore falsifies the DECODE
direction only.

Analysis-only: reads committed f1_capture_* JSONs, writes new results files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RES = BASE.parent / "results"

MODES = ["neural", "zero", "random", "mask0.5", "poisson"]
DT = 0.025
LAG = 48                                    # 48 ticks = 1.2 s (hub window)
WF_WINDOW = 100                             # 100 ticks = 2.5 s per test window
LAM_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1]
MI_PERM = 100
RNG_SEED = 2026


def vref(ticks: int):
    t = np.arange(ticks) * DT
    out = np.full(ticks, 0.5)
    out[(t >= 4.0) & (t < 8.0)] = 0.8
    return out


def _sch(xhat, y):
    rmse = float(np.sqrt(np.mean((xhat - y) ** 2)))
    rho = 0.0
    if np.std(xhat) > 1e-9 and np.std(y) > 1e-9:
        rho = float(np.corrcoef(xhat, y)[0, 1])
    return {"rmse": rmse, "rho": rho}


def meanpd(y):
    xhat = np.full(len(y), y.mean())
    return _sch(xhat, y)


def _discretize(x, n):
    lo, hi = np.percentile(x, [2.0, 98.0])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip(((x - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def _mi_counts(a, b, na, nb):
    h = np.zeros((na, nb))
    for i in range(len(a)):
        h[a[i], b[i]] += 1
    h = h / h.sum()
    pa = h.sum(axis=1)
    pb = h.sum(axis=0)
    mi = 0.0
    for i in range(na):
        for j in range(nb):
            if h[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += h[i, j] * np.log(h[i, j] / (pa[i] * pb[j]))
    return mi


def mi_bias_corrected(x, y, n_perm=MI_PERM):
    rng = np.random.default_rng(RNG_SEED)
    na = nb = 12
    a, b = _discretize(x, na), _discretize(y, nb)
    obs = _mi_counts(a, b, na, nb)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(b)
        perm[k] = _mi_counts(a, b, na, nb)
    bs = max(obs - perm.mean(), 0.0)
    p_one = float((perm >= obs).mean()) if perm.size else 1.0
    return bs, p_one


def lag_design(counts, lag_list):
    """Stack shifted spike counts. Row t = [counts[t-l0], counts[t-l1], ...]
    for lag values in lag_list (negative shifts; lags <= 0 are current/past).
    Zero-padded at the run start. Shape (T, n_ch * len(lag_list))."""
    T, C = counts.shape
    cols = []
    for k in lag_list:
        shifted = np.concatenate([np.zeros((k, C)), counts[:T - k]])
        cols.append(shifted)
    return np.hstack(cols)


def _standard(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, mu, sd


def _ridge_pred(Xas, Kt, XtsT, xb, ytc, lam):
    alpha = np.linalg.solve(Kt + lam * np.eye(Kt.shape[0]), ytc)
    return Xas @ (XtsT @ alpha) + xb, alpha


def _rmse(pred, ya):
    return float(np.sqrt(np.mean((pred - ya) ** 2)))


def ridge_instant(counts, target):
    """Instantaneous readout (current tick's counts), interleaved splits."""
    X = counts.astype(np.float64)
    y = target.astype(np.float64)
    idx = np.arange(X.shape[0])
    tr, va, te = idx[::3], idx[1::3], idx[2::3]
    Xts, mu, sd = _standard(X[tr])
    Kt = Xts @ Xts.T
    Xqs = (X[va] - mu) / sd
    Xes = (X[te] - mu) / sd
    xb = y[tr].mean()
    ytc = y[tr] - xb
    best = (np.inf, LAM_GRID[0])
    for lam in LAM_GRID:
        pred, _ = _ridge_pred(Xqs, Kt, Xts.T, xb, ytc, lam)
        r = _rmse(pred, y[va])
        if r < best[0]:
            best = (r, lam)
    pred, _ = _ridge_pred(Xes, Kt, Xts.T, xb, ytc, best[1])
    mi, mi_p = mi_bias_corrected(pred, y[te])
    rho = 0.0
    if np.std(pred) > 1e-9 and np.std(y[te]) > 1e-9:
        rho = float(np.corrcoef(pred, y[te])[0, 1])
    return {"rmse": _rmse(pred, y[te]), "mi": mi, "mi_p": mi_p,
            "rho": rho, "lam": best[1]}


def ridge_walkforward(counts, target):
    """Causal best-linear readout. Features = strictly-past spike counts
    (lags 1..LAG) zero-padded at the run start. For each window, train on all
    ticks before the window (lambda chosen on the last fifth of the training
    block) and evaluate on the window. Returns per-window RMSE and the local-
    mean floor so carried info is a like-for-like comparison."""
    X = lag_design(counts.astype(np.float64), list(range(1, LAG + 1)))
    y = target.astype(np.float64)
    T = X.shape[0]
    windows = [(w0, min(w0 + WF_WINDOW, T)) for w0 in range(WF_WINDOW, T, WF_WINDOW)]
    rows = []
    for w0, w1 in windows:
        tr = np.arange(w0)
        val = tr[-max(len(tr) // 5, 5):] if len(tr) >= 10 else tr
        te = np.arange(w0, w1)
        Xts, mu, sd = _standard(X[tr])
        Kt = Xts @ Xts.T
        Xqs = (X[val] - mu) / sd
        Xes = (X[te] - mu) / sd
        xb = y[tr].mean()
        ytc = y[tr] - xb
        best = (np.inf, LAM_GRID[0])
        for lam in LAM_GRID:
            pred, _ = _ridge_pred(Xqs, Kt, Xts.T, xb, ytc, lam)
            r = _rmse(pred, y[val])
            if r < best[0]:
                best = (r, lam)
        pred, _ = _ridge_pred(Xes, Kt, Xts.T, xb, ytc, best[1])
        rmse_w = _rmse(pred, y[te])
        local_mean = float(np.mean(y[te]))
        rmse_mean_w = float(np.sqrt(np.mean(
            (np.full(len(te), local_mean) - y[te]) ** 2)))
        rows.append({"t": int(w0), "rmse": rmse_w, "meanpd": rmse_mean_w,
                     "carried": rmse_mean_w - rmse_w, "lam": best[1],
                     "pred": pred})
    all_pred = np.concatenate([r["pred"] for r in rows])
    all_y = np.concatenate([y[np.arange(r["t"], r["t"] + WF_WINDOW)].take(
        np.arange(min(WF_WINDOW, T - r["t"]))) for r in rows])
    mi, mi_p = mi_bias_corrected(all_pred, all_y)
    rho = 0.0
    if np.std(all_pred) > 1e-9 and np.std(all_y) > 1e-9:
        rho = float(np.corrcoef(all_pred, all_y)[0, 1])
    return {"windows": rows, "rmse": float(np.mean([r["rmse"] for r in rows])),
            "meanpd": float(np.mean([r["meanpd"] for r in rows])),
            "carried": float(np.mean([r["carried"] for r in rows])),
            "mi": mi, "mi_p": mi_p, "rho": rho}


def load_runs(pre):
    runs = {}
    for p in sorted(RES.glob("f1_capture%s_*_s*.json" % pre)):
        stem = p.stem
        if pre:
            body = stem[len("f1_capture%s_" % pre):]
        else:
            body = stem[len("f1_capture_"):]
        mode, seed = body.rsplit("_s", 1)
        seed = int(seed)
        if mode not in MODES:
            continue
        with open(p, encoding="utf-8") as fp:
            r = json.load(fp)
        runs[(mode, seed)] = {
            "counts": np.asarray(r["counts_matrix"], dtype=float),
            "cmd": np.asarray(r["cmd_series"], dtype=float),
        }
    return runs


def analyze_run(d):
    T = d["counts"].shape[0]
    vr = vref(T)
    cmd = d["cmd"][:T]
    out = {"canon_vref": _sch(cmd, vr)}
    out["meanpd_vref_global"] = meanpd(vr)
    out["meanpd_cmd_global"] = meanpd(cmd)
    out["ridgeX0_vref"] = ridge_instant(d["counts"][:T], vr)
    out["ridgeWF_vref"] = ridge_walkforward(d["counts"][:T], vr)
    out["ridgeWF_cmd"] = ridge_walkforward(d["counts"][:T], cmd)
    for k in ("ridgeWF_vref", "ridgeWF_cmd"):
        out[k]["windows"] = [
            {kk: r[kk] for kk in ("t", "rmse", "meanpd", "carried", "lam")}
            for r in out[k]["windows"]]
    out["carried_vref"] = out["ridgeWF_vref"]["carried"]
    out["carried_cmd"] = out["ridgeWF_cmd"]["carried"]
    out["canon_minus_ridge"] = out["canon_vref"]["rmse"] - out["ridgeWF_vref"]["rmse"]
    return out


def _agg(per_run, mode):
    s = sorted(per_run[mode])
    row = {}
    for sfx in ("canon_vref", "ridgeX0_vref", "ridgeWF_vref", "ridgeWF_cmd"):
        for kk in ("rmse", "mi", "mi_p", "rho", "meanpd", "carried"):
            if sfx == "canon_vref" and kk not in ("rmse", "rho"):
                continue
            if sfx == "ridgeX0_vref" and kk not in ("rmse", "mi", "mi_p", "rho"):
                continue
            if sfx in ("ridgeWF_vref", "ridgeWF_cmd") and kk not in (
                    "rmse", "meanpd", "carried", "mi", "mi_p", "rho"):
                continue
            vals = [per_run[mode][x][sfx][kk] for x in s]
            row[f"{sfx}_{kk}_mean"] = float(np.mean(vals))
            row[f"{sfx}_{kk}_se"] = float(np.std(vals) / np.sqrt(len(vals)))
    for sfx in ("carried_vref", "carried_cmd", "canon_minus_ridge"):
        vals = [per_run[mode][x][sfx] for x in s]
        row[sfx] = float(np.mean(vals))
        row[f"{sfx}_se"] = float(np.std(vals) / np.sqrt(len(vals)))
    row["n"] = len(s)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="",
                        help="capture suffix, e.g. 'izh' reads f1_capture_izh_*")
    args = parser.parse_args()
    pre = ("_" + args.tag) if args.tag else ""
    runs = load_runs(pre)
    if not runs:
        print("no captures for tag=%r" % args.tag)
        return

    per_run = {m: {} for m in MODES}
    for (mode, seed), d in sorted(runs.items()):
        per_run[mode][seed] = analyze_run(d)

    pooled = {m: _agg(per_run, m) for m in MODES}

    floor = pooled["poisson"]["ridgeWF_vref_carried_mean"]
    floor_cmd = pooled["poisson"]["ridgeWF_cmd_carried_mean"]
    n = pooled["poisson"]["n"]
    crit = {
        "carried_vref_null_floor": float(floor),
        "carried_cmd_null_floor": float(floor_cmd),
        "carried_vref_neural": float(pooled["neural"]["carried_vref"]),
        "carried_vref_zero": float(pooled["zero"]["carried_vref"]),
        "carried_vref_random": float(pooled["random"]["carried_vref"]),
        "carried_vref_mask0.5": float(pooled["mask0.5"]["carried_vref"]),
        "carried_cmd_neural": float(pooled["neural"]["carried_cmd"]),
        "carried_vref_neural_minus_null": float(
            pooled["neural"]["carried_vref"] - floor),
        "carried_cmd_neural_minus_null": float(
            pooled["neural"]["carried_cmd"] - floor_cmd),
        "canon_minus_ridge_wf_neural": float(
            pooled["neural"]["canon_minus_ridge"]),
        "ridgeWF_vref_mi_sig_frac_neural": float(np.mean(
            [per_run['neural'][x]['ridgeWF_vref']['mi_p'] < 0.05
             for x in per_run['neural']])),
        "ridgeWF_vref_mi_sig_frac_poisson": float(np.mean(
            [per_run['poisson'][x]['ridgeWF_vref']['mi_p'] < 0.05
             for x in per_run['poisson']])),
        "ridgeX0_vref_mi_sig_frac_poisson": float(np.mean(
            [per_run['poisson'][x]['ridgeX0_vref']['mi_p'] < 0.05
             for x in per_run['poisson']])),
        "null_still_dead": bool(floor <= 0.02),
        "neural_above_null_vref": bool(
            pooled["neural"]["carried_vref"] - floor > 0.03),
        "neural_above_null_cmd": bool(
            pooled["neural"]["carried_cmd"] - floor_cmd > 0.03),
        "readout_improvable": bool(pooled["neural"]["canon_minus_ridge"] > 0.05),
        "n_seeds": int(n),
    }

    verdict = (
        "Gate D (closed loop, walk-forward lag=%d, n=%d): dead-null carries "
        "vref %.3f / cmd %.3f (stays dead=%s); neural carries vref %.3f "
        "(%+.3f vs null) and cmd %.3f (%+.3f vs null); zero %.3f, random %.3f, "
        "mask0.5 %.3f; the causal linear readout beats the loop's fixed readout "
        "by %.3f RMSE on vref (readout improvable=%s)."
        % (LAG, n, floor, floor_cmd, crit["null_still_dead"],
           pooled["neural"]["carried_vref"], crit["carried_vref_neural_minus_null"],
           pooled["neural"]["carried_cmd"], crit["carried_cmd_neural_minus_null"],
           pooled["zero"]["carried_vref"], pooled["random"]["carried_vref"],
           pooled["mask0.5"]["carried_vref"], crit["canon_minus_ridge_wf_neural"],
           crit["readout_improvable"]))

    summary = {
        "gate": "closed-loop readout-invariance (Gate D, ported from open-loop mini)",
        "tag": args.tag,
        "modes": MODES,
        "lag_ticks": LAG,
        "walkforward_window_ticks": WF_WINDOW,
        "targets": ["vref (task profile)", "cmd (loop-issued command)"],
        "decoders": ["canon (loop's own cmd_series)", "meanpd (floor)",
                     "ridgeX0 (instantaneous, reference)", "ridgeWF (causal walk-forward)"],
        "caveat": "closed loop: non-poisson spikes encode plant state through "
                  "the encoder; Gate D falsifies the decode direction.",
        "per_run": per_run,
        "pooled": pooled,
        "criteria": crit,
        "verdict": verdict,
    }
    out_json = RES / ("gate_d%s.json" % pre)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=1)
    _figure(summary, RES / ("gate_d%s.png" % pre))
    print("saved:", out_json)
    print("== Gate D pooled (carried = window-local-mean - walk-forward ridge) ==")
    for mode in MODES:
        p = pooled[mode]
        print(f"  {mode:<8} n={p['n']} canon={p['canon_vref_rmse_mean']:.3f} "
              f"ridgeX0={p['ridgeX0_vref_rmse_mean']:.3f} "
              f"ridgeWF={p['ridgeWF_vref_rmse_mean']:.3f} "
              f"meanpd={p['ridgeWF_vref_meanpd_mean']:.3f} "
              f"carried_vref={p['carried_vref']:+.3f} "
              f"carried_cmd={p['carried_cmd']:+.3f}")
    print("verdict:", verdict)


def _figure(summary, path):
    p = summary["pooled"]
    modes = summary["modes"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    x = np.arange(len(modes)); w = 0.22
    series = [("canon", "canon_vref_rmse", "#2c7fb8"),
              ("ridgeX0", "ridgeX0_vref_rmse", "#fdb863"),
              ("ridgeWF", "ridgeWF_vref_rmse", "#41ab5d")]
    for i, (label, key, color) in enumerate(series):
        vals = [p[m][f"{key}_mean"] for m in modes]
        errs = [p[m][f"{key}_se"] for m in modes]
        axes[0].bar(x + (i - 1) * w, vals, w, color=color, yerr=errs,
                    capsize=3, label=label)
    mp = [p[m]["ridgeWF_vref_meanpd_mean"] for m in modes]
    axes[0].plot(x, mp, "k--", label="window-local meanpd")
    axes[0].set_xticks(x, modes)
    axes[0].set_ylabel("RMSE vs vref")
    axes[0].set_title("canon vs best-causal-linear vs mean-floor")
    axes[0].legend(fontsize=8)
    colors = ["#2c7fb8" if m != "poisson" else "#fdb863" for m in modes]
    axes[1].bar(x, [p[m]["carried_vref"] for m in modes], 0.5, color=colors)
    axes[1].axhline(p["poisson"]["carried_vref"], color="k", ls="--", lw=0.8,
                    label="dead-null floor")
    axes[1].set_xticks(x, modes)
    axes[1].set_ylabel("carried_vref (win' meanpd - ridgeWF)")
    axes[1].set_title("Substrate info above trivial-mean readout")
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [p[m]["carried_cmd"] for m in modes], 0.5)
    axes[2].axhline(0.0, color="k", lw=0.6)
    axes[2].set_xticks(x, modes)
    axes[2].set_ylabel("carried_cmd (spikes -> command)")
    axes[2].set_title("Causal command info in the spike stream")
    fig.suptitle("Gate D (closed loop%s, walk-forward lag=%d): %s" % (
        (" [%s]" % summary["tag"]) if summary["tag"] else "",
        summary["lag_ticks"], summary["verdict"]))
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    print("saved:", path)


if __name__ == "__main__":
    main()