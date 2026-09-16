"""Build paper figures from the committed result JSONs.

Reads only the committed, immutable artefacts under results/ and emits
publication figures into paper/figures/. Runs with numpy + matplotlib only
(no cl-sdk, no MuJoCo, no network). Reproducible: same inputs, same figures.

Figures:
  fig1_pipeline.png    Closed-loop block diagram (drawn, not data)
  fig2_tracking.png    Tracking RMSE x 5 profiles x 5 modes (rate | Izhikevich)
  fig4_terrain.png     Terrain survival + distance (curb 5 cm, rough)
  fig5_information.png MI / TE causal evidence (rate | Izhikevich)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

DPI = 300
COL_W = 3.49   # inches (single IEEE column)
FULL_W = 7.17  # inches (two-column width)

MODES = ["neural", "zero", "random", "mask0.5", "poisson"]
MODE_LABELS = {
    "neural": "neural",
    "zero": "zero",
    "random": "random",
    "mask0.5": "mask0.5\n(lesion)",
    "poisson": "poisson\n(dead)",
}
MODE_COLORS = {
    "neural": "#2c7fb8",
    "zero": "#636363",
    "random": "#e08214",
    "mask0.5": "#7570b3",
    "poisson": "#e7298a",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
})


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def task_rows(track, profile: str):
    return track["profiles_data"][profile]["rows"]


def fig1_pipeline():
    """Closed-loop block diagram: sensors -> frames -> culture -> spikes ->
    hub decode -> stimulation -> plant (the task channel 62 injects vx)."""
    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.44))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    box = dict(boxstyle="round,pad=0.35", facecolor="#f4f7f9",
               edgecolor="#37474f", linewidth=1.1)
    arw = dict(arrowstyle="-|>", color="#37474f", lw=1.4,
               mutation_scale=16, shrinkA=0, shrinkB=0)

    def draw(x, y, w, h, text, box=box, fs=8, color="#37474f"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                bbox=box, zorder=3, color=color)
        return (x - w / 2, x + w / 2, y - h / 2, y + h / 2)

    # main horizontal chain on y = 3.0
    x0, ymain = 0.55, 3.0
    w1 = 1.5
    boxes = [
        (x0, "sensors\n(qpos, qvel)", "attitude, height,\njoint errors"),
        (x0 + 2.1, "electrode frames\n64 ch @ 25 kHz", "spike buffer"),
        (x0 + 4.2, "culture substrate\n(rate-code | Izhikevich)", "spike trains"),
        (x0 + 6.3, "Nengo LIF hub\ndecode + $decide$", "2-D command"),
        (x0 + 8.45, "stim -> plant\nG1 in MuJoCo", "physics"),
    ]
    for x, name, sub in boxes:
        ax.text(x, ymain + 0.42, name, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#e3f2fd",
                          edgecolor="#1565c0", linewidth=1.1), zorder=3)
        ax.text(x, ymain - 0.42, sub, ha="center", va="center", fontsize=6.5,
                style="italic", color="#455a64")
    for a, b in zip([b[0] for b in boxes[:-1]], [b[0] for b in boxes[1:]]):
        ax.annotate("", xy=(b - 0.42, ymain), xytext=(a + 0.42, ymain), arrowprops=arw)

    # task / reference top-left
    ax.text(0.55, 4.05, "task channel 62:\ncommanded vx profile", ha="left",
            va="center", fontsize=7, bbox=dict(boxstyle="round,pad=0.3",
            facecolor="#fff8e1", edgecolor="#f9a825", linewidth=1.0))
    ax.annotate("", xy=(x0 + 2.1 - 0.2, ymain + 0.55), xytext=(0.95, 3.85),
                arrowprops=dict(arrowstyle="-|>", color="#f9a825", lw=1.4))

    # bottom return path: plant -> sensors
    ax.annotate("", xy=(x0 - 0.1, ymain - 0.8), xytext=(boxes[-1][0], ymain - 0.8),
                arrowprops=arw)
    ax.annotate("", xy=(x0 - 0.1, ymain - 1.3), xytext=(x0 - 0.1, ymain - 0.8),
                arrowprops=arw)
    ax.text(boxes[-1][0] + 0.2, ymain - 0.8, "state feedback", ha="center",
            va="top", fontsize=7, style="italic", color="#455a64")

    # perturbation / terrain
    ax.text(boxes[-1][0], 1.1, "perturbation (impulse / push)\n| terrain scenes",
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fce4ec",
                      edgecolor="#c62828", linewidth=1.0))
    ax.annotate("", xy=(boxes[-1][0], ymain - 0.35), xytext=(boxes[-1][0], 1.4),
                arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=1.2))

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig1_pipeline.png", transparent=False)
    plt.close(fig)


def fig2_tracking():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42))
    for ax, name, tag in ((axes[0], "task_tracking.json", "rate-code substrate"),
                          (axes[1], "task_tracking_izh.json",
                           "Izhikevich substrate")):
        track = load(name)
        profiles = track["profiles"]
        rows = {m: [] for m in MODES}
        for p in profiles:
            r = task_rows(track, p)
            for m in MODES:
                rows[m].append((r[m]["rmse_mean"], r[m]["rmse_std"]))
        x = np.arange(len(profiles))
        width = 0.8 / len(MODES)
        for i, m in enumerate(MODES):
            means = [v[0] for v in rows[m]]
            errs = [v[1] for v in rows[m]]
            ax.bar(x + (i - len(MODES) / 2 + 0.5) * width, means, width,
                   yerr=errs, capsize=1.5, label=MODE_LABELS[m],
                   color=MODE_COLORS[m], edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles)
        ax.set_ylim(0, 0.75)
        ax.set_title(name.endswith("izh.json") and "Izhikevich substrate"
                     or "rate-code substrate", fontsize=9)
        ax.set_xlabel("task profile")
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].set_ylabel("tracking RMSE  (lower = better)")
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.02, -0.30),
                   frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig2_tracking.png")
    plt.close(fig)


def fig4_terrain():
    ts = load("terrain_stats.json")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.4))
    scenes = ["curb", "rough"]
    labels = ["curb 5 cm", "rough bumps"]
    x = np.arange(2)
    w = 0.36
    # panel a: survival fraction
    ax = axes[0]
    ref_s = [ts["scenes"][s]["reference"]["survival"] for s in scenes]
    neu_s = [ts["scenes"][s]["neural"]["survival"] for s in scenes]
    ax.bar(x - w / 2, ref_s, w, label="reference policy (open loop)",
           color="#546e7a", edgecolor="white")
    ax.bar(x + w / 2, neu_s, w, label="closed loop (neural)",
           color="#2c7fb8", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel("survival fraction")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.text(0, 1.08, "Fisher p=0.0076\n(neural degrades)", ha="center",
            fontsize=6.5, color="#c62828")
    ax.text(1, 1.08, "n.s.", ha="center", fontsize=6.5, color="#37474f")
    # panel b: forward distance
    ax = axes[1]
    ref_d = [ts["scenes"][s]["reference"]["dist_mean"] for s in scenes]
    neu_d = [ts["scenes"][s]["neural"]["dist_mean"] for s in scenes]
    ref_e = [ts["scenes"][s]["reference"]["dist_sd"] for s in scenes]
    neu_e = [ts["scenes"][s]["neural"]["dist_sd"] for s in scenes]
    ax.bar(x - w / 2, ref_d, w, yerr=ref_e, capsize=3,
           label="reference", color="#546e7a", edgecolor="white")
    ax.bar(x + w / 2, neu_d, w, yerr=neu_e, capsize=3,
           label="neural", color="#2c7fb8", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 7.5); ax.set_ylabel("forward distance (m)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.axhline(6.0, color="#555", ls="--", lw=0.8)
    ax.text(1, 6.9, "Welch t=8.63\np=0.0011 (neural > ref)",
            ha="center", fontsize=6.5, color="#00695c")
    axes[0].legend(loc="lower center", frameon=False, fontsize=6.5)
    axes[1].legend(loc="upper center", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig4_terrain.png")
    plt.close(fig)


def fig5_information():
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, FULL_W * 0.5))
    pairs = [("f2_signal_summary.json", "rate-code"),
             ("f2_signal_izh.json", "Izhikevich")]
    for r, (name, label) in enumerate(pairs):
        agg = load(name)["aggregate"]
        mi = [agg[m]["mi_cmd_plant"] for m in MODES]
        te = [agg[m]["te_c62_to_cmd_lag2"] for m in MODES]
        ax = axes[r][0]
        ax.bar(np.arange(len(MODES)), mi, color=[MODE_COLORS[m] for m in MODES],
               edgecolor="white")
        ax.set_title("%s: MI(cmd; plant)" % label, fontsize=8)
        ax.set_xticks(np.arange(len(MODES)))
        ax.set_xticklabels(list(MODE_LABELS.values()), fontsize=6)
        ax.set_ylim(0, 0.06); ax.grid(axis="y", ls=":", alpha=0.4)
        ax = axes[r][1]
        ax.bar(np.arange(len(MODES)), te, color=[MODE_COLORS[m] for m in MODES],
               edgecolor="white")
        ax.set_title("%s: TE(c62 -> cmd)" % label, fontsize=8)
        ax.set_xticks(np.arange(len(MODES)))
        ax.set_xticklabels(list(MODE_LABELS.values()), fontsize=6)
        ax.set_ylim(0, 0.02); ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0][0].set_ylabel("nats"); axes[1][0].set_ylabel("nats")
    axes[1][0].set_xlabel("mode"); axes[1][1].set_xlabel("mode")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig5_information.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_tracking()
    fig4_terrain()
    fig5_information()
    print("figures written to", OUT)