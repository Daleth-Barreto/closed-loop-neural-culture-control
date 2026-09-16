# Reproduction guide

Everything needed to reproduce the results in `paper/` is in this repository.
This document gives the environment, the command map, the protocol definitions,
and the provenance of every figure.

## Environment

Verified on Windows 11 with Python 3.12.13 and `uv` 0.11.32.

```
mujoco==3.13.0
nengo==4.1.0
torch==2.14.0+cpu
numpy==2.5.3
scipy==1.18.1
matplotlib==3.11.2
cl-sdk==1.0.0        (CC BY-NC, academic use)
```

Full pins are in `requirements.lock.txt` (`uv freeze`). One-shot bootstrap
(Windows PowerShell):

```powershell
.\scripts\bootstrap.ps1
```

POSIX users: `uv venv .venv --python 3.12` then
`uv pip install --python .venv/bin/python -r requirements.lock.txt`.

## Protocol definitions (fixed)

- Loop cadence: 40 ticks/s; contract tick = 25 ms; run duration = 12 s.
- Channels (64): 0-11 torque overlay, 12/13 roll/pitch, 23 pelvis height,
  62 task/vx override (the load-bearing channel), 63 hub context.
- Spike event: downward crossing of the frame threshold over consecutive
  frames (trigger level fixed at 1400 sample units).
- Seeds: `{1, 7, 13, 29, 55, 91}` (6). The Nengo hub is deterministic;
  seed variance comes from the simulator RNG, the substrate/event RNG, the
  lesion draw, and real-time hub chunking.
- Profiles (5): `baseline`, `multi_step`, `sine`, `descend`, `pulse`.
- Modes (5): `neural`, `zero`, `random`, `mask0.5`, `poisson`.
- Null rationale: `zero` issues no command (scores the profile itself);
  `random` couples a fixed random decoder (separates decoder coupling from
  substrate information); `mask0.5` zeroes 50% of electrodes per seed and can
  erase the task channel entirely; `poisson` replaces frames with Poisson
  draws matched in expected spike rate but independent of state.

## Command map

| Command | Produces | Reading |
|---|---|---|
| `python src/task_tracking.py` | `results/task_tracking.json`/`.png` | primary battery, rate-code substrate |
| `python src/task_tracking.py --substrate izh` | `results/task_tracking_izh.json`/`.png` | substrate interchangeability |
| `python src/capture_signal.py` | `results/f1_capture_*.json` | Gate A raw captures (5 modes x {7,29}) |
| `python src/signal_analysis.py` | `results/f2_signal_summary.json` + fig | Gate A MI/TE (rate) |
| `python src/capture_signal.py --substrate izh && python src/signal_analysis.py --tag izh` | `results/f2_signal_izh.*` | Gate A MI/TE (Izhikevich) |
| `python src/gate_d.py` | `results/gate_d.json`/`gate_d.png` | Gate D (rate): readout-invariance (6 seeds) |
| `python src/gate_d.py --tag izh` | `results/gate_d_izh.json`/`gate_d_izh.png` | Gate D (Izhikevich, 6 seeds) |
| `python src/terrain_probe.py` | `results/terrain_probe.json`/`terrain_probe2.json` | terrain battery |
| `python src/terrain_analysis.py` | `results/terrain_stats.json` + `terrain_summary.png` | Fisher/Welch stats + figure |
| `python src/probe_frontier.py` | `results/push_frontier.json` | lateral impulse frontier |
| `python src/envelope.py` | `results/envelope.json` + fig | authority x latency envelope |
| `python src/perturb_sweep.py` | `results/perturb_sweep.json` + fig | flat-ground lateral robustness |
| `python paper/figures/build_figures.py` | `paper/figures/fig*.png` | paper figures from JSONs |

`python src/task_tracking.py --help` documents the seed/profile/mode
overrides, including the reduced (fewer-seed) batteries.

## Figure provenance (paper -> data)

| Figure | Source JSON |
|---|---|
| fig1 pipeline (drawn schematic) | n/a |
| fig2 tracking (rate | Izhikevich) | `results/task_tracking.json`, `results/task_tracking_izh.json` |
| fig4 terrain | `results/terrain_stats.json` |
| fig5 information (MI / TE) | `results/f2_signal_summary.json`, `results/f2_signal_izh.json` |

No figure in this repository is hand-typed; every render reads a committed JSON
under `results/`.

## Gate D (readout-invariance, closed loop)

Post-ICRA falsification (analysis-only, committed f1 captures; **n = 6 seeds
{1,7,13,29,55,91}** per mode). Question: is the tracking advantage recoverable
from the recorded spike counts by the BEST causal linear readout, or does it
live only in the fixed `decide` readout? Decoders per (mode, seed):

- **canon** - the loop's own command (cmd_series) vs the analytic profile vref.
- **meanpd** - window-local mean predictor (floor).
- **ridgeX0** - instantaneous linear readout (reference).
- **ridgeWF** - walk-forward causal readout, strictly past lags 1..48 ticks
  (= the hub's 1.2 s window), trained chronologically, tested on each 100-tick
  window covering both profile transitions. L2 via the Woodbury identity;
  lambda chosen on the last fifth of each training block.

Carried info = window-mean RMSE - walk-forward RMSE, pooled per mode.
Captures: seeds 1,13,55,91 captured fresh (30 runs per substrate); seeds 7,29
reuse the committed captures. `results/gate_d_battery_index{,_izh}.json` is the
merged manifest.

| mode     | rate carried_vref | rate carried_cmd | izh carried_vref | izh carried_cmd |
|----------|-------------------|------------------|------------------|-----------------|
| neural   | -0.073 +/- 0.025  | -0.038 +/- 0.013 | -0.072 +/- 0.016 | -0.043 +/- 0.007 |
| zero     | -0.062 +/- 0.013  |  0.000 +/- 0.000 | -0.070 +/- 0.016 |  0.000 +/- 0.000 |
| random   | -0.078 +/- 0.027  | -0.078 +/- 0.018 | -0.069 +/- 0.013 | -0.102 +/- 0.018 |
| mask0.5  | -0.171 +/- 0.079  | -0.072 +/- 0.065 | -0.189 +/- 0.064 | -0.088 +/- 0.074 |
| poisson  | -0.114 +/- 0.011  | -0.014 +/- 0.007 | -0.114 +/- 0.011 | -0.014 +/- 0.005 |

Reading (paired over the 6 seeds; one-sided p for "mode better than null"):

1. The dead-substrate null is validated and reproducible: poisson carries vref
   -0.114 (identical in both substrates) and command -0.014; the matched-dead
   substrate is a trivial-mean decoder.
2. The neural substrate does carry TASK information significantly above the
   dead null: carried_vref margin +0.041 (rate, t(5)=3.00, p ~ 0.015) and
   +0.042 (izh, t(5)=8.27, p < 0.001). But in ABSOLUTE terms all modes sit at
   or below the within-window mean (carried_vref <= 0); the linear readout
   never beats a dummy predictor in any mode.
3. The neural advantage over the uncoupled/disabled loops is NOT linearly
   recoverable: neural vs zero (d ~ -0.01, n.s.) and vs random (d ~ 0, n.s.)
   on carried_vref. The canonical battery's tracking separation is therefore a
   whole-loop (plant + inertia + nonlinear readout) effect, NOT a
   readout-invariant property of the spike stream. Readout-invariance at the
   linear level is FALSIFIED.
4. The spike -> command causal edge (Gate A's TE) does not survive as
   linearly-decodable structure: neural carried_cmd is below the null
   (d = -0.024/-0.030, both p -> 1 for "above null"). The command is not in
   the spikes in any linear way; the ch62 -> cmd coupling runs through the
   fixed nonlinear `decide` path.
5. The fixed `decide` readout is far from optimal (canon RMSE 0.234-0.261 vs
   the walk-forward vref decode ~0.14-0.18 and the window-mean floor 0.067):
   readout improvable.

Status: FULL STATISTICS (n = 6). Any claim extending the ICRA paper must keep
this falsification in the intro/limitations: surviving claims are (a) a
validated matched dead-substrate null, (b) substrate-relative-but-not-
absolute task information, (c) a real, significant, signed whole-loop
behavioral effect; the claim that the substrate is a readout-invariant
computation is not supported.

## Canonical numbers

- Tracking (rate): neural best on all 5 profiles, 6/6 seeds same direction;
  one-sided exact p = 1/64 = 0.0156 and two-sided p = 2/64 = 0.03125 per
  comparison (5 profiles x 3 nulls). Baseline neural RMSE 0.238 +/- 0.010
  vs zero 0.612 / random 0.337 / mask0.5 0.397 / poisson 0.306; Cohen d_z
  = -38.9 / -8.6 / -2.4.
- Gate A (rate): MI(cmd; plant) 0.0447 neural, 0.0000 zero; lag-2
  TE(c62 -> cmd) 0.0079 neural, exactly 0.0000 for zero/mask0.5/poisson.
- Gate B (Izhikevich): full ordering reproduces at full 6-seed power; baseline
  neural RMSE 0.257 vs 0.612/0.337/0.416/0.307; Gate A MI 0.0258, TE 0.0011.
- Terrain: curb 5 cm reference 6/6 vs neural 1/6 (Fisher p=0.0076, neural
  worse); rough reference 0/6 vs neural 2/6, distance 3.89 vs 5.87 m
  (Welch t(5.1) = 8.63, p = 0.0011, neural greater).
- Authority: torque headroom ~30 N m vs ~112 N m tipping moment (sustained
  140 N push); lateral impulse closable for duration < 0.45 s at <= 140 N;
  longitudinal forward/back push asymmetry recorded.
- Latency: contract tick 25 ms of loop time; per-tick wall-clock cadence is
  logged in every run (may overrun the 25 ms wall-clock deadline on slower
  hosts); spike-window (information) latency up to 1.2 s (600 hub steps at
  2 ms). The loop is liquidity-limited, not contract-limited.

## Licenses

See `NOTICE` and `LICENSE`. `cl-sdk` is CC BY-NC (non-commercial); the G1
assets under `third_party/unitree_rl_gym/` retain Unitree's license.