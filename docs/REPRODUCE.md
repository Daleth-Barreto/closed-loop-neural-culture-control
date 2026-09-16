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
- Latency: contract tick 25 ms respected; spike-window (information) latency
  up to 1.2 s (600 hub steps at 2 ms). The loop is liquidity-limited, not
  contract-limited.

## Licenses

See `NOTICE` and `LICENSE`. `cl-sdk` is CC BY-NC (non-commercial); the G1
assets under `third_party/unitree_rl_gym/` retain Unitree's license.