# Closed-loop neural-culture control on a G1 humanoid

Closed-loop experimental scaffold and manuscript: does a load-bearing control
function actually travel through a spiking culture substrate and back into a
robot loop?

A Unitree G1 humanoid (MuJoCo, frozen RL policy) is driven through the full
measurement contract of a closed-loop cortical platform (64 electrodes,
25 kHz frames, spike threshold, committed stimulation). Sensory state is
encoded to electrodes; spike events are read from a substrate (rate-code
transducer or Izhikevich microcircuit); a fixed Nengo-LIF hub decodes a
forward-velocity command; stimulation commits that command back onto the plant
within the 25 ms contract tick.

The central result is a falsifiable ablation ordering: the intact loop tracks
velocity task profiles significantly better than a zero command, an uncoupled
random decoder, a 50% electrode lesion, and a dead Poisson substrate whose
spikes are statistically matched but carry no state information. Six seeds,
exact paired permutation statistics (one-sided p floor 1/64), mutual
information and transfer entropy along the causal chain, substrate
interchangeability, terrain outcomes with opposite-sign statistics, and an
explicit authority/latency boundary are all documented in `results/` and in the
manuscript (`paper/`).

## Layout

```
paper/                  manuscript (IEEEtran, ICRA style) + figure scripts
src/                    experiment code (batteries, gates, analysis)
deploy/                 Deploy12 G1 policy loader (vendored, self-contained)
config/                 scene configs for the flat and terrain scenes
data/g1_description/    terrain XML scenes (meshes referenced from third_party)
third_party/unitree_rl_gym/  G1 model assets + frozen policy checkpoint
results/                every committed JSON result and figure
docs/REPRODUCE.md       reproduction guide (environment, commands, provenance)
scripts/bootstrap.ps1   one-shot environment bootstrap (Windows)
requirements.lock.txt   pinned environment (uv freeze)
NOTICE                  third-party and license notes (cl-sdk is CC BY-NC)
```

## Quick start

```powershell
# Windows, PowerShell 5.1+, uv installed
.\scripts\bootstrap.ps1
.\.venv\Scripts\activate
python src\task_tracking.py          # primary battery (5 profiles x 5 modes x 6 seeds)
python paper\figures\build_figures.py  # regenerate paper figures from results/
```

See `docs/REPRODUCE.md` for the full command map, the seed/mode/protocol
definitions, and which committed JSON each figure is drawn from.

## Manuscript

- Signed version: `paper/main.tex` -> `main.pdf`
- Double-anonymous review version: `paper/main_anon.tex` -> `main_anon.pdf`

Compile with `pdflatex` (two passes). Figures are produced from the committed
JSONs by `paper/figures/build_figures.py`.

## License and third-party notices

- This repository's own code is MIT; see `LICENSE`.
- The experimental environment installs `cl-sdk` (Cortical Labs), which is
  **CC BY-NC**: academic / non-commercial use only. See `NOTICE`.
- The G1 model and the frozen policy checkpoint are vendored from
  `unitree_rl_gym` (Unitree Robotics) with its license included under
  `third_party/unitree_rl_gym/LICENSE`.

No biological tissue is used and no physical closed-loop platform is required:
the culture substrate is simulated in-process, and the whole pipeline runs on a
CPU-only machine.