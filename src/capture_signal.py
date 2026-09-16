"""F1: aligned capture of per-tick spike counts + plant state, per mode.

Saves a JSON per (mode, seed): counts_matrix[64], x (hub input 5-dim), and
plant state (h, vx, roll, pitch, cmd) aligned per tick. Task-driven profile
(PROFILE) is used so that the command signal is identifiable for MI/TE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import ablate_loop as al  # noqa: E402  (registers bridge_g1 datasource)

SEEDS = [7, 29]
PROFILE = [(0.0, 0.5), (4.0, 0.8), (8.0, 0.5)]
RES = BASE.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", default="rate",
                        choices=["rate", "izh"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="seeds to capture (default: %(default)s override -> [7, 29])")
    parser.add_argument("--index", default=None,
                        help="index filename override (default f1_capture_index.json)")
    args = parser.parse_args()
    seeds = SEEDS if args.seeds is None else args.seeds
    tag = "" if args.substrate == "rate" else ("_" + args.substrate)
    index_name = args.index or ("f1_capture_index%s.json" % tag)
    RES.mkdir(exist_ok=True)
    runs = []
    for mode in al.MODES:
        for seed in seeds:
            r = al.run_mode(mode, seed=seed, task=PROFILE, capture=True,
                            substrate=args.substrate)
            name = f"f1_capture{tag}_{mode}_s{seed}.json"
            with open(RES / name, "w", encoding="utf-8") as fp:
                json.dump(r, fp)
            runs.append({"mode": mode, "seed": seed, "file": name,
                         "ticks": r["ticks"], "wall": r["_wall_step"],
                         "fallen": r["walker"]["fallen"],
                         "n_spikes_mean": r["mean_nspk"]})
            print(f"mode={mode:<8} seed={seed:<3} ticks={r['ticks']:<4} "
                  f"wall={r['_wall_step']:>5.1f}s fallen={r['walker']['fallen']} "
                  f"nspk={r['mean_nspk']}", flush=True)
    with open(RES / index_name, "w",
              encoding="utf-8") as fp:
        json.dump({"duration_sec": al.DURATION_SEC, "tps": al.TPS,
                   "profile": PROFILE, "seeds": seeds, "threshold": al.THR,
                   "substrate": args.substrate,
                   "poisson_lam": al.POISSON_LAM,
                   "notes": "task-driven capture; counts[64]+state per tick "
                            "(steps: counts,x aligned by tick.iteration; state "
                            "hold-last from bridge TSV at 50 Hz)",
                   "runs": runs}, fp, indent=2)
    print("saved index:", RES / index_name)


if __name__ == "__main__":
    main()