"""
Simulate a match and save it as a JSON-lines file of StreamMessages.

Usage:
    python -m koora.simulator.run                         # 10 minutes, seed 7
    python -m koora.simulator.run --minutes 20 --seed 3
    python -m koora.simulator.run --minutes 90 --out data/matches/full.jsonl

Then open http://127.0.0.1:8000/viewer.html (with the API running) to watch it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from koora.simulator.engine import MatchSimulator

DEFAULT_DIR = Path("data") / "matches"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate a synthetic football match.")
    parser.add_argument("--minutes", type=float, default=10.0, help="match length in minutes (split into two halves)")
    parser.add_argument("--seed", type=int, default=7, help="random seed: same seed, same match")
    parser.add_argument("--out", type=Path, default=None, help="output file (default data/matches/sim-<seed>.jsonl)")
    parser.add_argument(
        "--frame-every",
        type=int,
        default=1,
        help="write every Nth tracking frame (1 = full 25 Hz, 2 = 12.5 Hz, smaller files)",
    )
    args = parser.parse_args(argv)

    out = args.out or DEFAULT_DIR / f"sim-{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    sim = MatchSimulator(minutes=args.minutes, seed=args.seed, match_id=out.stem, frame_every=args.frame_every)
    started = time.perf_counter()
    lines = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for line in sim.run():
            f.write(line + "\n")
            lines += 1
    elapsed = time.perf_counter() - started

    home, away = sim.teams
    print(f"Wrote {lines:,} messages to {out} in {elapsed:.1f} s", file=sys.stderr)
    print(f"{home.name} {sim.score[0]} - {sim.score[1]} {away.name}", file=sys.stderr)
    for key, value in sim.summary().items():
        print(f"  {key:22} {value}", file=sys.stderr)


if __name__ == "__main__":
    main()
