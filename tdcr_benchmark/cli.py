from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tdcr_benchmark.config import default_config_path
from tdcr_benchmark.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Python TDCR benchmark.")
    parser.add_argument("--config", default=str(default_config_path()), help="Path to config.yaml.")
    parser.add_argument("--output-root", default="outputs", help="Directory where timestamped benchmark outputs are written.")
    parser.add_argument("--tensions", nargs=6, type=float, help="Override the six tendon tensions in Newtons.")
    parser.add_argument("--no-timestamp", action="store_true", help="Write directly into --output-root instead of a timestamped subdirectory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tensions = np.array(args.tensions, dtype=float) if args.tensions is not None else None
    output_dir, results = run_benchmark(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        tensions=tensions,
        timestamped=not args.no_timestamp,
    )
    print(f"Wrote benchmark outputs to {output_dir}")
    for result in results:
        status = "success" if result.solver_success else "not converged"
        print(f"{result.model_name}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
