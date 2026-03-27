from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_thesis.config import ExperimentConfig
from surrogate_thesis.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the surrogate thesis experiment pipeline.")
    parser.add_argument(
        "--config",
        default="configs/default_experiment.json",
        help="Path to an experiment configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under artifacts/.",
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    summary = run_experiment(config=config, output_dir=args.output_dir)
    print(f"Run completed: {summary['run_name']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Best model: {summary['best_model']}")


if __name__ == "__main__":
    main()
