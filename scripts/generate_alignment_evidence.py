"""Generate CSV evidence that forecast targets are correctly time-aligned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Load saved artifacts and write alignment-check CSV files."""

    parser = argparse.ArgumentParser(
        description="Generate alignment evidence for saved forecasting artifacts."
    )
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        default="artifacts/decorator_architecture_current",
        help="Experiment artifact directory containing resolved_config.json and test_predictions.csv.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of leading test samples to include in alignment_check.csv.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    config = json.loads((artifact_dir / "resolved_config.json").read_text())
    dataset_config = config["dataset"]
    lookback = int(dataset_config["lookback"])
    horizon = int(dataset_config["horizon"])
    target_column = str(dataset_config["target_column"])

    raw_frame = pd.read_csv(artifact_dir / "data" / "raw_simulation.csv", parse_dates=["timestamp"])
    predictions = pd.read_csv(artifact_dir / "test_predictions.csv", parse_dates=["timestamp"])

    # The first file traces selected prediction rows back to the raw simulator
    # frame. The second checks whether small time shifts would improve MAE.
    alignment = build_alignment_check(
        raw_frame=raw_frame,
        predictions=predictions,
        target_column=target_column,
        lookback=lookback,
        horizon=horizon,
        sample_count=args.samples,
    )
    shift_check = build_shift_check(predictions)

    alignment_path = artifact_dir / "alignment_check.csv"
    shift_path = artifact_dir / "alignment_shift_check.csv"
    alignment.to_csv(alignment_path, index=False)
    shift_check.to_csv(shift_path, index=False)

    print(f"Wrote {alignment_path}")
    print(f"Wrote {shift_path}")


def build_alignment_check(
    *,
    raw_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    target_column: str,
    lookback: int,
    horizon: int,
    sample_count: int,
) -> pd.DataFrame:
    """Map prediction rows back to raw input and target indices."""

    raw_index_by_timestamp = pd.Series(raw_frame.index, index=raw_frame["timestamp"])
    rows: list[dict[str, object]] = []
    count = min(sample_count, len(predictions))

    for sample_index in range(count):
        timestamp = predictions.loc[sample_index, "timestamp"]
        target_index = int(raw_index_by_timestamp.loc[timestamp])
        # For horizon h, the last input row is h steps before the target row.
        input_start_index = target_index - lookback - horizon + 1
        input_end_index = input_start_index + lookback - 1
        y_true = float(predictions.loc[sample_index, "y_true"])
        raw_target = float(raw_frame.loc[target_index, target_column])
        last_input_target = float(raw_frame.loc[input_end_index, target_column])

        row = {
            "sample_index": sample_index,
            "input_start_index": input_start_index,
            "input_end_index": input_end_index,
            "target_index": target_index,
            "target_timestamp": timestamp,
            "last_input_target": last_input_target,
            "y_true": y_true,
            "raw_target_at_timestamp": raw_target,
            "target_matches_raw": bool(np.isclose(y_true, raw_target)),
        }
        if "persistence" in predictions:
            # Persistence should equal the last input target when alignment is
            # correct, so this is an additional sanity check.
            row["persistence_prediction"] = float(predictions.loc[sample_index, "persistence"])
            row["persistence_matches_last_input"] = bool(
                np.isclose(row["persistence_prediction"], last_input_target)
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_shift_check(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE under small target shifts to detect off-by-one alignment."""

    y_true = predictions["y_true"].to_numpy(dtype=float)
    model_columns = [column for column in predictions.columns if column not in {"timestamp", "y_true"}]
    rows: list[dict[str, object]] = []

    for model_name in model_columns:
        y_pred = predictions[model_name].to_numpy(dtype=float)
        shift_rows = []
        # The correct alignment should usually be shift 0; neighboring shifts
        # are included to expose accidental lag or lead errors.
        for target_shift_steps in [-2, -1, 0, 1, 2]:
            value = shifted_mae(y_true, y_pred, target_shift_steps)
            shift_rows.append((target_shift_steps, value))

        best_shift, best_mae = min(shift_rows, key=lambda item: item[1])
        for target_shift_steps, value in shift_rows:
            rows.append(
                {
                    "model_name": model_name,
                    "target_shift_steps": target_shift_steps,
                    "mae": value,
                    "best_shift_steps": best_shift,
                    "best_mae": best_mae,
                    "is_best_alignment": target_shift_steps == best_shift,
                }
            )

    return pd.DataFrame(rows)


def shifted_mae(y_true: np.ndarray, y_pred: np.ndarray, target_shift_steps: int) -> float:
    """Return MAE after shifting the target series relative to predictions."""

    if target_shift_steps < 0:
        y_pred_aligned = y_pred[-target_shift_steps:]
        y_true_aligned = y_true[:target_shift_steps]
    elif target_shift_steps > 0:
        y_pred_aligned = y_pred[:-target_shift_steps]
        y_true_aligned = y_true[target_shift_steps:]
    else:
        y_pred_aligned = y_pred
        y_true_aligned = y_true
    return float(np.mean(np.abs(y_true_aligned - y_pred_aligned)))


if __name__ == "__main__":
    main()
