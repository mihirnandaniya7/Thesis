from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_overview(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    output_path: str | Path,
    points: int,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    subset = slice(max(len(y_true) - points, 0), len(y_true))
    ts = pd.to_datetime(timestamps[subset])
    plt.figure(figsize=(12, 5))
    plt.plot(ts, y_true[subset, 0], label="ground_truth", linewidth=2.2, color="black")
    for name, values in predictions_by_model.items():
        plt.plot(ts, values[subset, 0], label=name, alpha=0.9)
    plt.title("Short-Term Load Forecasting on Test Split")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_residual_overview(
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(predictions_by_model)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 3.5 * n_models), squeeze=False)
    for axis, (name, predictions) in zip(axes[:, 0], predictions_by_model.items(), strict=False):
        residuals = y_true[:, 0] - predictions[:, 0]
        axis.hist(residuals, bins=30, color="#4472c4", alpha=0.8)
        axis.set_title(f"Residual Distribution: {name}")
        axis.set_xlabel("Residual (kW)")
        axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_runtime_comparison(metrics_frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = metrics_frame.copy()
    plt.figure(figsize=(10, 5))
    metric_column = (
        "full_test_runtime_ms" if "full_test_runtime_ms" in frame.columns else "latency_ms"
    )
    simulator_column = (
        "simulator_full_test_runtime_ms"
        if "simulator_full_test_runtime_ms" in frame.columns
        else "simulator_latency_ms"
    )
    plt.bar(frame["model_name"], frame[metric_column], color="#2a9d8f", label="model")
    if simulator_column in frame:
        plt.plot(
            frame["model_name"],
            frame[simulator_column],
            color="#e76f51",
            marker="o",
            linewidth=2,
            label="reference_simulator",
        )
    ylabel = "Runtime (ms)"
    title = "Model Runtime vs. Reference Simulator Runtime"
    if metric_column == "latency_ms":
        ylabel = "Latency (ms)"
        title = "Average Inference Latency vs. Simulator Step Runtime"
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_error_by_hour(
    hours: np.ndarray,
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rounded_hours = np.floor(hours).astype(int)
    unique_hours = np.arange(24)

    plt.figure(figsize=(11, 5))
    for name, predictions in predictions_by_model.items():
        absolute_error = np.abs(y_true[:, 0] - predictions[:, 0])
        hourly_mae = []
        for hour in unique_hours:
            mask = rounded_hours == hour
            hourly_mae.append(float(np.mean(absolute_error[mask])) if mask.any() else np.nan)
        plt.plot(unique_hours, hourly_mae, marker="o", label=name)
    plt.xticks(unique_hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Absolute Error (kW)")
    plt.title("Error by Hour of Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_decorator_threshold_sensitivity(
    sensitivity_frame: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = sensitivity_frame.sort_values(by="threshold_multiplier")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        frame["threshold_multiplier"],
        frame["decorator_mae"],
        marker="o",
        label="decorator_mae",
        color="#1d3557",
    )
    axes[0].plot(
        frame["threshold_multiplier"],
        frame["pure_surrogate_mae"],
        marker="s",
        linestyle="--",
        label="pure_surrogate_mae",
        color="#e63946",
    )
    axes[0].set_ylabel("MAE (kW)")
    axes[0].set_title(f"Decorator Threshold Sensitivity: {model_name}")
    axes[0].legend()

    axes[1].plot(
        frame["threshold_multiplier"],
        frame["surrogate_usage_ratio"],
        marker="o",
        label="surrogate_usage_ratio",
        color="#2a9d8f",
    )
    axes[1].plot(
        frame["threshold_multiplier"],
        frame["decorator_speedup"],
        marker="s",
        label="decorator_speedup",
        color="#f4a261",
    )
    axes[1].set_xlabel("Threshold Multiplier")
    axes[1].set_ylabel("Usage Ratio / Speedup")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_decorator_decision_trace(
    trace_frame: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = trace_frame.copy()
    frame["mode_flag"] = frame["mode"].map({"simulation": 0, "surrogate": 1})

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame["index"], frame["rolling_error_after"], color="#457b9d")
    axes[0].axhline(frame["threshold"].iloc[0], color="#e63946", linestyle="--", label="threshold")
    axes[0].set_ylabel("Rolling Error")
    axes[0].set_title(f"Decorator Decision Trace: {model_name}")
    axes[0].legend()

    axes[1].step(frame["index"], frame["mode_flag"], where="post", color="#2a9d8f")
    axes[1].set_yticks([0, 1], labels=["simulation", "surrogate"])
    axes[1].set_xlabel("Test Step")
    axes[1].set_ylabel("Chosen Path")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
