"""Plotting helpers for experiment and decorator artifacts."""

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
    """Plot recent ground-truth and model predictions from the test split."""

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
    """Plot residual distributions with shared axes for fair comparison."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    residuals_by_model = {
        name: y_true[:, 0] - predictions[:, 0]
        for name, predictions in predictions_by_model.items()
    }
    all_residuals = np.concatenate(list(residuals_by_model.values()))
    max_abs_residual = float(np.max(np.abs(all_residuals)))
    # Symmetric bins around zero make bias and spread comparable across models.
    x_limit = max(max_abs_residual * 1.05, 1e-6)
    bins = np.linspace(-x_limit, x_limit, 41)
    density_max = max(
        float(np.max(np.histogram(residuals, bins=bins, density=True)[0]))
        for residuals in residuals_by_model.values()
    )

    n_models = len(predictions_by_model)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 3.5 * n_models), squeeze=False)
    for axis, (name, residuals) in zip(axes[:, 0], residuals_by_model.items(), strict=False):
        axis.hist(residuals, bins=bins, density=True, color="#4472c4", alpha=0.8)
        axis.axvline(0.0, color="#1d1d1d", linewidth=1.2, linestyle="--")
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(0.0, density_max * 1.08)
        axis.set_title(f"Residual Distribution: {name}")
        axis.set_xlabel("Residual (kW)")
        axis.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_runtime_comparison(metrics_frame: pd.DataFrame, output_path: str | Path) -> None:
    """Compare model runtime against the reference simulator runtime."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = metrics_frame.copy()
    plt.figure(figsize=(10, 5))
    metric_column = (
        "full_test_runtime_ms" if "full_test_runtime_ms" in frame.columns else "latency_ms"
    )
    # Older artifacts may only contain latency columns, so the plot falls back
    # gracefully while newer runs use full test-set runtime.
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


def plot_normalized_error_comparison(metrics_frame: pd.DataFrame, output_path: str | Path) -> None:
    """Plot absolute and normalized forecast errors side by side."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = metrics_frame.copy()
    models = frame["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x - width / 2, frame["MAE"], width=width, label="MAE", color="#457b9d")
    axes[0].bar(x + width / 2, frame["RMSE"], width=width, label="RMSE", color="#e76f51")
    axes[0].set_xticks(x, labels=models)
    axes[0].set_ylabel("Error (kW)")
    axes[0].set_title("Absolute Error Metrics")
    axes[0].legend()

    normalized_columns = [
        ("NMAE", "#2a9d8f"),
        ("NRMSE", "#f4a261"),
        ("sMAPE", "#7b2cbf"),
    ]
    offsets = np.linspace(-width, width, num=len(normalized_columns))
    for offset, (column, color) in zip(offsets, normalized_columns, strict=False):
        axes[1].bar(x + offset, frame[column], width=width * 0.9, label=column, color=color)
    axes[1].set_xticks(x, labels=models)
    axes[1].set_ylabel("Error (%)")
    axes[1].set_title("Normalized and Percentage Error Metrics")
    axes[1].legend()

    fig.suptitle("Forecast Error Comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_error_by_hour(
    hours: np.ndarray,
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Show whether models fail more at specific hours of the day."""

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
    """Visualize decorator accuracy, usage, and speedup across thresholds."""

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
    """Plot rolling error and selected execution path over test steps."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = trace_frame.copy()
    frame["mode_flag"] = frame["mode"].map({"simulation": 0, "surrogate": 1})
    probe_points = frame[frame["probe_executed"].astype(bool)]

    # The first panel explains why mode changes happen; the second panel shows
    # the actual simulator/surrogate path selected by the decorator.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frame["index"], frame["rolling_error_after"], color="#457b9d")
    axes[0].axhline(
        frame["entry_threshold"].iloc[0],
        color="#e63946",
        linestyle="--",
        label="entry_threshold",
    )
    if "exit_threshold" in frame:
        axes[0].axhline(
            frame["exit_threshold"].iloc[0],
            color="#f4a261",
            linestyle=":",
            label="exit_threshold",
        )
    if not probe_points.empty:
        axes[0].scatter(
            probe_points["index"],
            probe_points["rolling_error_after"],
            color="#2a9d8f",
            s=18,
            label="probe_step",
            zorder=3,
        )
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
