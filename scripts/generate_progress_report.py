from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from textwrap import wrap

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages



RUNS = {
    "Baseline v1 (simple load forecasting)": ROOT / "artifacts/20260328_003311_baseline_v1",
    "Stage 2 net-load v1": ROOT / "artifacts/20260328_005028_stage2_net_load_v1",
    "Stage 2 Transformer follow-up v1": ROOT / "artifacts/20260328_005330_stage2_transformer_followup_v1",
}


def load_metrics(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "metrics.csv")
    numeric_columns = [col for col in frame.columns if col != "model_name"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric)
    return frame


def wrap_lines(text: str, width: int = 98) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(wrap(raw_line, width=width))
    return lines


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.06, 0.96, title, fontsize=18, fontweight="bold", va="top")
    y = 0.92
    for line in wrap_lines(body):
        plt.text(0.06, y, line, fontsize=10.5, va="top")
        y -= 0.022
        if y < 0.06:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, frame: pd.DataFrame, notes: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.03, 0.97, title, fontsize=18, fontweight="bold", va="top")

    display_frame = frame.copy()
    for column in display_frame.columns:
        if column == "model_name":
            continue
        display_frame[column] = display_frame[column].map(lambda value: f"{value:.4f}")

    table = plt.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.03, 0.32, 0.94, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    y = 0.25
    for line in wrap_lines(notes, width=140):
        plt.text(0.03, y, line, fontsize=10, va="top")
        y -= 0.03

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.05, 0.97, title, fontsize=18, fontweight="bold", va="top")
    image = plt.imread(image_path)
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.58])
    ax.imshow(image)
    ax.axis("off")
    y = 0.20
    for line in wrap_lines(caption, width=105):
        plt.text(0.05, y, line, fontsize=10.5, va="top")
        y -= 0.025
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_summary_text() -> str:
    return (
        "This report documents the implementation and experimental progress completed so far for the thesis "
        "'Surrogate Modeling based on Machine Learning Approaches for Short-Term Load Forecasting in Microgrids.'\n\n"
        "What was implemented:\n"
        "- A modular Python project structure for simulation, data preparation, models, training, evaluation, and controller logic.\n"
        "- A sequential synthetic reference simulator with daily structure, seasonality, stochastic noise, and occasional peak events.\n"
        "- Stage 2 microgrid extensions with photovoltaic generation, battery state-of-charge, battery dispatch, and net-load computation.\n"
        "- A supervised dataset pipeline with leakage-safe chronological train/validation/test splits and train-only normalization.\n"
        "- Four models: persistence, linear regression, LSTM, and an encoder-only Transformer.\n"
        "- Evaluation logic for accuracy and runtime, including single-sample latency, batched throughput, and full test-set runtime.\n"
        "- Plots and saved artifacts for each run, plus automated tests to ensure the pipeline is reproducible.\n\n"
        "Why this matters for the thesis:\n"
        "The work now follows the core surrogate-modeling idea from the proposal: a synthetic simulator produces the data, "
        "and machine-learning surrogates are trained to approximate simulator behavior. The richer Stage 2 experiments move "
        "the implementation closer to a realistic microgrid use case by forecasting net load instead of only simple load."
    )


def build_interpretation_text() -> str:
    return (
        "Interpretation of the results:\n"
        "- The original simple baseline run showed that LSTM and Transformer improved prediction accuracy over linear regression "
        "and persistence, but the neural models were slower than the simulator under the original latency measurement.\n"
        "- After switching to the richer Stage 2 microgrid task and improving runtime measurement, both LSTM and Transformer became "
        "faster than the simulator when evaluated on the full test-set runtime. This is a much better match to the thesis objective.\n"
        "- In the best Stage 2 run, LSTM achieved MAE 0.0623 with about 8.44x full-run speedup, while Transformer achieved MAE "
        "0.0649 with about 7.52x speedup. This means the surrogate concept is working: good accuracy and meaningful simulator replacement speedup.\n"
        "- The follow-up run made the sequence task harder using lookback 48, horizon 4, and a larger Transformer. That experiment did "
        "not help the Transformer. LSTM remained best, and the Transformer became both slower and less accurate than in the previous Stage 2 run.\n\n"
        "Conclusion at this point:\n"
        "The project now supports the main thesis claim much better than before. We can honestly say that ML surrogates, especially LSTM, "
        "can approximate the richer simulator while reducing full-run execution time. However, the Transformer has not yet become the best model. "
        "So the next technical step should be focused Transformer tuning rather than a larger and harder configuration by default."
    )


def build_next_steps_text() -> str:
    return (
        "Recommended next steps:\n"
        "- Keep the Stage 2 net-load setup as the main baseline because it aligns better with the proposal than the earlier simple load-only task.\n"
        "- Use the Stage 2 net-load v1 run as the current best thesis result because it balances richer dynamics, good surrogate accuracy, and clear speedup.\n"
        "- Tune Transformer more carefully instead of scaling it aggressively. Likely directions are a moderate lookback increase, a smaller hidden size than the follow-up run, and horizon 1 first.\n"
        "- Decide which runtime metric to emphasize in the thesis text. Full-run or batched speedup is the defensible metric here; single-step latency alone does not tell the whole story.\n"
        "- Start drafting the results chapter around three phases: simple baseline run, improved Stage 2 run, and failed Transformer follow-up run as a useful negative result.\n\n"
        "Files and artifacts stored in this repository:\n"
        "- artifacts/20260328_003311_baseline_v1\n"
        "- artifacts/20260328_005028_stage2_net_load_v1\n"
        "- artifacts/20260328_005330_stage2_transformer_followup_v1\n\n"
        "Generated automatically on: "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report summarizing thesis progress.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "Thesis_Progress_Report.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = load_metrics(RUNS["Baseline v1 (simple load forecasting)"])
    stage2 = load_metrics(RUNS["Stage 2 net-load v1"])
    followup = load_metrics(RUNS["Stage 2 Transformer follow-up v1"])

    with PdfPages(output_path) as pdf:
        add_text_page(
            pdf,
            "Thesis Progress Report",
            build_summary_text(),
        )

        add_table_page(
            pdf,
            "Experiment 1: Baseline v1 (Simple Load Forecasting)",
            baseline,
            (
                "This was the first complete end-to-end run. It showed that LSTM and Transformer improved predictive accuracy "
                "over the simple baselines, but under the original runtime measurement the neural surrogates still looked slower "
                "than the simulator. Best model by MAE: LSTM."
            ),
        )

        add_table_page(
            pdf,
            "Experiment 2: Stage 2 Net-Load v1",
            stage2,
            (
                "This is the most important result so far. The simulator was upgraded to a richer microgrid task with PV, battery, "
                "and net-load prediction. Runtime evaluation was also improved. In this run, both LSTM and Transformer beat the simulator "
                "on full-run runtime while staying more accurate than the simple baselines. Best model by MAE: LSTM."
            ),
        )

        add_table_page(
            pdf,
            "Experiment 3: Transformer Follow-up v1",
            followup,
            (
                "This follow-up made the task harder with lookback 48, horizon 4, and a larger Transformer. It is a useful negative result: "
                "LSTM remained best, while Transformer lost both accuracy and efficiency relative to Experiment 2. This suggests the architecture "
                "needs targeted tuning rather than simply more complexity."
            ),
        )

        add_text_page(
            pdf,
            "Interpretation Against Thesis Goal",
            build_interpretation_text(),
        )

        add_image_page(
            pdf,
            "Stage 2 Net-Load Runtime Comparison",
            RUNS["Stage 2 net-load v1"] / "plots/runtime_comparison.png",
            (
                "This plot comes from the strongest current thesis run. It illustrates why the Stage 2 net-load configuration is a better match "
                "for the thesis than the original simple run: the richer simulator is expensive enough that the learned surrogates show meaningful "
                "full-run speedup."
            ),
        )

        add_image_page(
            pdf,
            "Stage 2 Net-Load Prediction Overview",
            RUNS["Stage 2 net-load v1"] / "plots/prediction_overview.png",
            (
                "Prediction overview from the Stage 2 net-load experiment. It shows that the learned surrogates track the target dynamics closely, "
                "with LSTM slightly ahead of Transformer on this run."
            ),
        )

        add_image_page(
            pdf,
            "Transformer Follow-up Runtime Comparison",
            RUNS["Stage 2 Transformer follow-up v1"] / "plots/runtime_comparison.png",
            (
                "Runtime comparison for the harder follow-up run. Even though the Transformer still beats the simulator on full-run speed, "
                "its efficiency and accuracy deteriorate relative to the Stage 2 net-load v1 experiment."
            ),
        )

        add_text_page(
            pdf,
            "Recommended Next Steps",
            build_next_steps_text(),
        )

    print(f"PDF report written to: {output_path}")


if __name__ == "__main__":
    main()
