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
    "Transformer follow-up v1": ROOT / "artifacts/20260328_005330_stage2_transformer_followup_v1",
    "Transformer tuning v1": ROOT / "artifacts/20260328_010532_stage2_transformer_tuning_v1",
    "Transformer small v1": ROOT / "artifacts/20260328_010946_stage2_transformer_small_v1",
    "Stage 2 net-load v1 with decorator": ROOT / "artifacts/20260328_012307_stage2_net_load_v1",
}


def load_metrics(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "metrics.csv")
    numeric_columns = [col for col in frame.columns if col != "model_name"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric)
    return frame


def load_decorator_summary(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "decorator" / "decorator_summary.csv")
    numeric_columns = [
        col
        for col in frame.columns
        if col
        not in {
            "model_name",
            "sensitivity_csv",
            "trace_csv",
        }
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric)
    return frame


def wrap_lines(text: str, width: int = 98) -> list[str]:
    lines: list[str] = []
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
    plt.text(0.06, 0.965, title, fontsize=18, fontweight="bold", va="top")
    y = 0.925
    for line in wrap_lines(body):
        plt.text(0.06, y, line, fontsize=10.5, va="top")
        y -= 0.022
        if y < 0.055:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    frame: pd.DataFrame,
    notes: str,
    *,
    height: float = 0.5,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.03, 0.97, title, fontsize=18, fontweight="bold", va="top")

    display_frame = frame.copy()
    for column in display_frame.columns:
        if display_frame[column].dtype.kind in {"i", "f"}:
            display_frame[column] = display_frame[column].map(lambda value: f"{value:.4f}")

    table = plt.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.03, 0.30, 0.94, height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.25)

    y = 0.23
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


def build_implementation_text() -> str:
    return (
        "This report summarizes the full implementation progress completed so far for the thesis "
        "'Surrogate Modeling based on Machine Learning Approaches for Short-Term Load Forecasting in Microgrids.'\n\n"
        "Implemented system components:\n"
        "- Synthetic sequential reference simulator with daily load structure, seasonality, noise, and peak events.\n"
        "- Stage 2 microgrid extension with PV generation, battery state of charge, battery dispatch, and net-load target generation.\n"
        "- Leakage-safe supervised dataset builder with chronological 70/15/15 train, validation, and test splits.\n"
        "- Baseline models: persistence and linear regression.\n"
        "- Learned surrogates: LSTM and encoder-only Transformer.\n"
        "- Training pipeline with fixed seeds, checkpoints, early stopping, plots, and saved experiment artifacts.\n"
        "- Runtime evaluation with single-step latency, batched throughput, and full test-set runtime.\n"
        "- Decorator-based switching layer with threshold evaluation, rolling error tracking, trace export, and sensitivity plots.\n"
        "- Automated smoke and unit tests for simulator, pipeline, and decorator behavior.\n\n"
        "Most important thesis shift so far:\n"
        "The project moved from a simple load-prediction prototype to a richer surrogate-system prototype where a higher-fidelity "
        "microgrid simulation can be approximated by learned models and now also wrapped by a runtime switching mechanism."
    )


def build_improvements_text() -> str:
    return (
        "Main improvements achieved so far:\n"
        "- Accuracy improvement over naive baselines: on the first end-to-end run, LSTM reduced MAE from 0.0795 "
        "(persistence) to 0.0590 and Transformer reduced it to 0.0626.\n"
        "- Better thesis alignment after Stage 2: once the simulator was upgraded to PV, battery, and net-load behavior, both "
        "LSTM and Transformer became faster than the richer simulator on full test-set runtime while still outperforming simple baselines.\n"
        "- Strongest non-decorator result so far: in Stage 2 net-load v1, LSTM reached MAE 0.0623 with about 8.44x full-run speedup, "
        "and Transformer reached MAE 0.0649 with about 7.52x speedup.\n"
        "- Clear negative-result learning: aggressive Transformer scaling with lookback 48 and horizon 4 hurt both accuracy and efficiency, "
        "which helped narrow the design space.\n"
        "- Measured speed-accuracy tradeoff: a smaller Transformer improved speed to about 9.53x but lost too much accuracy, showing the "
        "design cannot be judged on runtime alone.\n"
        "- New decorator capability: adaptive switching is now implemented and measurable. On the latest decorator run, surrogate usage reached "
        "about 64 to 66 percent with the remaining steps falling back to the simulator based on the threshold logic.\n"
        "- Accuracy gain from the decorator path: for the evaluated neural models, decorator MAE dropped to about 0.036 while pure surrogate "
        "MAE stayed around 0.062 to 0.065, which shows the switching logic can improve fidelity when fallback is allowed.\n"
        "- Remaining limitation: the current decorator uses shadow evaluation every step, so it improves accuracy but still needs runtime "
        "optimization before it becomes a strong efficiency result by itself."
    )


def build_conclusion_text() -> str:
    return (
        "Current thesis status:\n"
        "- The surrogate-modeling pipeline is implemented end to end.\n"
        "- The richer Stage 2 net-load experiments already support the core claim that learned surrogates can approximate the simulator with "
        "useful runtime savings on the full run.\n"
        "- The decorator concept is now implemented as a real measurable component instead of only a placeholder controller.\n"
        "- The decorator currently improves result fidelity through simulation fallback, but its runtime still needs refinement.\n\n"
        "Recommended next step after this report:\n"
        "Use the current results to refine the concept architecture and presentation diagram by hand, then optimize the decorator to reduce "
        "shadow-check overhead. The most practical direction is a cheaper trust-estimation strategy such as periodic probe checks or less "
        "frequent fallback validation instead of evaluating the full shadow path every step.\n\n"
        "Generated automatically on: "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def build_key_results_table() -> pd.DataFrame:
    summary_rows = [
        {
            "run": "Baseline v1",
            "best_model": "lstm",
            "best_mae": 0.0589742474,
            "transformer_mae": 0.0625799075,
            "best_speedup": 0.3133364613,
            "transformer_speedup": 0.2007930731,
        },
        {
            "run": "Stage 2 net-load v1",
            "best_model": "lstm",
            "best_mae": 0.0623214543,
            "transformer_mae": 0.0648499951,
            "best_speedup": 8.4364192722,
            "transformer_speedup": 7.5150061146,
        },
        {
            "run": "Follow-up v1",
            "best_model": "lstm",
            "best_mae": 0.0640231669,
            "transformer_mae": 0.0700364485,
            "best_speedup": 4.5651881361,
            "transformer_speedup": 1.6626717054,
        },
        {
            "run": "Tuning v1",
            "best_model": "linear_regression",
            "best_mae": 0.0627580285,
            "transformer_mae": 0.0670660958,
            "best_speedup": 7536.4767977913,
            "transformer_speedup": 4.8013794719,
        },
        {
            "run": "Transformer small v1",
            "best_model": "lstm",
            "best_mae": 0.0623214543,
            "transformer_mae": 0.0705067292,
            "best_speedup": 8.8603472188,
            "transformer_speedup": 9.5311671881,
        },
    ]
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report summarizing thesis progress.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "Thesis_Implementation_Progress_Report.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = load_metrics(RUNS["Baseline v1 (simple load forecasting)"])
    stage2 = load_metrics(RUNS["Stage 2 net-load v1"])
    tuning = load_metrics(RUNS["Transformer tuning v1"])
    decorator_summary = load_decorator_summary(RUNS["Stage 2 net-load v1 with decorator"])[
        [
            "model_name",
            "threshold_multiplier",
            "pure_surrogate_mae",
            "decorator_mae",
            "surrogate_usage_ratio",
            "simulation_usage_ratio",
            "switch_count",
            "decorator_speedup",
        ]
    ]

    with PdfPages(output_path) as pdf:
        add_text_page(
            pdf,
            "Thesis Implementation Progress Report",
            build_implementation_text(),
        )

        add_text_page(
            pdf,
            "Improvements So Far",
            build_improvements_text(),
        )

        add_table_page(
            pdf,
            "Key Run Comparison",
            build_key_results_table(),
            (
                "This condensed table shows how the work progressed across the major experiment stages. The main thesis baseline remains "
                "Stage 2 net-load v1 because it balances richer microgrid behavior, strong neural-model accuracy, and clear full-run speedup."
            ),
            height=0.40,
        )

        add_table_page(
            pdf,
            "Detailed Metrics: Baseline v1",
            baseline,
            (
                "This first complete run established that the learned surrogates were more accurate than the simple baselines. "
                "Its weakness was runtime: under the original setup, the simulator was still cheaper than the neural surrogates."
            ),
        )

        add_table_page(
            pdf,
            "Detailed Metrics: Stage 2 Net-Load v1",
            stage2,
            (
                "This is still the strongest pure-surrogate result. The richer simulator and improved runtime measurement produced a result that "
                "matches the thesis idea much better: neural surrogates remained accurate and became faster than the simulator on the full run."
            ),
        )

        add_table_page(
            pdf,
            "Detailed Metrics: Transformer Tuning v1",
            tuning,
            (
                "This tuning pass is useful because it shows what did not help. Extending lookback to 48 without changing the overall setup "
                "did not improve the Transformer, and even allowed linear regression to become the strongest MAE model in that run."
            ),
        )

        add_table_page(
            pdf,
            "Decorator Switching Summary",
            decorator_summary,
            (
                "The decorator layer is now implemented and evaluated. It reduces error by allowing selective fallback to the high-fidelity "
                "simulator, but the current shadow-evaluation design still adds overhead, so decorator speedup remains below pure simulation."
            ),
            height=0.42,
        )

        add_image_page(
            pdf,
            "Stage 2 Runtime Comparison",
            RUNS["Stage 2 net-load v1"] / "plots/runtime_comparison.png",
            (
                "This plot is the clearest visual proof that the Stage 2 net-load configuration improved the thesis story. In this richer setup, "
                "the neural surrogates are not only accurate but also faster than the simulator on the full evaluation run."
            ),
        )

        add_image_page(
            pdf,
            "Decorator Threshold Sensitivity",
            RUNS["Stage 2 net-load v1 with decorator"] / "decorator" / "transformer_threshold_sensitivity.png",
            (
                "Threshold sensitivity for the Transformer decorator. This figure shows that decorator performance depends on how aggressively "
                "the system falls back to the simulator. It is the beginning of the trust-estimation analysis requested by the supervisor."
            ),
        )

        add_image_page(
            pdf,
            "Decorator Decision Trace",
            RUNS["Stage 2 net-load v1 with decorator"] / "decorator" / "lstm_decision_trace.png",
            (
                "Decision trace from the LSTM decorator evaluation. This makes the switching behavior tangible by showing when the surrogate "
                "is trusted and when the system falls back to the high-fidelity simulator."
            ),
        )

        add_text_page(
            pdf,
            "Conclusion And Next Step",
            build_conclusion_text(),
        )

    print(f"PDF report written to: {output_path}")


if __name__ == "__main__":
    main()
