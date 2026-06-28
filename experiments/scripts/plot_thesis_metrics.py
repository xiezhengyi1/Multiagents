from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import RESULTS_ROOT
from experiments.scripts.compute_thesis_metrics import _aggregate_runs


DEFAULT_METHOD_ORDER = ("ours_wo_rag", "ours_wo_closedloop", "ours")
DEFAULT_INPUT_CSV = RESULTS_ROOT / "thesis_metrics.csv"
DEFAULT_OUTPUT_PNG = RESULTS_ROOT / "figures" / "thesis_metrics_bar.png"
DEFAULT_OUTPUT_PDF = RESULTS_ROOT / "figures" / "thesis_metrics_bar.pdf"
DEFAULT_OUTPUT_SVG = RESULTS_ROOT / "figures" / "thesis_metrics_bar.svg"
METHOD_LABELS = {
    "ours_wo_rag": "w/o RAG",
    "ours_wo_closedloop": "w/o Closed\nLoop",
    "ours": "Ours",
}
BASE_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 10


@dataclass(frozen=True)
class MethodMetrics:
    method_slug: str
    label: str
    total_cases: int
    eic_rate: float
    crr_rate: float
    tcr_rate: float
    avg_retry_count: float


def _read_metric_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float:
    text = str(value or "").strip()
    return float(text) if text else 0.0


def _to_int(value: Any) -> int:
    text = str(value or "").strip()
    return int(text) if text else 0


def _resolve_method_order(rows: Sequence[Mapping[str, str]], requested_methods: Sequence[str]) -> List[str]:
    available_methods = {
        str(row.get("method_slug") or "").strip()
        for row in rows
        if str(row.get("method_slug") or "").strip()
    }
    ordered_methods = [method for method in requested_methods if method in available_methods]
    extra_methods = sorted(available_methods.difference(ordered_methods))
    return [*ordered_methods, *extra_methods]


def filter_metric_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    scenario_id: str = "",
    model_tag: str = "",
) -> List[Dict[str, str]]:
    normalized_scenario_id = str(scenario_id or "").strip()
    normalized_model_tag = str(model_tag or "").strip().lower()

    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        row_scenario_id = str(row.get("scenario_ids") or "").strip()
        summary_file = str(row.get("summary_file") or "").strip().lower()

        if normalized_scenario_id and row_scenario_id != normalized_scenario_id:
            continue
        if normalized_model_tag and normalized_model_tag not in summary_file:
            continue

        filtered_rows.append(dict(row))

    return filtered_rows


def build_method_metrics(
    rows: Sequence[Mapping[str, str]],
    method_order: Sequence[str] | None = None,
) -> List[MethodMetrics]:
    rows_by_method: Dict[str, Dict[str, str]] = {}
    duplicate_methods: set[str] = set()
    for row in rows:
        method_slug = str(row.get("method_slug") or "").strip()
        if not method_slug:
            continue
        if method_slug in rows_by_method:
            duplicate_methods.add(method_slug)
            continue
        rows_by_method[method_slug] = dict(row)

    if duplicate_methods:
        duplicates = ", ".join(sorted(duplicate_methods))
        raise ValueError(f"Multiple rows found for methods: {duplicates}")

    resolved_order = _resolve_method_order(rows, method_order or DEFAULT_METHOD_ORDER)
    metrics: List[MethodMetrics] = []
    for method_slug in resolved_order:
        row = rows_by_method.get(method_slug)
        if row is None:
            continue
        metrics.append(
            MethodMetrics(
                method_slug=method_slug,
                label=METHOD_LABELS.get(method_slug, method_slug.replace("_", " ").title()),
                total_cases=_to_int(row.get("total_cases")),
                eic_rate=_to_float(row.get("eic_rate")),
                crr_rate=_to_float(row.get("crr_rate")),
                tcr_rate=_to_float(row.get("tcr_rate")),
                avg_retry_count=_to_float(row.get("avg_retry_count")),
            )
        )
    return metrics


def aggregate_metrics_by_method(
    rows: Sequence[Mapping[str, str]],
    method_order: Sequence[str] | None = None,
) -> List[MethodMetrics]:
    grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_slug = str(row.get("method_slug") or "").strip()
        if not method_slug:
            continue
        grouped_records[method_slug].append(
            {
                "total_cases": _to_int(row.get("total_cases")),
                "eic_count": _to_int(row.get("eic_count")),
                "pgr_count": _to_int(row.get("pgr_count")),
                "dsr_count": _to_int(row.get("dsr_count")),
                "tcr_count": _to_int(row.get("tcr_count")),
                "crr_first_round_failed_count": _to_int(row.get("crr_first_round_failed_count")),
                "crr_recovered_count": _to_int(row.get("crr_recovered_count")),
                "avg_round_count": _to_float(row.get("avg_round_count")),
                "avg_retry_count": _to_float(row.get("avg_retry_count")),
                "avg_elapsed_ms": _to_float(row.get("avg_elapsed_ms")),
            }
        )

    resolved_order = _resolve_method_order(rows, method_order or DEFAULT_METHOD_ORDER)
    aggregated_metrics: List[MethodMetrics] = []
    for method_slug in resolved_order:
        records = grouped_records.get(method_slug)
        if not records:
            continue
        summary = _aggregate_runs(records)
        aggregated_metrics.append(
            MethodMetrics(
                method_slug=method_slug,
                label=METHOD_LABELS.get(method_slug, method_slug.replace("_", " ").title()),
                total_cases=int(summary["total_cases"]),
                eic_rate=float(summary["eic_rate"]),
                crr_rate=float(summary["crr_rate"]),
                tcr_rate=float(summary["tcr_rate"]),
                avg_retry_count=float(summary["avg_retry_count"]),
            )
        )
    return aggregated_metrics


def _annotate_bars(axis: Any, bars: Sequence[Any], *, value_format: str, pad_ratio: float) -> None:
    upper_bound = axis.get_ylim()[1]
    offset = upper_bound * pad_ratio
    for bar in bars:
        height = float(bar.get_height())
        if height <= 0:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            format(height, value_format),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )


def plot_metrics_chart(
    metrics: Sequence[MethodMetrics],
    *,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    if not metrics:
        raise ValueError("No metrics available to plot")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE + 1,
            "axes.titlesize": BASE_FONT_SIZE + 2,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE,
            "svg.fonttype": "path",
        }
    )

    positions = list(range(len(metrics)))
    bar_width = 0.18
    rate_offsets = (-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width)
    retry_offset = 1.5 * bar_width

    eic_values = [item.eic_rate for item in metrics]
    crr_values = [item.crr_rate for item in metrics]
    tcr_values = [item.tcr_rate for item in metrics]
    retry_values = [item.avg_retry_count for item in metrics]

    figure, axis = plt.subplots(figsize=(8.6, 5.1))
    retry_axis = axis.twinx()

    eic_bars = axis.bar(
        [value + rate_offsets[0] for value in positions],
        eic_values,
        width=bar_width,
        color="#4C84D3",
        edgecolor="black",
        hatch="///",
        linewidth=0.9,
        label="EIC",
        zorder=3,
    )
    crr_bars = axis.bar(
        [value + rate_offsets[1] for value in positions],
        crr_values,
        width=bar_width,
        color="#B5B5B5",
        edgecolor="black",
        hatch="...",
        linewidth=0.9,
        label="CRR",
        zorder=3,
    )
    tcr_bars = axis.bar(
        [value + rate_offsets[2] for value in positions],
        tcr_values,
        width=bar_width,
        color="#F39C3D",
        edgecolor="black",
        hatch="\\\\",
        linewidth=0.9,
        label="TCR",
        zorder=3,
    )
    retry_bars = retry_axis.bar(
        [value + retry_offset for value in positions],
        retry_values,
        width=bar_width,
        color="#4CAF50",
        edgecolor="black",
        hatch="xx",
        linewidth=0.9,
        label="Retries",
        zorder=3,
    )

    axis.set_ylabel("Rate")
    axis.set_ylim(0.0, max(1.0, max([*eic_values, *crr_values, *tcr_values]) + 0.12))
    retry_axis.set_ylabel("Avg. Retries")
    retry_axis.set_ylim(0.0, max(1.0, max(retry_values) + 0.15))

    axis.set_xticks(positions)
    axis.set_xticklabels([item.label for item in metrics])
    axis.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    axis.spines["top"].set_visible(False)
    retry_axis.spines["top"].set_visible(False)

    # 中文注释：左轴展示三类成功率，右轴单独展示平均重试次数，避免把 rate 和 count 混在同一刻度里。
    legend_handles = [eic_bars[0], crr_bars[0], tcr_bars[0], retry_bars[0]]
    axis.legend(
        legend_handles,
        ["EIC", "CRR", "TCR", "Retries"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    _annotate_bars(axis, eic_bars, value_format=".2f", pad_ratio=0.012)
    _annotate_bars(axis, crr_bars, value_format=".2f", pad_ratio=0.012)
    _annotate_bars(axis, tcr_bars, value_format=".2f", pad_ratio=0.012)
    _annotate_bars(retry_axis, retry_bars, value_format=".2f", pad_ratio=0.018)

    figure.tight_layout()
    figure.subplots_adjust(top=0.78)

    for output_path in (output_png, output_pdf, output_svg):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a paper-ready grouped bar chart from thesis_metrics.csv.",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(DEFAULT_METHOD_ORDER),
        help="Optional method order. Missing methods are ignored and extra methods are appended alphabetically.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = _read_metric_rows(args.input_csv)
    metrics = aggregate_metrics_by_method(rows, method_order=args.methods)
    plot_metrics_chart(
        metrics,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        output_svg=args.output_svg,
    )

    print(f"Plotted thesis metrics for {len(metrics)} methods")
    print(f"PNG -> {args.output_png}")
    print(f"PDF -> {args.output_pdf}")
    print(f"SVG -> {args.output_svg}")


if __name__ == "__main__":
    main()