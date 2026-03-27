from __future__ import annotations

"""Helpers for comparing real-data threshold sets via the manuscript's main losses."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from method import Distribution, fitness
from tools.downstream_tools import pooled_quantile_thresholds
from tools.processing_aireadi import load_ai_readi_cohort

__all__ = [
    "CONSENSUS_THRESHOLD_SETS",
    "EXACT_DE_THRESHOLD_SETS",
    "MAIN_OBJECTIVE_BY_DATASET",
    "RealLossDataset",
    "compute_real_loss_tables",
    "create_naive_thresholds_latex_table",
    "create_real_loss_latex_table",
    "export_real_loss_tables",
    "format_thresholds",
    "load_real_loss_datasets",
    "pooled_quantile_thresholds",
    "style_loss_comparison_table",
]

CONSENSUS_THRESHOLD_SETS: dict[int, list[int]] = {
    2: [70, 181],
    4: [54, 70, 181, 251],
}

# Exact notebook thresholds from the original real-data analyses.
EXACT_DE_THRESHOLD_SETS: dict[str, dict[int, list[float]]] = {
    "healthy": {
        2: [71.69, 127.91],
        4: [75.83, 100.69, 123.70, 154.96],
    },
    "t1d": {
        2: [210.48, 288.43],
        4: [84.88, 171.20, 232.62, 301.58],
    },
    "combined": {
        2: [149.70, 257.25],
        4: [81.49, 125.04, 192.40, 274.25],
    },
    "ai_readi": {
        2: [95.7497, 169.4792],
        4: [89.1604, 127.7855, 171.9481, 231.3521],
    },
}

MAIN_OBJECTIVE_BY_DATASET: dict[str, str] = {
    "healthy": "Loss1",
    "t1d": "Loss1",
    "combined": "Loss2",
    "ai_readi": "Loss2",
}

DATASET_SPECS: tuple[tuple[str, str], ...] = (
    ("healthy", "Healthy"),
    ("t1d", "T1D"),
    ("combined", "Combined"),
    ("ai_readi", "AI-READI"),
)

NAIVE_THRESHOLD_COUNTS: tuple[int, ...] = (2, 4, 9)
METHOD_ORDER: tuple[str, ...] = ("DE", "Consensus", "Naive")
LOSS_COLUMN_ORDER: tuple[tuple[str, str], ...] = (
    ("K=2", "DE"),
    ("K=2", "Consensus"),
    ("K=2", "Naive"),
    ("K=4", "DE"),
    ("K=4", "Consensus"),
    ("K=4", "Naive"),
    ("K=9", "Naive"),
)
METHODS_BY_K: dict[str, tuple[str, ...]] = {
    "K=2": ("DE", "Consensus", "Naive"),
    "K=4": ("DE", "Consensus", "Naive"),
    "K=9": ("Naive",),
}
LOSS_TEX_LABELS: dict[str, str] = {"L1": r"$L_1$", "L2": r"$L_2$"}


@dataclass
class RealLossDataset:
    key: str
    label: str
    glucose_lists: list[np.ndarray]
    data_class: Distribution


def _load_grouped_glucose_lists(csv_path: Path) -> list[np.ndarray]:
    data = pd.read_csv(csv_path, usecols=["id", "gl"])
    grouped = data.groupby("id", sort=False).agg({"gl": list}).reset_index()
    return [np.asarray(values, dtype=float) for values in grouped["gl"]]


def load_real_loss_datasets(repo_root: Path | str) -> dict[str, RealLossDataset]:
    repo_root = Path(repo_root)
    data_dir = repo_root / "data"

    healthy_lists = _load_grouped_glucose_lists(data_dir / "shah2019_filtered.csv")
    t1d_lists = _load_grouped_glucose_lists(data_dir / "brown2019_filtered.csv")
    combined_lists = healthy_lists + t1d_lists

    ai_readi = load_ai_readi_cohort(repo_root)
    ai_readi_lists = [np.asarray(values, dtype=float) for values in ai_readi["gl"]]

    healthy_series = pd.Series([values.tolist() for values in healthy_lists], dtype=object)
    t1d_series = pd.Series([values.tolist() for values in t1d_lists], dtype=object)
    combined_series = pd.Series([values.tolist() for values in combined_lists], dtype=object)
    ai_readi_series = pd.Series([values.tolist() for values in ai_readi_lists], dtype=object)

    return {
        "healthy": RealLossDataset("healthy", "Healthy", healthy_lists, Distribution(healthy_series, ran=(39.0, 401.0), M=200)),
        "t1d": RealLossDataset("t1d", "T1D", t1d_lists, Distribution(t1d_series, ran=(39.0, 401.0), M=200)),
        "combined": RealLossDataset("combined", "Combined", combined_lists, Distribution(combined_series, ran=(39.0, 401.0), M=200)),
        "ai_readi": RealLossDataset("ai_readi", "AI-READI", ai_readi_lists, Distribution(ai_readi_series, ran=(40.0, 400.0), M=200)),
    }


def _format_threshold_value(value: float) -> str:
    rounded = int(round(float(value)))
    if np.isclose(value, rounded, atol=1e-8):
        return str(rounded)
    return f"{float(value):.1f}"


def format_thresholds(thresholds: list[float] | np.ndarray, tex: bool = False) -> str:
    inner = ", ".join(_format_threshold_value(value) for value in thresholds)
    if tex:
        return rf"$\{{{inner}\}}$"
    return "{" + inner + "}"


def _tex_threshold_display(display_value: str) -> str:
    return "$" + display_value.replace("{", r"\{").replace("}", r"\}") + "$"


def _evaluate_loss(data_class: Distribution, thresholds: list[float] | np.ndarray, objective: str) -> float:
    return float(fitness(list(np.asarray(thresholds, dtype=float)), data_class, loss=objective))


def _sorted_threshold_df(threshold_df: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {label: idx for idx, (_, label) in enumerate(DATASET_SPECS)}
    ordered = threshold_df.copy()
    ordered["_dataset_order"] = ordered["Dataset"].map(dataset_order)
    ordered = ordered.sort_values(["_dataset_order", "K"], kind="stable")
    return ordered.drop(columns=["_dataset_order"]).reset_index(drop=True)


def _sorted_loss_df(loss_df: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {label: idx for idx, (_, label) in enumerate(DATASET_SPECS)}
    method_order = {label: idx for idx, label in enumerate(METHOD_ORDER)}
    ordered = loss_df.copy()
    ordered["_dataset_order"] = ordered["Dataset"].map(dataset_order)
    ordered["_method_order"] = ordered["Method"].map(method_order)
    ordered = ordered.sort_values(["_dataset_order", "K", "_method_order"], kind="stable")
    return ordered.drop(columns=["_dataset_order", "_method_order"]).reset_index(drop=True)


def _build_naive_thresholds_wide_df(threshold_summary_df: pd.DataFrame) -> pd.DataFrame:
    naive_df = threshold_summary_df.loc[
        threshold_summary_df["Method"] == "Naive",
        ["Dataset", "K", "Thresholds display"],
    ].copy()
    naive_df["K"] = naive_df["K"].map(lambda value: f"K={int(value)}")
    wide_df = naive_df.pivot(index="Dataset", columns="K", values="Thresholds display")
    wide_df = wide_df.reindex(index=[label for _, label in DATASET_SPECS], columns=["K=2", "K=4", "K=9"])
    wide_df.index.name = "Dataset"
    wide_df.columns.name = None
    return wide_df


def _build_reported_loss_wide_df(loss_comparison_df: pd.DataFrame) -> pd.DataFrame:
    loss_df = loss_comparison_df.copy()
    loss_df["K"] = loss_df["K"].map(lambda value: f"K={int(value)}")
    wide_df = loss_df.pivot(index=["Dataset", "Reported loss"], columns=["K", "Method"], values="Value")
    wide_df = wide_df.reindex(columns=pd.MultiIndex.from_tuples(LOSS_COLUMN_ORDER))
    ordered_index = pd.MultiIndex.from_tuples(
        [
            (dataset_label, MAIN_OBJECTIVE_BY_DATASET[dataset_key].replace("Loss", "L"))
            for dataset_key, dataset_label in DATASET_SPECS
        ],
        names=["Dataset", "Loss"],
    )
    wide_df = wide_df.reindex(ordered_index)
    wide_df.columns.names = [None, None]
    return wide_df


def _optimal_loss_mask(loss_wide_df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=loss_wide_df.index, columns=loss_wide_df.columns)
    for row_index in loss_wide_df.index:
        for k_label, methods in METHODS_BY_K.items():
            if len(methods) <= 1:
                continue
            row_values = loss_wide_df.loc[row_index, k_label]
            best_value = row_values.min()
            winners = row_values.index[np.isclose(row_values.to_numpy(dtype=float), best_value)]
            for method in winners:
                mask.loc[row_index, (k_label, method)] = True
    return mask


def style_loss_comparison_table(loss_wide_df: pd.DataFrame):
    mask = _optimal_loss_mask(loss_wide_df)

    def apply_styles(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.where(mask, "font-weight: bold;", ""), index=data.index, columns=data.columns)

    return (
        loss_wide_df.style
        .format("{:.1f}")
        .apply(apply_styles, axis=None)
    )


def compute_real_loss_tables(repo_root: Path | str) -> dict[str, pd.DataFrame]:
    datasets = load_real_loss_datasets(repo_root)

    threshold_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []

    for dataset_key, dataset_label in DATASET_SPECS:
        bundle = datasets[dataset_key]
        bundle.data_class.Wdist_matrix()
        objective = MAIN_OBJECTIVE_BY_DATASET[dataset_key]

        naive_threshold_map: dict[int, list[float]] = {}
        for threshold_count in NAIVE_THRESHOLD_COUNTS:
            _, naive_thresholds = pooled_quantile_thresholds(bundle.glucose_lists, threshold_count)
            naive_threshold_map[threshold_count] = naive_thresholds.tolist()

        for threshold_count in (2, 4):
            threshold_specs = (
                ("DE", EXACT_DE_THRESHOLD_SETS[dataset_key][threshold_count], "Exact notebook DE thresholds"),
                ("Consensus", CONSENSUS_THRESHOLD_SETS[threshold_count], "Consensus thresholds"),
                ("Naive", naive_threshold_map[threshold_count], "Pooled empirical quantiles"),
            )
            for method, thresholds, source in threshold_specs:
                threshold_rows.append(
                    {
                        "Dataset": dataset_label,
                        "Dataset key": dataset_key,
                        "K": threshold_count,
                        "Method": method,
                        "Source": source,
                        "Thresholds": list(thresholds),
                        "Thresholds display": format_thresholds(thresholds),
                    }
                )
                loss_rows.append(
                    {
                        "Dataset": dataset_label,
                        "Dataset key": dataset_key,
                        "Reported loss": objective.replace("Loss", "L"),
                        "K": threshold_count,
                        "Method": method,
                        "Value": _evaluate_loss(bundle.data_class, thresholds, objective),
                    }
                )

        threshold_rows.append(
            {
                "Dataset": dataset_label,
                "Dataset key": dataset_key,
                "K": 9,
                "Method": "Naive",
                "Source": "Pooled empirical deciles",
                "Thresholds": naive_threshold_map[9],
                "Thresholds display": format_thresholds(naive_threshold_map[9]),
            }
        )
        loss_rows.append(
            {
                "Dataset": dataset_label,
                "Dataset key": dataset_key,
                "Reported loss": objective.replace("Loss", "L"),
                "K": 9,
                "Method": "Naive",
                "Value": _evaluate_loss(bundle.data_class, naive_threshold_map[9], objective),
            }
        )

    threshold_summary_df = _sorted_threshold_df(pd.DataFrame(threshold_rows))
    loss_comparison_df = _sorted_loss_df(pd.DataFrame(loss_rows))
    naive_thresholds_wide_df = _build_naive_thresholds_wide_df(threshold_summary_df)
    reported_loss_wide_df = _build_reported_loss_wide_df(loss_comparison_df)

    return {
        "threshold_summary_df": threshold_summary_df,
        "loss_comparison_df": loss_comparison_df,
        "naive_thresholds_wide_df": naive_thresholds_wide_df,
        "reported_loss_wide_df": reported_loss_wide_df,
    }


def create_naive_thresholds_latex_table(naive_thresholds_wide_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pooled-quantile naive thresholds for the four real-data CGM cohorts. The $K=2$ column uses cohort-specific tertile cutoffs, the $K=4$ column uses cohort-specific quintile cutoffs, and the $K=9$ column uses cohort-specific decile cutoffs.}",
        r"\label{tab:real_naive_thresholds}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Naive thresholds (mg/dL)} \\",
        r"\cmidrule(lr){2-4}",
        r"Dataset & $K=2$ & $K=4$ & $K=9$ \\",
        r"\midrule",
    ]
    for dataset_label in naive_thresholds_wide_df.index:
        row = naive_thresholds_wide_df.loc[dataset_label]
        lines.append(
            f"{dataset_label} & {_tex_threshold_display(row['K=2'])} & {_tex_threshold_display(row['K=4'])} & {_tex_threshold_display(row['K=9'])}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _format_loss_entry(value: float, bold: bool) -> str:
    formatted = f"{float(value):.1f}"
    return rf"\textbf{{{formatted}}}" if bold else formatted


def create_real_loss_latex_table(loss_wide_df: pd.DataFrame) -> str:
    optimal_mask = _optimal_loss_mask(loss_wide_df)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with consensus and pooled-quantile naive thresholds for the manuscript's main real-data experiments. Healthy and T1D rows report $L_1$ because those analyses use the $L_1$ criterion, whereas Combined and AI-READI rows report $L_2$. Within each dataset, the smallest value inside each comparable $K$ block is shown in bold. The $K=9$ block reports the reviewer-requested naive decile baseline only.}",
        r"\label{tab:real_loss_comparison}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$K=2$} & \multicolumn{3}{c}{$K=4$} & \multicolumn{1}{c}{$K=9$} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-9}",
        r"Dataset & Loss & DE & Consensus & Naive & DE & Consensus & Naive & Naive \\",
        r"\midrule",
    ]
    dataset_labels = [label for _, label in DATASET_SPECS]
    for idx, dataset_label in enumerate(dataset_labels):
        row = loss_wide_df.xs(dataset_label, level="Dataset")
        loss_label = row.index[0]
        row_parts = [dataset_label, LOSS_TEX_LABELS.get(loss_label, loss_label)]
        series = row.iloc[0]
        for column in LOSS_COLUMN_ORDER:
            value = series[column]
            is_optimal = bool(optimal_mask.loc[(dataset_label, loss_label), column])
            row_parts.append(_format_loss_entry(value, is_optimal))
        lines.append(" & ".join(row_parts) + r" \\")
        if idx < len(dataset_labels) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def export_real_loss_tables(
    repo_root: Path | str,
    tables: dict[str, pd.DataFrame],
    include_csv: bool = False,
) -> dict[str, str]:
    repo_root = Path(repo_root)
    output_dir = repo_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    naive_tex_path = output_dir / "real_naive_thresholds_table.tex"
    loss_tex_path = output_dir / "real_loss_comparison_table.tex"

    naive_thresholds_wide_df = tables["naive_thresholds_wide_df"]
    reported_loss_wide_df = tables["reported_loss_wide_df"]

    export_paths = {
        "naive_tex": str(naive_tex_path),
        "loss_tex": str(loss_tex_path),
    }

    if include_csv:
        naive_csv_path = output_dir / "real_naive_thresholds_table.csv"
        loss_csv_path = output_dir / "real_loss_comparison_table.csv"
        naive_thresholds_wide_df.reset_index().to_csv(naive_csv_path, index=False)

        loss_export_df = reported_loss_wide_df.copy()
        loss_export_df.columns = [f"{k_label} {method}" for k_label, method in loss_export_df.columns]
        loss_export_df = loss_export_df.reset_index()
        for column in loss_export_df.columns[2:]:
            loss_export_df[column] = loss_export_df[column].map(lambda value: f"{value:.1f}")
        loss_export_df.to_csv(loss_csv_path, index=False)

        export_paths.update({
            "naive_csv": str(naive_csv_path),
            "loss_csv": str(loss_csv_path),
        })

    naive_tex_path.write_text(create_naive_thresholds_latex_table(naive_thresholds_wide_df), encoding="utf-8")
    loss_tex_path.write_text(create_real_loss_latex_table(reported_loss_wide_df), encoding="utf-8")

    return export_paths
