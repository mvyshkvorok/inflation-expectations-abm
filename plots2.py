from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import (
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    HISTORY_CSV_PATH,
    H1_METRICS_CSV_PATH,
    H2_METRICS_CSV_PATH,
    H4_METRICS_CSV_PATH,
    HOUSEHOLD_TYPE_LABELS,
)

PCT = 100.0
DPI = 220

GROUP_DIRS = {
    "h1": "h1_information",
    "h2": "h2_price_experience",
    "h3": "h3_cb_communication",
    "h4": "h4_social_network",
}

ORDER = {
    "h1": ["no_update", "slow_update", "baseline_update", "fast_update"],
    "h2": ["homogeneous_agents", "basket_only", "basket_salience", "behavior_only", "full_income_proxy_profile"],
    "h3": ["no_trust", "low_trust", "baseline_trust", "high_trust", "target_signal", "inflation_signal"],
    "h4": ["no_network", "random_network", "homophilic_network", "echo_chamber_network", "mixed_high_influence_network"],
}

LABELS = {
    "inflation_target": "таргет ЦБ",
    "official_aggregate_inflation": "официальная инфляция",
    "aggregate_inflation": "эффективная инфляция в модели",
    "cb_signal": "сигнал ЦБ",
    "mean_expectation": "средние ожидания",
    "mean_perceived_inflation": "воспринимаемая инфляция",
    "mean_price_experience_inflation": "личный ценовой опыт",
    "mean_official_basket_inflation": "инфляция личной корзины",
    "real_expected_inflation": "опросные ожидания",
    "real_perceived_inflation": "опросная воспринимаемая инфляция",
    "std_expectation": "разброс внутри населения",
    "between_type_disagreement": "разрыв между типами",
    "between_type_perceived_gap": "разрыв воспринимаемой инфляции",
    "between_type_price_experience_gap": "разрыв личного ценового опыта",
    "update_share": "доля обновивших ожидания",
    "mean_abs_social_gap": "разрыв с соседями",
    "mean_personal_weight": "вес личного опыта",
    "mean_cb_weight": "вес сигнала ЦБ",
    "mean_social_weight": "вес соцсети",
}

SCENARIO_LABELS = {
    "no_update": "нет обновления",
    "slow_update": "редкое обновление",
    "baseline_update": "базовая частота",
    "fast_update": "частое обновление",
    "homogeneous_agents": "одинаковые агенты",
    "basket_only": "только разные корзины",
    "basket_salience": "корзины + заметность",
    "behavior_only": "только поведенческие параметры",
    "full_income_proxy_profile": "полный proxy-профиль",
    "no_trust": "нет доверия",
    "low_trust": "низкое доверие",
    "baseline_trust": "базовое доверие",
    "high_trust": "высокое доверие",
    "target_signal": "сигнал = таргет",
    "inflation_signal": "сигнал = инфляция",
    "no_network": "без сети",
    "random_network": "случайная сеть",
    "homophilic_network": "гомофильная сеть",
    "echo_chamber_network": "эхо-камера",
    "mixed_high_influence_network": "смешанная сеть + сильное влияние",
}

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = [None, "o", "s", "^", "D", "v"]


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def make_baseline_plots(history_path: str | Path = HISTORY_CSV_PATH, output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    configure_style()
    df = pd.read_csv(history_path)
    output = Path(output_dir) / "baseline"
    output.mkdir(parents=True, exist_ok=True)

    plots = [
        plot_columns(df, ["mean_expectation", "real_expected_inflation", "official_aggregate_inflation", "cb_signal"], output / "01_expectations_inflation_signal.png", "Ожидания, инфляция и сигнал ЦБ", target=True),
        plot_columns(df, ["mean_perceived_inflation", "real_perceived_inflation", "mean_price_experience_inflation", "official_aggregate_inflation"], output / "02_perceived_inflation_and_experience.png", "Воспринимаемая инфляция и личный ценовой опыт"),
        plot_type_lines(df, "mean_expectation_", output / "03_expectations_by_group.png", "Ожидания по proxy-группам", target=True),
        plot_type_deviations(df, output / "04_group_deviation_from_average.png", "Отклонение ожиданий групп от среднего"),
        plot_columns(df, ["std_expectation", "between_type_disagreement", "between_type_perceived_gap", "between_type_price_experience_gap"], output / "05_heterogeneity_components.png", "Компоненты гетерогенности"),
        plot_columns(df, ["mean_personal_weight", "mean_cb_weight", "mean_social_weight"], output / "06_information_weights.png", "Веса информационных каналов", percent=False),
    ]
    return [path for path in plots if path is not None]


def make_group_plots(group: str, output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    configure_style()
    key = normalize_group(group)
    data = load_scenario_histories(key)
    if not data:
        return []

    output = Path(output_dir) / key
    output.mkdir(parents=True, exist_ok=True)

    if key == "h1":
        return [path for path in [
            plot_scenarios(data, "mean_expectation", output / "01_expectations_with_survey.png", "H1: ожидания и опросные ожидания", target=True, reference="real_expected_inflation"),
            plot_scenarios(data, "between_type_disagreement", output / "02_disagreement.png", "H1: межтиповой разрыв"),
            plot_scenarios(data, "update_share", output / "03_update_share.png", "H1: доля обновивших ожидания"),
            plot_scenarios(data, "mean_perceived_inflation", output / "04_perceived_inflation.png", "H1: воспринимаемая инфляция", reference="real_perceived_inflation"),
        ] if path is not None]

    if key == "h2":
        return make_h2_plots(data, output)

    if key == "h3":
        return [path for path in [
            plot_scenarios(data, "mean_expectation", output / "01_expectations_with_survey.png", "H3: ожидания и опросные ожидания", target=True, reference="real_expected_inflation"),
            plot_scenarios(data, "between_type_disagreement", output / "02_disagreement.png", "H3: межтиповой разрыв"),
            plot_scenarios(data, "mean_perceived_inflation", output / "03_perceived_inflation.png", "H3: воспринимаемая инфляция", reference="real_perceived_inflation"),
            plot_scenarios(data, "cb_signal", output / "04_cb_signal.png", "H3: сигнал ЦБ", target=True),
            plot_scenarios(data, "mean_cb_weight", output / "05_cb_weight.png", "H3: вес сигнала ЦБ", percent=False),
        ] if path is not None]

    return [path for path in [
        plot_scenarios(data, "mean_expectation", output / "01_expectations.png", "H4: средние ожидания", target=True),
        plot_scenarios(data, "between_type_disagreement", output / "02_disagreement.png", "H4: межтиповой разрыв"),
        plot_scenarios(data, "mean_abs_social_gap", output / "03_social_gap.png", "H4: разрыв с соседями"),
        plot_scenarios(data, "mean_social_weight", output / "04_social_weight.png", "H4: вес социальной информации", percent=False),
        plot_h4_metrics(output / "05_network_metrics.png"),
    ] if path is not None]


def make_h2_plots(data: list[tuple[str, pd.DataFrame]], output: Path) -> list[Path]:
    plots = [
        plot_h2_decomposition(output / "01_decomposition.png"),
        plot_scenarios(data, "mean_expectation", output / "02_expectations.png", "H2: средние ожидания", target=True),
        plot_scenarios(data, "mean_perceived_inflation", output / "03_perceived_inflation.png", "H2: воспринимаемая инфляция"),
        plot_scenarios(data, "between_type_disagreement", output / "04_disagreement.png", "H2: межтиповой разрыв ожиданий"),
    ]

    full = dict(data).get("full_income_proxy_profile")
    if full is not None:
        plots.extend([
            plot_type_lines(full, "mean_expectation_", output / "05_full_profile_groups.png", "H2: ожидания групп в полном proxy-профиле", target=True),
            plot_type_deviations(full, output / "06_full_profile_deviation.png", "H2: отклонение групп от среднего"),
        ])
    return [path for path in plots if path is not None]


def make_all_plots(output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    paths: list[Path] = []
    if Path(HISTORY_CSV_PATH).exists():
        paths.extend(make_baseline_plots(output_dir=output_dir))
    for group in ("h1", "h2", "h3", "h4"):
        paths.extend(make_group_plots(group, output_dir=output_dir))
    return paths


def load_scenario_histories(group: str) -> list[tuple[str, pd.DataFrame]]:
    key = normalize_group(group)
    directory = EXPERIMENTS_DIR / GROUP_DIRS[key]
    if not directory.exists():
        return []

    data = []
    for name in ORDER[key]:
        path = directory / f"{name}_history.csv"
        if path.exists():
            data.append((name, pd.read_csv(path)))
    return data


def plot_columns(
    df: pd.DataFrame,
    columns: list[str],
    path: Path,
    title: str,
    target: bool = False,
    percent: bool = True,
) -> Path | None:
    columns = [column for column in columns if has_data(df, column)]
    if not columns:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x, xlabel = x_axis(df)
    multiplier = PCT if percent else 1.0
    for index, column in enumerate(columns):
        ax.plot(x, numeric(df[column]) * multiplier, linewidth=2.1, linestyle=line_style(index), marker=marker(index), markevery=12, label=label(column))
    add_target(ax, df, target, percent)
    finish_plot(fig, ax, path, title, xlabel, "%" if percent else "вес")
    return path


def plot_scenarios(
    data: list[tuple[str, pd.DataFrame]],
    column: str,
    path: Path,
    title: str,
    target: bool = False,
    reference: str | None = None,
    percent: bool = True,
) -> Path | None:
    if not any(has_data(df, column) for _, df in data):
        return None

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    multiplier = PCT if percent else 1.0
    xlabel = "период"
    for index, (name, df) in enumerate(data):
        if not has_data(df, column):
            continue
        x, xlabel = x_axis(df)
        ax.plot(x, numeric(df[column]) * multiplier, linewidth=2.1, linestyle=line_style(index), marker=marker(index), markevery=12, label=scenario_label(name))

    if reference and data and has_data(data[0][1], reference):
        x, xlabel = x_axis(data[0][1])
        ax.plot(x, numeric(data[0][1][reference]) * multiplier, linewidth=2.0, linestyle="--", color="black", alpha=0.75, label=label(reference))
    add_target(ax, data[0][1], target, percent)
    finish_plot(fig, ax, path, title, xlabel, "%" if percent else "вес")
    return path


def plot_type_lines(df: pd.DataFrame, prefix: str, path: Path, title: str, target: bool = False) -> Path | None:
    columns = [f"{prefix}{group}" for group in HOUSEHOLD_TYPE_LABELS if has_data(df, f"{prefix}{group}")]
    if not columns:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x, xlabel = x_axis(df)
    for index, column in enumerate(columns):
        group = column.replace(prefix, "")
        ax.plot(x, numeric(df[column]) * PCT, linewidth=2.1, linestyle=line_style(index), marker=marker(index), markevery=12, label=type_label(group))
    add_target(ax, df, target, True)
    finish_plot(fig, ax, path, title, xlabel, "%")
    return path


def plot_type_deviations(df: pd.DataFrame, path: Path, title: str) -> Path | None:
    columns = [f"mean_expectation_{group}" for group in HOUSEHOLD_TYPE_LABELS if has_data(df, f"mean_expectation_{group}")]
    if not columns or not has_data(df, "mean_expectation"):
        return None

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x, xlabel = x_axis(df)
    baseline = numeric(df["mean_expectation"])
    for index, column in enumerate(columns):
        group = column.replace("mean_expectation_", "")
        ax.plot(x, (numeric(df[column]) - baseline) * PCT, linewidth=2.1, linestyle=line_style(index), marker=marker(index), markevery=12, label=type_label(group))
    ax.axhline(0, linestyle="--", linewidth=1.2)
    finish_plot(fig, ax, path, title, xlabel, "п.п.")
    return path


def plot_h2_decomposition(path: Path) -> Path | None:
    metrics_path = H2_METRICS_CSV_PATH
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    required = ["scenario", "mean_type_price_experience_gap", "mean_type_perceived_gap", "mean_model_disagreement"]
    if df.empty or any(column not in df.columns for column in required):
        return None

    df = df[df["scenario"].isin(ORDER["h2"])].copy()
    df["_order"] = df["scenario"].map({name: index for index, name in enumerate(ORDER["h2"])})
    df = df.sort_values("_order")

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    x = range(len(df))
    width = 0.25
    series = [
        ("mean_type_price_experience_gap", "ценовой опыт"),
        ("mean_type_perceived_gap", "восприятие"),
        ("mean_model_disagreement", "ожидания"),
    ]
    for offset, (column, text) in zip([-width, 0, width], series):
        ax.bar([value + offset for value in x], pd.to_numeric(df[column]) * PCT, width=width, label=text)
    ax.set_xticks(list(x))
    ax.set_xticklabels([scenario_label(name) for name in df["scenario"]], rotation=15, ha="right")
    finish_plot(fig, ax, path, "H2: декомпозиция гетерогенности", "сценарий", "п.п.")
    return path


def plot_h4_metrics(path: Path) -> Path | None:
    if not H4_METRICS_CSV_PATH.exists():
        return None
    df = pd.read_csv(H4_METRICS_CSV_PATH)
    required = ["scenario", "network_realized_homophily", "mean_abs_social_gap", "mean_social_weight"]
    if df.empty or any(column not in df.columns for column in required):
        return None

    df = df[df["scenario"].isin(ORDER["h4"])].copy()
    df["_order"] = df["scenario"].map({name: index for index, name in enumerate(ORDER["h4"])})
    df = df.sort_values("_order")

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    x = range(len(df))
    width = 0.25
    series = [
        ("network_realized_homophily", "гомофилия", 1.0),
        ("mean_abs_social_gap", "разрыв с соседями, п.п.", PCT),
        ("mean_social_weight", "вес соцсети", 1.0),
    ]
    for offset, (column, text, multiplier) in zip([-width, 0, width], series):
        ax.bar([value + offset for value in x], pd.to_numeric(df[column]) * multiplier, width=width, label=text)
    ax.set_xticks(list(x))
    ax.set_xticklabels([scenario_label(name) for name in df["scenario"]], rotation=15, ha="right")
    finish_plot(fig, ax, path, "H4: параметры сети", "сценарий", "значение")
    return path


def add_target(ax: plt.Axes, df: pd.DataFrame, enabled: bool, percent: bool) -> None:
    if enabled and has_data(df, "inflation_target"):
        multiplier = PCT if percent else 1.0
        value = numeric(df["inflation_target"]).dropna().iloc[0] * multiplier
        ax.axhline(value, linestyle="--", linewidth=1.4, label=label("inflation_target"))


def finish_plot(fig: plt.Figure, ax: plt.Axes, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def x_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.notna().sum() > 2:
            return dates, "год"
    if "period" in df.columns:
        return pd.to_numeric(df["period"], errors="coerce"), "период"
    return pd.Series(range(len(df))), "период"


def has_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and numeric(df[column]).notna().any()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def label(column: str) -> str:
    return LABELS.get(column, column)


def scenario_label(name: str) -> str:
    return SCENARIO_LABELS.get(name, name.replace("_", " "))


def type_label(group: str) -> str:
    return HOUSEHOLD_TYPE_LABELS.get(group, group.replace("_", " "))


def line_style(index: int) -> str:
    return LINE_STYLES[index % len(LINE_STYLES)]


def marker(index: int) -> str | None:
    return MARKERS[index % len(MARKERS)]


def normalize_group(group: str) -> str:
    aliases = {
        "h1": "h1",
        "information": "h1",
        "h2": "h2",
        "price_experience": "h2",
        "h3": "h3",
        "cb": "h3",
        "communication": "h3",
        "h4": "h4",
        "social": "h4",
        "social_network": "h4",
    }
    if group not in aliases:
        raise ValueError(f"unknown plot group: {group}")
    return aliases[group]


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if command == "all":
        paths = make_all_plots()
    elif command in ("h1", "h2", "h3", "h4", "information", "price_experience", "cb", "communication", "social"):
        paths = make_group_plots(command)
    else:
        paths = make_baseline_plots()

    for saved_path in paths:
        print(saved_path)
