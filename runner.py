from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from typing import Any

from agents import CentralBank, Household, clamp, create_households, create_social_network, weighted_sum
from settings import (
    AGGREGATE_WEIGHTS,
    EXPERIMENTS_DIR,
    H1_METRICS_CSV_PATH,
    H2_METRICS_CSV_PATH,
    H3_METRICS_CSV_PATH,
    H4_METRICS_CSV_PATH,
    HISTORY_CSV_PATH,
    HOUSEHOLD_TYPES,
    INFLATION_PATHS_PATH,
    METRICS_CSV_PATH,
    VALIDATION_DATA_PATH,
    create_settings,
    load_inflation_paths,
)

SCENARIO = {
    "name": "baseline",
    "use_real_data": True,
    "periods": 123,
    "households_amount": 1000,
    "random_seed": 42,
    "inflation_paths_path": INFLATION_PATHS_PATH,
    "validation_data_path": VALIDATION_DATA_PATH,
    "history_path": HISTORY_CSV_PATH,
    "metrics_path": METRICS_CSV_PATH,
    "settings": {},
}

VALIDATION_COLUMNS = (
    "date",
    "real_expected_inflation",
    "real_perceived_inflation",
    "aggregate_inflation",
    "key_rate",
    "inflation_target",
)

VALIDATION_OUTPUT_NAMES = {
    "real_expected_inflation": "real_expected_inflation",
    "real_perceived_inflation": "real_perceived_inflation",
    "aggregate_inflation": "real_aggregate_inflation",
    "key_rate": "key_rate",
    "inflation_target": "real_inflation_target",
}

GROUP_DIRS = {
    "h1": "h1_information",
    "h2": "h2_price_experience",
    "h3": "h3_cb_communication",
    "h4": "h4_social_network",
}

CLEAN_SETTINGS = {
    "use_social_network": False,
    "network_degree": 0,
    "network_homophily": 0.0,
    "social_influence": 0.0,
    "expectations_to_inflation_strength": 0.0,
}

H2_SETTINGS = {
    **CLEAN_SETTINGS,
    "initial_expectation_mode": "fixed_mean",
    "initial_expectation_std": 0.0,
}

COMMON_BEHAVIOR = {
    "update_frequency": 0.40,
    "cb_trust": 0.25,
    "perception_sensitivity": 1.30,
    "perception_bias": 0.035,
    "perception_memory": 0.0,
}


HOUSEHOLD_FIELDS = (
    "expectation",
    "perceived_inflation",
    "current_perceived_inflation",
    "official_basket_inflation",
    "price_experience_inflation",
    "neighbor_expectation",
    "anchor",
    "social_gap",
    "personal_weight",
    "cb_weight",
    "social_weight",
    "updated",
)


def collect_household_values(households: list[Household]) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {field: [] for field in HOUSEHOLD_FIELDS}
    for household in households:
        for field in HOUSEHOLD_FIELDS:
            values[field].append(getattr(household, field))
    return values


class InflationExpectationsModel:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.rng = random.Random(settings.get("random_seed"))
        self.cb = CentralBank(settings)
        self.households = create_households(settings, rng=self.rng)
        self.network_metrics = create_social_network(self.households, settings, rng=self.rng)
        self.households_by_type = self.group_households()
        self.last_mean_expectation = mean([household.expectation for household in self.households])
        self.history = self.empty_history()
        self.metrics: dict[str, Any] = {}

    def group_households(self) -> dict[str, list[Household]]:
        groups = {name: [] for name in self.settings["household_types"]}
        for household in self.households:
            groups[household.household_type].append(household)
        return groups

    def empty_history(self) -> dict[str, list[float]]:
        columns = {
            "period": [],
            "inflation_target": [],
            "official_aggregate_inflation": [],
            "inflation_feedback": [],
            "food_inflation": [],
            "nonfood_inflation": [],
            "services_inflation": [],
            "aggregate_inflation": [],
            "cb_signal": [],
            "mean_expectation": [],
            "std_expectation": [],
            "min_expectation": [],
            "max_expectation": [],
            "mean_official_basket_inflation": [],
            "mean_price_experience_inflation": [],
            "mean_current_perceived_inflation": [],
            "mean_perceived_inflation": [],
            "mean_neighbor_expectation": [],
            "mean_social_anchor": [],
            "mean_social_gap": [],
            "mean_abs_social_gap": [],
            "mean_personal_weight": [],
            "mean_cb_weight": [],
            "mean_social_weight": [],
            "target_gap": [],
            "update_share": [],
            "between_type_disagreement": [],
            "between_type_perceived_gap": [],
            "between_type_price_experience_gap": [],
            "between_type_official_basket_gap": [],
        }
        for household_type in self.settings["household_types"]:
            columns[f"mean_expectation_{household_type}"] = []
            columns[f"std_expectation_{household_type}"] = []
            columns[f"mean_perceived_inflation_{household_type}"] = []
            columns[f"mean_price_experience_inflation_{household_type}"] = []
            columns[f"mean_official_basket_inflation_{household_type}"] = []
            columns[f"update_share_{household_type}"] = []
        return columns

    def input_group_inflation(self, period: int) -> dict[str, float]:
        return {good: float(self.settings["inflation_paths"][good][period]) for good in self.settings["goods"]}

    def inflation_feedback(self, input_inflation: dict[str, float]) -> float:
        strength = float(self.settings.get("expectations_to_inflation_strength", 0.0))
        if strength == 0.0:
            return 0.0
        official = weighted_sum(input_inflation, self.settings["aggregate_weights"])
        feedback = strength * (self.last_mean_expectation - official)
        return clamp(feedback, -self.settings["max_inflation_feedback"], self.settings["max_inflation_feedback"])

    def step(self, period: int) -> None:
        for household in self.households:
            household.prepare_step()

        input_inflation = self.input_group_inflation(period)
        feedback = self.inflation_feedback(input_inflation)
        group_inflation = {good: input_inflation[good] + feedback for good in self.settings["goods"]}
        cb_signal = self.cb.step(group_inflation)

        for household in self.households:
            household.step(group_inflation, cb_signal, self.settings)

        self.record(period, input_inflation, group_inflation, feedback, cb_signal)
        self.last_mean_expectation = self.history["mean_expectation"][-1]

    def run(self) -> dict[str, list[float]]:
        for period in range(self.settings["periods"]):
            self.step(period)
        return self.history

    def record(
        self,
        period: int,
        input_inflation: dict[str, float],
        group_inflation: dict[str, float],
        feedback: float,
        cb_signal: float,
    ) -> None:
        values = collect_household_values(self.households)
        expectations = values["expectation"]
        mean_expectation = mean(expectations)

        self.append_history({
            "period": period,
            "inflation_target": float(self.settings["inflation_target"]),
            "official_aggregate_inflation": weighted_sum(input_inflation, self.settings["aggregate_weights"]),
            "inflation_feedback": feedback,
            "food_inflation": group_inflation["food"],
            "nonfood_inflation": group_inflation["nonfood"],
            "services_inflation": group_inflation["services"],
            "aggregate_inflation": self.cb.aggregate_inflation,
            "cb_signal": cb_signal,
            "mean_expectation": mean_expectation,
            "std_expectation": std(expectations),
            "min_expectation": min(expectations),
            "max_expectation": max(expectations),
            "mean_official_basket_inflation": mean(values["official_basket_inflation"]),
            "mean_price_experience_inflation": mean(values["price_experience_inflation"]),
            "mean_current_perceived_inflation": mean(values["current_perceived_inflation"]),
            "mean_perceived_inflation": mean(values["perceived_inflation"]),
            "mean_neighbor_expectation": mean(values["neighbor_expectation"]),
            "mean_social_anchor": mean(values["anchor"]),
            "mean_social_gap": mean(values["social_gap"]),
            "mean_abs_social_gap": mean([abs(value) for value in values["social_gap"]]),
            "mean_personal_weight": mean(values["personal_weight"]),
            "mean_cb_weight": mean(values["cb_weight"]),
            "mean_social_weight": mean(values["social_weight"]),
            "target_gap": mean_expectation - self.settings["inflation_target"],
            "update_share": share(values["updated"]),
        })

        type_means = []
        type_perceived = []
        type_price_experience = []
        type_official = []

        for household_type, households in self.households_by_type.items():
            type_values = collect_household_values(households)
            type_expectations = type_values["expectation"]
            type_perceived_values = type_values["perceived_inflation"]
            type_price_values = type_values["price_experience_inflation"]
            type_official_values = type_values["official_basket_inflation"]

            type_means.append(mean(type_expectations))
            type_perceived.append(mean(type_perceived_values))
            type_price_experience.append(mean(type_price_values))
            type_official.append(mean(type_official_values))

            self.append_history({
                f"mean_expectation_{household_type}": type_means[-1],
                f"std_expectation_{household_type}": std(type_expectations),
                f"mean_perceived_inflation_{household_type}": type_perceived[-1],
                f"mean_price_experience_inflation_{household_type}": type_price_experience[-1],
                f"mean_official_basket_inflation_{household_type}": type_official[-1],
                f"update_share_{household_type}": share(type_values["updated"]),
            })

        self.append_history({
            "between_type_disagreement": max(type_means) - min(type_means),
            "between_type_perceived_gap": max(type_perceived) - min(type_perceived),
            "between_type_price_experience_gap": max(type_price_experience) - min(type_price_experience),
            "between_type_official_basket_gap": max(type_official) - min(type_official),
        })

    def append_history(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self.history[name].append(value)


def build_settings(scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario = {**SCENARIO, **(scenario or {})}
    path = Path(scenario["inflation_paths_path"])
    inflation_paths = load_inflation_paths(path) if scenario["use_real_data"] and path.exists() else None

    config = {
        "periods": scenario["periods"],
        "households_amount": scenario["households_amount"],
        "random_seed": scenario["random_seed"],
        **scenario.get("settings", {}),
    }
    if inflation_paths is not None:
        config["inflation_paths"] = inflation_paths
    return create_settings(**config)


def run_experiment(scenario: dict[str, Any] | None = None) -> tuple[InflationExpectationsModel, Path]:
    scenario = {**SCENARIO, **(scenario or {})}
    model = InflationExpectationsModel(build_settings(scenario))
    history = model.run()
    validation = read_validation_data(scenario["validation_data_path"])
    history_path = save_history(history, validation, scenario["history_path"])
    model.metrics = calculate_metrics(history, validation)
    model.metrics.update(model.network_metrics)
    model.metrics["scenario"] = scenario["name"]
    save_metrics(model.metrics, scenario["metrics_path"])
    return model, history_path


def read_validation_data(path: str | Path = VALIDATION_DATA_PATH) -> dict[str, list[Any]]:
    data = {column: [] for column in VALIDATION_COLUMNS}
    file_path = Path(path)
    if not file_path.exists():
        return data

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            data["date"].append(row.get("date", ""))
            for column in VALIDATION_COLUMNS[1:]:
                value = row.get(column, "")
                data[column].append(float(value) if value not in ("", None) else None)
    return data


def save_history(history: dict[str, list[float]], validation: dict[str, list[Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", *history.keys(), *VALIDATION_OUTPUT_NAMES.values()]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(history["period"])):
            row = {"date": value_at(validation["date"], index, "")}
            row.update({name: values[index] for name, values in history.items()})
            for source, target in VALIDATION_OUTPUT_NAMES.items():
                row[target] = value_at(validation[source], index, "")
            writer.writerow(row)
    return path


def calculate_metrics(history: dict[str, list[float]], validation: dict[str, list[Any]]) -> dict[str, Any]:
    metrics = {
        "periods": len(history["period"]),
        "mean_model_expectation": mean(history["mean_expectation"]),
        "final_model_expectation": history["mean_expectation"][-1],
        "mean_model_perceived_inflation": mean(history["mean_perceived_inflation"]),
        "final_model_perceived_inflation": history["mean_perceived_inflation"][-1],
        "mean_model_disagreement": mean(history["between_type_disagreement"]),
        "max_model_disagreement": max(history["between_type_disagreement"]),
        "final_model_disagreement": history["between_type_disagreement"][-1],
        "mean_std_expectation": mean(history["std_expectation"]),
        "max_std_expectation": max(history["std_expectation"]),
        "mean_type_perceived_gap": mean(history["between_type_perceived_gap"]),
        "mean_type_price_experience_gap": mean(history["between_type_price_experience_gap"]),
        "mean_type_official_basket_gap": mean(history["between_type_official_basket_gap"]),
        "mean_update_share": mean(history["update_share"]),
        "mean_inflation_feedback": mean(history["inflation_feedback"]),
        "max_abs_inflation_feedback": max(abs(value) for value in history["inflation_feedback"]),
        "mean_abs_social_gap": mean(history["mean_abs_social_gap"]),
        "mean_social_weight": mean(history["mean_social_weight"]),
    }
    metrics.update(error_metrics("expectation", history["mean_expectation"], validation["real_expected_inflation"]))
    metrics.update(error_metrics("perceived_inflation", history["mean_perceived_inflation"], validation["real_perceived_inflation"]))
    metrics.update(error_metrics("official_inflation", history["official_aggregate_inflation"], validation["aggregate_inflation"]))
    return metrics


def error_metrics(name: str, model_values: list[float], real_values: list[Any]) -> dict[str, Any]:
    pairs = paired_values(model_values, real_values)
    if not pairs:
        return {
            f"{name}_mean_error": "",
            f"{name}_mae": "",
            f"{name}_rmse": "",
            f"{name}_correlation": "",
            f"{name}_model_peak_period": "",
            f"{name}_real_peak_period": "",
            f"{name}_final_error": "",
        }

    model = [left for left, _ in pairs]
    real = [right for _, right in pairs]
    errors = [left - right for left, right in pairs]
    return {
        f"{name}_mean_error": mean(errors),
        f"{name}_mae": mean([abs(value) for value in errors]),
        f"{name}_rmse": mean([value ** 2 for value in errors]) ** 0.5,
        f"{name}_correlation": correlation(model, real),
        f"{name}_model_peak_period": model.index(max(model)),
        f"{name}_real_peak_period": real.index(max(real)),
        f"{name}_final_error": errors[-1],
    }


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return path


def run_scenario_group(group_name: str, scenarios: dict[str, dict[str, Any]], summary_path: str | Path) -> Path:
    output_dir = EXPERIMENTS_DIR / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_csv_files(output_dir)

    rows = []
    for name, settings_override in scenarios.items():
        scenario = {
            **SCENARIO,
            "name": name,
            "history_path": output_dir / f"{name}_history.csv",
            "metrics_path": output_dir / f"{name}_metrics.csv",
            "settings": settings_override,
        }
        model, _ = run_experiment(scenario)
        rows.append(dict(model.metrics))

    summary = output_dir / f"{group_name}_summary.csv"
    save_metrics_table(rows, summary)
    save_metrics_table(rows, summary_path)
    return summary


def clean_csv_files(directory: Path) -> None:
    for path in directory.glob("*.csv"):
        path.unlink()


def save_metrics_table(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def information_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "no_update": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "update_frequency", 0.0)},
        "slow_update": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "update_frequency", 0.15)},
        "baseline_update": {**CLEAN_SETTINGS, "household_types": copy.deepcopy(HOUSEHOLD_TYPES)},
        "fast_update": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "update_frequency", 0.80)},
    }


def price_experience_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "homogeneous_agents": {**H2_SETTINGS, "household_types": homogeneous_types(), "use_price_salience": False},
        "basket_only": {**H2_SETTINGS, "household_types": basket_only_types(), "use_price_salience": False},
        "basket_salience": {**H2_SETTINGS, "household_types": basket_salience_types(), "use_price_salience": True},
        "behavior_only": {**H2_SETTINGS, "household_types": behavior_only_types(), "use_price_salience": False},
        "full_income_proxy_profile": {**H2_SETTINGS, "household_types": copy.deepcopy(HOUSEHOLD_TYPES), "use_price_salience": True},
    }


def cb_communication_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "no_trust": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "cb_trust", 0.0)},
        "low_trust": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "cb_trust", 0.15)},
        "baseline_trust": {**CLEAN_SETTINGS, "household_types": copy.deepcopy(HOUSEHOLD_TYPES)},
        "high_trust": {**CLEAN_SETTINGS, "household_types": set_type_parameter(HOUSEHOLD_TYPES, "cb_trust", 0.60)},
        "target_signal": {**CLEAN_SETTINGS, "signal_alpha": 1.0},
        "inflation_signal": {**CLEAN_SETTINGS, "signal_alpha": 0.0},
    }


def social_network_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "no_network": {"use_social_network": False, "network_degree": 0, "network_homophily": 0.0, "social_influence": 0.0, "expectations_to_inflation_strength": 0.0},
        "random_network": {"use_social_network": True, "network_degree": 8, "network_homophily": 0.33, "social_influence": 0.20, "expectations_to_inflation_strength": 0.0},
        "homophilic_network": {"use_social_network": True, "network_degree": 8, "network_homophily": 0.75, "social_influence": 0.20, "expectations_to_inflation_strength": 0.0},
        "echo_chamber_network": {"use_social_network": True, "network_degree": 8, "network_homophily": 0.90, "social_influence": 0.40, "expectations_to_inflation_strength": 0.0},
        "mixed_high_influence_network": {"use_social_network": True, "network_degree": 8, "network_homophily": 0.30, "social_influence": 0.40, "expectations_to_inflation_strength": 0.0},
    }


def homogeneous_types() -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(HOUSEHOLD_TYPES)
    for params in result.values():
        params["basket_weights"] = dict(AGGREGATE_WEIGHTS)
        params["salience_weights"] = dict(AGGREGATE_WEIGHTS)
        params.update(COMMON_BEHAVIOR)
    return result


def basket_only_types() -> dict[str, dict[str, Any]]:
    result = homogeneous_types()
    for name, params in result.items():
        params["basket_weights"] = dict(HOUSEHOLD_TYPES[name]["basket_weights"])
        params["salience_weights"] = dict(HOUSEHOLD_TYPES[name]["basket_weights"])
    return result


def basket_salience_types() -> dict[str, dict[str, Any]]:
    result = homogeneous_types()
    for name, params in result.items():
        params["basket_weights"] = dict(HOUSEHOLD_TYPES[name]["basket_weights"])
        params["salience_weights"] = dict(HOUSEHOLD_TYPES[name]["salience_weights"])
    return result


def behavior_only_types() -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(HOUSEHOLD_TYPES)
    for params in result.values():
        params["basket_weights"] = dict(AGGREGATE_WEIGHTS)
        params["salience_weights"] = dict(AGGREGATE_WEIGHTS)
    return result


def set_type_parameter(types: dict[str, dict[str, Any]], parameter: str, value: float) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(types)
    for params in result.values():
        params[parameter] = value
    return result


def paired_values(model_values: list[float], real_values: list[Any]) -> list[tuple[float, float]]:
    return [(float(left), float(right)) for left, right in zip(model_values, real_values) if right not in ("", None)]


def value_at(values: list[Any], index: int, default: Any = "") -> Any:
    return values[index] if index < len(values) and values[index] is not None else default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return (sum((value - center) ** 2 for value in values) / len(values)) ** 0.5


def share(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_std = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_std = sum((y - right_mean) ** 2 for y in right) ** 0.5
    return numerator / (left_std * right_std) if left_std and right_std else 0.0


def run_all() -> None:
    run_experiment()
    run_scenario_group(GROUP_DIRS["h1"], information_scenarios(), H1_METRICS_CSV_PATH)
    run_scenario_group(GROUP_DIRS["h2"], price_experience_scenarios(), H2_METRICS_CSV_PATH)
    run_scenario_group(GROUP_DIRS["h3"], cb_communication_scenarios(), H3_METRICS_CSV_PATH)
    run_scenario_group(GROUP_DIRS["h4"], social_network_scenarios(), H4_METRICS_CSV_PATH)


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    scenario_groups = {
        "h1": (GROUP_DIRS["h1"], information_scenarios, H1_METRICS_CSV_PATH),
        "h2": (GROUP_DIRS["h2"], price_experience_scenarios, H2_METRICS_CSV_PATH),
        "h3": (GROUP_DIRS["h3"], cb_communication_scenarios, H3_METRICS_CSV_PATH),
        "h4": (GROUP_DIRS["h4"], social_network_scenarios, H4_METRICS_CSV_PATH),
    }

    if command in scenario_groups:
        group_dir, scenario_factory, metrics_path = scenario_groups[command]
        print(run_scenario_group(group_dir, scenario_factory(), metrics_path))
    elif command == "all":
        run_all()
        print(EXPERIMENTS_DIR)
    else:
        _, saved_path = run_experiment()
        print(saved_path)
