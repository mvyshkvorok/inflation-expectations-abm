from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

GOODS = ("food", "nonfood", "services")

AGGREGATE_WEIGHTS = {
    "food": 0.388,
    "nonfood": 0.346,
    "services": 0.266,
}

# Proxy socio-income profiles. Income is not a monetary flow in the model;
# it is represented indirectly through basket structure, price salience,
# information updating and the weight of the official CB signal.
HOUSEHOLD_TYPES = {
    "low_income": {
        "share": 0.30,
        "income_proxy": "low",
        "basket_weights": {"food": 0.47, "nonfood": 0.25, "services": 0.28},
        "salience_weights": {"food": 0.68, "nonfood": 0.18, "services": 0.14},
        "update_frequency": 0.30,
        "cb_trust": 0.12,
        "perception_sensitivity": 1.45,
        "perception_bias": 0.040,
        "perception_memory": 0.58,
    },
    "middle_income": {
        "share": 0.50,
        "income_proxy": "middle",
        "basket_weights": {"food": 0.39, "nonfood": 0.35, "services": 0.26},
        "salience_weights": {"food": 0.55, "nonfood": 0.25, "services": 0.20},
        "update_frequency": 0.40,
        "cb_trust": 0.25,
        "perception_sensitivity": 1.30,
        "perception_bias": 0.035,
        "perception_memory": 0.50,
    },
    "high_income": {
        "share": 0.20,
        "income_proxy": "high",
        "basket_weights": {"food": 0.24, "nonfood": 0.33, "services": 0.43},
        "salience_weights": {"food": 0.42, "nonfood": 0.25, "services": 0.33},
        "update_frequency": 0.52,
        "cb_trust": 0.38,
        "perception_sensitivity": 1.18,
        "perception_bias": 0.030,
        "perception_memory": 0.42,
    },
}

HOUSEHOLD_TYPE_LABELS = {
    "low_income": "низкодоходные / ценочувствительные",
    "middle_income": "среднедоходные / сбалансированные",
    "high_income": "высокодоходные / финансово включенные",
}

DEFAULT_SETTINGS = {
    "periods": 123,
    "households_amount": 1000,
    "goods": GOODS,
    "inflation_paths": None,
    "inflation_target": 0.04,
    "signal_alpha": 0.85,
    "inertia": 0.80,
    "aggregate_weights": AGGREGATE_WEIGHTS,
    "household_types": HOUSEHOLD_TYPES,
    "initial_expectation_mode": "fixed_mean_with_noise",
    "initial_expectation_mean": 0.15,
    "initial_expectation_std": 0.025,
    "min_expectation": -0.05,
    "max_expectation": 0.50,
    "random_seed": 42,
    "perception_memory": 0.50,
    "use_price_salience": True,
    "expectations_to_inflation_strength": 0.0,
    "max_inflation_feedback": 0.03,
    "use_social_network": False,
    "network_degree": 8,
    "network_homophily": 0.75,
    "social_influence": 0.20,
}


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for path in (current, *current.parents):
        if (path / "data" / "processed" / "v1_inflation_paths.json").exists():
            return path
        if (path / "v1_inflation_paths.json").exists():
            return path
    return current


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed" if (PROJECT_ROOT / "data" / "processed").exists() else PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"

INFLATION_PATHS_PATH = DATA_DIR / "v1_inflation_paths.json"
VALIDATION_DATA_PATH = DATA_DIR / "v1_monthly_dataset.csv"
HISTORY_CSV_PATH = OUTPUT_DIR / "history.csv"
METRICS_CSV_PATH = OUTPUT_DIR / "metrics.csv"

H1_METRICS_CSV_PATH = EXPERIMENTS_DIR / "h1_information_metrics.csv"
H2_METRICS_CSV_PATH = EXPERIMENTS_DIR / "h2_price_experience_metrics.csv"
H3_METRICS_CSV_PATH = EXPERIMENTS_DIR / "h3_cb_communication_metrics.csv"
H4_METRICS_CSV_PATH = EXPERIMENTS_DIR / "h4_social_network_metrics.csv"


def default_inflation_paths(periods: int, values: dict[str, float] | None = None) -> dict[str, list[float]]:
    values = values or {"food": 0.06, "nonfood": 0.05, "services": 0.07}
    return {good: [float(values[good])] * periods for good in GOODS}


def load_inflation_paths(path: str | Path = INFLATION_PATHS_PATH) -> dict[str, list[float]]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    paths = {good: [float(value) for value in data[good]] for good in GOODS}
    validate_inflation_paths(paths, len(paths[GOODS[0]]))
    return paths


def create_settings(**overrides: Any) -> dict[str, Any]:
    model_settings = copy.deepcopy(DEFAULT_SETTINGS)
    model_settings.update(overrides)

    if model_settings["inflation_paths"] is None:
        model_settings["inflation_paths"] = default_inflation_paths(int(model_settings["periods"]))
    else:
        model_settings["inflation_paths"] = copy.deepcopy(model_settings["inflation_paths"])
        model_settings["periods"] = len(model_settings["inflation_paths"][GOODS[0]])

    model_settings["goods"] = tuple(model_settings.get("goods", GOODS))
    model_settings["periods"] = int(model_settings["periods"])
    model_settings["households_amount"] = int(model_settings["households_amount"])
    model_settings["network_degree"] = int(model_settings.get("network_degree", 0))
    validate_settings(model_settings)
    return model_settings


def create_real_data_settings(
    data_path: str | Path = INFLATION_PATHS_PATH,
    households_amount: int = 1000,
    random_seed: int | None = 42,
    **overrides: Any,
) -> dict[str, Any]:
    return create_settings(
        inflation_paths=load_inflation_paths(data_path),
        households_amount=households_amount,
        random_seed=random_seed,
        **overrides,
    )


def validate_settings(model_settings: dict[str, Any]) -> None:
    if model_settings["periods"] <= 0:
        raise ValueError("periods must be positive")
    if model_settings["households_amount"] <= 0:
        raise ValueError("households_amount must be positive")
    if model_settings["min_expectation"] >= model_settings["max_expectation"]:
        raise ValueError("min_expectation must be lower than max_expectation")
    if float(model_settings["initial_expectation_std"]) < 0:
        raise ValueError("initial_expectation_std must be non-negative")
    if float(model_settings["max_inflation_feedback"]) < 0:
        raise ValueError("max_inflation_feedback must be non-negative")
    if int(model_settings["network_degree"]) < 0:
        raise ValueError("network_degree must be non-negative")

    for name in ("signal_alpha", "inertia", "perception_memory", "expectations_to_inflation_strength", "network_homophily", "social_influence"):
        validate_probability(name, model_settings[name])

    validate_inflation_paths(model_settings["inflation_paths"], model_settings["periods"])
    validate_weights("aggregate_weights", model_settings["aggregate_weights"], model_settings["goods"])
    validate_household_types(model_settings["household_types"], model_settings["goods"])


def validate_inflation_paths(paths: dict[str, list[float]], periods: int) -> None:
    if set(paths) != set(GOODS):
        raise ValueError(f"inflation_paths must contain {GOODS}")
    for good, values in paths.items():
        if len(values) != periods:
            raise ValueError(f"{good} path length must be {periods}, got {len(values)}")


def validate_household_types(types: dict[str, dict[str, Any]], goods: tuple[str, ...]) -> None:
    required = {"share", "basket_weights", "update_frequency", "cb_trust", "perception_sensitivity", "perception_bias"}
    total_share = 0.0
    for name, params in types.items():
        missing = required - set(params)
        if missing:
            raise ValueError(f"household type {name} misses {sorted(missing)}")
        validate_probability(f"{name}.share", params["share"])
        validate_probability(f"{name}.update_frequency", params["update_frequency"])
        validate_probability(f"{name}.cb_trust", params["cb_trust"])
        if "perception_memory" in params:
            validate_probability(f"{name}.perception_memory", params["perception_memory"])
        validate_weights(f"{name}.basket_weights", params["basket_weights"], goods)
        if "salience_weights" in params:
            validate_weights(f"{name}.salience_weights", params["salience_weights"], goods)
        if float(params["perception_sensitivity"]) <= 0:
            raise ValueError(f"{name}.perception_sensitivity must be positive")
        total_share += float(params["share"])
    if not near(total_share, 1.0):
        raise ValueError(f"household type shares must sum to 1.0, got {total_share}")


def validate_weights(name: str, weights: dict[str, float], goods: tuple[str, ...]) -> None:
    if set(weights) != set(goods):
        raise ValueError(f"{name} must contain {goods}")
    total = sum(float(value) for value in weights.values())
    if not near(total, 1.0):
        raise ValueError(f"{name} must sum to 1.0, got {total}")


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def near(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance
