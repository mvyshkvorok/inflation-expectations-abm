from __future__ import annotations

import random
from typing import Any


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def weighted_sum(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(weights[key]) * float(values[key]) for key in weights)


class CentralBank:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.target = float(settings["inflation_target"])
        self.signal_alpha = float(settings["signal_alpha"])
        self.aggregate_weights = dict(settings["aggregate_weights"])
        self.aggregate_inflation = self.target
        self.signal = self.target

    def step(self, group_inflation: dict[str, float]) -> float:
        self.aggregate_inflation = weighted_sum(group_inflation, self.aggregate_weights)
        self.signal = self.signal_alpha * self.target + (1.0 - self.signal_alpha) * self.aggregate_inflation
        return self.signal


class Household:
    def __init__(self, id_: int, household_type: str, settings: dict[str, Any], rng: random.Random) -> None:
        params = settings["household_types"][household_type]

        self.id = id_
        self.household_type = household_type
        self.rng = rng

        self.basket_weights = dict(params["basket_weights"])
        self.salience_weights = dict(params.get("salience_weights", self.basket_weights))
        self.update_frequency = float(params["update_frequency"])
        self.cb_trust = float(params["cb_trust"])
        self.perception_sensitivity = float(params["perception_sensitivity"])
        self.perception_bias = float(params["perception_bias"])
        self.perception_memory = float(params.get("perception_memory", settings["perception_memory"]))

        self.expectation = self.initial_expectation(settings)
        self.previous_expectation = self.expectation
        self.neighbors: list[Household] = []

        self.official_basket_inflation = self.expectation
        self.price_experience_inflation = self.expectation
        self.current_perceived_inflation = self.expectation
        self.perceived_inflation = self.expectation

        self.neighbor_expectation = 0.0
        self.social_gap = 0.0
        self.anchor = self.expectation
        self.personal_weight = 1.0 - self.cb_trust
        self.cb_weight = self.cb_trust
        self.social_weight = 0.0
        self.updated = False

    def initial_expectation(self, settings: dict[str, Any]) -> float:
        mode = settings["initial_expectation_mode"]
        target = float(settings["inflation_target"])
        mean = float(settings["initial_expectation_mean"])
        std = float(settings["initial_expectation_std"])

        if mode == "target":
            value = target
        elif mode == "target_with_noise":
            value = self.rng.gauss(target, std)
        elif mode == "fixed_mean":
            value = mean
        elif mode == "fixed_mean_with_noise":
            value = self.rng.gauss(mean, std)
        else:
            raise ValueError(f"unknown initial_expectation_mode: {mode}")

        return clamp(value, settings["min_expectation"], settings["max_expectation"])

    def prepare_step(self) -> None:
        self.previous_expectation = self.expectation

    def perceive(self, group_inflation: dict[str, float], settings: dict[str, Any]) -> float:
        self.official_basket_inflation = weighted_sum(group_inflation, self.basket_weights)
        price_weights = self.salience_weights if settings.get("use_price_salience", False) else self.basket_weights
        self.price_experience_inflation = weighted_sum(group_inflation, price_weights)

        current = self.perception_sensitivity * self.price_experience_inflation + self.perception_bias
        self.current_perceived_inflation = clamp(current, settings["min_expectation"], settings["max_expectation"])
        self.perceived_inflation = clamp(
            self.perception_memory * self.perceived_inflation
            + (1.0 - self.perception_memory) * self.current_perceived_inflation,
            settings["min_expectation"],
            settings["max_expectation"],
        )
        return self.perceived_inflation

    def social_expectation(self) -> float | None:
        if not self.neighbors:
            return None
        return sum(neighbor.previous_expectation for neighbor in self.neighbors) / len(self.neighbors)

    def update_anchor(self, perceived: float, cb_signal: float, settings: dict[str, Any]) -> float:
        neighbor_expectation = self.social_expectation()
        social_influence = float(settings.get("social_influence", 0.0)) if neighbor_expectation is not None else 0.0

        self.cb_weight = self.cb_trust
        self.social_weight = (1.0 - self.cb_trust) * social_influence
        self.personal_weight = (1.0 - self.cb_trust) * (1.0 - social_influence)
        self.neighbor_expectation = neighbor_expectation or 0.0
        self.social_gap = 0.0 if neighbor_expectation is None else self.previous_expectation - neighbor_expectation

        self.anchor = (
            self.personal_weight * perceived
            + self.cb_weight * float(cb_signal)
            + self.social_weight * self.neighbor_expectation
        )
        return self.anchor

    def step(self, group_inflation: dict[str, float], cb_signal: float, settings: dict[str, Any]) -> None:
        perceived = self.perceive(group_inflation, settings)
        anchor = self.update_anchor(perceived, cb_signal, settings)

        self.updated = self.rng.random() < self.update_frequency
        if self.updated:
            inertia = float(settings["inertia"])
            self.expectation = clamp(
                inertia * self.expectation + (1.0 - inertia) * anchor,
                settings["min_expectation"],
                settings["max_expectation"],
            )


def create_households(settings: dict[str, Any], rng: random.Random | None = None, shuffle: bool = True) -> list[Household]:
    rng = rng or random.Random(settings.get("random_seed"))
    households: list[Household] = []
    next_id = 0

    for household_type, count in type_counts(settings).items():
        for _ in range(count):
            households.append(Household(next_id, household_type, settings, rng))
            next_id += 1

    if shuffle:
        rng.shuffle(households)
    return households


def type_counts(settings: dict[str, Any]) -> dict[str, int]:
    amount = int(settings["households_amount"])
    raw = {name: amount * float(params["share"]) for name, params in settings["household_types"].items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = amount - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def create_social_network(households: list[Household], settings: dict[str, Any], rng: random.Random | None = None) -> dict[str, float | int]:
    for household in households:
        household.neighbors = []

    if not settings.get("use_social_network", False):
        return network_stats(households)

    degree = int(settings.get("network_degree", 0))
    if degree <= 0 or len(households) <= 1:
        return network_stats(households)

    rng = rng or random.Random(settings.get("random_seed"))
    homophily = float(settings.get("network_homophily", 0.0))
    by_type = group_households(households)
    household_types = list(by_type)
    other_types_by_type = {
        household_type: [name for name in household_types if name != household_type]
        for household_type in household_types
    }
    max_degree = min(degree, len(households) - 1)

    for household in households:
        selected: set[int] = set()
        other_types = other_types_by_type[household.household_type]
        while len(selected) < max_degree:
            if rng.random() < homophily and len(by_type[household.household_type]) > 1:
                pool = by_type[household.household_type]
            else:
                pool = by_type[rng.choice(other_types)] if other_types else households

            candidate = draw_neighbor(pool, household.id, selected, rng) or draw_neighbor(households, household.id, selected, rng)
            if candidate is None:
                break
            household.neighbors.append(candidate)
            selected.add(candidate.id)

    return network_stats(households)


def draw_neighbor(pool: list[Household], self_id: int, selected: set[int], rng: random.Random) -> Household | None:
    if not pool:
        return None
    for _ in range(30):
        candidate = rng.choice(pool)
        if candidate.id != self_id and candidate.id not in selected:
            return candidate
    for candidate in pool:
        if candidate.id != self_id and candidate.id not in selected:
            return candidate
    return None


def group_households(households: list[Household]) -> dict[str, list[Household]]:
    groups: dict[str, list[Household]] = {}
    for household in households:
        groups.setdefault(household.household_type, []).append(household)
    return groups


def network_stats(households: list[Household]) -> dict[str, float | int]:
    if not households:
        return {"network_nodes": 0, "network_edges": 0, "network_average_degree": 0.0, "network_realized_homophily": 0.0}

    edge_count = sum(len(household.neighbors) for household in households)
    same_type_edges = sum(
        1
        for household in households
        for neighbor in household.neighbors
        if household.household_type == neighbor.household_type
    )
    return {
        "network_nodes": len(households),
        "network_edges": edge_count,
        "network_average_degree": edge_count / len(households),
        "network_realized_homophily": same_type_edges / edge_count if edge_count else 0.0,
    }
