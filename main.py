"""Reproducible two-team sports-day allocation with a balanced genetic algorithm."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED = ("gender", "department", "weight", "height", "age")
DEFAULTS = {
    "csv_path": "HR.csv", "output_dir": "ga_output",
    "seed": 42, "population_size": 200, "generations": 300,
    "elite_size": 5, "tournament_size": 3, "mutation_rate": 0.3,
    "dept_weight": 5.0, "gender_weight": 5.0, "physical_weight": 2.0,
}


def load_config(path: Path) -> dict:
    config = DEFAULTS.copy()
    if path.exists():
        settings = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(settings, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(settings) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(settings)
    for name in ("population_size", "generations", "elite_size", "tournament_size"):
        if type(config[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if config["population_size"] < 2 or config["generations"] < 1:
        raise ValueError("population_size must be >= 2; generations must be >= 1")
    if not 1 <= config["elite_size"] < config["population_size"]:
        raise ValueError("elite_size must be between 1 and population_size - 1")
    if not 1 <= config["tournament_size"] <= config["population_size"]:
        raise ValueError("tournament_size must be within population_size")
    for name in ("mutation_rate", "dept_weight", "gender_weight", "physical_weight"):
        if not isinstance(config[name], (int, float)) or not np.isfinite(config[name]):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= config["mutation_rate"] <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if any(config[name] < 0 for name in ("dept_weight", "gender_weight", "physical_weight")):
        raise ValueError("Fitness weights cannot be negative")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return config


def load_people(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) < 2:
        raise ValueError("At least two people are required")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required columns cannot contain missing values")
    df = df.copy()
    for column in ("weight", "height", "age"):
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all() or (df[column] <= 0).any():
            raise ValueError(f"{column} must contain positive finite values")
    for column in ("gender", "department"):
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"{column} contains empty values")
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df


def balanced_chromosome(n: int, rng: np.random.Generator) -> np.ndarray:
    chromosome = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int8)
    rng.shuffle(chromosome)
    return chromosome


def rebalance(child: np.ndarray, target_ones: int, rng: np.random.Generator) -> np.ndarray:
    child = child.copy()
    delta = int(child.sum()) - target_ones
    if delta:
        source = 1 if delta > 0 else 0
        indices = rng.choice(np.flatnonzero(child == source), size=abs(delta), replace=False)
        child[indices] = 1 - source
    return child


def optimize(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(cfg["seed"])
    n = len(df)
    target_ones = n - n // 2
    departments = pd.get_dummies(df["department"]).to_numpy(dtype=float)
    genders = pd.get_dummies(df["gender"]).to_numpy(dtype=float)
    physical = df[["weight", "height", "age", "bmi"]].to_numpy(dtype=float)
    # Normalize physical features so height (cm) does not dominate age/BMI.
    spread = physical.std(axis=0)
    spread[spread == 0] = 1.0
    physical = (physical - physical.mean(axis=0)) / spread

    def score_population(chromosomes: np.ndarray) -> np.ndarray:
        """Compute all candidate scores via NumPy matrix products (no row-wise pandas)."""
        # Every chromosome has a fixed count of zeroes and ones.
        # Matrix multiplication counts Team B members in each category.
        members_b = chromosomes.astype(np.float64)
        dept_b = members_b @ departments
        gender_b = members_b @ genders
        dept_gap = np.abs(departments.sum(axis=0) - 2 * dept_b).sum(axis=1)
        gender_gap = np.abs(genders.sum(axis=0) - 2 * gender_b).sum(axis=1)
        size_b = target_ones
        size_a = n - size_b
        sum_b = members_b @ physical
        sum_a = physical.sum(axis=0) - sum_b
        physical_gap = np.abs(sum_a / size_a - sum_b / size_b).sum(axis=1)
        return (cfg["dept_weight"] * dept_gap
                + cfg["gender_weight"] * gender_gap
                + cfg["physical_weight"] * physical_gap)

    population = [balanced_chromosome(n, rng) for _ in range(cfg["population_size"])]
    best_score = float("inf")
    best = population[0].copy()
    history: list[float] = []
    for _ in range(cfg["generations"]):
        scores = score_population(np.asarray(population, dtype=np.int8))
        indices = np.argsort(scores)
        if scores[indices[0]] < best_score:
            best_score = float(scores[indices[0]])
            best = population[int(indices[0])].copy()
        history.append(best_score)
        next_generation = [population[int(i)].copy() for i in indices[: cfg["elite_size"]]]

        def select() -> np.ndarray:
            contenders = rng.choice(len(population), size=cfg["tournament_size"], replace=False)
            return population[int(contenders[np.argmin(scores[contenders])])]

        while len(next_generation) < cfg["population_size"]:
            parent_a, parent_b = select(), select()
            child = np.where(rng.random(n) < 0.5, parent_a, parent_b).astype(np.int8)
            child = rebalance(child, target_ones, rng)
            if rng.random() < cfg["mutation_rate"]:
                i = int(rng.choice(np.flatnonzero(child == 0)))
                j = int(rng.choice(np.flatnonzero(child == 1)))
                child[i], child[j] = child[j], child[i]
            next_generation.append(child)
        population = next_generation
    return best, history


def write_results(df: pd.DataFrame, history: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "team_assignment.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("Team", sort=True).agg(
        Total=("Team", "size"), Age_Avg=("age", "mean"),
        Weight_Avg=("weight", "mean"), Height_Avg=("height", "mean"),
        BMI_Avg=("bmi", "mean"),
    ).round(2)
    gender_counts = pd.crosstab(df["Team"], df["gender"])
    summary = summary.join(gender_counts)
    summary.to_csv(output_dir / "fairness_summary.csv", encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "Sports_Day_Teams.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Employees", index=False)
        summary.to_excel(writer, sheet_name="Summary")
        pd.crosstab(df["department"], df["Team"]).to_excel(writer, sheet_name="Departments")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(np.arange(1, len(history) + 1), history)
    axes[0, 0].set(title="Best Fitness (lower is better)", xlabel="Generation", ylabel="Penalty")
    pd.crosstab(df["department"], df["Team"]).plot.bar(ax=axes[0, 1], title="Department Distribution")
    pd.crosstab(df["gender"], df["Team"]).plot.bar(ax=axes[1, 0], title="Gender Distribution")
    df.groupby("Team")["bmi"].mean().plot.bar(ax=axes[1, 1], title="Average BMI")
    axes[1, 1].set_ylabel("BMI")
    fig.tight_layout()
    fig.savefig(output_dir / "fairness_charts.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocate two balanced sports-day teams using a GA")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument("--csv", type=Path, help="Override CSV input path")
    parser.add_argument("--output", type=Path, help="Override output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    csv_path = args.csv or Path(cfg["csv_path"])
    if not csv_path.is_absolute():
        csv_path = args.config.resolve().parent / csv_path
    output_dir = args.output or Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = args.config.resolve().parent / output_dir
    df = load_people(csv_path)
    best, history = optimize(df, cfg)
    df["Team"] = np.where(best == 0, "Team A", "Team B")
    write_results(df, history, output_dir)
    counts = df["Team"].value_counts()
    print(f"Done: {len(df)} people, Team A={counts['Team A']}, Team B={counts['Team B']}")
    print(f"Best fitness: {history[-1]:.4f} | Results: {output_dir}")


if __name__ == "__main__":
    main()
