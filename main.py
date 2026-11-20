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
    "patience": 60,
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
    for name in ("population_size", "generations", "elite_size", "tournament_size", "patience"):
        if type(config[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if config["population_size"] < 2 or config["generations"] < 1 or config["patience"] < 0:
        raise ValueError("population_size >= 2, generations >= 1, patience >= 0 required")
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
    departments = pd.get_dummies(df["department"]).to_numpy(dtype=np.float64)
    genders = pd.get_dummies(df["gender"]).to_numpy(dtype=np.float64)
    physical = df[["weight", "height", "age", "bmi"]].to_numpy(dtype=np.float64)
    # All four physical measures contribute comparably regardless of their units.
    scale = physical.std(axis=0)
    scale[scale == 0] = 1.0
    physical = (physical - physical.mean(axis=0)) / scale

    # Compute category counts and physical sums in one matrix multiplication.
    nd, ng = departments.shape[1], genders.shape[1]
    features = np.concatenate((departments, genders, physical), axis=1)
    totals = features.sum(axis=0)
    count_a, count_b = n // 2, target_ones
    physical_factor = float(cfg["physical_weight"])

    def score_population(chromosomes: np.ndarray) -> np.ndarray:
        # Chromosomes always contain exactly count_b ones.
        sums_b = chromosomes @ features
        category_gaps = np.abs(totals[:nd + ng] - 2.0 * sums_b[:, :nd + ng])
        physical_b = sums_b[:, nd + ng:]
        physical_gap = np.abs((totals[nd + ng:] - physical_b) / count_a
                              - physical_b / count_b).sum(axis=1)
        return (float(cfg["dept_weight"]) * category_gaps[:, :nd].sum(axis=1)
                + float(cfg["gender_weight"]) * category_gaps[:, nd:].sum(axis=1)
                + physical_factor * physical_gap)

    population = [balanced_chromosome(n, rng) for _ in range(cfg["population_size"])]
    best_score = float("inf")
    best = population[0].copy()
    history: list[float] = []
    stagnant = 0
    for _ in range(cfg["generations"]):
        scores = score_population(np.asarray(population, dtype=np.int8))
        indices = np.argsort(scores)
        if scores[indices[0]] < best_score:
            best_score = float(scores[indices[0]])
            best = population[int(indices[0])].copy()
            stagnant = 0
        else:
            stagnant += 1
        history.append(best_score)
        # Optional early stopping: patience=0 runs all generations.
        if cfg["patience"] and stagnant >= cfg["patience"]:
            break
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
    team_order = ["Team A", "Team B"]
    comparison_columns = ("age", "weight", "height")
    team_averages = df.groupby("Team", sort=True)[list(comparison_columns)].mean()
    comparison_rows = []
    for column in comparison_columns:
        team_a = float(team_averages.loc["Team A", column])
        team_b = float(team_averages.loc["Team B", column])
        difference = abs(team_a - team_b)
        comparison_rows.append({
            "Metric": column,
            "Team_A_Avg": team_a,
            "Team_B_Avg": team_b,
            "Absolute_Difference": difference,
            "Difference_Percent_of_Overall_Avg": (
                difference / float(df[column].mean()) * 100
            ),
        })
    initial_fitness = history[0]
    final_fitness = history[-1]
    fitness_improvement = initial_fitness - final_fitness
    convergence = pd.DataFrame([{
        "Generations_Run": len(history),
        "Initial_Fitness": initial_fitness,
        "Final_Fitness": final_fitness,
        "Fitness_Improvement": fitness_improvement,
        "Improvement_Percent": fitness_improvement / initial_fitness * 100,
    }])
    convergence.to_csv(output_dir / "convergence_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "team_comparison.csv", index=False, encoding="utf-8-sig"
    )
    with pd.ExcelWriter(output_dir / "Sports_Day_Teams.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Employees", index=False)
        summary.to_excel(writer, sheet_name="Summary")
        pd.crosstab(df["department"], df["Team"]).to_excel(writer, sheet_name="Departments")
        pd.DataFrame(comparison_rows).to_excel(writer, sheet_name="Team Comparison", index=False)
        convergence.to_excel(writer, sheet_name="Convergence", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gender_order = sorted(df["gender"].unique())
    gender_colors = {
        gender: plt.get_cmap("Set2")(index % 8)
        for index, gender in enumerate(gender_order)
    }
    x = np.arange(len(team_order))
    bar_width = 0.8 / len(gender_order)

    def grouped_bars(axis, values: pd.DataFrame, title: str, ylabel: str) -> None:
        for index, gender in enumerate(gender_order):
            heights = values[gender].to_numpy(dtype=float)
            bars = axis.bar(
                x + (index - (len(gender_order) - 1) / 2) * bar_width,
                heights,
                bar_width,
                label=gender,
                color=gender_colors[gender],
            )
            for bar, value in zip(bars, heights):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.1f}" if not float(value).is_integer() else f"{value:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
        axis.set(
            title=title,
            ylabel=ylabel,
           xticks=x,
            xticklabels=team_order,
        )
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(title="Gender")

    gender_counts = (
        pd.crosstab(df["Team"], df["gender"])
        .reindex(index=team_order, columns=gender_order, fill_value=0)
    )
    grouped_bars(axes[0, 0], gender_counts, "Male/Female Count by Team", "People")

    for axis, column, title, ylabel in (
        (axes[0, 1], "age", "Average Age by Team and Gender", "Age (years)"),
        (axes[0, 2], "weight", "Average Weight by Team and Gender", "Weight (kg)"),
        (axes[1, 0], "height", "Average Height by Team and Gender", "Height (cm)"),
    ):
        averages = (
            df.pivot_table(
                index="Team",
                columns="gender",
                values=column,
                aggfunc="mean",
            )
            .reindex(index=team_order, columns=gender_order)
        )
        grouped_bars(axis, averages, title, ylabel)

    axes[1, 1].plot(np.arange(1, len(history) + 1), history, color="#4472C4", linewidth=2)
    axes[1, 1].set(
        title="Fitness Improvement (lower is better)",
        xlabel="Generation",
        ylabel="Penalty",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].set_axisbelow(True)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        (
            "Convergence Summary\n\n"
            f"Generations run: {len(history)}\n"
            f"Initial fitness: {initial_fitness:.4f}\n"
            f"Final fitness: {final_fitness:.4f}\n"
            f"Improvement: {fitness_improvement:.4f}\n"
            f"Improvement: {fitness_improvement / initial_fitness * 100:.2f}%"
        ),
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F2F2F2", "edgecolor": "#BFBFBF"},
    )
    fig.suptitle("Sports Day Team Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fairness_charts.png", dpi=160)
    plt.close(fig)

    generations = np.arange(1, len(history) + 1)
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        generations,
        history,
        color="#4472C4",
        linewidth=2.5,
        marker="o",
        markersize=3,
        markevery=max(1, len(history) // 20),
    )
    axis.scatter(
        [generations[0], generations[-1]],
        [history[0], history[-1]],
        color=["#ED7D31", "#70AD47"],
        zorder=3,
        s=70,
    )
    axis.annotate(
        f"Start: {history[0]:.2f}",
        (generations[0], history[0]),
        xytext=(10, 10),
        textcoords="offset points",
        color="#C55A11",
    )
    axis.annotate(
        f"Final: {history[-1]:.2f}",
        (generations[-1], history[-1]),
        xytext=(-70, 15),
        textcoords="offset points",
        color="#548235",
    )
    axis.set(
        title="Genetic Algorithm Convergence",
        xlabel="Generation",
        ylabel="Fitness (lower is better)",
        xlim=(1, len(history) + 5),
        ylim=(0, initial_fitness * 1.08),
    )
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.text(
        0.98,
        0.95,
        (
            f"Generations: {len(history)}\n"
            f"Start: {initial_fitness:.2f}\n"
            f"Final: {final_fitness:.2f}\n"
            f"Improvement: {fitness_improvement / initial_fitness * 100:.2f}%"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_chart.png", dpi=180)
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
