"""Genetic Algorithm สำหรับแบ่งคนออกเป็นสองทีมกีฬาอย่างสมดุล"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = ("gender", "department", "weight", "height", "age")
TEAM_NAMES = ("Team A", "Team B")
COMPARISON_COLUMNS = ("age", "weight", "height")

DEFAULT_CONFIG = {
    "csv_path": "HR.csv",
    "output_dir": "ga_output",
    "seed": 42,
    "population_size": 200,
    "generations": 300,
    "elite_size": 5,
    "tournament_size": 3,
    "mutation_rate": 0.3,
    "dept_weight": 5.0,
    "gender_weight": 5.0,
    "physical_weight": 2.0,
    "patience": 60,
}


def load_config(config_path: Path) -> dict:
    """อ่านและตรวจสอบการตั้งค่าจากไฟล์ YAML"""
    # เริ่มจากค่าตั้งต้น แล้วแทนค่าด้วยข้อมูลที่ระบุในไฟล์ YAML
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(settings, dict):
            raise ValueError("Configuration must be a YAML mapping")

        unknown_keys = set(settings) - set(config)
        if unknown_keys:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown_keys)}")
        config.update(settings)

    integer_keys = (
        "population_size",
        "generations",
        "elite_size",
        "tournament_size",
        "patience",
    )
    for key in integer_keys:
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")

    if config["population_size"] < 2:
        raise ValueError("population_size must be at least 2")
    if config["generations"] < 1:
        raise ValueError("generations must be at least 1")
    if config["patience"] < 0:
        raise ValueError("patience cannot be negative")
    if not 1 <= config["elite_size"] < config["population_size"]:
        raise ValueError("elite_size must be between 1 and population_size - 1")
    if not 1 <= config["tournament_size"] <= config["population_size"]:
        raise ValueError("tournament_size must be within population_size")

    numeric_keys = (
        "mutation_rate",
        "dept_weight",
        "gender_weight",
        "physical_weight",
    )
    for key in numeric_keys:
        value = config[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")

    if not 0 <= config["mutation_rate"] <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if any(config[key] < 0 for key in ("dept_weight", "gender_weight", "physical_weight")):
        raise ValueError("Fitness weights cannot be negative")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")

    return config


def load_people(csv_path: Path) -> pd.DataFrame:
    """โหลดข้อมูล CSV และตรวจสอบคอลัมน์และค่าที่จำเป็น"""
    # หยุดการทำงานถ้าไม่พบไฟล์ข้อมูลหรือไม่มีคอลัมน์ที่โปรแกรมต้องใช้
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    if len(df) < 2:
        raise ValueError("At least two people are required")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns cannot contain missing values")

    df = df.copy()

    # ค่าข้อมูลกายภาพต้องเป็นตัวเลขบวก
    for column in ("weight", "height", "age"):
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all() or (df[column] <= 0).any():
            raise ValueError(f"{column} must contain positive finite values")

    # ตัดช่องว่างหน้า-หลังและปฏิเสธค่าหมวดหมู่ที่ว่าง
    for column in ("gender", "department"):
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"{column} contains empty values")

    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df


def create_chromosome(person_count: int, rng: np.random.Generator) -> np.ndarray:
    """สร้างชุดการแบ่งทีมแบบสุ่ม โดยกำหนดจำนวนคนแต่ละทีมให้สมดุล"""
    # ค่า 0 หมายถึงทีม A และค่า 1 หมายถึงทีม B
    chromosome = np.array(
        [0] * (person_count // 2) + [1] * (person_count - person_count // 2),
        dtype=np.int8,
    )
    rng.shuffle(chromosome)
    return chromosome


def rebalance_chromosome(
    chromosome: np.ndarray,
    target_team_b_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """ปรับจำนวนสมาชิกทีมหลังจากทำ Crossover"""
    chromosome = chromosome.copy()
    difference = int(chromosome.sum()) - target_team_b_count

    if difference == 0:
        return chromosome

    source_value = 1 if difference > 0 else 0
    source_positions = np.flatnonzero(chromosome == source_value)
    positions_to_change = rng.choice(
        source_positions,
        size=abs(difference),
        replace=False,
    )
    chromosome[positions_to_change] = 1 - source_value
    return chromosome


def build_fitness_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """แปลงข้อมูลแผนก เพศ และข้อมูลกายภาพเป็นค่าที่ใช้คำนวณ Fitness"""
    # แปลงข้อมูลหมวดหมู่ข้อความเป็นคอลัมน์ตัวเลข 0 และ 1
    departments = pd.get_dummies(df["department"]).to_numpy(dtype=np.float64)
    genders = pd.get_dummies(df["gender"]).to_numpy(dtype=np.float64)
    # จัดเก็บส่วนสูง น้ำหนัก อายุ และ BMI เป็นกลุ่มข้อมูลกายภาพ
    physical = df[["weight", "height", "age", "bmi"]].to_numpy(dtype=np.float64)

    # ทำ Normalize ข้อมูลกายภาพ เพื่อให้หน่วยและช่วงค่าของข้อมูลต่างกันน้อยลง
    scale = physical.std(axis=0)
    scale[scale == 0] = 1.0
    physical = (physical - physical.mean(axis=0)) / scale

    features = np.concatenate((departments, genders, physical), axis=1)
    totals = features.sum(axis=0)
    category_count = departments.shape[1] + genders.shape[1]
    return features, totals, category_count, physical.shape[1]


def run_genetic_algorithm(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, list[float]]:
    """ค้นหาการแบ่งทีมที่สมดุลที่สุดด้วย Genetic Algorithm"""
    # กำหนด Seed เพื่อให้ผลการสุ่มซ้ำได้เหมือนเดิม
    rng = np.random.default_rng(config["seed"])
    person_count = len(df)
    team_b_count = person_count - person_count // 2
    team_a_count = person_count // 2

    features, totals, category_count, _ = build_fitness_features(df)
    department_count = pd.get_dummies(df["department"]).shape[1]

    def calculate_fitness(chromosomes: np.ndarray) -> np.ndarray:
        # ค่า Fitness ยิ่งต่ำ แปลว่าการแบ่งทีมสมดุลยิ่งมาก
        team_b_features = chromosomes @ features
        # เปรียบเทียบสัดส่วนแผนกและเพศระหว่างทั้งสองทีม
        category_gaps = np.abs(
            totals[:category_count] - 2.0 * team_b_features[:, :category_count]
        )

        # เปรียบเทียบค่าเฉลี่ยข้อมูลกายภาพหลัง Normalize ระหว่างทีม
        team_b_physical = team_b_features[:, category_count:]
        team_a_physical = totals[category_count:] - team_b_physical
        physical_gaps = np.abs(
            team_a_physical / team_a_count - team_b_physical / team_b_count
        ).sum(axis=1)

        department_gap = category_gaps[:, :department_count].sum(axis=1)
        gender_gap = category_gaps[:, department_count:].sum(axis=1)
        return (
            float(config["dept_weight"]) * department_gap
            + float(config["gender_weight"]) * gender_gap
            + float(config["physical_weight"]) * physical_gaps
        )

    population = [
        create_chromosome(person_count, rng)
        for _ in range(config["population_size"])
    ]

    # เก็บ Chromosome ที่ดีที่สุดพร้อมค่า Fitness ตลอดทุก Generation
    best_chromosome = population[0].copy()
    best_fitness = float("inf")
    fitness_history: list[float] = []
    stagnant_generations = 0

    for _ in range(config["generations"]):
        # 1. ประเมิน Fitness ของ Chromosome ทุกตัวใน Population
        chromosomes = np.asarray(population, dtype=np.int8)
        fitness_values = calculate_fitness(chromosomes)
        ranked_positions = np.argsort(fitness_values)

        # 2. บันทึกผลลัพธ์ที่ดีที่สุดของ Generation นี้หรือรุ่นก่อนหน้า
        current_fitness = float(fitness_values[ranked_positions[0]])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_chromosome = population[int(ranked_positions[0])].copy()
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        fitness_history.append(best_fitness)

        # หยุดก่อนกำหนดเมื่อผลลัพธ์ไม่ดีขึ้นตามจำนวนที่ตั้งไว้ใน patience
        if config["patience"] and stagnant_generations >= config["patience"]:
            break

        # 3. เก็บชุดการแบ่งทีมที่ดีที่สุดไว้จำนวน Elite Size ตัว
        elite = [
            population[int(position)].copy()
            for position in ranked_positions[: config["elite_size"]]
        ]

        def tournament_selection() -> np.ndarray:
            # Tournament Selection เลือกคำตอบที่ดีที่สุดจากตัวเลือกสุ่มจำนวนเล็กน้อย
            candidates = rng.choice(
                len(population),
                size=config["tournament_size"],
                replace=False,
            )
            winner = candidates[np.argmin(fitness_values[candidates])]
            return population[int(winner)]

        next_population = elite
        while len(next_population) < config["population_size"]:
            # 4. เลือกพ่อแม่สองตัวเพื่อสร้างลูก
            parent_a = tournament_selection()
            parent_b = tournament_selection()

            # 5. Crossover: คัดลอกยีนทีม A/ทีม B แบบสุ่มจากพ่อแม่ทั้งสอง
            child = np.where(
                rng.random(person_count) < 0.5,
                parent_a,
                parent_b,
            ).astype(np.int8)
            child = rebalance_chromosome(child, team_b_count, rng)

            # 6. Mutation: สลับคนหนึ่งคนระหว่างสองทีม
            if rng.random() < config["mutation_rate"]:
                team_a_positions = np.flatnonzero(child == 0)
                team_b_positions = np.flatnonzero(child == 1)
                position_a = int(rng.choice(team_a_positions))
                position_b = int(rng.choice(team_b_positions))
                child[position_a], child[position_b] = (
                    child[position_b],
                    child[position_a],
                )

            next_population.append(child)

        population = next_population

    return best_chromosome, fitness_history


def calculate_team_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """เปรียบเทียบค่าเฉลี่ยอายุ น้ำหนัก และส่วนสูงระหว่างทีม"""
    # สร้างข้อมูลหนึ่งแถวสำหรับตัวชี้วัดกายภาพแต่ละรายการ
    team_averages = df.groupby("Team")[list(COMPARISON_COLUMNS)].mean()
    rows = []

    for column in COMPARISON_COLUMNS:
        team_a_average = float(team_averages.loc["Team A", column])
        team_b_average = float(team_averages.loc["Team B", column])
        difference = abs(team_a_average - team_b_average)
        overall_average = float(df[column].mean())

        rows.append({
            "Metric": column,
            "Team_A_Avg": team_a_average,
            "Team_B_Avg": team_b_average,
            "Absolute_Difference": difference,
            "Difference_Percent_of_Overall_Avg": (
                difference / overall_average * 100
            ),
        })

    return pd.DataFrame(rows)


def calculate_balance_accuracy(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """คำนวณคะแนนสมดุลแยกหมวดและคะแนนรวมเป็นร้อยละ"""
    # คะแนนจำนวนสมาชิก: ทีมที่มีจำนวนคนเท่ากันได้รับ 100%
    team_counts = df["Team"].value_counts().reindex(TEAM_NAMES)
    team_size_accuracy = 100 * (
        1 - abs(team_counts["Team A"] - team_counts["Team B"]) / len(df)
    )

    # คะแนนเพศ: เปรียบเทียบสัดส่วนของเพศแต่ละประเภทในทั้งสองทีม
    gender_accuracy_values = []
    for gender in sorted(df["gender"].astype(str).unique()):
        gender_df = df[df["gender"].astype(str) == gender]
        proportions = gender_df["Team"].value_counts(normalize=True).reindex(
            TEAM_NAMES,
            fill_value=0,
        )
        gender_accuracy_values.append(
            100 * (1 - abs(proportions["Team A"] - proportions["Team B"]))
        )
    gender_accuracy = float(np.mean(gender_accuracy_values))

    # คะแนนแผนก: เปรียบเทียบสัดส่วนของแผนกแต่ละแห่งในทั้งสองทีม
    department_accuracy_values = []
    for department in df["department"].astype(str).unique():
        department_df = df[df["department"].astype(str) == department]
        proportions = department_df["Team"].value_counts(normalize=True).reindex(
            TEAM_NAMES,
            fill_value=0,
        )
        department_accuracy_values.append(
            100 * (1 - abs(proportions["Team A"] - proportions["Team B"]))
        )
    department_accuracy = float(np.mean(department_accuracy_values))

    # คะแนนกายภาพ: เปรียบเทียบค่าเฉลี่ยอายุ น้ำหนัก และส่วนสูง
    team_averages = df.groupby("Team")[list(COMPARISON_COLUMNS)].mean()
    physical_accuracy_values = []
    for column in COMPARISON_COLUMNS:
        difference = abs(
            float(team_averages.loc["Team A", column])
            - float(team_averages.loc["Team B", column])
        )
        overall_average = float(df[column].mean())
        physical_accuracy_values.append(
            100 * max(0.0, 1 - difference / overall_average)
        )
    physical_accuracy = float(np.mean(physical_accuracy_values))

    # สรุปแบบถ่วงน้ำหนัก: จำนวนคน 15%, เพศ 25%, แผนก 25%, กายภาพ 35%
    overall_accuracy = (
        0.15 * team_size_accuracy
        + 0.25 * gender_accuracy
        + 0.25 * department_accuracy
        + 0.35 * physical_accuracy
    )
    overall_accuracy = float(np.clip(overall_accuracy, 0, 100))

    scores = pd.DataFrame({
        "Metric": [
            "Team Size",
            "Gender",
            "Department",
            "Age / Weight / Height",
            "Overall Balance Accuracy",
        ],
        "Accuracy_Percent": [
            team_size_accuracy,
            gender_accuracy,
            department_accuracy,
            physical_accuracy,
            overall_accuracy,
        ],
    })
    return scores, overall_accuracy


def setup_matplotlib():
    """ตั้งค่า Matplotlib สำหรับสร้างกราฟลงไฟล์โดยไม่ต้องเปิดหน้าต่าง"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_grouped_bars(axis, plt, values: pd.DataFrame, title: str, ylabel: str) -> None:
    """วาดกราฟแท่งเปรียบเทียบและแสดงค่าบนแท่งกราฟ"""
    team_order = list(TEAM_NAMES)
    gender_order = sorted(values.columns)
    gender_colors = {
        gender: plt.get_cmap("Set2")(index % 8)
        for index, gender in enumerate(gender_order)
    }

    x_positions = np.arange(len(team_order))
    bar_width = 0.8 / len(gender_order)

    for index, gender in enumerate(gender_order):
        heights = values[gender].to_numpy(dtype=float)
        bars = axis.bar(
            x_positions + (index - (len(gender_order) - 1) / 2) * bar_width,
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
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    axis.set(
        title=title,
        ylabel=ylabel,
        xticks=x_positions,
        xticklabels=team_order,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(title="Gender")


def write_charts(
    df: pd.DataFrame,
    history: list[float],
    balance_scores: pd.DataFrame,
    overall_accuracy: float,
    output_dir: Path,
) -> None:
    """สร้างกราฟเปรียบเทียบ การลู่เข้า และคะแนนความสมดุล"""
    # ใช้กราฟแบบไม่โต้ตอบ เพราะบันทึกเป็นไฟล์ภาพ
    plt = setup_matplotlib()
    team_order = list(TEAM_NAMES)
    gender_order = sorted(df["gender"].astype(str).unique())

    # กราฟชุดที่ 1: เปรียบเทียบทีมอย่างชัดเจน
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    gender_counts = pd.crosstab(df["Team"], df["gender"]).reindex(
        index=team_order,
        columns=gender_order,
        fill_value=0,
    )
    plot_grouped_bars(
        axes[0, 0],
        plt,
        gender_counts,
        "Male/Female Count by Team",
        "People",
    )

    chart_settings = (
        (axes[0, 1], "age", "Average Age by Team and Gender", "Age (years)"),
        (axes[0, 2], "weight", "Average Weight by Team and Gender", "Weight (kg)"),
        (axes[1, 0], "height", "Average Height by Team and Gender", "Height (cm)"),
    )
    for axis, column, title, ylabel in chart_settings:
        averages = df.pivot_table(
            index="Team",
            columns="gender",
            values=column,
            aggfunc="mean",
        ).reindex(index=team_order, columns=gender_order)
        plot_grouped_bars(axis, plt, averages, title, ylabel)

    generations = np.arange(1, len(history) + 1)
    axes[1, 1].plot(generations, history, color="#4472C4", linewidth=2)
    axes[1, 1].set(
        title="Fitness Improvement (lower is better)",
        xlabel="Generation",
        ylabel="Penalty",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].set_axisbelow(True)

    axes[1, 2].axis("off")
    initial_fitness = history[0]
    final_fitness = history[-1]
    improvement_percent = (initial_fitness - final_fitness) / initial_fitness * 100
    axes[1, 2].text(
        0.02,
        0.95,
        (
            "Convergence Summary\n\n"
            f"Generations run: {len(history)}\n"
            f"Initial fitness: {initial_fitness:.4f}\n"
            f"Final fitness: {final_fitness:.4f}\n"
            f"Improvement: {improvement_percent:.2f}%"
        ),
        va="top",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F2F2F2",
            "edgecolor": "#BFBFBF",
        },
    )
    fig.suptitle("Sports Day Team Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "fairness_charts.png", dpi=160)
    plt.close(fig)

    # กราฟชุดที่ 2: กราฟการลู่เข้าของ Genetic Algorithm
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
            f"Improvement: {improvement_percent:.2f}%"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_chart.png", dpi=180)
    plt.close(fig)

    # กราฟชุดที่ 3: คะแนนสมดุลรวมและแยกตามหมวด
    category_scores = balance_scores.iloc[:-1]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1, 2.2]},
    )

    axes[0].pie(
        [overall_accuracy, 100 - overall_accuracy],
        colors=["#4472C4", "#E7E6E6"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white"},
    )
    axes[0].text(
        0,
        0.03,
        f"{overall_accuracy:.2f}%",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )
    axes[0].text(
        0,
        -0.18,
        "Overall Balance\nAccuracy",
        ha="center",
        va="center",
        fontsize=11,
    )
    axes[0].set_title("Total Team Balance", fontsize=13, fontweight="bold")

    bars = axes[1].barh(
        category_scores["Metric"],
        category_scores["Accuracy_Percent"],
        color=["#5B9BD5", "#70AD47", "#ED7D31", "#A5A5A5"],
    )
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Accuracy / Balance (%)")
    axes[1].set_title("Balance by Category", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_axisbelow(True)

    for bar, value in zip(bars, category_scores["Accuracy_Percent"]):
        axes[1].text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=10,
        )

    fig.suptitle("Team Division Balance Accuracy", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "balance_accuracy_chart.png", dpi=180)
    plt.close(fig)


def write_reports(
    df: pd.DataFrame,
    history: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """สร้างรายงาน CSV และ Excel และแสดงคะแนนความสมดุล"""
    # สร้างโฟลเดอร์ผลลัพธ์หากยังไม่มี
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "team_assignment.csv", index=False, encoding="utf-8-sig")

    summary = df.groupby("Team", sort=True).agg(
        Total=("Team", "size"),
        Age_Avg=("age", "mean"),
        Weight_Avg=("weight", "mean"),
        Height_Avg=("height", "mean"),
        BMI_Avg=("bmi", "mean"),
    ).round(2)
    summary = summary.join(pd.crosstab(df["Team"], df["gender"]))
    summary.to_csv(output_dir / "fairness_summary.csv", encoding="utf-8-sig")

    team_comparison = calculate_team_comparison(df)
    team_comparison.to_csv(
        output_dir / "team_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    initial_fitness = history[0]
    final_fitness = history[-1]
    convergence = pd.DataFrame([{
        "Generations_Run": len(history),
        "Initial_Fitness": initial_fitness,
        "Final_Fitness": final_fitness,
        "Fitness_Improvement": initial_fitness - final_fitness,
        "Improvement_Percent": (
            (initial_fitness - final_fitness) / initial_fitness * 100
        ),
    }])
    convergence.to_csv(
        output_dir / "convergence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    balance_scores, overall_accuracy = calculate_balance_accuracy(df)
    balance_scores.to_csv(
        output_dir / "balance_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        output_dir / "Sports_Day_Teams.xlsx",
        engine="openpyxl",
    ) as writer:
        df.to_excel(writer, sheet_name="All Employees", index=False)
        summary.to_excel(writer, sheet_name="Summary")
        pd.crosstab(df["department"], df["Team"]).to_excel(
            writer,
            sheet_name="Departments",
        )
        team_comparison.to_excel(writer, sheet_name="Team Comparison", index=False)
        convergence.to_excel(writer, sheet_name="Convergence", index=False)
        balance_scores.to_excel(writer, sheet_name="Balance Accuracy", index=False)

    print(f"Balance accuracy: {overall_accuracy:.2f}%")
    return balance_scores, overall_accuracy


def parse_arguments() -> argparse.Namespace:
    """อ่านตัวเลือกที่รับจากบรรทัดคำสั่ง"""
    parser = argparse.ArgumentParser(
        description="Allocate two balanced sports-day teams using a GA"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yml"),
    )
    parser.add_argument("--csv", type=Path, help="Override CSV input path")
    parser.add_argument("--output", type=Path, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)

    csv_path = args.csv or Path(config["csv_path"])
    if not csv_path.is_absolute():
        csv_path = args.config.resolve().parent / csv_path

    output_dir = args.output or Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = args.config.resolve().parent / output_dir

    df = load_people(csv_path)
    best_chromosome, history = run_genetic_algorithm(df, config)
    df["Team"] = np.where(best_chromosome == 0, "Team A", "Team B")

    balance_scores, overall_accuracy = write_reports(df, history, output_dir)
    write_charts(df, history, balance_scores, overall_accuracy, output_dir)

    team_counts = df["Team"].value_counts()
    print(
        f"Done: {len(df)} people, "
        f"Team A={team_counts['Team A']}, "
        f"Team B={team_counts['Team B']}"
    )
    print(
        f"Best fitness: {history[-1]:.4f} "
        f"| Results: {output_dir}"
    )


if __name__ == "__main__":
    main()
