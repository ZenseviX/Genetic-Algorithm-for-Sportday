# Genetic Algorithm for Sportday

An optimized, reproducible **two-team Sports Day allocator** using a genetic algorithm (GA). This project is based on the original repository [Zuplexes/Genetic-Algorithm-for-Sportday](https://github.com/Zuplexes/Genetic-Algorithm-for-Sportday); existing input, reference exports, charts and walkthrough are retained.

## Requirements

Python **3.10 or newer**.

```bash
python -m pip install -r requirements.txt
python main.py
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python main.py
```

## Commands

```bash
python main.py --config config.yml
python main.py --csv HR.csv --output my_results
python -m unittest discover -s tests -v
```

Paths in `config.yml` are relative to the config file directory. CLI paths `--csv` and `--output` also resolve against the config file directory. `config.yml` controls the seed, number of generations, population size, tournament size, elitism and fairness weights.

## Input

`HR.csv` must contain these columns (extra columns are preserved):

| Column | Meaning |
|---|---|
| `gender` | Gender category |
| `department` | Department category |
| `weight` | Weight in kg (> 0) |
| `height` | Height in cm (> 0) |
| `age` | Age in years (> 0) |

Every row is one participant. A derived `bmi` column is computed. At least two participants are required.

## Outputs

The script writes these files to the configured output directory, by default `ga_output/`:

- `team_assignment.csv` — all input rows plus BMI and the assigned team.
- `fairness_summary.csv` — team counts, physical averages and gender counts.
- `Sports_Day_Teams.xlsx` — participants, summary and department breakdown sheets.
- `fairness_charts.png` — best-score convergence and fairness visualizations.

**Heads up:** Running the default command overwrites generated files under `ga_output/`, including the legacy CSV and chart already present in the repository. To preserve historical exports, use `--output new_results`.

## Algorithm

- Balanced chromosomes: teams differ by **at most one person**, including after crossover and mutation.
- Vectorized fitness evaluation with NumPy matrix operations for faster scoring of entire populations.
- Seeded random generator for reproducibility; tournament selection, elite preservation, uniform crossover and swap mutation.
- Fitness balances department and gender distributions and standardized physical attributes (weight, height, age, BMI). Physical attributes are standardized to avoid their measurement units dominating the objective.
- The optimization is heuristic: it does **not** guarantee an optimal distribution, and group counts are not required to be equal for each department or gender. Only overall team sizes are a hard constraint.

## Repository preservation / migration

The original `HR.csv`, `Sports_Day_Teams.xlsx`, `walkthrough.html`, `walkthrough-model.json`, `.idea/`, `claude.md`, plus the existing `ga_output*` files are included unchanged in this distribution. The original `uv.lock` is archived at `legacy/uv.lock.original` because it no longer matches the updated dependency metadata; `uv sync` can generate a new `uv.lock` when internet access is available. The legacy walkthrough and exports describe earlier runs; they are **not regenerated or automatically synchronized** with the current GA implementation. The new README documents the current CLI and source.

To publish as a **new GitHub repository**, extract the ZIP, open its folder, then run:

```bash
git init
git add .
git commit -m "Initial import: optimized sportday GA"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPO.git
git push -u origin main
```

Create the empty repository on GitHub first. Replace the URL above with your own repository URL. This ZIP is a source snapshot, not a Git clone, so it does not contain `.git/` history.
