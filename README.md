# Genetic Algorithm for Sportday

A reproducible Python genetic algorithm that divides participants into **two balanced sports-day teams** using gender, department, age, height, weight and BMI. Team sizes differ by at most one person.

## Files

- `main.py` — genetic algorithm, CSV validation, Excel/CSV/chart export
- `HR.csv` — example input (358 people)
- `config.yml` — settings and fairness weights
- `requirements.txt` — Python packages
- `.gitignore` — excludes generated reports and caches

## Setup / run

Requires Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python main.py
```

Files are saved in `ga_output/`: `team_assignment.csv`, `fairness_summary.csv`, `Sports_Day_Teams.xlsx`, and `fairness_charts.png`.

```powershell
python main.py --csv HR.csv --output custom_results
python main.py --config config.yml
```

Settings in `config.yml`: `seed` for reproducible runs; `population_size`, `generations`, `elite_size`, `tournament_size`, and `mutation_rate` control search; `dept_weight`, `gender_weight` and `physical_weight` control relative fairness; `patience=60` stops after 60 non-improving generations (`0` disables early stopping). Paths are resolved relative to the configuration file.

Input CSV columns: `gender,department,weight,height,age`. Numeric values must be finite and positive. Extra columns are preserved. Each row is one participant.

## How it works

Balanced binary assignments are evolved with tournament selection, elitism, uniform crossover, balance repair and swap mutation. Fitness is evaluated for each population using **one NumPy matrix multiplication** over combined one-hot and normalized numeric features, avoiding repeated per-individual DataFrame computations. The lowest fitness is preferred; fitness values are meaningful primarily for comparisons using the same weights and dataset.

The GA is heuristic, not an exact fairness guarantee for each demographic category. The data is illustrative; the tool does not measure sports ability or fitness. Outputs are regenerated each run and are intentionally ignored by Git.

Source inspiration: [Zuplexes/Genetic-Algorithm-for-Sportday](https://github.com/Zuplexes/Genetic-Algorithm-for-Sportday).
