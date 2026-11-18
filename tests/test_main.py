import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main import DEFAULTS, balanced_chromosome, load_config, load_people, optimize, rebalance, write_results


class GeneticAlgorithmTests(unittest.TestCase):
    def test_balancing_keeps_exact_team_sizes(self):
        rng = np.random.default_rng(1)
        for n in (2, 3, 10, 11):
            child = np.ones(n, dtype=np.int8)
            result = rebalance(child, n - n // 2, rng)
            self.assertEqual(int(result.sum()), n - n // 2)
            self.assertEqual(len(result), n)

    def test_seeded_optimization_is_reproducible(self):
        df = pd.DataFrame({"gender": ["Male", "Female"] * 5,
                           "department": ["A"] * 5 + ["B"] * 5,
                           "weight": range(50, 60), "height": range(150, 160),
                           "age": range(20, 30)})
        df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
        cfg = {**DEFAULTS, "population_size": 12, "generations": 10, "elite_size": 2}
        a, ha = optimize(df, cfg)
        b, hb = optimize(df, cfg)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ha, hb)
        self.assertLessEqual(abs(int(a.sum()) - int((a == 0).sum())), 1)
        self.assertTrue(all(ha[i] >= ha[i + 1] for i in range(len(ha) - 1)))

    def test_invalid_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "HR.csv"
            path.write_text("gender,department,weight,height,age\nMale,A,70,0,23\nFemale,B,55,160,21\n")
            with self.assertRaises(ValueError):
                load_people(path)

    def test_complete_export_and_team_balance_for_odd_count(self):
        df = pd.DataFrame({
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "department": ["A", "B", "B", "A", "B"],
            "weight": [70, 62, 65, 54, 75],
            "height": [172, 162, 170, 159, 181],
            "age": [20, 21, 22, 23, 24],
        })
        df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
        cfg = {**DEFAULTS, "population_size": 8, "generations": 6, "elite_size": 2}
        chrom, history = optimize(df, cfg)
        self.assertEqual(sorted([int(sum(chrom == 0)), int(sum(chrom == 1))]), [2, 3])
        df["Team"] = np.where(chrom == 0, "Team A", "Team B")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_results(df, history, output)
            for name in ("team_assignment.csv", "fairness_summary.csv", "Sports_Day_Teams.xlsx", "fairness_charts.png"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(len(pd.read_csv(output / "team_assignment.csv")), 5)
            with pd.ExcelFile(output / "Sports_Day_Teams.xlsx") as book:
                self.assertEqual(book.sheet_names, ["All Employees", "Summary", "Departments"])

    def test_invalid_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yml"
            path.write_text("population_size: 3\nelite_size: 3\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
