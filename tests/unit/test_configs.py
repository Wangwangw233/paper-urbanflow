"""Configuration files are parseable and portable."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Configs(unittest.TestCase):
    def test_yaml_loads(self):
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
            with self.subTest(path=path.name):
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(doc, dict)

    def test_archived_json_config_loads(self):
        path = ROOT / "configs" / "experiments" / "chengdu_nonlinear.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["window"]["n_phase"], 18)
        self.assertEqual(doc["window"]["test"], [16302, 16303, 16304])

    def test_no_workstation_paths(self):
        for path in sorted((ROOT / "configs").rglob("*")):
            if path.suffix not in {".yaml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", path.name)


if __name__ == "__main__":
    unittest.main()
