import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelHealthTests(unittest.TestCase):
    def test_fpl_metadata_is_plain_utf8_json(self):
        path = PROJECT_ROOT / "data" / "production_artifacts" / "fpl_points_v3_candidate_metadata.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["feature_count"], 20)
        self.assertEqual(payload["status"], "candidate_only_not_production")

    def test_production_feature_contract(self):
        path = PROJECT_ROOT / "models" / "saved" / "production_features_v3.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["feature_count"], len(payload["features"]))
        self.assertEqual(payload["feature_count"], 32)


if __name__ == "__main__":
    unittest.main()
