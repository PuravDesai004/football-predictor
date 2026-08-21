import unittest

import pandas as pd

from src.model_validation import make_gameweek_split


class ModelValidationTests(unittest.TestCase):
    def test_multi_season_split_keeps_seasons_chronological(self):
        frame = pd.DataFrame(
            [
                {"season": "2021-22", "gameweek": 1, "fixture_id": 1},
                {"season": "2021-22", "gameweek": 2, "fixture_id": 2},
                {"season": "2022-23", "gameweek": 1, "fixture_id": 3},
                {"season": "2022-23", "gameweek": 2, "fixture_id": 4},
            ]
        )
        train, test = make_gameweek_split(frame, train_fraction=0.5)
        self.assertEqual(set(train["season"]), {"2021-22"})
        self.assertEqual(set(test["season"]), {"2022-23"})


if __name__ == "__main__":
    unittest.main()
