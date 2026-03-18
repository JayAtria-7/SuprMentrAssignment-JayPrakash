import tempfile
import unittest
from pathlib import Path

import numpy as np

from spam_classifier import TrainConfig, fit_model, load_artifact, load_dataset, predict_texts, save_artifact


class SpamClassifierTests(unittest.TestCase):
    def test_train_save_load_predict(self) -> None:
        df = load_dataset(Path("sample_data/sms_sample.csv"))
        self.assertGreaterEqual(len(df), 10)
        self.assertIn("text", df.columns)
        self.assertIn("y", df.columns)
        self.assertEqual(set(np.unique(df["y"])), {0, 1})

        artifact, report = fit_model(df, TrainConfig(test_size=0.25, random_state=123, balanced=True))
        self.assertIn("pipeline", artifact)
        self.assertIn("classification_report", report)

        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "model.joblib"
            save_artifact(artifact, model_path)
            loaded = load_artifact(model_path)

            out = predict_texts(
                loaded,
                [
                    "Please review the report and send feedback.",
                    "WINNER!!! Click https://free-prize.example to claim your reward now!!!",
                ],
                threshold=0.5,
            )
            self.assertEqual(list(out.columns), ["text", "spam_probability", "prediction"])
            self.assertEqual(len(out), 2)
            self.assertTrue(out["spam_probability"].between(0.0, 1.0).all())
            self.assertTrue(set(out["prediction"]).issubset({"ham", "spam"}))


if __name__ == "__main__":
    unittest.main()

