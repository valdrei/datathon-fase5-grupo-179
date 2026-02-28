"""Testes adicionais para app.main (startup/shutdown)."""

from pathlib import Path
import sys
import asyncio
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.main as app_main


class MockPredictionLogger:
    def get_prediction_statistics(self):
        return {"total_predictions": 1}


def test_startup_load_models_with_files(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    model_dir = app_dir / "model"
    model_dir.mkdir(parents=True)

    joblib.dump({"model": True}, model_dir / "model_random_forest_latest.pkl")
    joblib.dump({"pre": True}, model_dir / "preprocessor_latest.pkl")
    joblib.dump({"eng": True}, model_dir / "feature_engineer_latest.pkl")

    monkeypatch.setattr(app_main, "__file__", str(app_dir / "main.py"))

    asyncio.run(app_main.load_models())

    assert app_main.model is not None
    assert app_main.preprocessor is not None
    assert app_main.feature_engineer is not None
    assert app_main.prediction_logger is not None
    assert app_main.drift_detector is not None


def test_startup_without_model_files_and_shutdown(tmp_path, monkeypatch):
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)

    monkeypatch.setattr(app_main, "__file__", str(app_dir / "main.py"))

    asyncio.run(app_main.load_models())
    assert app_main.prediction_logger is not None
    assert app_main.drift_detector is not None

    app_main.prediction_logger = MockPredictionLogger()
    asyncio.run(app_main.shutdown_event())
