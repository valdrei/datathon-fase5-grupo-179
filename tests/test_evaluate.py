"""Testes adicionais para src.evaluate."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import ModelEvaluator, evaluate_model


class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))


def test_calculate_metrics_and_print(capsys):
    evaluator = ModelEvaluator()
    y_true = pd.Series([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 1.5, 3.5])

    metrics = evaluator.calculate_metrics(y_true, y_pred)
    assert "mse" in metrics
    assert "accuracy_within_tolerance" in metrics

    evaluator.print_metrics()
    output = capsys.readouterr().out
    assert "MÉTRICAS DE AVALIAÇÃO DO MODELO" in output


def test_confidence_message_levels():
    evaluator = ModelEvaluator()

    evaluator.metrics = {"r2": 0.8, "mae": 0.3, "accuracy_within_tolerance": 90}
    assert "ALTA CONFIANÇA" in evaluator.get_model_confidence_message()

    evaluator.metrics = {"r2": 0.55, "mae": 0.55, "accuracy_within_tolerance": 55}
    assert "MODERADA" in evaluator.get_model_confidence_message()

    evaluator.metrics = {"r2": 0.2, "mae": 1.0, "accuracy_within_tolerance": 20}
    assert "REQUER MELHORIAS" in evaluator.get_model_confidence_message()


def test_analyze_predictions_and_wrapper(capsys):
    evaluator = ModelEvaluator()
    y_true = pd.Series([-2.5, -1.5, 0.0, 1.0, 2.0])
    y_pred = np.array([-2.4, -1.0, 0.2, 0.9, 2.2])

    analysis = evaluator.analyze_predictions_by_class(y_true, y_pred)
    assert "count" in analysis.columns

    X_test = pd.DataFrame({"f1": [1, 2, 3]})
    y_test = pd.Series([0.0, 0.0, 0.0])
    metrics, message = evaluate_model(DummyModel(), X_test, y_test)
    assert "mse" in metrics
    assert isinstance(message, str)
    assert len(capsys.readouterr().out) > 0
