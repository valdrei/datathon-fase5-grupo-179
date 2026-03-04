"""Testes adicionais para src.train."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import ModelTrainer


def make_dataset(n=40):
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.integers(0, 5, size=n),
    })
    y = 0.5 * X["f1"] - 0.3 * X["f2"] + 0.1 * X["f3"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_get_model_and_param_grid_variants():
    trainer = ModelTrainer("random_forest")
    assert trainer.get_model() is not None
    assert isinstance(trainer.get_param_grid(), dict)

    trainer.model_name = "gradient_boosting"
    assert trainer.get_model() is not None

    trainer.model_name = "ridge"
    assert trainer.get_model() is not None

    trainer.model_name = "lasso"
    assert trainer.get_model() is not None

    trainer.model_name = "inexistente"
    model = trainer.get_model()
    assert model is not None
    assert trainer.model_name == "random_forest"


def test_train_evaluate_cross_validate_and_persistence(tmp_path):
    X, y = make_dataset()
    trainer = ModelTrainer("random_forest")

    trainer.train_model(X, y, tune_hyperparameters=False)
    assert trainer.model is not None

    metrics = trainer.evaluate_model(X, y)
    assert "rmse" in metrics

    cv_metrics = trainer.cross_validate(X, y, cv=3)
    assert "cv_r2_mean" in cv_metrics

    trainer.save_model(output_dir=str(tmp_path))
    latest = tmp_path / "model_random_forest_latest.pkl"
    assert latest.exists()

    loaded = ModelTrainer("random_forest")
    loaded.load_model(str(latest))
    assert loaded.model is not None
    preds = loaded.model.predict(X.head(5))
    assert len(preds) == 5


def test_train_with_hyperparameter_tuning(tmp_path):
    """Test hyperparameter tuning for different model types."""
    X, y = make_dataset(n=50)
    
    for model_name in ["random_forest", "gradient_boosting"]:
        trainer = ModelTrainer(model_name)
        trainer.train_model(X, y, tune_hyperparameters=True)
        assert trainer.model is not None
        assert trainer.best_params is not None
        assert trainer.feature_importance is not None


def test_evaluate_model_error_before_training():
    """Test evaluate_model raises error if model not trained."""
    X, y = make_dataset()
    trainer = ModelTrainer("random_forest")
    try:
        trainer.evaluate_model(X, y)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "não foi treinado" in str(e)


def test_evaluate_model_with_zero_division_mape():
    """Test evaluate_model handles zero values for MAPE calculation."""
    X, y = make_dataset(n=30)
    trainer = ModelTrainer("random_forest")
    trainer.train_model(X, y, tune_hyperparameters=False)
    
    y_test = pd.Series([0.0] * len(X))
    metrics = trainer.evaluate_model(X, y_test)
    assert np.isinf(metrics["mape"]) or metrics["mape"] == np.inf


def test_cross_validate_creates_model_if_none():
    """Test cross_validate initializes model if not already trained."""
    X, y = make_dataset()
    trainer = ModelTrainer("ridge")
    assert trainer.model is None
    
    cv_metrics = trainer.cross_validate(X, y, cv=3)
    assert "cv_r2_mean" in cv_metrics
    assert trainer.model is not None


def test_save_model_error_if_not_trained():
    """Test save_model raises error if model not trained."""
    trainer = ModelTrainer("random_forest")
    try:
        trainer.save_model()
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "não foi treinado" in str(e)


def test_save_model_creates_metadata_and_importance(tmp_path):
    """Test save_model creates metadata JSON and feature importance CSV."""
    X, y = make_dataset()
    trainer = ModelTrainer("random_forest")
    trainer.train_model(X, y, tune_hyperparameters=False)
    
    trainer.save_model(output_dir=str(tmp_path))
    
    # Check metadata file
    metadata_files = list(tmp_path.glob("metadata_*.json"))
    assert len(metadata_files) > 0
    
    with open(metadata_files[0]) as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "random_forest"
    assert "timestamp" in metadata
    
    # Check feature importance file
    importance_files = list(tmp_path.glob("feature_importance_*.csv"))
    assert len(importance_files) > 0


def test_ridge_lasso_param_grids():
    """Test Ridge and Lasso return correct param grids."""
    trainer_ridge = ModelTrainer("ridge")
    grid_ridge = trainer_ridge.get_param_grid()
    assert "alpha" in grid_ridge
    
    trainer_lasso = ModelTrainer("lasso")
    grid_lasso = trainer_lasso.get_param_grid()
    assert "alpha" in grid_lasso


def test_train_pipeline_full_flow(tmp_path):
    """Test full training pipeline with actual data."""
    from src.train import train_pipeline
    import tempfile
    
    # Create minimal CSV for pipeline
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "Defas": [0.5, 1.0, -0.5, 1.5] * 10,
        "Gênero": ["Menina", "Menino"] * 20,
        "IDA": np.random.normal(5, 2, 40),
        "IEG": np.random.normal(6, 1.5, 40),
        "Fase": np.random.randint(1, 10, 40),
    })
    df.to_csv(csv_file, index=False)
    
    trainer, preprocessor, engineer = train_pipeline(
        data_path=str(csv_file),
        model_name="random_forest",
        test_size=0.2,
        tune_hyperparameters=False,
        output_dir=str(tmp_path)
    )
    
    assert trainer.model is not None
    assert len(trainer.metrics) > 0
    assert (tmp_path / "preprocessor_latest.pkl").exists()
    assert (tmp_path / "feature_engineer_latest.pkl").exists()