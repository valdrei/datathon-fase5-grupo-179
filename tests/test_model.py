"""
Testes unitários para o módulo de treinamento e predição do modelo.
Testa funcionalidades de treinamento, avaliação e predição.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_training_data():
    """Gera dados de exemplo para treinamento."""
    np.random.seed(42)
    
    n_samples = 100
    data = {
        'RA': [f'RA{i:03d}' for i in range(n_samples)],
        'Nome': [f'Aluno {i}' for i in range(n_samples)],
        'Fase': np.random.randint(4, 11, n_samples),
        'Turma': np.random.choice(['A', 'B', 'C'], n_samples),
        'Idade 22': np.random.randint(12, 22, n_samples),
        'Gênero': np.random.choice(['Menino', 'Menina'], n_samples),
        'Ano nasc': np.random.randint(2001, 2011, n_samples),
        'Ano ingresso': np.random.randint(2015, 2022, n_samples),
        'Instituição de ensino': np.random.choice(['Escola Pública', 'Escola Particular'], n_samples),
        'Pedra 20': np.random.choice(['Quartzo', 'Ágata', 'Ametista', 'Topázio', np.nan], n_samples),
        'Pedra 21': np.random.choice(['Quartzo', 'Ágata', 'Ametista', 'Topázio', np.nan], n_samples),
        'Pedra 22': np.random.choice(['Quartzo', 'Ágata', 'Ametista', 'Topázio'], n_samples),
        'INDE 22': np.random.uniform(3, 9, n_samples),
        'Cg': np.random.uniform(1, 1000, n_samples),
        'Cf': np.random.randint(1, 50, n_samples),
        'Ct': np.random.randint(1, 20, n_samples),
        'Nº Av': np.random.randint(2, 6, n_samples),
        'IAA': np.random.uniform(4, 10, n_samples),
        'IEG': np.random.uniform(3, 10, n_samples),
        'IPS': np.random.uniform(3, 10, n_samples),
        'IDA': np.random.uniform(3, 10, n_samples),
        'Matem': np.random.uniform(2, 10, n_samples),
        'Portug': np.random.uniform(2, 10, n_samples),
        'Inglês': np.random.uniform(3, 10, n_samples),
        'IPV': np.random.uniform(3, 10, n_samples),
        'IAN': np.random.uniform(3, 10, n_samples),
        'Fase ideal': [f'Fase {np.random.randint(5, 12)}' for _ in range(n_samples)],
        'Defasagem': np.random.uniform(-3, 2, n_samples),
        'Indicado': np.random.choice(['Sim', 'Não'], n_samples),
        'Atingiu PV': np.random.choice(['Sim', 'Não'], n_samples),
        'Rec Psicologia': np.random.choice(['Requer avaliação', 'Acompanhamento'], n_samples),
        'Avaliador1': [f'Avaliador-{i}' for i in np.random.randint(1, 20, n_samples)],
        'Rec Av1': np.random.choice(['Promovido de Fase', 'Mantido na Fase atual'], n_samples),
        'Avaliador2': [f'Avaliador-{i}' for i in np.random.randint(1, 20, n_samples)],
        'Rec Av2': np.random.choice(['Promovido de Fase', 'Mantido na Fase atual'], n_samples),
        'Avaliador3': [f'Avaliador-{i}' for i in np.random.randint(1, 20, n_samples)],
        'Rec Av3': np.random.choice(['Promovido de Fase', 'Mantido na Fase atual'], n_samples),
        'Destaque IEG': ['Melhorar entrega de lições'] * n_samples,
        'Destaque IDA': ['Empenhar-se mais nas aulas'] * n_samples,
        'Destaque IPV': ['Integrar-se mais aos Princípios'] * n_samples,
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Cria um preprocessador."""
    return DataPreprocessor()


@pytest.fixture
def feature_engineer():
    """Cria um feature engineer."""
    return FeatureEngineer()


@pytest.fixture
def model_trainer():
    """Cria um treinador de modelo."""
    return ModelTrainer()


@pytest.fixture
def model_evaluator():
    """Cria um avaliador de modelo."""
    return ModelEvaluator()


def test_model_trainer_initialization(model_trainer):
    """Testa inicialização do ModelTrainer."""
    assert model_trainer is not None
    assert hasattr(model_trainer, 'model')
    assert hasattr(model_trainer, 'best_params')
    assert hasattr(model_trainer, 'feature_importance')


def test_get_model(model_trainer):
    """Testa obtenção de modelo."""
    model = model_trainer.get_model('random_forest')
    assert model is not None
    assert isinstance(model, RandomForestRegressor)
    
    model = model_trainer.get_model('gradient_boosting')
    assert model is not None


def test_training_pipeline(sample_training_data, preprocessor, feature_engineer, model_trainer):
    """Testa pipeline completo de treinamento."""
    # Pré-processar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    
    # Feature engineering
    df_features = feature_engineer.engineer_features(df_clean)
    
    # Preparar features e target
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Verificar que temos dados
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert X.shape[0] == y.shape[0]
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar modelo simples (sem Grid Search para testes rápidos)
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Verificar que modelo foi treinado
    assert model is not None
    assert hasattr(model, 'feature_importances_')
    
    # Fazer predições
    predictions = model.predict(X_test)
    
    # Verificar predições
    assert predictions.shape[0] == y_test.shape[0]
    assert not np.isnan(predictions).any()


def test_model_prediction(sample_training_data, preprocessor, feature_engineer):
    """Testa predição com modelo treinado."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Treinar modelo simples
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Fazer predição em um único exemplo
    single_prediction = model.predict(X[:1])
    
    assert single_prediction.shape[0] == 1
    assert isinstance(single_prediction[0], (int, float, np.number))


def test_model_evaluation(sample_training_data, preprocessor, feature_engineer, model_evaluator):
    """Testa avaliação de modelo."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Fazer predições
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = model_evaluator.calculate_metrics(y_test, y_pred)
    
    # Verificar métricas
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    
    # Verificar que métricas são numéricas
    for key, value in metrics.items():
        if key not in ['p25', 'p50', 'p75', 'accuracy_within_0.5', 'accuracy_within_1.0']:
            assert isinstance(value, (int, float, np.number))
            assert not np.isnan(value)


def test_model_confidence_assessment(model_evaluator):
    """Testa avaliação de confiança do modelo."""
    # Métricas de alta confiança
    high_confidence_metrics = {
        'r2': 0.85,
        'mae': 0.3,
        'accuracy_within_0.5': 0.75
    }
    
    confidence = model_evaluator.get_model_confidence_message(high_confidence_metrics)
    assert 'ALTA CONFIANÇA' in confidence
    
    # Métricas de confiança moderada
    moderate_confidence_metrics = {
        'r2': 0.60,
        'mae': 0.5,
        'accuracy_within_0.5': 0.55
    }
    
    confidence = model_evaluator.get_model_confidence_message(moderate_confidence_metrics)
    assert 'MODERADA' in confidence
    
    # Métricas baixas
    low_confidence_metrics = {
        'r2': 0.40,
        'mae': 0.8,
        'accuracy_within_0.5': 0.40
    }
    
    confidence = model_evaluator.get_model_confidence_message(low_confidence_metrics)
    assert 'REQUER MELHORIAS' in confidence or 'MELHORIAS' in confidence


def test_feature_importance(sample_training_data, preprocessor, feature_engineer):
    """Testa extração de feature importance."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Verificar feature importance
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == X.shape[1]
    assert np.isclose(model.feature_importances_.sum(), 1.0)
    assert (model.feature_importances_ >= 0).all()


def test_cross_validation(sample_training_data, preprocessor, feature_engineer, model_trainer):
    """Testa cross-validation."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Criar modelo simples
    model = RandomForestRegressor(n_estimators=5, random_state=42, max_depth=3)
    
    # Executar cross-validation
    cv_results = model_trainer.cross_validate(model, X, y, cv=3)
    
    # Verificar resultados
    assert 'fit_time' in cv_results
    assert 'test_neg_mean_squared_error' in cv_results
    assert 'test_r2' in cv_results
    assert len(cv_results['fit_time']) == 3  # 3 folds


def test_model_persistence(sample_training_data, preprocessor, feature_engineer, tmp_path):
    """Testa salvamento e carregamento de modelo."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Salvar modelo
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Verificar que arquivo foi criado
    assert model_path.exists()
    
    # Carregar modelo
    loaded_model = joblib.load(model_path)
    
    # Verificar que modelo carregado funciona
    predictions_original = model.predict(X[:5])
    predictions_loaded = loaded_model.predict(X[:5])
    
    # Predições devem ser idênticas
    assert np.allclose(predictions_original, predictions_loaded)


def test_prediction_bounds(sample_training_data, preprocessor, feature_engineer):
    """Testa se predições estão dentro de limites razoáveis."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    
    # Fazer predições
    predictions = model.predict(X)
    
    # Verificar limites razoáveis (defasagem entre -5 e 5)
    assert predictions.min() >= -10  # Limite inferior razoável
    assert predictions.max() <= 10   # Limite superior razoável


def test_error_handling_invalid_data():
    """Testa tratamento de dados inválidos."""
    # Dados vazios
    empty_df = pd.DataFrame()
    preprocessor = DataPreprocessor()
    
    with pytest.raises(Exception):
        preprocessor.prepare_features_target(empty_df)


def test_model_reproducibility(sample_training_data, preprocessor, feature_engineer):
    """Testa reprodutibilidade do modelo com random_state."""
    # Preparar dados
    df_clean = preprocessor.clean_data(sample_training_data)
    df_clean = preprocessor.handle_missing_values(df_clean)
    df_features = feature_engineer.engineer_features(df_clean)
    X, y = preprocessor.prepare_features_target(df_features)
    
    # Treinar primeiro modelo
    model1 = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model1.fit(X, y)
    pred1 = model1.predict(X[:10])
    
    # Treinar segundo modelo com mesma seed
    model2 = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    model2.fit(X, y)
    pred2 = model2.predict(X[:10])
    
    # Predições devem ser idênticas
    assert np.allclose(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
