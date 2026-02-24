"""
Testes unitários para API FastAPI.
"""

import pytest
import tempfile
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Fixture com cliente de teste."""
    # Mock dos componentes necessários para testes
    import app.main as app_main
    
    # Criar mocks
    class MockModel:
        def predict(self, X):
            import numpy as np
            return np.array([-1.0] * len(X))
        
        @property
        def feature_importances_(self):
            return [0.1, 0.2, 0.3]
    
    class MockPreprocessor:
        numeric_features = ['INDE_22', 'IAA']
        categorical_features = ['Gênero']
        
        def preprocess_pipeline(self, df, fit=False):
            import pandas as pd
            return df.iloc[:, :5], None
        
        def prepare_features_target(self, df):
            import pandas as pd
            return df.iloc[:, :5], None
        
        def scale_features(self, df, fit=False):
            return df
    
    class MockFeatureEngineer:
        def engineer_features(self, df):
            return df
    
    class MockLogger:
        def __init__(self):
            self.predictions_file = Path(tempfile.mkdtemp()) / "predictions.jsonl"

        def log_prediction(self, **kwargs):
            pass

        def get_prediction_statistics(self, last_n=None):
            return {}
    
    class MockDriftDetector:
        reference_data = None
        def detect_drift(self, df):
            return False
    
    # Injetar mocks
    app_main.model = MockModel()
    app_main.preprocessor = MockPreprocessor()
    app_main.feature_engineer = MockFeatureEngineer()
    app_main.prediction_logger = MockLogger()
    app_main.drift_detector = MockDriftDetector()
    
    return TestClient(app_main.app)


class TestAPIEndpoints:
    """Testes para endpoints da API."""
    
    def test_root_endpoint(self, client):
        """Testa endpoint raiz."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "online"
    
    def test_health_check(self, client):
        """Testa health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["model_loaded"] == True
    
    def test_model_info(self, client):
        """Testa endpoint de informações do modelo."""
        response = client.get("/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert data["model_type"] == "MockModel"
    
    def test_predict_endpoint_valid_data(self, client):
        """Testa predição com dados válidos."""
        payload = {
            "Fase": 7,
            "Turma": "A",
            "Idade 22": 19,
            "Gênero": "Menina",
            "Ano ingresso": 2016,
            "Instituição de ensino": "Escola Pública",
            "Pedra 20": "Ametista",
            "Pedra 21": "Ametista",
            "Pedra 22": "Quartzo",
            "INDE 22": 5.783,
            "Cg": 753.0,
            "Cf": 18,
            "Ct": 10,
            "Nº Av": 4,
            "Avaliador1": "Avaliador-5",
            "Rec Av1": "Mantido na Fase atual",
            "Avaliador2": "Avaliador-27",
            "Rec Av2": "Promovido de Fase + Bolsa",
            "Avaliador3": "Avaliador-28",
            "Rec Av3": "Promovido de Fase",
            "Avaliador4": "Avaliador-31",
            "Rec Av4": "Mantido na Fase atual",
            "IAA": 8.3,
            "IEG": 4.1,
            "IPS": 5.6,
            "Rec Psicologia": "Requer avaliação",
            "IDA": 4.0,
            "Matem": 2.7,
            "Portug": 3.5,
            "Inglês": 6.0,
            "Indicado": "Sim",
            "Atingiu PV": "Não",
            "IPV": 7.278,
            "IAN": 5.0,
            "Fase ideal": "Fase 8 (Universitários)",
            "Destaque IEG": "Melhorar: Melhorar a sua entrega de lições de casa.",
            "Destaque IDA": "Melhorar: Empenhar-se mais nas aulas e avaliações.",
            "Destaque IPV": "Melhorar: Integrar-se mais aos Princípios Passos Mágicos."
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "defasagem_prevista" in data
        assert "risco" in data
        assert "confianca" in data
        assert "recomendacao" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_invalid_data(self, client):
        """Testa predição com dados inválidos."""
        payload = {
            "Fase": 7,
            # Faltam campos obrigatórios
        }
        
        response = client.post("/predict", json=payload)
        
        # Deve retornar erro de validação
        assert response.status_code == 422


class TestRiskClassification:
    """Testes para classificação de risco."""
    
    def test_classify_risk_low(self):
        """Testa classificação de risco baixo."""
        from app.routes import classify_risk
        
        assert classify_risk(0) == "Baixo"
        assert classify_risk(1) == "Baixo"
    
    def test_classify_risk_moderate(self):
        """Testa classificação de risco moderado."""
        from app.routes import classify_risk
        
        assert classify_risk(-0.5) == "Moderado"
        assert classify_risk(-0.9) == "Moderado"
    
    def test_classify_risk_high(self):
        """Testa classificação de risco alto."""
        from app.routes import classify_risk
        
        assert classify_risk(-1.5) == "Alto"
        assert classify_risk(-1.9) == "Alto"
    
    def test_classify_risk_critical(self):
        """Testa classificação de risco crítico."""
        from app.routes import classify_risk
        
        assert classify_risk(-2.0) == "Crítico"
        assert classify_risk(-3.0) == "Crítico"


class TestRecommendations:
    """Testes para geração de recomendações."""
    
    def test_generate_recommendation(self):
        """Testa geração de recomendações."""
        from app.routes import generate_recommendation
        
        # Testar diferentes níveis de risco
        rec_low = generate_recommendation(0.5, "Baixo")
        assert "adequado" in rec_low.lower() or "regular" in rec_low.lower()
        
        rec_moderate = generate_recommendation(-0.5, "Moderado")
        assert "reforço" in rec_moderate.lower() or "acompanhamento" in rec_moderate.lower()
        
        rec_high = generate_recommendation(-1.5, "Alto")
        assert "intensivo" in rec_high.lower() or "tutoria" in rec_high.lower()
        
        rec_critical = generate_recommendation(-2.5, "Crítico")
        assert "crítica" in rec_critical.lower() or "imediata" in rec_critical.lower()


def test_api_imports():
    """Testa se todos os imports necessários estão disponíveis."""
    try:
        from app.routes import (
            classify_risk, generate_recommendation
        )
        from app.main import app
        from app.routes import StudentData, PredictionResponse
        assert True
    except ImportError as e:
        pytest.fail(f"Erro ao importar módulos da API: {str(e)}")
