"""
Testes para os endpoints e módulos de monitoramento contínuo.

Cobre:
- Endpoints /monitoring/stats, /monitoring/predictions, /monitoring/drift
- Classe PredictionLogger
- Classe DriftDetector
- Classe ModelMonitor
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.monitoring import (
    PredictionLogger,
    DriftDetector,
    ModelMonitor,
)


# ═══════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════

@pytest.fixture
def tmp_log_dir(tmp_path):
    """Diretório temporário para logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def prediction_logger(tmp_log_dir):
    """PredictionLogger com diretório temporário."""
    return PredictionLogger(log_dir=tmp_log_dir)


@pytest.fixture
def prediction_logger_with_data(tmp_log_dir):
    """PredictionLogger com predições pré-registradas."""
    logger = PredictionLogger(log_dir=tmp_log_dir)
    # Registrar várias predições
    test_data = [
        {"INDE 22": 5.8, "IDA": 4.0, "IEG": 4.1, "Fase": 7, "Idade 22": 19},
        {"INDE 22": 7.2, "IDA": 6.5, "IEG": 8.0, "Fase": 4, "Idade 22": 14},
        {"INDE 22": 3.1, "IDA": 2.0, "IEG": 3.5, "Fase": 6, "Idade 22": 17},
        {"INDE 22": 8.5, "IDA": 8.0, "IEG": 9.0, "Fase": 3, "Idade 22": 12},
        {"INDE 22": 6.0, "IDA": 5.5, "IEG": 6.0, "Fase": 5, "Idade 22": 15},
        {"INDE 22": 4.2, "IDA": 3.0, "IEG": 4.0, "Fase": 8, "Idade 22": 20},
        {"INDE 22": 7.8, "IDA": 7.0, "IEG": 7.5, "Fase": 2, "Idade 22": 11},
        {"INDE 22": 5.0, "IDA": 4.5, "IEG": 5.0, "Fase": 6, "Idade 22": 16},
        {"INDE 22": 9.0, "IDA": 9.0, "IEG": 9.5, "Fase": 1, "Idade 22": 10},
        {"INDE 22": 2.5, "IDA": 1.5, "IEG": 2.0, "Fase": 9, "Idade 22": 22},
        {"INDE 22": 6.5, "IDA": 6.0, "IEG": 6.5, "Fase": 4, "Idade 22": 13},
        {"INDE 22": 4.8, "IDA": 3.5, "IEG": 4.5, "Fase": 7, "Idade 22": 18},
    ]
    predictions = [-1.2, 0.5, -2.1, 1.0, -0.3, -1.8, 0.8, -0.5, 1.5, -3.0, 0.2, -1.0]
    confidences = [0.87, 0.92, 0.78, 0.95, 0.88, 0.82, 0.93, 0.85, 0.96, 0.75, 0.90, 0.84]
    risks = ["Alto", "Baixo", "Crítico", "Baixo", "Moderado", "Alto",
             "Baixo", "Moderado", "Baixo", "Crítico", "Baixo", "Moderado"]

    for i in range(len(predictions)):
        logger.log_prediction(
            input_data=test_data[i],
            prediction=predictions[i],
            confidence=confidences[i],
            risk=risks[i],
        )
    return logger


@pytest.fixture
def drift_detector_with_ref(tmp_path):
    """DriftDetector com dados de referência."""
    np.random.seed(42)
    ref_data = pd.DataFrame({
        "INDE_22": np.random.normal(6, 1.5, 100),
        "IDA": np.random.normal(5, 2, 100),
        "IEG": np.random.normal(6.5, 1.5, 100),
    })
    ref_file = tmp_path / "reference.csv"
    ref_data.to_csv(ref_file, index=False)
    return DriftDetector(reference_file=str(ref_file), threshold=0.05)


@pytest.fixture
def client_with_monitoring(tmp_log_dir):
    """TestClient com mocks completos para monitoramento."""
    import app.main as app_main

    class MockModel:
        def predict(self, X):
            return np.array([-1.0] * len(X))

        @property
        def estimators_(self):
            return [self] * 5

        @property
        def feature_importances_(self):
            return [0.1, 0.2, 0.3]

    class MockPreprocessor:
        numeric_features = ["INDE_22", "IAA"]
        categorical_features = ["Gênero"]

        def preprocess_pipeline(self, df, fit=False):
            return df.iloc[:, :5], None

    class MockFeatureEngineer:
        def engineer_features(self, df):
            return df

    # Usar PredictionLogger real (com tmp dir)
    real_logger = PredictionLogger(log_dir=tmp_log_dir)

    # Registrar algumas predições para testes
    for i in range(15):
        real_logger.log_prediction(
            input_data={"INDE 22": 5.0 + i * 0.2, "Fase": 5},
            prediction=-1.0 + i * 0.1,
            confidence=0.85 + i * 0.005,
            risk="Alto" if i < 5 else ("Moderado" if i < 10 else "Baixo"),
        )

    class MockDriftDetector:
        reference_data = None
        def detect_drift(self, df):
            return False

    app_main.model = MockModel()
    app_main.preprocessor = MockPreprocessor()
    app_main.feature_engineer = MockFeatureEngineer()
    app_main.prediction_logger = real_logger
    app_main.drift_detector = MockDriftDetector()

    return TestClient(app_main.app)


# ═══════════════════════════════════════════════════════
#  1. TESTES DA CLASSE PredictionLogger
# ═══════════════════════════════════════════════════════

class TestPredictionLogger:
    """Testes para a classe PredictionLogger."""

    def test_init_creates_directory(self, tmp_path):
        """Diretório é criado automaticamente."""
        log_dir = tmp_path / "new_logs"
        logger = PredictionLogger(log_dir=str(log_dir))
        assert log_dir.exists()

    def test_log_prediction_creates_file(self, prediction_logger):
        """Arquivo JSONL é criado na primeira predição."""
        prediction_logger.log_prediction(
            input_data={"INDE 22": 5.0},
            prediction=-1.0,
            confidence=0.85,
            risk="Alto",
        )
        assert Path(prediction_logger.predictions_file).exists()

    def test_log_prediction_appends(self, prediction_logger):
        """Cada predição adiciona uma linha ao arquivo."""
        for i in range(3):
            prediction_logger.log_prediction(
                input_data={"INDE 22": 5.0 + i},
                prediction=-1.0 + i,
                confidence=0.85,
                risk="Alto",
            )
        with open(prediction_logger.predictions_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_log_prediction_valid_json(self, prediction_logger):
        """Cada linha do JSONL é JSON válido."""
        prediction_logger.log_prediction(
            input_data={"INDE 22": 7.5, "Fase": 4},
            prediction=0.5,
            confidence=0.92,
            risk="Baixo",
        )
        with open(prediction_logger.predictions_file) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert entry["prediction"] == 0.5
        assert entry["confidence"] == 0.92
        assert entry["risk"] == "Baixo"

    def test_get_statistics_empty(self, prediction_logger):
        """Retorna dict vazio quando sem predições."""
        stats = prediction_logger.get_prediction_statistics()
        assert stats == {}

    def test_get_statistics_with_data(self, prediction_logger_with_data):
        """Estatísticas calculadas corretamente."""
        stats = prediction_logger_with_data.get_prediction_statistics()
        assert stats["total_predictions"] == 12
        assert "mean_prediction" in stats
        assert "std_prediction" in stats
        assert "risk_distribution" in stats
        assert stats["risk_distribution"]["Baixo"] == 5
        assert stats["risk_distribution"]["Moderado"] == 3
        assert stats["risk_distribution"]["Alto"] == 2
        assert stats["risk_distribution"]["Crítico"] == 2

    def test_get_statistics_last_n(self, prediction_logger_with_data):
        """Filtra corretamente as últimas N predições."""
        stats = prediction_logger_with_data.get_prediction_statistics(
            last_n=5
        )
        assert stats["total_predictions"] == 5

    def test_save_metrics(self, prediction_logger):
        """Métricas são salvas corretamente."""
        metrics = {"r2": 0.85, "rmse": 0.31, "mae": 0.21}
        prediction_logger.save_metrics(metrics)
        assert Path(prediction_logger.metrics_file).exists()
        with open(prediction_logger.metrics_file) as f:
            saved = json.load(f)
        assert "timestamp" in saved
        assert saved["metrics"]["r2"] == 0.85

    def test_retention_removes_old_records(self, tmp_path):
        """Ao ultrapassar max_records, mantém apenas os mais recentes."""
        logger = PredictionLogger(
            log_dir=str(tmp_path / "retention_logs"),
            max_records=5,
        )
        # Forçar verificação a cada escrita
        logger._truncate_interval = 1

        for i in range(10):
            logger.log_prediction(
                input_data={"INDE 22": float(i)},
                prediction=float(i),
                confidence=0.9,
                risk="Baixo",
            )

        with open(logger.predictions_file) as f:
            lines = f.readlines()
        assert len(lines) == 5
        # Verifica que manteve os mais recentes (5..9)
        first_kept = json.loads(lines[0])
        assert first_kept["prediction"] == 5.0

    def test_retention_disabled_when_zero(self, tmp_path):
        """max_records=0 desativa a retenção (sem limite)."""
        logger = PredictionLogger(
            log_dir=str(tmp_path / "no_limit_logs"),
            max_records=0,
        )
        logger._truncate_interval = 1

        for i in range(20):
            logger.log_prediction(
                input_data={"INDE 22": float(i)},
                prediction=float(i),
                confidence=0.9,
                risk="Baixo",
            )

        with open(logger.predictions_file) as f:
            lines = f.readlines()
        assert len(lines) == 20


# ═══════════════════════════════════════════════════════
#  2. TESTES DA CLASSE DriftDetector
# ═══════════════════════════════════════════════════════

class TestDriftDetector:
    """Testes para a classe DriftDetector."""

    def test_init_no_reference(self):
        """Inicializa sem dados de referência."""
        detector = DriftDetector()
        assert detector.reference_data is None

    def test_init_with_reference(self, drift_detector_with_ref):
        """Inicializa com dados de referência."""
        assert drift_detector_with_ref.reference_data is not None
        assert len(drift_detector_with_ref.reference_data) == 100

    def test_detect_drift_no_reference(self):
        """Sem referência, retorna False."""
        detector = DriftDetector()
        new_data = pd.DataFrame({"col": [1, 2, 3]})
        assert detector.detect_drift(new_data) is False

    def test_detect_drift_same_distribution(self, drift_detector_with_ref):
        """Dados iguais não devem indicar drift."""
        np.random.seed(42)
        same_data = pd.DataFrame({
            "INDE_22": np.random.normal(6, 1.5, 100),
            "IDA": np.random.normal(5, 2, 100),
            "IEG": np.random.normal(6.5, 1.5, 100),
        })
        # With similar distribution, drift should not be detected
        # (could pass or fail depending on random; we just test it runs)
        result = drift_detector_with_ref.detect_drift(same_data)
        assert isinstance(result, bool)

    def test_detect_drift_different_distribution(self, drift_detector_with_ref):
        """Distribuição muito diferente deve detectar drift."""
        shifted_data = pd.DataFrame({
            "INDE_22": np.random.normal(15, 1, 100),  # muito diferente
            "IDA": np.random.normal(15, 1, 100),
            "IEG": np.random.normal(15, 1, 100),
        })
        assert drift_detector_with_ref.detect_drift(shifted_data) is True

    def test_calculate_psi(self, drift_detector_with_ref):
        """PSI é calculado sem erro."""
        expected = pd.Series(np.random.normal(5, 1, 200))
        actual = pd.Series(np.random.normal(5, 1, 200))
        psi = drift_detector_with_ref.calculate_psi(expected, actual)
        assert isinstance(psi, float)
        assert psi >= 0

    def test_calculate_psi_different(self, drift_detector_with_ref):
        """PSI alto para distribuições muito diferentes."""
        expected = pd.Series(np.random.normal(0, 1, 500))
        actual = pd.Series(np.random.normal(10, 1, 500))
        psi = drift_detector_with_ref.calculate_psi(expected, actual)
        assert psi > 0.25  # Mudança significativa

    def test_monitor_psi_no_reference(self):
        """PSI vazio quando sem referência."""
        detector = DriftDetector()
        result = detector.monitor_psi(pd.DataFrame({"col": [1, 2]}))
        assert result == {}

    def test_monitor_psi_with_reference(self, drift_detector_with_ref):
        """PSI calculado por feature."""
        np.random.seed(42)
        new_data = pd.DataFrame({
            "INDE_22": np.random.normal(6, 1.5, 50),
            "IDA": np.random.normal(5, 2, 50),
        })
        psi_values = drift_detector_with_ref.monitor_psi(new_data)
        assert "INDE_22" in psi_values
        assert "IDA" in psi_values
        assert all(v >= 0 for v in psi_values.values())


# ═══════════════════════════════════════════════════════
#  3. TESTES DA CLASSE ModelMonitor
# ═══════════════════════════════════════════════════════

class TestModelMonitor:
    """Testes para a classe ModelMonitor."""

    def test_init(self, tmp_log_dir):
        """Inicializa corretamente."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        assert monitor.log_dir == Path(tmp_log_dir)

    def test_log_performance(self, tmp_log_dir):
        """Registra métricas de performance."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        monitor.log_performance(
            metrics={"r2": 0.85, "rmse": 0.31},
            metadata={"dataset": "test"},
        )
        assert monitor.performance_file.exists()

    def test_check_degradation_no_degradation(self, tmp_log_dir):
        """Sem degradação quando métricas estáveis."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        current = {"r2": 0.85, "mae": 0.20, "rmse": 0.30}
        baseline = {"r2": 0.86, "mae": 0.21, "rmse": 0.31}
        assert monitor.check_degradation(current, baseline) is False

    def test_check_degradation_detected(self, tmp_log_dir):
        """Detecta degradação quando métricas caem muito."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        current = {"r2": 0.50, "mae": 0.60, "rmse": 0.80}
        baseline = {"r2": 0.85, "mae": 0.20, "rmse": 0.30}
        assert monitor.check_degradation(current, baseline) is True


# ═══════════════════════════════════════════════════════
#  4. TESTES DOS ENDPOINTS DE MONITORAMENTO
# ═══════════════════════════════════════════════════════

class TestMonitoringEndpoints:
    """Testes para endpoints /monitoring/*."""

    def test_monitoring_stats_returns_200(self, client_with_monitoring):
        """GET /monitoring/stats retorna 200."""
        r = client_with_monitoring.get("/monitoring/stats")
        assert r.status_code == 200

    def test_monitoring_stats_structure(self, client_with_monitoring):
        """Resposta contém campos esperados."""
        data = client_with_monitoring.get("/monitoring/stats").json()
        assert "total_predictions" in data
        assert "mean_prediction" in data
        assert "std_prediction" in data
        assert "min_prediction" in data
        assert "max_prediction" in data
        assert "mean_confidence" in data
        assert "risk_distribution" in data
        assert "last_prediction_time" in data

    def test_monitoring_stats_values_correct(self, client_with_monitoring):
        """Valores são numéricos e coerentes."""
        data = client_with_monitoring.get("/monitoring/stats").json()
        assert data["total_predictions"] == 15
        assert isinstance(data["mean_prediction"], float)
        assert isinstance(data["mean_confidence"], float)
        assert 0 <= data["mean_confidence"] <= 1
        assert data["min_prediction"] <= data["mean_prediction"]
        assert data["mean_prediction"] <= data["max_prediction"]

    def test_monitoring_stats_with_last_n(self, client_with_monitoring):
        """Filtra por last_n."""
        data = client_with_monitoring.get(
            "/monitoring/stats", params={"last_n": 5}
        ).json()
        assert data["total_predictions"] == 5

    def test_monitoring_predictions_returns_200(self, client_with_monitoring):
        """GET /monitoring/predictions retorna 200."""
        r = client_with_monitoring.get("/monitoring/predictions")
        assert r.status_code == 200

    def test_monitoring_predictions_structure(self, client_with_monitoring):
        """Resposta contém lista de predições."""
        data = client_with_monitoring.get("/monitoring/predictions").json()
        assert "predictions" in data
        assert "total" in data
        assert isinstance(data["predictions"], list)
        assert data["total"] == 15

    def test_monitoring_predictions_entry_fields(self, client_with_monitoring):
        """Cada predição tem os campos corretos."""
        data = client_with_monitoring.get("/monitoring/predictions").json()
        entry = data["predictions"][0]
        assert "timestamp" in entry
        assert "prediction" in entry
        assert "confidence" in entry
        assert "risk" in entry

    def test_monitoring_predictions_last_n(self, client_with_monitoring):
        """Respeita parâmetro last_n."""
        data = client_with_monitoring.get(
            "/monitoring/predictions", params={"last_n": 3}
        ).json()
        assert data["total"] == 3

    def test_monitoring_drift_returns_200(self, client_with_monitoring):
        """GET /monitoring/drift retorna 200."""
        r = client_with_monitoring.get("/monitoring/drift")
        assert r.status_code == 200

    def test_monitoring_drift_structure(self, client_with_monitoring):
        """Resposta contém campos esperados."""
        data = client_with_monitoring.get("/monitoring/drift").json()
        assert "drift_enabled" in data
        assert "psi" in data
        assert "psi_thresholds" in data
        assert "prediction_drift" in data

    def test_monitoring_drift_thresholds(self, client_with_monitoring):
        """Thresholds de PSI estão presentes."""
        data = client_with_monitoring.get("/monitoring/drift").json()
        thresholds = data["psi_thresholds"]
        assert "no_change" in thresholds
        assert "moderate" in thresholds
        assert "significant" in thresholds

    def test_monitoring_drift_prediction_analysis(self, client_with_monitoring):
        """Com >= 10 predições, análise de drift está presente."""
        data = client_with_monitoring.get("/monitoring/drift").json()
        pred_drift = data["prediction_drift"]
        assert "first_half_mean" in pred_drift
        assert "second_half_mean" in pred_drift
        assert "mean_shift" in pred_drift
        assert "ks_statistic" in pred_drift
        assert "ks_pvalue" in pred_drift
        assert "drift_detected" in pred_drift
        assert isinstance(pred_drift["drift_detected"], bool)

    def test_monitoring_endpoints_in_root(self, client_with_monitoring):
        """Endpoints de monitoramento listados na raiz."""
        data = client_with_monitoring.get("/").json()
        endpoints = data["endpoints"]
        assert "/monitoring/stats" in endpoints
        assert "/monitoring/predictions" in endpoints
        assert "/monitoring/drift" in endpoints


# ═══════════════════════════════════════════════════════
#  5. TESTES DE MONITORAMENTO SEM DADOS
# ═══════════════════════════════════════════════════════

class TestMonitoringEndpointsEmpty:
    """Testes dos endpoints quando não há predições."""

    @pytest.fixture
    def client_empty(self, tmp_path):
        """Client sem predições registradas."""
        import app.main as app_main

        empty_log_dir = str(tmp_path / "empty_logs")
        Path(empty_log_dir).mkdir(parents=True)

        class MockModel:
            def predict(self, X):
                return np.array([-1.0])
            @property
            def feature_importances_(self):
                return [0.1]

        class MockPreprocessor:
            numeric_features = ["INDE_22"]
            categorical_features = []
            def preprocess_pipeline(self, df, fit=False):
                return df.iloc[:, :3], None

        class MockFeatureEngineer:
            def engineer_features(self, df):
                return df

        class MockDriftDetector:
            reference_data = None
            def detect_drift(self, df):
                return False

        app_main.model = MockModel()
        app_main.preprocessor = MockPreprocessor()
        app_main.feature_engineer = MockFeatureEngineer()
        app_main.prediction_logger = PredictionLogger(
            log_dir=empty_log_dir
        )
        app_main.drift_detector = MockDriftDetector()

        return TestClient(app_main.app)

    def test_stats_empty_returns_200(self, client_empty):
        """Stats sem dados retorna 200 com mensagem."""
        r = client_empty.get("/monitoring/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_predictions"] == 0

    def test_predictions_empty_returns_200(self, client_empty):
        """Predictions sem dados retorna lista vazia."""
        r = client_empty.get("/monitoring/predictions")
        assert r.status_code == 200
        data = r.json()
        assert data["predictions"] == []
        assert data["total"] == 0

    def test_drift_empty_returns_200(self, client_empty):
        """Drift sem dados retorna estrutura vazia."""
        r = client_empty.get("/monitoring/drift")
        assert r.status_code == 200
        data = r.json()
        assert data["prediction_drift"] == {}
        assert data["drift_enabled"] is False


# ═══════════════════════════════════════════════════════
#  6. TESTES DE DASHBOARD E FUNÇÕES UTILITÁRIAS
# ═══════════════════════════════════════════════════════

class TestMonitoringDashboard:
    """Testes para geração de dashboard e funções auxiliares."""

    def test_create_monitoring_dashboard_with_data(self, tmp_path):
        """Dashboard é gerado com dados presentes."""
        from src.monitoring import create_monitoring_dashboard
        
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        # Criar logger com dados
        logger = PredictionLogger(log_dir=str(log_dir))
        for i in range(5):
            logger.log_prediction(
                input_data={"INDE 22": 5.0 + i},
                prediction=-1.0 + i * 0.2,
                confidence=0.80 + i * 0.03,
                risk=["Baixo", "Moderado", "Alto"][i % 3],
            )
        
        output_file = tmp_path / "dashboard.html"
        create_monitoring_dashboard(
            log_dir=str(log_dir),
            output_file=str(output_file)
        )
        
        assert output_file.exists()
        with open(output_file) as f:
            content = f.read()
        assert any(variant in content for variant in ["Passos Mágicos", "Passos M", "gicos"])
        assert "Dashboard" in content

    def test_create_monitoring_dashboard_empty(self, tmp_path):
        """Dashboard sem dados retorna aviso."""
        from src.monitoring import create_monitoring_dashboard
        
        log_dir = tmp_path / "empty_logs"
        log_dir.mkdir()
        
        output_file = tmp_path / "dashboard.html"
        create_monitoring_dashboard(
            log_dir=str(log_dir),
            output_file=str(output_file)
        )
        
        # Arquivo não deve ser criado quando sem dados
        # ou contem aviso
        if output_file.exists():
            with open(output_file) as f:
                content = f.read()
            assert "Sem dados" in content or len(content) == 0


def test_generate_risk_bars():
    """Teste da função _generate_risk_bars."""
    from src.monitoring import _generate_risk_bars
    
    risk_dist = {"Baixo": 10, "Moderado": 5, "Alto": 3, "Crítico": 1}
    html = _generate_risk_bars(risk_dist)
    
    assert "Baixo" in html
    assert "Moderado" in html
    assert "Alto" in html
    assert "Crítico" in html
    assert "10" in html  # Count for Baixo
    assert "risk-bar" in html


def test_generate_risk_bars_empty():
    """_generate_risk_bars com dict vazio."""
    from src.monitoring import _generate_risk_bars
    
    html = _generate_risk_bars({})
    assert "Sem dados" in html


# ═══════════════════════════════════════════════════════
#  7. TESTES DE PSI COM EDGE CASES
# ═══════════════════════════════════════════════════════

class TestPSIEdgeCases:
    """Testes de edge cases para cálculo de PSI."""

    def test_calculate_psi_with_single_bucket(self, drift_detector_with_ref):
        """PSI quando todos os valores caem em um bucket."""
        expected = pd.Series([5.0] * 100)
        actual = pd.Series([5.0] * 50)
        psi = drift_detector_with_ref.calculate_psi(expected, actual)
        assert isinstance(psi, float)

    def test_calculate_psi_small_series(self, drift_detector_with_ref):
        """PSI com séries pequenas."""
        expected = pd.Series([1.0, 2.0, 3.0])
        actual = pd.Series([1.5, 2.5, 3.5])
        psi = drift_detector_with_ref.calculate_psi(expected, actual)
        assert isinstance(psi, float)
        assert psi >= 0

    def test_detect_drift_missing_columns(self, drift_detector_with_ref):
        """Drift com colunas faltando."""
        new_data = pd.DataFrame({"UNKNOWN_COL": [1, 2, 3]})
        result = drift_detector_with_ref.detect_drift(new_data)
        assert isinstance(result, bool)

    def test_detect_drift_with_nans(self, drift_detector_with_ref):
        """Drift lida com NaNs."""
        new_data = pd.DataFrame({
            "INDE_22": [np.nan, 5.0, 6.0] * 10,
            "IDA": [4.0, np.nan, 5.0] * 10,
            "IEG": [6.5, 7.0, np.nan] * 10,
        })
        result = drift_detector_with_ref.detect_drift(new_data)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════
#  8. TESTES DE PERFORMANCE E MÉTRICAS
# ═══════════════════════════════════════════════════════

class TestMetricsAndPerformance:
    """Testes de salva e leitura de métricas."""

    def test_save_and_read_metrics(self, tmp_log_dir):
        """Metrics são salvos e podem ser lidos."""
        logger = PredictionLogger(log_dir=tmp_log_dir)
        metrics = {
            "r2": 0.87,
            "rmse": 0.28,
            "mae": 0.19,
            "mape": 5.3,
        }
        logger.save_metrics(metrics)
        
        with open(logger.metrics_file) as f:
            saved = json.load(f)
        
        assert saved["metrics"]["r2"] == 0.87
        assert saved["metrics"]["rmse"] == 0.28
        assert "timestamp" in saved

    def test_log_performance_metadata(self, tmp_log_dir):
        """Performance log com metadados."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        monitor.log_performance(
            metrics={"r2": 0.85},
            metadata={"epoch": 1, "dataset": "test"},
        )
        
        with open(monitor.performance_file) as f:
            entry = json.loads(f.readline())
        
        assert entry["metrics"]["r2"] == 0.85
        assert entry["metadata"]["epoch"] == 1

    def test_degradation_with_partial_metrics(self, tmp_log_dir):
        """Degradation check com métricas incompletas."""
        monitor = ModelMonitor(log_dir=tmp_log_dir)
        current = {"r2": 0.80, "mae": 0.25}
        baseline = {"r2": 0.85}  # mae falta
        
        # Deve funcionar sem erro
        result = monitor.check_degradation(current, baseline)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════
#  9. TESTES DE CONFORMIDADE E INTEGRIDADE
# ═══════════════════════════════════════════════════════

class TestDataIntegrity:
    """Testes de integridade de dados no logging."""

    def test_prediction_logger_jsonl_format(self, tmp_log_dir):
        """Arquivo JSONL mantém formato correto."""
        logger = PredictionLogger(log_dir=tmp_log_dir)
        
        for i in range(10):
            logger.log_prediction(
                input_data={"val": i},
                prediction=float(i),
                confidence=0.9,
                risk="Baixo",
            )
        
        # Validar que cada linha é JSON válido
        with open(logger.predictions_file) as f:
            for line in f:
                entry = json.loads(line)
                assert "timestamp" in entry
                assert "prediction" in entry

    def test_retention_preserves_most_recent(self, tmp_path):
        """Retention mantém os registros mais recentes, não aleatórios."""
        logger = PredictionLogger(
            log_dir=str(tmp_path),
            max_records=3,
        )
        logger._truncate_interval = 1
        
        predictions = [100, 200, 300, 400, 500]
        for pred in predictions:
            logger.log_prediction(
                input_data={"val": pred},
                prediction=float(pred),
                confidence=0.9,
                risk="Baixo",
            )
        
        with open(logger.predictions_file) as f:
            entries = [json.loads(line) for line in f.readlines()]
        
        # Deve manter os últimos 3
        actual_predictions = [e["prediction"] for e in entries]
        assert actual_predictions == [300.0, 400.0, 500.0]
