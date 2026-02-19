"""
Testes de integração Frontend ↔ API.

Valida que os payloads montados pelo frontend (simplificado e completo)
são aceitos pelo schema da API e geram predições válidas.

Uso:
    pytest tests/test_frontend.py -v

Nota: Usa TestClient do FastAPI (não precisa da API rodando).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.routes import StudentData, classify_risk, generate_recommendation
from pydantic import ValidationError


# ─── Defaults do frontend (espelho de frontend/app_streamlit.py) ───
DEFAULTS = {
    "Fase": 2, "Turma": "A", "Idade 22": 12, "Gênero": "Menina",
    "Ano ingresso": 2021, "Instituição de ensino": "Escola Pública",
    "Pedra 20": "Ametista", "Pedra 21": "Ametista", "Pedra 22": "Ametista",
    "INDE 22": 7.197, "Cg": 430.5, "Cf": 67, "Ct": 6, "Nº Av": 3,
    "Avaliador1": "Avaliador-6", "Rec Av1": "Mantido na Fase atual",
    "Avaliador2": "Avaliador-27", "Rec Av2": "Mantido na Fase atual",
    "Avaliador3": "Avaliador-30", "Rec Av3": "Mantido na Fase atual",
    "Avaliador4": "Avaliador-31", "Rec Av4": "Promovido de Fase + Bolsa",
    "IAA": 8.8, "IEG": 8.3, "IPS": 7.5, "IDA": 6.3, "IPV": 7.333, "IAN": 5.0,
    "Matem": 6.0, "Portug": 6.7, "Inglês": 6.3,
    "Rec Psicologia": "Não atendido",
    "Indicado": "Não", "Atingiu PV": "Não",
    "Fase ideal": "Fase 2 (5º e 6º ano)",
    "Destaque IEG": "Destaque: A sua boa entrega das lições de casa.",
    "Destaque IDA": "Melhorar: Empenhar-se mais nas aulas e avaliações.",
    "Destaque IPV": "Melhorar: Integrar-se mais aos Princípios Passos Mágicos.",
}

# Campos obrigatórios da API (não-opcionais no schema StudentData)
REQUIRED_FIELDS = [
    "Fase", "Turma", "Idade 22", "Gênero", "Ano ingresso",
    "Instituição de ensino", "Pedra 22", "INDE 22", "Cg", "Cf", "Ct", "Nº Av",
    "Avaliador1", "Rec Av1", "Avaliador2", "Rec Av2", "Avaliador3", "Rec Av3",
    "IAA", "IEG", "IPS", "Rec Psicologia", "IDA", "Matem", "Portug",
    "Indicado", "Atingiu PV", "IPV", "IAN", "Fase ideal",
    "Destaque IEG", "Destaque IDA", "Destaque IPV",
]


# ─── Fixture: TestClient com mocks (não depende de modelo real) ───
@pytest.fixture
def client():
    """TestClient com mocks injetados."""
    import app.main as app_main

    class MockModel:
        def predict(self, X):
            import numpy as np
            return np.array([-0.5] * len(X))

        @property
        def estimators_(self):
            return [self] * 5  # simula 5 árvores

        @property
        def feature_importances_(self):
            return [0.1, 0.2, 0.3]

    class MockPreprocessor:
        numeric_features = ["INDE_22", "IAA"]
        categorical_features = ["Gênero"]

        def preprocess_pipeline(self, df, fit=False):
            import pandas as pd
            return df.iloc[:, :5], None

    class MockFeatureEngineer:
        def engineer_features(self, df):
            return df

    class MockLogger:
        def log_prediction(self, **kwargs):
            pass

    class MockDriftDetector:
        def detect_drift(self, df):
            return False

    app_main.model = MockModel()
    app_main.preprocessor = MockPreprocessor()
    app_main.feature_engineer = MockFeatureEngineer()
    app_main.prediction_logger = MockLogger()
    app_main.drift_detector = MockDriftDetector()

    return TestClient(app_main.app)


# ═════════════════════════════════════════════════════════════
#  1. VALIDAÇÃO DO SCHEMA – defaults do frontend passam?
# ═════════════════════════════════════════════════════════════
class TestFrontendDefaults:
    """Garante que os DEFAULTS do frontend passam na validação do schema."""

    def test_defaults_passam_no_schema(self):
        """DEFAULTS completos devem ser aceitos pelo StudentData."""
        student = StudentData(**DEFAULTS)
        assert student is not None

    def test_defaults_contem_todos_os_campos_obrigatorios(self):
        """Verifica se DEFAULTS cobre todos os campos required da API."""
        for field in REQUIRED_FIELDS:
            assert field in DEFAULTS, f"Campo obrigatório '{field}' ausente nos DEFAULTS do frontend"

    def test_defaults_tipos_corretos(self):
        """Valida tipos dos campos numéricos nos DEFAULTS."""
        assert isinstance(DEFAULTS["Fase"], int)
        assert isinstance(DEFAULTS["Idade 22"], int)
        assert isinstance(DEFAULTS["Ano ingresso"], int)
        assert isinstance(DEFAULTS["Cf"], int)
        assert isinstance(DEFAULTS["Ct"], int)
        assert isinstance(DEFAULTS["Nº Av"], int)
        assert isinstance(DEFAULTS["INDE 22"], (int, float))
        assert isinstance(DEFAULTS["IAN"], (int, float))
        assert isinstance(DEFAULTS["IPV"], (int, float))

    def test_defaults_dentro_dos_limites(self):
        """Valida que defaults respeitam ge/le do schema."""
        assert 0 <= DEFAULTS["Fase"] <= 10
        assert 5 <= DEFAULTS["Idade 22"] <= 25
        assert 2010 <= DEFAULTS["Ano ingresso"] <= 2023
        assert 0 <= DEFAULTS["Nº Av"] <= 10
        assert 0 <= DEFAULTS["INDE 22"] <= 10
        assert 0 <= DEFAULTS["IAA"] <= 10
        assert 0 <= DEFAULTS["IEG"] <= 10
        assert 0 <= DEFAULTS["IPS"] <= 10
        assert 0 <= DEFAULTS["IDA"] <= 10
        assert 0 <= DEFAULTS["IPV"] <= 10
        assert 0 <= DEFAULTS["IAN"] <= 10
        assert 0 <= DEFAULTS["Matem"] <= 10
        assert 0 <= DEFAULTS["Portug"] <= 10
        assert 0 <= DEFAULTS["Inglês"] <= 10


# ═════════════════════════════════════════════════════════════
#  2. PAYLOAD SIMPLIFICADO (só campos essenciais + defaults)
# ═════════════════════════════════════════════════════════════
class TestPayloadSimplificado:
    """Simula o que o frontend monta quando o usuário preenche só o essencial."""

    def _build_simplified_payload(self, **overrides):
        """Monta payload como o frontend faz: defaults + campos essenciais."""
        payload = {**DEFAULTS}
        payload.update(overrides)
        return payload

    def test_payload_minimo_aceito_api(self, client):
        """Payload com defaults + campos essenciais deve retornar 200."""
        payload = self._build_simplified_payload(
            IAN=3.0, IPV=5.0, Fase=4, **{"Idade 22": 14},
            **{"Fase ideal": "Fase 4 (7º e 8º ano)"}, Cf=30,
            **{"INDE 22": 6.0, "Nº Av": 4},
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "defasagem_prevista" in data
        assert "risco" in data
        assert "confianca" in data
        assert "recomendacao" in data
        assert "timestamp" in data

    def test_payload_ian_baixo(self, client):
        """IAN muito baixo → deve funcionar (risco alto esperado)."""
        payload = self._build_simplified_payload(IAN=1.0)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_payload_ian_alto(self, client):
        """IAN muito alto → deve funcionar (risco baixo esperado)."""
        payload = self._build_simplified_payload(IAN=9.5)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_payload_extremos_validos(self, client):
        """Todos os valores nos limites extremos (mas válidos)."""
        payload = self._build_simplified_payload(
            Fase=0, **{"Idade 22": 5, "Ano ingresso": 2010, "Nº Av": 0},
            **{"INDE 22": 0.0}, IAN=0.0, IPV=0.0, Cf=0,
            IAA=0.0, IEG=0.0, IPS=0.0, IDA=0.0,
            Matem=0.0, Portug=0.0, **{"Inglês": 0.0},
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_payload_extremos_maximos(self, client):
        """Todos os valores no limite máximo."""
        payload = self._build_simplified_payload(
            Fase=10, **{"Idade 22": 25, "Ano ingresso": 2023, "Nº Av": 10},
            **{"INDE 22": 10.0}, IAN=10.0, IPV=10.0,
            IAA=10.0, IEG=10.0, IPS=10.0, IDA=10.0,
            Matem=10.0, Portug=10.0, **{"Inglês": 10.0},
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ═════════════════════════════════════════════════════════════
#  3. CAMPOS FALTANTES → API deve rejeitar (422)
# ═════════════════════════════════════════════════════════════
class TestCamposFaltantes:
    """Garante que a remoção de qualquer campo obrigatório causa 422."""

    @pytest.mark.parametrize("campo", REQUIRED_FIELDS)
    def test_remover_campo_obrigatorio(self, client, campo):
        """Cada campo obrigatório removido individualmente deve dar 422."""
        payload = {**DEFAULTS}
        del payload[campo]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422, (
            f"Remover '{campo}' deveria dar 422, mas deu {response.status_code}"
        )


# ═════════════════════════════════════════════════════════════
#  4. CAMPOS FORA DO LIMITE → API deve rejeitar (422)
# ═════════════════════════════════════════════════════════════
class TestCamposForaDoLimite:
    """Testa que valores fora dos limites ge/le são rejeitados."""

    @pytest.mark.parametrize("campo,valor", [
        ("Fase", -1),
        ("Fase", 11),
        ("Idade 22", 4),
        ("Idade 22", 26),
        ("Ano ingresso", 2009),
        ("Ano ingresso", 2024),
        ("Nº Av", -1),
        ("Nº Av", 11),
        ("INDE 22", -0.1),
        ("INDE 22", 10.1),
        ("IAN", -0.1),
        ("IAN", 10.1),
        ("IAA", -0.1),
        ("IPV", 10.1),
    ])
    def test_valor_fora_do_limite(self, client, campo, valor):
        """Valores fora do range devem retornar 422."""
        payload = {**DEFAULTS, campo: valor}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422, (
            f"'{campo}'={valor} deveria dar 422, mas deu {response.status_code}"
        )


# ═════════════════════════════════════════════════════════════
#  5. RESPOSTA DA API – formato correto
# ═════════════════════════════════════════════════════════════
class TestRespostaAPI:
    """Valida a estrutura e valores da resposta da API."""

    def test_resposta_contem_todos_campos(self, client):
        """Resposta deve ter exatamente os 5 campos do PredictionResponse."""
        response = client.post("/predict", json=DEFAULTS)
        data = response.json()
        expected_keys = {"defasagem_prevista", "risco", "confianca", "recomendacao", "timestamp"}
        assert set(data.keys()) == expected_keys

    def test_risco_valor_valido(self, client):
        """Risco deve ser um dos 4 níveis."""
        response = client.post("/predict", json=DEFAULTS)
        data = response.json()
        assert data["risco"] in ["Baixo", "Moderado", "Alto", "Crítico"]

    def test_confianca_entre_0_e_1(self, client):
        """Confiança deve estar entre 0 e 1."""
        response = client.post("/predict", json=DEFAULTS)
        data = response.json()
        assert 0 <= data["confianca"] <= 1

    def test_defasagem_tipo_numerico(self, client):
        """Defasagem deve ser numérica."""
        response = client.post("/predict", json=DEFAULTS)
        data = response.json()
        assert isinstance(data["defasagem_prevista"], (int, float))

    def test_timestamp_formato_iso(self, client):
        """Timestamp deve estar em formato ISO."""
        from datetime import datetime
        response = client.post("/predict", json=DEFAULTS)
        data = response.json()
        # Não deve lançar exceção
        datetime.fromisoformat(data["timestamp"])


# ═════════════════════════════════════════════════════════════
#  6. CAMPOS OPCIONAIS (Pedra 20, Pedra 21, Avaliador4, etc.)
# ═════════════════════════════════════════════════════════════
class TestCamposOpcionais:
    """Testa que campos opcionais podem ser None sem quebrar."""

    def test_pedras_anteriores_none(self, client):
        """Pedra 20 e 21 podem ser None (aluno novo)."""
        payload = {**DEFAULTS, "Pedra 20": None, "Pedra 21": None}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_avaliador4_none(self, client):
        """Avaliador4 e Rec Av4 podem ser None."""
        payload = {**DEFAULTS, "Avaliador4": None, "Rec Av4": None}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_ingles_none(self, client):
        """Nota de Inglês pode ser None."""
        payload = {**DEFAULTS, "Inglês": None}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_todos_opcionais_none(self, client):
        """Todos os campos opcionais como None simultaneamente."""
        payload = {
            **DEFAULTS,
            "Pedra 20": None, "Pedra 21": None,
            "Avaliador4": None, "Rec Av4": None,
            "Inglês": None,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ═════════════════════════════════════════════════════════════
#  7. SINCRONIZAÇÃO FRONTEND ↔ API (chaves)
# ═════════════════════════════════════════════════════════════
class TestSincronizacaoChaves:
    """Garante que frontend e API usam exatamente as mesmas chaves."""

    def test_defaults_chaves_reconhecidas_pelo_schema(self):
        """Toda chave dos DEFAULTS deve ser um alias válido no StudentData."""
        schema = StudentData.model_json_schema()
        # Coletar aliases e nomes de campo
        valid_keys = set()
        for prop_name, prop_info in schema.get("properties", {}).items():
            valid_keys.add(prop_name)
            # Pydantic v2 alias aparece como título ou no schema
        # Também aceitar construção via alias
        try:
            StudentData(**DEFAULTS)
            all_valid = True
        except ValidationError as e:
            all_valid = False
            pytest.fail(f"DEFAULTS contém chaves inválidas para o schema: {e}")
        assert all_valid

    def test_nenhuma_chave_extra_nos_defaults(self):
        """DEFAULTS não deve ter chaves que não existam no schema."""
        schema_fields = set()
        for field_name, field_info in StudentData.model_fields.items():
            alias = field_info.alias or field_name
            schema_fields.add(alias)
            schema_fields.add(field_name)

        for key in DEFAULTS:
            assert key in schema_fields, (
                f"Chave '{key}' nos DEFAULTS do frontend não existe no schema da API"
            )
