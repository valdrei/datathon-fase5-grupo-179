"""
Testes unitários para frontend/app_streamlit.py.

Testa as funções helper e lógica de negócio do frontend Streamlit,
mockando as dependências de st (streamlit) e requests.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import json

# Inserir raiz do projeto no path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════
#  Fixtures e helpers
# ═══════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def mock_streamlit():
    """
    Mocka o módulo streamlit inteiro antes do import do frontend.
    Isso evita que o Streamlit tente inicializar o servidor.
    """
    mock_st = MagicMock()
    mock_st.set_page_config = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.info = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.success = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.write = MagicMock()
    mock_st.json = MagicMock()
    mock_st.image = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.plotly_chart = MagicMock()
    mock_st.dataframe = MagicMock()
    mock_st.download_button = MagicMock()
    mock_st.button = MagicMock(return_value=False)
    mock_st.rerun = MagicMock()
    mock_st.spinner = MagicMock()
    mock_st.progress = MagicMock()
    mock_st.empty = MagicMock()
    mock_st.expander = MagicMock()

    # sidebar como context manager
    sidebar_mock = MagicMock()
    sidebar_mock.__enter__ = MagicMock(return_value=sidebar_mock)
    sidebar_mock.__exit__ = MagicMock(return_value=False)
    sidebar_mock.radio = MagicMock(return_value="ℹ️ Sobre")
    sidebar_mock.image = MagicMock()
    sidebar_mock.markdown = MagicMock()
    sidebar_mock.success = MagicMock()
    sidebar_mock.error = MagicMock()
    sidebar_mock.caption = MagicMock()
    mock_st.sidebar = sidebar_mock

    # radio retorna "Sobre" para cair na página mais simples
    mock_st.radio = MagicMock(return_value="ℹ️ Sobre")

    # form como context manager
    form_mock = MagicMock()
    form_mock.__enter__ = MagicMock(return_value=form_mock)
    form_mock.__exit__ = MagicMock(return_value=False)
    mock_st.form = MagicMock(return_value=form_mock)

    # columns retorna lista de mocks
    col_mock = MagicMock()
    mock_st.columns = MagicMock(return_value=[col_mock, col_mock, col_mock, col_mock])

    # file uploader
    mock_st.file_uploader = MagicMock(return_value=None)

    # expander como context manager
    exp_mock = MagicMock()
    exp_mock.__enter__ = MagicMock(return_value=exp_mock)
    exp_mock.__exit__ = MagicMock(return_value=False)
    mock_st.expander = MagicMock(return_value=exp_mock)

    # cache_data decorator (passa a função direto)
    mock_st.cache_data = lambda **kwargs: lambda f: f

    # session_state
    mock_st.session_state = {}

    with patch.dict('sys.modules', {'streamlit': mock_st}):
        yield mock_st


@pytest.fixture
def frontend_module(mock_streamlit):
    """Importa o módulo do frontend com streamlit mockado."""
    # Remover do cache se já importou antes para forçar re-execução do código top-level
    for mod_name in list(sys.modules.keys()):
        if 'frontend' in mod_name:
            del sys.modules[mod_name]

    import frontend.app_streamlit as app_st
    return app_st


# ═══════════════════════════════════════════════════════
#  1. Testes das funções helper puras
# ═══════════════════════════════════════════════════════

class TestRiskColor:
    """Testa a função risk_color."""

    def test_risk_color_baixo(self, frontend_module):
        assert frontend_module.risk_color("Baixo") == "#38ef7d"

    def test_risk_color_moderado(self, frontend_module):
        assert frontend_module.risk_color("Moderado") == "#F2C94C"

    def test_risk_color_alto(self, frontend_module):
        assert frontend_module.risk_color("Alto") == "#f45c43"

    def test_risk_color_critico(self, frontend_module):
        assert frontend_module.risk_color("Crítico") == "#6f0000"

    def test_risk_color_desconhecido(self, frontend_module):
        assert frontend_module.risk_color("Inexistente") == "#888"

    def test_risk_color_vazio(self, frontend_module):
        assert frontend_module.risk_color("") == "#888"


class TestRiskEmoji:
    """Testa a função risk_emoji."""

    def test_risk_emoji_baixo(self, frontend_module):
        assert frontend_module.risk_emoji("Baixo") == "✅"

    def test_risk_emoji_moderado(self, frontend_module):
        assert frontend_module.risk_emoji("Moderado") == "⚠️"

    def test_risk_emoji_alto(self, frontend_module):
        assert frontend_module.risk_emoji("Alto") == "🔴"

    def test_risk_emoji_critico(self, frontend_module):
        assert frontend_module.risk_emoji("Crítico") == "🚨"

    def test_risk_emoji_desconhecido(self, frontend_module):
        assert frontend_module.risk_emoji("Qualquer") == "❓"


# ═══════════════════════════════════════════════════════
#  2. Testes de check_api_health
# ═══════════════════════════════════════════════════════

class TestCheckApiHealth:
    """Testa a função check_api_health."""

    def test_api_healthy(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch('requests.get', return_value=mock_response):
            result = frontend_module.check_api_health()
            assert result == {"status": "healthy"}

    def test_api_unhealthy_status(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch('requests.get', return_value=mock_response):
            result = frontend_module.check_api_health()
            assert result is None

    def test_api_connection_error(self, frontend_module):
        import requests as req
        with patch('requests.get', side_effect=req.exceptions.ConnectionError):
            result = frontend_module.check_api_health()
            assert result is None


# ═══════════════════════════════════════════════════════
#  3. Testes de get_model_info
# ═══════════════════════════════════════════════════════

class TestGetModelInfo:
    """Testa a função get_model_info."""

    def test_model_info_success(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model_type": "RandomForest",
            "features_count": 29,
        }

        with patch('requests.get', return_value=mock_response):
            result = frontend_module.get_model_info()
            assert result["model_type"] == "RandomForest"

    def test_model_info_error(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch('requests.get', return_value=mock_response):
            result = frontend_module.get_model_info()
            assert result is None

    def test_model_info_exception(self, frontend_module):
        with patch('requests.get', side_effect=Exception("timeout")):
            result = frontend_module.get_model_info()
            assert result is None


# ═══════════════════════════════════════════════════════
#  4. Testes de make_prediction
# ═══════════════════════════════════════════════════════

class TestMakePrediction:
    """Testa a função make_prediction."""

    def test_prediction_success(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "defasagem_prevista": -1.5,
            "risco": "Alto",
            "confianca": 0.87,
            "recomendacao": "Atenção especial",
        }

        with patch('requests.post', return_value=mock_response):
            result = frontend_module.make_prediction({"IAN": 3.0})
            assert result["defasagem_prevista"] == -1.5
            assert result["risco"] == "Alto"

    def test_prediction_api_error(self, frontend_module):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation Error"

        with patch('requests.post', return_value=mock_response):
            result = frontend_module.make_prediction({"bad": "data"})
            assert result is None

    def test_prediction_connection_error(self, frontend_module):
        import requests as req
        with patch('requests.post', side_effect=req.exceptions.ConnectionError):
            result = frontend_module.make_prediction({})
            assert result is None

    def test_prediction_generic_exception(self, frontend_module):
        with patch('requests.post', side_effect=RuntimeError("boom")):
            result = frontend_module.make_prediction({})
            assert result is None


# ═══════════════════════════════════════════════════════
#  5. Testes de constantes e configuração
# ═══════════════════════════════════════════════════════

class TestFrontendConstants:
    """Testa constantes e configurações do frontend."""

    def test_api_url_default(self, frontend_module):
        assert "localhost" in frontend_module.API_URL or "http" in frontend_module.API_URL

    def test_defaults_dict_exists(self, frontend_module):
        assert hasattr(frontend_module, 'DEFAULTS')
        assert isinstance(frontend_module.DEFAULTS, dict)

    def test_defaults_has_required_keys(self, frontend_module):
        required = ["Fase", "IAN", "IPV", "INDE 22", "IAA", "IEG", "IPS", "IDA"]
        for key in required:
            assert key in frontend_module.DEFAULTS, f"DEFAULTS missing key: {key}"

    def test_defaults_ian_value(self, frontend_module):
        assert isinstance(frontend_module.DEFAULTS["IAN"], (int, float))
        assert 0 <= frontend_module.DEFAULTS["IAN"] <= 10

    def test_defaults_fase_value(self, frontend_module):
        assert isinstance(frontend_module.DEFAULTS["Fase"], int)
        assert 0 <= frontend_module.DEFAULTS["Fase"] <= 10

    def test_defaults_genero(self, frontend_module):
        assert frontend_module.DEFAULTS["Gênero"] in ["Menina", "Menino"]

    def test_defaults_indicadores_numericos(self, frontend_module):
        indicadores = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]
        for ind in indicadores:
            val = frontend_module.DEFAULTS[ind]
            assert isinstance(val, (int, float)), f"{ind} should be numeric"
            assert 0 <= val <= 10, f"{ind}={val} out of [0,10]"

    def test_defaults_pedras(self, frontend_module):
        pedras_validas = ["Quartzo", "Ágata", "Ametista", "Topázio"]
        for key in ["Pedra 20", "Pedra 21", "Pedra 22"]:
            assert frontend_module.DEFAULTS[key] in pedras_validas

    def test_defaults_notas(self, frontend_module):
        for nota in ["Matem", "Portug", "Inglês"]:
            val = frontend_module.DEFAULTS[nota]
            assert 0 <= val <= 10

    def test_icon_path_is_path_object(self, frontend_module):
        assert hasattr(frontend_module, 'ICON_PATH')


# ═══════════════════════════════════════════════════════
#  6. Testes das funções de monitoramento (fetch_*)
#     Nota: fetch_* são definidas dentro de blocos condicionais
#     do streamlit (elif page == "Monitoramento"), portanto são
#     testadas indiretamente via cobertura de import do módulo.
# ═══════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════
#  7. Testes de cobertura de páginas
#     Re-importa o frontend com diferentes pages ativas
#     para cobrir os blocos condicionais (if page == "...")
# ═══════════════════════════════════════════════════════

def _make_col_mock():
    """Cria um mock de coluna Streamlit com widgets que retornam valores reais."""
    col = MagicMock()
    col.slider = MagicMock(return_value=5.0)
    col.number_input = MagicMock(return_value=5)
    col.selectbox = MagicMock(return_value="Ametista")
    col.metric = MagicMock()
    return col


def _mock_columns_factory():
    """Retorna side_effect para st.columns que gera o número correto de colunas."""
    def _columns(n, **kwargs):
        col = _make_col_mock()
        if isinstance(n, int):
            return [col] * n
        elif isinstance(n, (list, tuple)):
            return [col] * len(n)
        return [col] * 4
    return _columns


def _reimport_frontend():
    """Remove o frontend do cache de módulos e re-importa."""
    for mod_name in list(sys.modules.keys()):
        if 'frontend' in mod_name:
            del sys.modules[mod_name]
    import frontend.app_streamlit as app_st
    return app_st


class TestPageCoveragePredicao:
    """Cobre linhas 165-315 (formulário de predição individual + resultado)."""

    def test_prediction_page_with_submission(self, mock_streamlit):
        mock_streamlit.radio.return_value = "🔮 Predição Individual"
        mock_streamlit.form_submit_button = MagicMock(return_value=True)
        mock_streamlit.columns = MagicMock(side_effect=_mock_columns_factory())

        # spinner como context manager
        spinner_ctx = MagicMock()
        spinner_ctx.__enter__ = MagicMock(return_value=spinner_ctx)
        spinner_ctx.__exit__ = MagicMock(return_value=False)
        mock_streamlit.spinner = MagicMock(return_value=spinner_ctx)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "defasagem_prevista": -1.5,
            "risco": "Alto",
            "confianca": 0.87,
            "recomendacao": "Atenção especial",
        }

        with patch('requests.post', return_value=mock_resp), \
             patch('requests.get', return_value=MagicMock(
                 status_code=200,
                 json=MagicMock(return_value={"status": "healthy"}))):
            app_st = _reimport_frontend()

        assert hasattr(app_st, 'DEFAULTS')
        assert mock_streamlit.form.called


class TestPageCoverageDashboard:
    """Cobre linhas 476-519 (dashboard do modelo)."""

    def test_dashboard_page_with_model_info(self, mock_streamlit):
        mock_streamlit.radio.return_value = "📈 Dashboard do Modelo"
        mock_streamlit.columns = MagicMock(side_effect=_mock_columns_factory())

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_type": "RandomForest",
            "features_count": 29,
            "top_features": [
                {"feature": "IAN", "importance": 0.786},
                {"feature": "IPV", "importance": 0.05},
            ],
        }

        with patch('requests.get', return_value=mock_resp):
            app_st = _reimport_frontend()

        assert hasattr(app_st, 'DEFAULTS')


class _SessionState(dict):
    """Dict que suporta acesso por atributo (como o Streamlit real)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


class TestPageCoverageLote:
    """Cobre linhas 322-335 (predição em lote – setup sem upload)."""

    def test_batch_page_no_upload(self, mock_streamlit):
        mock_streamlit.radio.return_value = "📊 Predição em Lote (CSV)"
        mock_streamlit.file_uploader = MagicMock(return_value=None)
        mock_streamlit.session_state = _SessionState()

        with patch('requests.get', return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "healthy"}))):
            app_st = _reimport_frontend()

        assert hasattr(app_st, 'DEFAULTS')


class TestPageCoverageMonitoramento:
    """Cobre linhas 526-717 (monitoramento contínuo)."""

    def test_monitoring_page_with_full_data(self, mock_streamlit):
        mock_streamlit.radio.return_value = "🛡️ Monitoramento"
        mock_streamlit.cache_data = lambda **kwargs: lambda f: f
        mock_streamlit.button = MagicMock(return_value=False)
        mock_streamlit.columns = MagicMock(side_effect=_mock_columns_factory())

        stats = {
            "total_predictions": 50,
            "mean_prediction": -0.5,
            "std_prediction": 0.3,
            "mean_confidence": 0.85,
            "min_prediction": -2.0,
            "max_prediction": 1.0,
            "last_prediction_time": "2025-01-01",
            "risk_distribution": {
                "Baixo": 20, "Moderado": 15,
                "Alto": 10, "Crítico": 5,
            },
        }
        preds = {
            "predictions": [
                {"timestamp": f"2025-01-{(i//12)+1:02d}T{i%12+8:02d}:00:00",
                 "prediction": -0.5 + i * 0.1,
                 "confidence": 0.85}
                for i in range(20)
            ]
        }
        drift = {
            "prediction_drift": {
                "total_predictions": 20,
                "drift_detected": False,
                "first_half_mean": -0.5,
                "second_half_mean": -0.4,
                "mean_shift": 0.1,
                "first_half_std": 0.3,
                "second_half_std": 0.25,
                "ks_pvalue": 0.8,
            },
            "drift_enabled": True,
            "psi_thresholds": {
                "no_change": "< 0.10",
                "moderate": "0.10 - 0.25",
                "significant": "> 0.25",
            },
        }

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "monitoring/stats" in url:
                resp.json.return_value = stats
            elif "monitoring/predictions" in url:
                resp.json.return_value = preds
            elif "monitoring/drift" in url:
                resp.json.return_value = drift
            else:
                resp.json.return_value = {"status": "healthy"}
            return resp

        with patch('requests.get', side_effect=mock_get):
            app_st = _reimport_frontend()

        assert hasattr(app_st, 'DEFAULTS')

    def test_monitoring_page_no_data(self, mock_streamlit):
        mock_streamlit.radio.return_value = "🛡️ Monitoramento"
        mock_streamlit.cache_data = lambda **kwargs: lambda f: f
        mock_streamlit.button = MagicMock(return_value=False)

        import requests as _req

        def mock_get_no_monitoring(url, **kwargs):
            if "health" in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"status": "healthy"}
                return resp
            raise _req.exceptions.ConnectionError("no api")

        with patch('requests.get', side_effect=mock_get_no_monitoring):
            app_st = _reimport_frontend()

        assert hasattr(app_st, 'DEFAULTS')
