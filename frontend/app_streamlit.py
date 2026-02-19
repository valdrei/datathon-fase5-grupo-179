"""
Frontend Streamlit – Instituto Passos Mágicos
Consome a API FastAPI para predição de defasagem escolar.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ─────────────────────── CONFIG ───────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Passos Mágicos – Predição de Defasagem",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── CSS ───────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .risk-baixo { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .risk-moderado { background: linear-gradient(135deg, #F2994A, #F2C94C); }
    .risk-alto { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .risk-critico { background: linear-gradient(135deg, #200122, #6f0000); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── HELPERS ───────────────────────
def check_api_health() -> dict | None:
    """Verifica se a API está saudável."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.ConnectionError:
        return None
    return None


def get_model_info() -> dict | None:
    """Obtém informações do modelo via API."""
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def make_prediction(payload: dict) -> dict | None:
    """Envia dados para a API e retorna a predição."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"Erro na API: {r.status_code} – {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar à API. Verifique se ela está rodando.")
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
    return None


def risk_color(risk: str) -> str:
    """Retorna cor CSS para o nível de risco."""
    return {
        "Baixo": "#38ef7d",
        "Moderado": "#F2C94C",
        "Alto": "#f45c43",
        "Crítico": "#6f0000",
    }.get(risk, "#888")


def risk_emoji(risk: str) -> str:
    return {
        "Baixo": "✅",
        "Moderado": "⚠️",
        "Alto": "🔴",
        "Crítico": "🚨",
    }.get(risk, "❓")


# ─────────────────────── SIDEBAR ───────────────────────
with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2023/04/logo-passos-magicos.png", width=220)
    st.markdown("---")

    page = st.radio(
        "Navegação",
        ["🔮 Predição Individual", "📊 Predição em Lote (CSV)", "📈 Dashboard do Modelo", "ℹ️ Sobre"],
        index=0,
    )

    st.markdown("---")
    # Status da API
    health = check_api_health()
    if health and health.get("status") == "healthy":
        st.success("API Online ✅")
    else:
        st.error("API Offline ❌")
        st.caption(f"URL: {API_URL}")


# ─────────────────────── DEFAULTS (medianas/modas do PEDE2022) ─────────────────
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


# ═══════════════════════════════════════════════════════
#  PÁGINA 1 – PREDIÇÃO INDIVIDUAL
# ═══════════════════════════════════════════════════════
if page == "🔮 Predição Individual":
    st.markdown('<p class="main-header">🔮 Predição de Defasagem Escolar</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Instituto Passos Mágicos – Preencha os dados essenciais do aluno</p>', unsafe_allow_html=True)

    st.info(
        "💡 **Formulário simplificado:** apenas os campos mais relevantes para a predição. "
        "O indicador **IAN** (Adequação ao Nível) sozinho responde por **78,6%** da predição. "
        "Os demais campos são preenchidos automaticamente com valores medianos do PEDE 2022."
    )

    pedras = ["Quartzo", "Ágata", "Ametista", "Topázio"]
    generos = ["Menina", "Menino"]
    sim_nao = ["Sim", "Não"]
    fases_ideais = [
        "Fase 1 (4º ano)", "Fase 2 (5º e 6º ano)", "Fase 3 (6º e 7º ano)",
        "Fase 4 (7º e 8º ano)", "Fase 5 (8º e 9º ano)", "Fase 6 (Ensino Médio)",
        "Fase 7 (Ensino Médio)", "Fase 8 (Universitários)",
    ]

    with st.form("prediction_form"):
        # ── Campos principais (alta importância) ──
        st.subheader("🎯 Dados Essenciais")
        st.caption("Estes campos são os que mais impactam a predição do modelo.")

        c1, c2 = st.columns(2)
        ian = c1.slider("IAN – Adequação ao Nível ⭐", 0.0, 10.0, 5.0, 0.1,
                         help="Feature #1 (78,6% de importância). Mede quão adequado o aluno está ao nível esperado.")
        ipv = c2.slider("IPV – Ponto de Virada", 0.0, 10.0, 7.3, 0.1,
                         help="Indicador de Ponto de Virada (1,0%)")

        c3, c4, c5 = st.columns(3)
        fase = c3.number_input("Fase Atual", 0, 10, 2)
        idade = c4.number_input("Idade (em 2022)", 5, 25, 12)
        fase_ideal = c5.selectbox("Fase Ideal", fases_ideais, index=1)

        c6, c7 = st.columns(2)
        cf = c6.number_input("Cf (Ranking na Fase)", 0, 500, 67,
                              help="Feature #7 (1,2%). Posição do aluno no ranking da fase.")
        inde = c7.number_input("INDE (Índice de Desenv.)", 0.0, 10.0, 7.2, step=0.1)

        n_av = st.number_input("Nº de Avaliações", 0, 10, 3)

        # ── Campos opcionais (expander) ──
        with st.expander("📋 Campos Adicionais (opcionais – preenchidos automaticamente se vazios)"):
            st.caption("Altere apenas se tiver os dados reais do aluno.")

            st.markdown("**Indicadores de Performance**")
            ca, cb, cc, cd = st.columns(4)
            iaa = ca.number_input("IAA", 0.0, 10.0, DEFAULTS["IAA"], step=0.1)
            ieg = cb.number_input("IEG", 0.0, 10.0, DEFAULTS["IEG"], step=0.1)
            ips = cc.number_input("IPS", 0.0, 10.0, DEFAULTS["IPS"], step=0.1)
            ida = cd.number_input("IDA", 0.0, 10.0, DEFAULTS["IDA"], step=0.1)

            st.markdown("**Notas Acadêmicas**")
            ce, cf2, cg2 = st.columns(3)
            matem = ce.number_input("Matemática", 0.0, 10.0, DEFAULTS["Matem"], step=0.1)
            portug = cf2.number_input("Português", 0.0, 10.0, DEFAULTS["Portug"], step=0.1)
            ingles = cg2.number_input("Inglês", 0.0, 10.0, DEFAULTS["Inglês"], step=0.1)

            st.markdown("**Dados Básicos**")
            ch, ci = st.columns(2)
            genero = ch.selectbox("Gênero", generos)
            ano_ingresso = ci.number_input("Ano de Ingresso", 2010, 2023, DEFAULTS["Ano ingresso"])

            st.markdown("**Classificação (Pedras)**")
            cj, ck, cl = st.columns(3)
            pedra_20 = cj.selectbox("Pedra 2020", pedras, index=2)
            pedra_21 = ck.selectbox("Pedra 2021", pedras, index=2)
            pedra_22 = cl.selectbox("Pedra 2022", pedras, index=2)

            st.markdown("**Rankings**")
            cm, cn = st.columns(2)
            cg_val = cm.number_input("Cg (Classif. Geral)", 0.0, 2000.0, DEFAULTS["Cg"], step=1.0)
            ct = cn.number_input("Ct (Classif. Turma)", 0, 500, DEFAULTS["Ct"])

            indicado = st.selectbox("Indicado p/ Bolsa", sim_nao, index=1)
            atingiu_pv = st.selectbox("Atingiu Ponto de Virada", sim_nao, index=1)

        submitted = st.form_submit_button("🔮 Realizar Predição", use_container_width=True, type="primary")

    if submitted:
        payload = {**DEFAULTS}  # começa com defaults
        # Sobrescreve com campos essenciais
        payload.update({
            "IAN": ian, "IPV": ipv, "Fase": fase, "Idade 22": idade,
            "Fase ideal": fase_ideal, "Cf": cf, "INDE 22": inde, "Nº Av": n_av,
        })
        # Sobrescreve com campos avançados (se alterados)
        payload.update({
            "IAA": iaa, "IEG": ieg, "IPS": ips, "IDA": ida,
            "Matem": matem, "Portug": portug, "Inglês": ingles,
            "Gênero": genero, "Ano ingresso": ano_ingresso,
            "Pedra 20": pedra_20, "Pedra 21": pedra_21, "Pedra 22": pedra_22,
            "Cg": cg_val, "Ct": ct,
            "Indicado": indicado, "Atingiu PV": atingiu_pv,
        })

        with st.spinner("Consultando modelo..."):
            result = make_prediction(payload)

        if result:
            st.markdown("---")
            st.subheader("📊 Resultado da Predição")

            r1, r2, r3 = st.columns(3)
            r1.metric("Defasagem Prevista", f"{result['defasagem_prevista']:.2f}")
            r2.metric("Nível de Risco", f"{risk_emoji(result['risco'])} {result['risco']}")
            r3.metric("Confiança", f"{result['confianca']:.0%}")

            # Gauge de risco
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["defasagem_prevista"],
                title={"text": "Defasagem Prevista"},
                gauge={
                    "axis": {"range": [-4, 2]},
                    "bar": {"color": risk_color(result["risco"])},
                    "steps": [
                        {"range": [-4, -2], "color": "#ffcccc"},
                        {"range": [-2, -1], "color": "#ffe0b2"},
                        {"range": [-1, 0], "color": "#fff9c4"},
                        {"range": [0, 2], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": result["defasagem_prevista"],
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Radar dos indicadores
            indicators = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]
            values = [iaa, ieg, ips, ida, ipv, ian]
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=indicators + [indicators[0]],
                fill="toself",
                fillcolor="rgba(102, 126, 234, 0.3)",
                line=dict(color="#667eea"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                title="Perfil do Aluno – Indicadores",
                height=400,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Recomendação
            st.info(f"**Recomendação:** {result['recomendacao']}")


# ═══════════════════════════════════════════════════════
#  PÁGINA 2 – PREDIÇÃO EM LOTE
# ═══════════════════════════════════════════════════════
elif page == "📊 Predição em Lote (CSV)":
    st.markdown('<p class="main-header">📊 Predição em Lote</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Faça upload de um CSV com múltiplos alunos para predição simultânea</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV (mesmo formato do PEDE2022)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} alunos carregados**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Processar Predições", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Processando...")

            for i, row in df.iterrows():
                payload = row.to_dict()
                # Converter NaN para None
                payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}

                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                    if r.status_code == 200:
                        pred = r.json()
                        results.append({
                            "Aluno": payload.get("Nome", f"Aluno {i+1}"),
                            "Defasagem": pred["defasagem_prevista"],
                            "Risco": pred["risco"],
                            "Confiança": pred["confianca"],
                            "Recomendação": pred["recomendacao"],
                        })
                    else:
                        results.append({
                            "Aluno": payload.get("Nome", f"Aluno {i+1}"),
                            "Defasagem": None, "Risco": "Erro",
                            "Confiança": None, "Recomendação": f"Erro: {r.status_code}",
                        })
                except Exception as e:
                    results.append({
                        "Aluno": payload.get("Nome", f"Aluno {i+1}"),
                        "Defasagem": None, "Risco": "Erro",
                        "Confiança": None, "Recomendação": str(e),
                    })

                progress.progress((i + 1) / len(df), text=f"Processando {i+1}/{len(df)}...")

            progress.empty()
            df_results = pd.DataFrame(results)

            st.subheader("📋 Resultados")
            st.dataframe(df_results, use_container_width=True)

            # Distribuição de risco
            if "Risco" in df_results.columns:
                fig_risk = px.pie(
                    df_results, names="Risco",
                    title="Distribuição de Risco",
                    color="Risco",
                    color_discrete_map={
                        "Baixo": "#38ef7d", "Moderado": "#F2C94C",
                        "Alto": "#f45c43", "Crítico": "#6f0000", "Erro": "#999",
                    },
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            # Download
            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Resultados (CSV)", csv_out, "predicoes_passos_magicos.csv", "text/csv")


# ═══════════════════════════════════════════════════════
#  PÁGINA 3 – DASHBOARD DO MODELO
# ═══════════════════════════════════════════════════════
elif page == "📈 Dashboard do Modelo":
    st.markdown('<p class="main-header">📈 Dashboard do Modelo</p>', unsafe_allow_html=True)

    model_info = get_model_info()

    if model_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tipo do Modelo", model_info.get("model_type", "N/A"))
        c2.metric("Nº de Features", model_info.get("features_count", "N/A"))
        c3.metric("R² Score", "0.8591")

        st.markdown("---")

        # Métricas do treinamento
        st.subheader("📊 Métricas de Treinamento")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", "0.8591", help="Coeficiente de determinação no conjunto de teste")
        m2.metric("RMSE", "0.3065", help="Root Mean Squared Error")
        m3.metric("MAE", "0.2099", help="Mean Absolute Error")
        m4.metric("CV R²", "0.5491 ± 0.16", help="R² médio da validação cruzada (5 folds)")

        st.subheader("🏆 Top 10 Features Mais Importantes")
        if "top_features" in model_info:
            df_feat = pd.DataFrame(model_info["top_features"])
            fig_feat = px.bar(
                df_feat, x="importance", y="feature",
                orientation="h", title="Feature Importance",
                color="importance", color_continuous_scale="Viridis",
            )
            fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Informações de feature importance não disponíveis via API.")

        # Hiperparâmetros
        st.subheader("⚙️ Hiperparâmetros (GridSearchCV)")
        params = {
            "max_depth": 15,
            "min_samples_leaf": 2,
            "min_samples_split": 5,
            "n_estimators": 200,
        }
        st.json(params)
    else:
        st.warning("Não foi possível conectar à API para obter informações do modelo.")


# ═══════════════════════════════════════════════════════
#  PÁGINA 4 – SOBRE
# ═══════════════════════════════════════════════════════
elif page == "ℹ️ Sobre":
    st.markdown('<p class="main-header">ℹ️ Sobre o Projeto</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 🏫 Instituto Passos Mágicos

    O **Instituto Passos Mágicos** é uma organização que atua na transformação da vida de
    crianças e jovens em situação de vulnerabilidade social através da educação de qualidade.

    ### 🎯 Objetivo do Projeto

    Este projeto foi desenvolvido como parte do **Datathon – Pós-Graduação FIAP (Fase 5)**.
    O objetivo é criar um modelo preditivo capaz de identificar alunos com risco de
    **defasagem escolar**, permitindo intervenções preventivas.

    ### 🔬 Metodologia

    1. **Análise Exploratória** dos dados PEDE (Pesquisa Extensiva do Desempenho Educacional)
    2. **Feature Engineering** com 29 variáveis derivadas (evolução de pedras, indicadores combinados, rankings)
    3. **Modelo Random Forest** com otimização de hiperparâmetros via GridSearchCV
    4. **API REST** (FastAPI) para servir predições em produção
    5. **Frontend interativo** (Streamlit) para visualização dos resultados

    ### 📊 Dados

    | Dataset | Alunos | Variáveis |
    |---------|--------|-----------|
    | PEDE 2022 | 860 | 42 |
    | PEDE 2023 | 1.014 | 48 |
    | PEDE 2024 | 1.156 | 50 |

    ### 🛠️ Stack Tecnológico

    - **Python 3.12** – Linguagem principal
    - **scikit-learn** – Treinamento do modelo
    - **FastAPI** – API REST
    - **Streamlit** – Frontend
    - **Docker** – Containerização
    - **Render** – Deploy em nuvem

    ### 👥 Equipe

    Pós-Graduação em Data Analytics – FIAP 2024/2026
    """)

    st.markdown("---")
    st.caption("Passos Mágicos © 2026 – Projeto acadêmico FIAP")
