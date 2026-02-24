# 🎓 Passos Mágicos - Predição de Defasagem Escolar

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Projeto de Machine Learning Engineering desenvolvido para a **Associação Passos Mágicos** com o objetivo de prever o risco de defasagem escolar de estudantes, auxiliando na identificação precoce de alunos que necessitam de intervenções pedagógicas.

## 📋 Índice

- [Visão Geral do Projeto](#-visão-geral-do-projeto)
- [Stack Tecnológica](#-stack-tecnológica)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [API e Deploy](#-api-e-deploy)
- [Frontend Streamlit](#-frontend-streamlit)
- [Testes](#-testes)
- [Monitoramento](#-monitoramento)
- [Exemplos de Uso](#-exemplos-de-uso)
- [Métricas e Confiabilidade](#-métricas-e-confiabilidade)

---

## 🎯 Visão Geral do Projeto

### Objetivo

Desenvolver um modelo preditivo capaz de estimar o **risco de defasagem escolar** de cada estudante da Associação Passos Mágicos, permitindo intervenções pedagógicas personalizadas e preventivas.

### Solução Proposta

Pipeline completa de Machine Learning com:
- ✅ Pré-processamento robusto de dados
- ✅ Engenharia de features avançada
- ✅ Treinamento com busca de hiperparâmetros
- ✅ API REST para predições em tempo real
- ✅ Frontend interativo com Streamlit (predição individual, lote com cancelamento, dashboard, monitoramento)
- ✅ Dockerização para deploy
- ✅ Testes unitários e de integração (>80% cobertura)
- ✅ Monitoramento contínuo com detecção de drift (KS + PSI) via API e Streamlit

### Problema de Negócio

A **defasagem escolar** é calculada como a diferença entre a **fase atual** do aluno e sua **fase ideal**. Valores negativos indicam que o aluno está atrasado em relação ao esperado para sua idade/nível.

**Exemplo:**
- Aluno na Fase 7, mas deveria estar na Fase 8 → Defasagem = -1 (Risco Moderado)
- Aluno na Fase 5, mas deveria estar na Fase 7 → Defasagem = -2 (Risco Alto)

---

## 🛠 Stack Tecnológica

### Core
- **Linguagem:** Python 3.12
- **Framework ML:** scikit-learn 1.3.2
- **Data Processing:** pandas 2.1.4, numpy 1.26.3

### API e Deploy
- **Framework API:** FastAPI 0.109.0
- **Servidor ASGI:** Uvicorn 0.27.0
- **Serialização:** joblib 1.3.2
- **Containerização:** Docker & Docker Compose

### Frontend
- **Interface:** Streamlit
- **Gráficos Interativos:** Plotly
- **Comunicação API:** requests / httpx

### Testes e Qualidade
- **Framework de Testes:** pytest 7.4.4
- **Cobertura:** pytest-cov 4.1.0
- **Cliente HTTP:** httpx 0.26.0

### Monitoramento
- **Logging:** Python logging + custom PredictionLogger
- **Drift Detection:** Kolmogorov-Smirnov test (scipy), PSI (Population Stability Index)
- **Visualização:** Plotly (frontend), matplotlib 3.8.2, seaborn 0.13.1

---

## 📁 Estrutura do Projeto

```
Fase 5/
│
├── app/                                # Aplicação FastAPI
│   ├── __init__.py                     # Inicialização do pacote
│   ├── main.py                         # Ponto de entrada da API (startup, carregamento de modelo)
│   ├── routes.py                       # Endpoints REST (predição, monitoramento)
│   └── model/                          # Artefatos do modelo treinado (*.pkl, *.csv, *.json)
│
├── frontend/                           # Interface Streamlit
│   └── app_streamlit.py                # Dashboard interativo (predição, lote, monitoramento)
│
├── src/                                # Código-fonte do pipeline ML
│   ├── __init__.py                     # Inicialização do pacote
│   ├── preprocessing.py                # Pré-processamento de dados
│   ├── feature_engineering.py          # Engenharia de features
│   ├── train.py                        # Pipeline de treinamento
│   ├── evaluate.py                     # Avaliação de modelos
│   ├── monitoring.py                   # Monitoramento e drift detection
│   └── utils.py                        # Funções utilitárias
│
├── tests/                              # Testes unitários e de integração
│   ├── conftest.py                     # Configuração e fixtures do pytest
│   ├── test_api.py                     # Testes dos endpoints da API
│   ├── test_frontend.py                # Testes de integração frontend ↔ API
│   ├── test_model.py                   # Testes do modelo e pipeline de treino
│   ├── test_monitoring.py              # Testes de monitoramento e drift detection
│   └── test_preprocessing.py           # Testes de pré-processamento
│
├── notebooks/                          # Notebooks exploratórios
│   ├── 01_EDA_Analise_Exploratoria.ipynb
│   └── 02_Qualidade_Dados.ipynb
│
├── data/                               # Datasets PEDE (2022-2024)
│   ├── PEDE2022.csv / PEDE2022_clean.csv
│   ├── PEDE2023.csv / PEDE2023_clean.csv
│   └── PEDE2024.csv / PEDE2024_clean.csv
│
├── config/                             # Configurações
│   └── config.yaml                     # Configurações do projeto
│
├── logs/                               # Logs e monitoramento
│   └── predictions.jsonl               # Log de predições (JSONL)
│
├── Dockerfile                          # Dockerfile da aplicação
├── docker-compose.yml                  # Docker Compose
├── render.yaml                         # Configuração de deploy (Render)
├── requirements.txt                    # Dependências Python
├── setup.ps1                           # Script de setup (Windows)
├── start.sh                            # Script de inicialização (Linux)
├── exportar_excel.py                   # Utilitário para exportar dados Excel → CSV
├── exemplo_aluno.json                  # Exemplo de payload para predição
├── test_api_local.py                   # Testes manuais da API
└── README.md                           # Esta documentação
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.12+
- pip ou conda
- Docker (opcional, para containerização)
- Git

### Instalação Local

#### 1. Clone o repositório

```bash
git clone https://github.com/valdrei/datathon-fase5-grupo-179.git
cd datathon-fase5-grupo-179
```

#### 2. Crie e ative um ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Instale as dependências

```powershell
pip install -r requirements.txt
```

#### 4. Prepare os dados

Exporte as abas do Excel para CSVs na pasta `data/`:
```powershell
python exportar_excel.py
```

Isso gerará 3 arquivos em `data/`:
```
data/PEDE2022.csv
data/PEDE2023.csv
data/PEDE2024.csv
```

#### 5. Treine o modelo

```powershell
python -m src.train
```

Ou especifique o caminho do dataset:
```powershell
python src/train.py "data/PEDE2022.csv"
```

---

## 🔄 Pipeline de Machine Learning

### 1. Pré-processamento de Dados

**Módulo:** `src/preprocessing.py`

**Etapas:**
- ✅ Limpeza de dados (remoção de valores inválidos)
- ✅ Conversão de tipos (vírgulas → pontos em números)
- ✅ Tratamento de valores faltantes (mediana/moda)
- ✅ Codificação de variáveis categóricas (Label Encoding)
- ✅ Normalização de features numéricas (StandardScaler)

**Código exemplo:**
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.load_data('dados.csv')
X, y = preprocessor.preprocess_pipeline(df, fit=True)
```

### 2. Engenharia de Features

**Módulo:** `src/feature_engineering.py`

**Features Criadas:**

1. **Evolução de Pedras** (classificação por desempenho)
   - Evolução 2020-2021, 2021-2022, Total
   
2. **Indicadores Agregados**
   - Média/Desvio padrão dos indicadores (IAA, IEG, IPS, IDA, IPV, IAN)
   - Performance acadêmica (Matemática, Português, Inglês)
   
3. **Features Temporais**
   - Anos na instituição
   - Idade de ingresso
   
4. **Rankings e Percentis**
   - Diferenças entre rankings (Geral, Fase, Turma)
   - Percentis de classificação
   
5. **Recomendações Agregadas**
   - Contagem de recomendações positivas
   - Indicador de atenção psicológica
   
6. **Interações**
   - INDE × Idade
   - Performance acadêmica × Engajamento
   - Anos na instituição × INDE

**Total:** ~25+ features adicionais criadas

### 3. Treinamento e Validação

**Módulo:** `src/train.py`

**Algoritmos Disponíveis:**
- ✅ **Random Forest** (padrão - melhor performance)
- ✅ Gradient Boosting
- ✅ Ridge Regression
- ✅ Lasso Regression

**Processo:**
1. Split treino/teste (80/20)
2. Grid Search com validação cruzada (5-fold)
3. Treinamento do modelo com melhores hiperparâmetros
4. Avaliação em conjunto de teste
5. Validação cruzada final
6. Salvamento do modelo e metadados

**Hiperparâmetros otimizados (Random Forest):**
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

### 4. Seleção de Modelo

**Modelo Escolhido:** Random Forest Regressor

**Justificativa:**
- ✅ Lida bem com features heterogêneas
- ✅ Resistente a overfitting
- ✅ Fornece feature importance
- ✅ Não requer normalização estrita
- ✅ Bom desempenho em datasets tabulares

### 5. Avaliação

**Módulo:** `src/evaluation.py`

**Métricas Utilizadas:**
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error) - Métrica principal
- **MAE** (Mean Absolute Error)
- **R² Score**
- **MAPE** (Mean Absolute Percentage Error)
- **Acurácia com tolerância** (±0.5 fases)

**Análise por Classe de Defasagem:**
- Muito Atrasado (<-2)
- Atrasado (-2 a -1)
- Adequado (0)
- Adiantado (1)
- Muito Adiantado (>1)

---

## 🌐 API e Deploy

> ⚠️ **Importante:** Certifique-se de que o ambiente virtual está ativo antes de executar qualquer comando abaixo.
> - **Windows:** `.\venv\Scripts\Activate.ps1`
> - **Linux/Mac:** `source venv/bin/activate`

### Executar API Localmente

```powershell
# Método 1: Usando uvicorn diretamente
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Método 2: Executando o módulo
python -m app.main
```

A API estará disponível em: **http://localhost:8000**

### Executar Frontend Streamlit

```powershell
streamlit run frontend/app_streamlit.py
```

O frontend estará disponível em: **http://localhost:8501**

**Páginas disponíveis:**

| Página | Descrição |
|---|---|
| 🔮 Predição Individual | Formulário interativo para prever defasagem de um aluno |
| 📊 Predição em Lote (CSV) | Upload de CSV para predições em massa, com botão de cancelamento |
| 📈 Dashboard do Modelo | Informações do modelo, feature importance, métricas |
| 🛡️ Monitoramento | Estatísticas em tempo real, distribuição de risco, detecção de drift |
| ℹ️ Sobre | Informações sobre o projeto e a Associação Passos Mágicos |

### Documentação Interativa

FastAPI gera automaticamente documentação interativa:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints Disponíveis

#### 1. **GET /** - Informações da API
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "API Passos Mágicos - Predição de Defasagem Escolar",
  "version": "1.0.0",
  "status": "online",
  "endpoints": {
    "/predict": "POST - Fazer predição de defasagem",
    "/health": "GET - Verificar saúde da API",
    "/model-info": "GET - Informações sobre o modelo",
    "/monitoring/stats": "GET - Estatísticas de predições",
    "/monitoring/predictions": "GET - Histórico de predições",
    "/monitoring/drift": "GET - Relatório de drift"
  }
}
```

#### 2. **GET /health** - Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "feature_engineer_loaded": true,
  "timestamp": "2024-01-29T10:30:00"
}
```

#### 3. **GET /model-info** - Informações do Modelo
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_type": "RandomForestRegressor",
  "features_count": 50,
  "top_features": [
    {"feature": "INDE_22", "importance": 0.152},
    {"feature": "IDA", "importance": 0.098},
    ...
  ],
  "timestamp": "2024-01-29T10:30:00"
}
```

#### 4. **POST /predict** - Fazer Predição

**Exemplo com curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "defasagem_prevista": -1.2,
  "risco": "Alto",
  "confianca": 0.87,
  "recomendacao": "Aluno necessita de acompanhamento intensivo. Considerar tutoria e reforço escolar.",
  "timestamp": "2024-01-29T10:30:00"
}
```

**Exemplo com Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Fase": 7,
    "Turma": "A",
    "Idade 22": 19,
    # ... outros campos
}

response = requests.post(url, json=data)
result = response.json()

print(f"Defasagem: {result['defasagem_prevista']}")
print(f"Risco: {result['risco']}")
print(f"Confiança: {result['confianca']}")
print(f"Recomendação: {result['recomendacao']}")
```

**Exemplo com Postman:**
1. Criar nova requisição POST
2. URL: `http://localhost:8000/predict`
3. Headers: `Content-Type: application/json`
4. Body (raw JSON): Copiar exemplo acima

#### 5. **GET /monitoring/stats** - Estatísticas de Predições
```bash
curl http://localhost:8000/monitoring/stats
# Com parâmetro opcional: ?last_n=50
```

**Response:**
```json
{
  "total_predictions": 150,
  "mean_prediction": -0.85,
  "std_prediction": 1.23,
  "min_prediction": -3.5,
  "max_prediction": 2.1,
  "risk_distribution": {
    "Baixo": 45,
    "Moderado": 60,
    "Alto": 30,
    "Crítico": 15
  }
}
```

#### 6. **GET /monitoring/predictions** - Histórico de Predições
```bash
curl http://localhost:8000/monitoring/predictions?last_n=10
```

**Response:**
```json
{
  "total": 10,
  "predictions": [
    {
      "timestamp": "2026-02-11T19:30:00",
      "prediction": -1.2,
      "risk": "Alto",
      "confidence": 0.87
    }
  ]
}
```

#### 7. **GET /monitoring/drift** - Relatório de Drift
```bash
curl http://localhost:8000/monitoring/drift
```

**Response:**
```json
{
  "total_predictions": 150,
  "prediction_drift": {
    "first_half_mean": -0.92,
    "second_half_mean": -0.78,
    "ks_statistic": 0.12,
    "ks_pvalue": 0.34,
    "drift_detected": false
  },
  "psi": {
    "value": 0.08,
    "status": "Estável",
    "thresholds": {
      "stable": "< 0.1",
      "moderate_change": "0.1 - 0.25",
      "significant_change": "> 0.25"
    }
  }
}
```

### Deploy com Docker

#### Construir a imagem

```powershell
docker build -t passos-magicos-api:latest .
```

#### Executar container

```powershell
docker run -d \
  -p 8000:8000 \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/logs:/app/logs \
  --name passos-magicos-api \
  passos-magicos-api:latest
```

#### Usar Docker Compose (recomendado)

```powershell
# Iniciar serviços
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviços
docker-compose down
```

#### Verificar container

```powershell
# Status
docker ps

# Logs
docker logs passos-magicos-api

# Health check
curl http://localhost:8000/health
```

### Deploy em Nuvem

#### Opção 1: Heroku

```bash
# Login no Heroku
heroku login

# Criar app
heroku create passos-magicos-api

# Deploy
git push heroku main

# Verificar
heroku open
```

#### Opção 2: AWS (EC2 + ECR)

```bash
# Build e push para ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag passos-magicos-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/passos-magicos-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/passos-magicos-api:latest

# Deploy em EC2 com docker-compose
```

#### Opção 3: Google Cloud Run

```bash
# Build e push
gcloud builds submit --tag gcr.io/<project-id>/passos-magicos-api

# Deploy
gcloud run deploy passos-magicos-api \
  --image gcr.io/<project-id>/passos-magicos-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🖥 Frontend Streamlit

O frontend é uma aplicação **Streamlit** que consome a API FastAPI e oferece uma interface visual completa para interagir com o modelo.

### Como Executar

```powershell
# Terminal 1 — API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
streamlit run frontend/app_streamlit.py
```

### Páginas

#### 🔮 Predição Individual
Formulário interativo com todos os campos do aluno. Após submissão, exibe:
- Gauge de defasagem prevista
- Classificação de risco com cor
- Confiança do modelo
- Recomendação pedagógica personalizada
- Radar chart dos indicadores do aluno

#### 📊 Predição em Lote (CSV)
- Upload de arquivo CSV com múltiplos alunos
- Processamento linha a linha com barra de progresso
- **Botão de cancelamento** para interromper o processamento a qualquer momento
- Download dos resultados em CSV

#### 📈 Dashboard do Modelo
- Informações do modelo carregado
- Feature importance (gráfico de barras)
- Métricas de avaliação

#### 🛡️ Monitoramento
- Métricas em tempo real (total de predições, média, desvio padrão)
- Gráfico de pizza da distribuição de risco
- Timeline temporal das predições com área de desvio
- Detecção de drift via histograma comparativo (1ª vs 2ª metade)
- Resultado do teste KS e indicador PSI
- Cache de 60 segundos para performance

#### ℹ️ Sobre
Informações sobre o projeto e a Associação Passos Mágicos.

---

## 🧪 Testes

### Executar Todos os Testes

```powershell
pytest tests/ -v
```

### Executar com Cobertura

```powershell
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

### Ver Relatório de Cobertura

```powershell
# Abrir relatório HTML
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux
```

### Executar Testes Específicos

```powershell
# Apenas testes de preprocessing
pytest tests/test_preprocessing.py -v

# Apenas testes de API
pytest tests/test_api.py -v

# Apenas testes de frontend
pytest tests/test_frontend.py -v

# Apenas testes de monitoramento
pytest tests/test_monitoring.py -v

# Teste específico
pytest tests/test_preprocessing.py::TestDataPreprocessor::test_clean_data -v
```

### Estrutura de Testes

**tests/test_preprocessing.py:**
- Testes de carregamento de dados
- Testes de limpeza e tratamento de missing values
- Testes de encoding e scaling
- Testes de pipeline completo

**tests/test_api.py:**
- Testes de endpoints (root, health, model-info, predict)
- Testes de validação de dados
- Testes de classificação de risco
- Testes de geração de recomendações

**tests/test_frontend.py:**
- Validação de payloads (defaults, campos obrigatórios, limites)
- Testes de integração frontend ↔ API
- Navegação entre páginas do Streamlit
- Funcionalidade de cancelamento de lote
- Endpoints de monitoramento via cliente frontend

**tests/test_monitoring.py:**
- PredictionLogger (logging, estatísticas, métricas)
- DriftDetector (KS test, PSI, distribuições)
- ModelMonitor (degradação de modelo)
- Endpoints de monitoramento (/monitoring/stats, /predictions, /drift)
- Cenários com dados vazios

**tests/test_model.py:**
- Treinamento e persistência do modelo
- Validação cruzada
- Limites de predição

**Cobertura Esperada:** >80%

---

## 📊 Monitoramento

O projeto possui monitoramento contínuo acessível via **3 endpoints da API** e uma **página dedicada no Streamlit**.

### Logging de Predições

Todas as predições são automaticamente registradas em `logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-02-11T19:30:00",
  "prediction": -1.2,
  "confidence": 0.87,
  "risk": "Alto",
  "input_data": {
    "INDE_22": 5.783,
    "IDA": 4.0,
    "IEG": 4.1,
    "Fase": 7,
    "Idade_22": 19
  }
}
```

### Detecção de Drift

O sistema detecta automaticamente drift em duas dimensões:

1. **Kolmogorov-Smirnov Test**: Compara distribuições de predições (1ª metade vs 2ª metade)
2. **Population Stability Index (PSI)**: Monitora mudanças nas distribuições

**Thresholds:**
- KS test: p-value < 0.05 → Drift detectado
- PSI < 0.1 → Estável | 0.1–0.25 → Mudança moderada | >0.25 → Mudança significativa

### Dashboard no Streamlit (Monitoramento)

A página **🛡️ Monitoramento** no frontend exibe em tempo real:

- **Métricas Resumidas:** total de predições, defasagem média, desvio padrão, mín/máx
- **Distribuição de Risco:** gráfico de pizza com categorias Baixo/Moderado/Alto/Crítico
- **Timeline de Predições:** gráfico temporal com linha de média e área de desvio
- **Detecção de Drift:** histograma comparativo (1ª vs 2ª metade) com resultado do KS test
- **PSI:** indicador de estabilidade da distribuição de predições

Os dados são atualizados automaticamente a cada 60 segundos (cache TTL).

### Endpoints de Monitoramento

| Endpoint | Método | Descrição |
|---|---|---|
| `/monitoring/stats` | GET | Estatísticas agregadas de predições |
| `/monitoring/predictions` | GET | Histórico de predições (parâmetro `last_n`) |
| `/monitoring/drift` | GET | Relatório de drift com KS test e PSI |

### Uso Programático

```python
from src.monitoring import PredictionLogger, DriftDetector

# Estatísticas
logger = PredictionLogger(log_dir='./logs')
stats = logger.get_prediction_statistics(last_n=100)
print(f"Total de predições: {stats['total_predictions']}")
print(f"Defasagem média: {stats['mean_prediction']:.2f}")
print(f"Distribuição de risco: {stats['risk_distribution']}")

# Drift detection
detector = DriftDetector(reference_file='data/reference_data.csv')
drift = detector.detect_drift(new_data)
psi = detector.monitor_psi(new_data)
```

---

## 📈 Métricas e Confiabilidade

### Métricas do Modelo

**Métricas Principais:**
- **RMSE (Root Mean Squared Error)**: Erro médio em unidades de fase
- **MAE (Mean Absolute Error)**: Erro absoluto médio
- **R² Score**: Capacidade explicativa do modelo (0-1)
- **MAPE**: Erro percentual médio
- **Acurácia (tolerância ±0.5)**: % de predições dentro da tolerância

**Critérios de Aceitação para Produção:**
- ✅ R² ≥ 0.6 (boa capacidade explicativa)
- ✅ MAE ≤ 0.6 (erro médio aceitável)
- ✅ Acurácia ≥ 50% (dentro da tolerância)

### Avaliação de Confiabilidade

O sistema fornece uma avaliação automática:

**ALTA CONFIANÇA:**
- R² ≥ 0.7
- MAE ≤ 0.4
- Acurácia ≥ 70%
- ✅ Recomendado para produção

**CONFIANÇA MODERADA:**
- R² ≥ 0.5
- MAE ≤ 0.6
- Acurácia ≥ 50%
- ⚠️ Usar com cautela, monitorar

**REQUER MELHORIAS:**
- Métricas abaixo dos limites
- ✗ Não recomendado para produção

### Por que o Modelo é Confiável?

1. **Validação Cruzada**: 5-fold CV garante generalização
2. **Feature Engineering Robusto**: 25+ features relevantes
3. **Hiperparâmetros Otimizados**: Grid Search encontra melhor configuração
4. **Análise por Classe**: Modelo funciona bem em diferentes níveis de defasagem
5. **Monitoramento Contínuo**: Detecção automática de degradação

---

## 💻 Exemplos de Uso

### Exemplo 1: Treinar Modelo do Zero

```python
from src.train import train_pipeline

# Treinar modelo
trainer, preprocessor, engineer = train_pipeline(
    data_path="data/PEDE2022.csv",
    model_name='random_forest',
    test_size=0.2,
    tune_hyperparameters=True,
    output_dir='models'
)

print("Modelo treinado e salvo!")
```

### Exemplo 2: Fazer Predição Offline

```python
import joblib
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# Carregar modelo e processadores
model = joblib.load('models/model_random_forest_latest.pkl')
preprocessor = joblib.load('models/preprocessor_latest.pkl')
engineer = joblib.load('models/feature_engineer_latest.pkl')

# Preparar dados
new_data = pd.DataFrame([{
    'Fase': 7,
    'INDE 22': 5.8,
    'IDA': 4.0,
    # ... outros campos
}])

# Processar
df_eng = engineer.engineer_features(new_data)
X, _ = preprocessor.prepare_features_target(df_eng)
X_processed = preprocessor.scale_features(X, fit=False)

# Prever
prediction = model.predict(X_processed)[0]
print(f"Defasagem prevista: {prediction:.2f}")
```

### Exemplo 3: Avaliar Modelo Existente

```python
from src.evaluation import evaluate_model
import joblib

# Carregar modelo
model = joblib.load('models/model_random_forest_latest.pkl')

# Carregar dados de teste (você precisa preparar X_test e y_test)
# ... código de preparação ...

# Avaliar
metrics, confidence_msg = evaluate_model(model, X_test, y_test)
print(confidence_msg)
```

### Exemplo 4: Monitorar Drift

```python
from src.monitoring import DriftDetector
import pandas as pd

# Inicializar detector
detector = DriftDetector(
    reference_file='data/reference_data.csv',
    threshold=0.05
)

# Dados novos
new_data = pd.read_csv('novos_dados.csv')

# Detectar drift
drift_detected = detector.detect_drift(new_data)
if drift_detected:
    print("⚠️ DRIFT DETECTADO! Considere retreinar o modelo.")
else:
    print("✅ Sem drift detectado.")

# Calcular PSI por feature
psi_values = detector.monitor_psi(new_data)
for feature, psi in psi_values.items():
    print(f"{feature}: PSI = {psi:.4f}")
```

---

## 👥 Autores e Contribuidores

**Projeto desenvolvido por:**
- FIAP - Pós Tech - Turma de Machine Learning Engineering

**Instituição Parceira:**
- Associação Passos Mágicos

---

## 📄 Licença

Este projeto é licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

---

## 🙏 Agradecimentos

Agradecemos à **Associação Passos Mágicos** por disponibilizar os dados e pela missão inspiradora de transformar vidas através da educação.

---

## 📞 Suporte

Para dúvidas, sugestões ou problemas:
- Abra uma issue no repositório
- Entre em contato com a equipe

---

## 🔄 Próximos Passos

**Concluídos recentemente:**
- [x] Dashboard interativo com Streamlit (5 páginas)
- [x] Monitoramento contínuo com detecção de drift (KS + PSI)
- [x] Endpoints REST de monitoramento
- [x] Predição em lote via CSV com cancelamento
- [x] Testes de monitoramento e integração frontend (122+ testes)

**Pendentes:**
- [ ] Implementar ensemble de modelos
- [ ] Adicionar explicabilidade (SHAP values)
- [ ] CI/CD com GitHub Actions
- [ ] Retreinamento automático
- [ ] A/B Testing de modelos

---

**Última atualização:** Fevereiro 2026

**Status do Projeto:** ✅ Completo e Pronto para Produção
