# 📓 Notebooks - Passos Mágicos

Notebooks Jupyter para análise exploratória do projeto Passos Mágicos.

## Notebooks Disponíveis

| Notebook | Descrição |
|----------|-----------|
| **01_EDA_Analise_Exploratoria.ipynb** | Análise exploratória: estatísticas descritivas, distribuições, correlações, outliers, padrões temporais |
| **02_Qualidade_Dados.ipynb** | Qualidade dos dados: valores ausentes, tipos, inconsistências, duplicatas, recomendações de limpeza |

**Ordem recomendada:** execute o **01** antes do **02**.

## Como Executar

```powershell
# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux/Mac

# Instalar Jupyter (se necessário)
pip install jupyter ipykernel

# Iniciar
jupyter notebook notebooks/
```

## Dependências

- **Datasets:** `data/PEDE2022.csv`, `data/PEDE2023.csv`, `data/PEDE2024.csv`
- **Módulos:** `src/preprocessing.py`, `src/feature_engineering.py`
- **Bibliotecas:** pandas, numpy, matplotlib, seaborn, scikit-learn (já incluídas no `requirements.txt`)
