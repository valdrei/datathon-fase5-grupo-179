# 📓 Guia dos Notebooks - Passos Mágicos

## 📚 Notebooks Disponíveis

Este diretório contém notebooks Jupyter para análise exploratória e experimentação do projeto Passos Mágicos.

---

## 📋 Lista de Notebooks

### 1. **01_EDA_Analise_Exploratoria.ipynb**
**Análise Exploratória de Dados (EDA)**

**Objetivo**: Compreensão inicial e profunda do dataset

**Conteúdo**:
- 📊 Carregamento e visualização inicial dos dados
- 📈 Estatísticas descritivas completas
- 🎯 Análise da variável target (Defasagem)
- 📉 Distribuições de features numéricas e categóricas
- 🔗 Análise de correlações
- 👥 Performance por demographics (gênero, idade, instituição)
- ⏱️ Padrões temporais (evolução ao longo do tempo)
- ⚠️ Detecção de outliers

**Quando usar**: Início do projeto, para entender os dados

---

### 2. **02_Qualidade_Dados.ipynb**
**Análise de Qualidade de Dados**

**Objetivo**: Avaliar e documentar a qualidade do dataset

**Conteúdo**:
- 🔍 Identificação de valores ausentes (missing values)
- ✅ Validação de tipos de dados
- 🚫 Detecção de inconsistências
- 🔄 Análise de duplicatas
- 📊 Identificação de outliers
- 💡 Recomendações de limpeza
- 🧹 Aplicação do DataPreprocessor

**Quando usar**: Antes de iniciar o preprocessing, para planejar estratégias de limpeza

---

### 3. **03_Feature_Analysis.ipynb** *(A criar)*
**Análise de Features**

**Objetivo**: Análise detalhada das features criadas

**Conteúdo**:
- 🔧 Aplicação do FeatureEngineer
- 📊 Análise de features derivadas
- 🎯 Importância de features (feature importance)
- 🔗 Análise de multicolinearidade
- ✂️ Seleção de features
- 📈 Comparação: features originais vs derivadas

**Quando usar**: Após feature engineering, antes do treinamento

---

### 4. **04_Model_Experiments.ipynb** *(A criar)*
**Experimentos de Modelagem**

**Objetivo**: Testar diferentes modelos e hiperparâmetros

**Conteúdo**:
- 🤖 Comparação de algoritmos (RF, GBM, Ridge, Lasso)
- 🔍 Grid Search e tuning de hiperparâmetros
- 📊 Validação cruzada
- 📈 Curvas de aprendizado
- ⚖️ Análise de bias-variance
- 🎯 Métricas de performance

**Quando usar**: Durante a fase de modelagem e experimentação

---

### 5. **05_Results_Analysis.ipynb** *(A criar)*
**Análise de Resultados**

**Objetivo**: Analisar resultados do modelo treinado

**Conteúdo**:
- 📊 Métricas finais do modelo
- 📉 Análise de resíduos
- 🎯 Predições vs Valores reais
- 👥 Performance por segmentos
- ⚠️ Análise de erros
- 💡 Insights e recomendações

**Quando usar**: Após treinamento do modelo final

---

## 🚀 Como Usar os Notebooks

### 1. Configuração do Ambiente

```powershell
# Navegar até a pasta do projeto
cd "c:\Users\Drei\OneDrive\Documentos\Pós_FIAP\Fase 5"

# Ativar ambiente virtual (se houver)
# .\venv\Scripts\activate

# Instalar Jupyter se necessário
pip install jupyter notebook ipykernel

# Iniciar Jupyter Notebook
jupyter notebook
```

### 2. Ordem Recomendada

Execute os notebooks nesta ordem para melhor compreensão:

1. **01_EDA_Analise_Exploratoria.ipynb** → Entender os dados
2. **02_Qualidade_Dados.ipynb** → Avaliar qualidade
3. **03_Feature_Analysis.ipynb** → Analisar features (a criar)
4. **04_Model_Experiments.ipynb** → Experimentar modelos (a criar)
5. **05_Results_Analysis.ipynb** → Analisar resultados (a criar)

### 3. Dependências dos Notebooks

Todos os notebooks dependem de:
- ✅ **Datasets**: `data/PEDE2022.csv`, `data/PEDE2023.csv`, `data/PEDE2024.csv`
- ✅ **Módulos**: `src/preprocessing.py`, `src/feature_engineering.py`, etc.
- ✅ **Bibliotecas**: pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 📦 Bibliotecas Necessárias

```python
# Análise de dados
pandas>=2.1.4
numpy>=1.26.3

# Visualização
matplotlib>=3.8.2
seaborn>=0.13.1
plotly>=5.18.0

# Machine Learning
scikit-learn>=1.3.2

# Jupyter
jupyter>=1.0.0
ipykernel>=6.28.0

# Extras para análise
missingno>=0.5.2  # Visualização de missing values
```

Instalar todas:
```powershell
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter ipykernel missingno
```

---

## 🎯 Objetivos de Cada Notebook

| Notebook | Objetivo Principal | Output Esperado |
|----------|-------------------|-----------------|
| 01_EDA | Entender distribuições e padrões | Insights sobre os dados |
| 02_Qualidade | Identificar problemas de qualidade | Plano de limpeza |
| 03_Feature_Analysis | Validar features criadas | Lista de features importantes |
| 04_Model_Experiments | Encontrar melhor modelo | Modelo otimizado |
| 05_Results_Analysis | Avaliar performance final | Relatório de resultados |

---

## 📊 Estrutura de Visualizações

Cada notebook segue um padrão visual:

### Títulos e Seções
```markdown
# 📊 Título Principal
## Seção
### Subseção
```

### Código Python
```python
# Comentários claros
resultado = funcao()
print(f"✅ Resultado: {resultado}")
```

### Visualizações
- **Cores consistentes**: Paleta `viridis`, `coolwarm`, `Reds_r`
- **Tamanho padrão**: `figsize=(12, 6)` ou `(14, 6)`
- **Títulos informativos**: Sempre com `fontsize=14, fontweight='bold'`

---

## 💡 Dicas de Uso

### 1. Executar Células
- **Run Cell**: `Ctrl + Enter`
- **Run and Next**: `Shift + Enter`
- **Run All**: Menu → Cell → Run All

### 2. Salvar Visualizações
```python
# Salvar figura
plt.savefig('../outputs/grafico.png', dpi=300, bbox_inches='tight')
```

### 3. Exportar Resultados
```python
# Exportar para CSV
df_results.to_csv('../outputs/resultados.csv', index=False)

# Exportar para Excel
df_results.to_excel('../outputs/resultados.xlsx', index=False)
```

### 4. Limpar Outputs
```powershell
# Limpar outputs de todos os notebooks
jupyter nbconvert --clear-output --inplace *.ipynb
```

---

## 🔧 Troubleshooting

### Problema: Módulo não encontrado
```python
import sys
sys.path.append('../src')  # Adicionar path dos módulos
```

### Problema: Dataset não encontrado
```python
# Verificar path relativo
from pathlib import Path
data_dir = Path('../data')
print(f"PEDE2022 existe: {(data_dir / 'PEDE2022.csv').exists()}")
print(f"PEDE2023 existe: {(data_dir / 'PEDE2023.csv').exists()}")
print(f"PEDE2024 existe: {(data_dir / 'PEDE2024.csv').exists()}")
```

### Problema: Kernel morreu
- Reduzir tamanho das visualizações
- Usar `plt.close()` após cada plot
- Reiniciar kernel: Menu → Kernel → Restart

---

## 📝 Template de Notebook

Estrutura recomendada para novos notebooks:

```markdown
# 📊 Título do Notebook

## Objetivo
Descrever o objetivo do notebook

## Conteúdo
- Item 1
- Item 2

---
```

```python
# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('../src')

# 2. Configurações
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')

# 3. Carregar dados
from pathlib import Path
data_dir = Path('../data')
datasets = {
    '2022': pd.read_csv(data_dir / 'PEDE2022.csv'),
    '2023': pd.read_csv(data_dir / 'PEDE2023.csv'),
    '2024': pd.read_csv(data_dir / 'PEDE2024.csv'),
}

# 4. Análises...
```

---

## 🎓 Próximos Notebooks a Criar

- [ ] **03_Feature_Analysis.ipynb** - Análise de features
- [ ] **04_Model_Experiments.ipynb** - Experimentos de modelos
- [ ] **05_Results_Analysis.ipynb** - Análise de resultados
- [ ] **06_Model_Interpretation.ipynb** - Interpretabilidade (SHAP, LIME)
- [ ] **07_Production_Tests.ipynb** - Testes para produção

---

## 📚 Recursos Adicionais

### Documentação
- [Pandas Docs](https://pandas.pydata.org/docs/)
- [Matplotlib Docs](https://matplotlib.org/stable/contents.html)
- [Seaborn Docs](https://seaborn.pydata.org/)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)

### Tutoriais de EDA
- [Kaggle EDA Tutorial](https://www.kaggle.com/learn/data-visualization)
- [Towards Data Science - EDA](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)

---

**Versão**: 1.0  
**Data**: 30/01/2026  
**Status**: 2 notebooks criados, 3 pendentes
