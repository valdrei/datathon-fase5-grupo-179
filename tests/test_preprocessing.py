"""
Testes unitários para o módulo de pré-processamento.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo."""
    data = {
        'RA': ['RA-1', 'RA-2', 'RA-3'],
        'Fase': [7, 6, 5],
        'Turma': ['A', 'B', 'A'],
        'Nome': ['Aluno-1', 'Aluno-2', 'Aluno-3'],
        'Ano nasc': [2003, 2005, 2006],
        'Idade 22': [19, 17, 16],
        'Gênero': ['Menina', 'Menina', 'Menino'],
        'Ano ingresso': [2016, 2017, 2018],
        'Instituição de ensino': ['Escola Pública', 'Rede Decisão', 'Escola Pública'],
        'Pedra 20': ['Ametista', 'Ametista', 'Quartzo'],
        'Pedra 21': ['Ametista', 'Ágata', 'Ágata'],
        'Pedra 22': ['Quartzo', 'Ametista', 'Ágata'],
        'INDE 22': [5.783, 7.055, 6.591],
        'Cg': [753.0, 469.0, 629.0],
        'Cf': [18, 8, 13],
        'Ct': [10, 3, 6],
        'Nº Av': [4, 4, 4],
        'IAA': [8.3, 8.8, 0.0],
        'IEG': [4.1, 5.2, 7.9],
        'IPS': [5.6, 6.3, 5.6],
        'Rec Psicologia': ['Requer avaliação', 'Sem limitações', 'Sem limitações'],
        'IDA': [4.0, 6.8, 5.6],
        'Matem': [2.7, 6.3, 5.8],
        'Portug': [3.5, 4.5, 4.0],
        'Inglês': [6.0, 9.7, 6.9],
        'Indicado': ['Sim', 'Não', 'Não'],
        'Atingiu PV': ['Não', 'Não', 'Não'],
        'IPV': [7.278, 6.778, 7.556],
        'IAN': [5.0, 10.0, 10.0],
        'Fase ideal': ['Fase 8 (Universitários)', 'Fase 7 (3º EM)', 'Fase 7 (3º EM)'],
        'Defas': [-1, 0, 1],
        'Destaque IEG': ['Melhorar', 'Melhorar', 'Destaque'],
        'Destaque IDA': ['Melhorar', 'Melhorar', 'Melhorar'],
        'Destaque IPV': ['Melhorar', 'Melhorar', 'Destaque']
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Fixture com preprocessor inicializado."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Testes para classe DataPreprocessor."""
    
    def test_init(self, preprocessor):
        """Testa inicialização do preprocessor."""
        assert preprocessor is not None
        assert isinstance(preprocessor.label_encoders, dict)
        assert len(preprocessor.label_encoders) == 0
    
    def test_clean_data(self, preprocessor, sample_data):
        """Testa limpeza de dados."""
        df_clean = preprocessor.clean_data(sample_data)
        
        # Verificar que dados foram limpos
        assert len(df_clean) <= len(sample_data)
        assert 'Defas' in df_clean.columns
        
        # Verificar conversão de booleanos
        assert df_clean['Indicado'].dtype in [np.int64, np.float64]
        assert df_clean['Atingiu PV'].dtype in [np.int64, np.float64]
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Testa tratamento de valores faltantes."""
        # Adicionar alguns NaN
        df_with_nan = sample_data.copy()
        df_with_nan.loc[0, 'IDA'] = np.nan
        df_with_nan.loc[1, 'Matem'] = np.nan
        
        df_filled = preprocessor.handle_missing_values(df_with_nan)
        
        # Verificar que NaNs foram preenchidos
        assert df_filled['IDA'].isna().sum() == 0
        assert df_filled['Matem'].isna().sum() == 0
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Testa encoding de features categóricas."""
        df_encoded = preprocessor.encode_categorical_features(sample_data, fit=True)
        
        # Verificar que features categóricas foram codificadas
        if 'Gênero' in df_encoded.columns:
            assert df_encoded['Gênero'].dtype in [np.int64, np.int32]
        
        # Verificar que encoders foram criados
        assert len(preprocessor.label_encoders) > 0
    
    def test_scale_features(self, preprocessor, sample_data):
        """Testa normalização de features."""
        df_scaled = preprocessor.scale_features(sample_data, fit=True)
        
        # Verificar que features numéricas existem
        assert len(preprocessor.numeric_features) > 0
        
        # Verificar que valores foram escalados
        for col in preprocessor.numeric_features:
            if col in df_scaled.columns:
                mean = df_scaled[col].mean()
                std = df_scaled[col].std()
                # Valores escalados devem ter média próxima de 0 e std próxima de 1
                assert abs(mean) < 1e-10 or abs(std - 1.0) < 0.1
    
    def test_prepare_features_target(self, preprocessor, sample_data):
        """Testa separação de features e target."""
        X, y = preprocessor.prepare_features_target(sample_data)
        
        # Verificar shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Verificar que target está correto
        assert 'Defas' not in X.columns
        assert all(y == sample_data['Defas'])
        
        # Verificar que colunas desnecessárias foram removidas
        assert 'RA' not in X.columns
        assert 'Nome' not in X.columns
    
    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Testa pipeline completo de pré-processamento."""
        X, y = preprocessor.preprocess_pipeline(sample_data, fit=True)
        
        # Verificar outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0
        
        # Verificar que processadores foram ajustados
        assert len(preprocessor.label_encoders) > 0
        assert len(preprocessor.numeric_features) > 0


class TestDataLoading:
    """Testes para carregamento de dados."""
    
    def test_load_data_csv(self, preprocessor, tmp_path):
        """Testa carregamento de arquivo CSV."""
        # Criar arquivo CSV temporário
        csv_file = tmp_path / "test_data.csv"
        data = {
            'INDE 22': ['5,783', '7,055'],
            'IAA': ['8,3', '8,8'],
            'Defas': [-1, 0]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        # Carregar dados
        df_loaded = preprocessor.load_data(str(csv_file))
        
        # Verificar que foi carregado
        assert len(df_loaded) == 2
        assert 'INDE 22' in df_loaded.columns
        
        # Verificar que vírgulas foram convertidas
        assert df_loaded['INDE 22'].dtype == np.float64


def test_preprocess_data_function():
    """Testa função auxiliar preprocess_data."""
    # Este teste requer um arquivo CSV real, então vamos pular se não existir
    csv_path = Path(__file__).parent.parent / "data" / "PEDE2022.csv"
    
    if not csv_path.exists():
        pytest.skip("Arquivo CSV de dados não encontrado")
    
    from preprocessing import preprocess_data
    
    X, y, preprocessor = preprocess_data(str(csv_path), fit=True)
    
    assert X is not None
    assert y is not None
    assert preprocessor is not None
    assert len(X) == len(y)
    assert len(X) > 0