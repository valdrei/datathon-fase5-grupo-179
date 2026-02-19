"""
Módulo de pré-processamento de dados.
Responsável por limpeza, transformação e preparação dos dados.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pré-processamento dos dados da Passos Mágicos."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Carrega o dataset do CSV.
        
        Args:
            filepath: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados carregados
        """
        logger.info(f"Carregando dados de {filepath}")
        df = pd.read_csv(filepath)
        
        # Substituir vírgulas por pontos em colunas numéricas
        numeric_cols = ['INDE 22', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês', 'IPV', 'IAN']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Converter Cg para numeric
        if 'Cg' in df.columns:
            df['Cg'] = pd.to_numeric(df['Cg'].astype(str).str.replace(',', '.'), errors='coerce')
        
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa os dados removendo valores inválidos e inconsistências.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame limpo
        """
        logger.info("Iniciando limpeza dos dados")
        df_clean = df.copy()
        
        # Remover linhas completamente vazias
        df_clean = df_clean.dropna(how='all')
        
        # Converter booleanos
        bool_cols = ['Indicado', 'Atingiu PV']
        for col in bool_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Sim': 1, 'Não': 0, 'sim': 1, 'não': 0})
        
        # Tratar valores missing em Defas (target)
        if 'Defas' in df_clean.columns:
            # Remover linhas onde Defas é NaN
            df_clean = df_clean.dropna(subset=['Defas'])
        
        logger.info(f"Dados após limpeza: {df_clean.shape[0]} linhas")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores faltantes no dataset.
        
        Args:
            df: DataFrame com valores faltantes
            
        Returns:
            DataFrame com valores faltantes tratados
        """
        logger.info("Tratando valores faltantes")
        df_filled = df.copy()
        
        # Identificar colunas numéricas e categóricas
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_filled.select_dtypes(include=['object']).columns.tolist()
        
        # Preencher numéricas com mediana
        for col in numeric_cols:
            if df_filled[col].isna().sum() > 0:
                median_val = df_filled[col].median()
                if pd.isna(median_val):
                    logger.warning(f"Coluna {col} está totalmente vazia — ignorada")
                    continue
                df_filled[col] = df_filled[col].fillna(median_val)
                logger.info(f"Preenchido {col} com mediana: {median_val}")
        
        # Preencher categóricas com moda
        for col in categorical_cols:
            if df_filled[col].isna().sum() > 0:
                mode_series = df_filled[col].mode()
                if mode_series.empty:
                    logger.warning(f"Coluna {col} está totalmente vazia — ignorada")
                    continue
                mode_val = mode_series[0]
                df_filled[col] = df_filled[col].fillna(mode_val)
                logger.info(f"Preenchido {col} com moda: {mode_val}")
        
        return df_filled
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica features categóricas usando Label Encoding.
        
        Args:
            df: DataFrame com features categóricas
            fit: Se True, ajusta os encoders. Se False, usa encoders já ajustados.
            
        Returns:
            DataFrame com features codificadas
        """
        logger.info("Codificando features categóricas")
        df_encoded = df.copy()
        
        categorical_cols = ['Gênero', 'Instituição de ensino', 'Pedra 20', 'Pedra 21', 
                           'Pedra 22', 'Rec Psicologia', 'Rec Av1', 'Rec Av2', 
                           'Rec Av3', 'Rec Av4', 'Fase ideal', 'Destaque IEG', 
                           'Destaque IDA', 'Destaque IPV']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        # Lidar com valores não vistos durante o treinamento
                        le = self.label_encoders[col]
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        self.categorical_features = [col for col in categorical_cols if col in df_encoded.columns]
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normaliza features numéricas usando StandardScaler.
        
        Args:
            df: DataFrame com features numéricas
            fit: Se True, ajusta o scaler. Se False, usa scaler já ajustado.
            
        Returns:
            DataFrame com features normalizadas
        """
        logger.info("Normalizando features numéricas")
        df_scaled = df.copy()
        
        # Features numéricas para normalizar
        numeric_features = ['INDE 22', 'Cg', 'Cf', 'Ct', 'Nº Av', 'IAA', 'IEG', 
                           'IPS', 'IDA', 'Matem', 'Portug', 'Inglês', 'IPV', 'IAN', 
                           'Idade 22', 'Ano ingresso']
        
        numeric_features = [col for col in numeric_features if col in df_scaled.columns]
        self.numeric_features = numeric_features
        
        if numeric_features:
            if fit:
                df_scaled[numeric_features] = self.scaler.fit_transform(df_scaled[numeric_features])
            else:
                df_scaled[numeric_features] = self.scaler.transform(df_scaled[numeric_features])
        
        return df_scaled
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target.
        
        Args:
            df: DataFrame completo
            
        Returns:
            Tupla (features, target)
        """
        logger.info("Separando features e target")
        
        # Colunas a serem removidas (não são features preditivas)
        cols_to_drop = ['RA', 'Fase', 'Turma', 'Nome', 'Ano nasc', 'Avaliador1', 
                       'Avaliador2', 'Avaliador3', 'Avaliador4', 'Defas']
        
        # Target
        y = df['Defas'].copy()
        
        # Features
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline completo de pré-processamento.
        
        Args:
            df: DataFrame original
            fit: Se True, ajusta transformadores. Se False, usa transformadores já ajustados.
            
        Returns:
            Tupla (features processadas, target)
        """
        logger.info("=== Iniciando pipeline de pré-processamento ===")
        
        # Limpeza
        df_clean = self.clean_data(df)
        
        # Tratar valores faltantes
        df_filled = self.handle_missing_values(df_clean)
        
        # Codificar categóricas
        df_encoded = self.encode_categorical_features(df_filled, fit=fit)
        
        # Separar features e target antes da normalização
        X, y = self.prepare_features_target(df_encoded)
        
        # Normalizar features
        X_scaled = self.scale_features(X, fit=fit)
        
        logger.info("=== Pipeline de pré-processamento concluído ===")
        return X_scaled, y


def preprocess_data(filepath: str, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Função auxiliar para pré-processar dados.
    
    Args:
        filepath: Caminho para o arquivo CSV
        fit: Se True, ajusta transformadores
        
    Returns:
        Tupla (features, target, preprocessor)
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(filepath)
    X, y = preprocessor.preprocess_pipeline(df, fit=fit)
    return X, y, preprocessor
