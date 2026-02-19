"""
Módulo de engenharia de features.
Cria novas features relevantes para melhorar o desempenho do modelo.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe para criação de features avançadas."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_pedra_evolution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de evolução das pedras ao longo dos anos.
        
        Args:
            df: DataFrame com colunas Pedra 20, Pedra 21, Pedra 22
            
        Returns:
            DataFrame com novas features
        """
        logger.info("Criando features de evolução de pedras")
        df_new = df.copy()
        
        # Mapear pedras para valores numéricos
        pedra_map = {
            'Quartzo': 1,
            'Ágata': 2, 'Agata': 2,
            'Ametista': 3,
            'Topázio': 4
        }
        
        for col in ['Pedra 20', 'Pedra 21', 'Pedra 22']:
            if col in df_new.columns:
                col_numeric = col.replace(' ', '_') + '_num'
                df_new[col_numeric] = df_new[col].astype(str).map(pedra_map)
                df_new[col_numeric] = df_new[col_numeric].fillna(0)
        
        # Evolução de pedras
        if 'Pedra_20_num' in df_new.columns and 'Pedra_21_num' in df_new.columns:
            df_new['evolucao_pedra_20_21'] = df_new['Pedra_21_num'] - df_new['Pedra_20_num']
        
        if 'Pedra_21_num' in df_new.columns and 'Pedra_22_num' in df_new.columns:
            df_new['evolucao_pedra_21_22'] = df_new['Pedra_22_num'] - df_new['Pedra_21_num']
        
        if 'Pedra_20_num' in df_new.columns and 'Pedra_22_num' in df_new.columns:
            df_new['evolucao_pedra_total'] = df_new['Pedra_22_num'] - df_new['Pedra_20_num']
        
        return df_new
    
    def create_performance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria indicadores de performance combinados.
        
        Args:
            df: DataFrame com indicadores individuais
            
        Returns:
            DataFrame com novos indicadores
        """
        logger.info("Criando indicadores de performance combinados")
        df_new = df.copy()
        
        # Média dos indicadores principais
        indicator_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']
        available_cols = [col for col in indicator_cols if col in df_new.columns]
        
        if available_cols:
            df_new['media_indicadores'] = df_new[available_cols].mean(axis=1)
            df_new['std_indicadores'] = df_new[available_cols].std(axis=1)
            df_new['min_indicadores'] = df_new[available_cols].min(axis=1)
            df_new['max_indicadores'] = df_new[available_cols].max(axis=1)
        
        # Performance acadêmica
        academic_cols = ['Matem', 'Portug', 'Inglês']
        available_academic = [col for col in academic_cols if col in df_new.columns]
        
        if available_academic:
            df_new['media_academica'] = df_new[available_academic].mean(axis=1)
            df_new['std_academica'] = df_new[available_academic].std(axis=1)
        
        # Ratio de engajamento vs aprendizagem
        if 'IEG' in df_new.columns and 'IDA' in df_new.columns:
            df_new['ratio_engajamento_aprendizagem'] = df_new['IEG'] / (df_new['IDA'] + 0.001)
        
        # Ratio de performance vs adequação
        if 'IDA' in df_new.columns and 'IAN' in df_new.columns:
            df_new['ratio_aprendizagem_adequacao'] = df_new['IDA'] / (df_new['IAN'] + 0.001)
        
        return df_new
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais.
        
        Args:
            df: DataFrame com ano de ingresso e idade
            
        Returns:
            DataFrame com features temporais
        """
        logger.info("Criando features temporais")
        df_new = df.copy()
        
        # Anos na instituição (calculado em relação a 2022)
        if 'Ano ingresso' in df_new.columns:
            df_new['anos_na_instituicao'] = 2022 - df_new['Ano ingresso']
        
        # Idade de ingresso
        if 'Idade 22' in df_new.columns and 'anos_na_instituicao' in df_new.columns:
            df_new['idade_ingresso'] = df_new['Idade 22'] - df_new['anos_na_instituicao']
        
        return df_new
    
    def create_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em rankings.
        
        Args:
            df: DataFrame com rankings Cg, Cf, Ct
            
        Returns:
            DataFrame com features de ranking
        """
        logger.info("Criando features de ranking")
        df_new = df.copy()
        
        # Performance relativa nos rankings
        if 'Cg' in df_new.columns and 'Cf' in df_new.columns:
            df_new['diff_ranking_geral_fase'] = df_new['Cg'] - df_new['Cf']
        
        if 'Cf' in df_new.columns and 'Ct' in df_new.columns:
            df_new['diff_ranking_fase_turma'] = df_new['Cf'] - df_new['Ct']
        
        # Percentil de ranking (inverso porque menor ranking é melhor)
        if 'Cg' in df_new.columns:
            df_new['percentil_ranking_geral'] = df_new['Cg'].rank(pct=True, ascending=False)
        
        return df_new
    
    def create_recommendation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em recomendações dos avaliadores.
        
        Args:
            df: DataFrame com recomendações
            
        Returns:
            DataFrame com features de recomendação
        """
        logger.info("Criando features de recomendações")
        df_new = df.copy()
        
        # Contagem de recomendações positivas
        rec_cols = ['Rec Av1', 'Rec Av2', 'Rec Av3', 'Rec Av4']
        available_recs = [col for col in rec_cols if col in df_new.columns]
        
        if available_recs:
            # Criar uma matriz de recomendações
            for col in available_recs:
                # Simplificar recomendações em categorias
                df_new[f'{col}_positiva'] = df_new[col].astype(str).apply(
                    lambda x: 1 if 'Promovido' in x or 'Bolsa' in x else 0
                )
            
            # Contagem total de recomendações positivas
            positive_cols = [f'{col}_positiva' for col in available_recs]
            df_new['total_rec_positivas'] = df_new[positive_cols].sum(axis=1)
            df_new['prop_rec_positivas'] = df_new['total_rec_positivas'] / len(available_recs)
        
        # Feature de recomendação psicológica
        if 'Rec Psicologia' in df_new.columns:
            df_new['requer_atencao_psico'] = df_new['Rec Psicologia'].astype(str).apply(
                lambda x: 1 if 'Requer' in x or 'requer' in x else 0
            )
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis importantes.
        
        Args:
            df: DataFrame com features existentes
            
        Returns:
            DataFrame com features de interação
        """
        logger.info("Criando features de interação")
        df_new = df.copy()
        
        # Interação INDE com idade
        if 'INDE 22' in df_new.columns and 'Idade 22' in df_new.columns:
            df_new['inde_x_idade'] = df_new['INDE 22'] * df_new['Idade 22']
        
        # Interação performance acadêmica com engajamento
        if 'media_academica' in df_new.columns and 'IEG' in df_new.columns:
            df_new['academica_x_engajamento'] = df_new['media_academica'] * df_new['IEG']
        
        # Interação anos na instituição com performance
        if 'anos_na_instituicao' in df_new.columns and 'INDE 22' in df_new.columns:
            df_new['anos_x_inde'] = df_new['anos_na_instituicao'] * df_new['INDE 22']
        
        return df_new
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de engenharia de features.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame com features engenheiradas
        """
        logger.info("=== Iniciando pipeline de feature engineering ===")
        
        df_engineered = df.copy()
        
        # Aplicar todas as transformações
        df_engineered = self.create_pedra_evolution(df_engineered)
        df_engineered = self.create_performance_indicators(df_engineered)
        df_engineered = self.create_temporal_features(df_engineered)
        df_engineered = self.create_ranking_features(df_engineered)
        df_engineered = self.create_recommendation_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        
        # Armazenar nomes das novas features
        original_cols = df.columns.tolist()
        new_cols = [col for col in df_engineered.columns if col not in original_cols]
        self.feature_names = new_cols
        
        logger.info(f"Features criadas: {len(new_cols)}")
        logger.info(f"Features: {new_cols}")
        logger.info("=== Pipeline de feature engineering concluído ===")
        
        return df_engineered


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Função auxiliar para aplicar feature engineering.
    
    Args:
        df: DataFrame original
        
    Returns:
        Tupla (DataFrame com features, FeatureEngineer)
    """
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    return df_engineered, engineer
