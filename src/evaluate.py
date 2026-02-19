"""
Módulo de avaliação de modelos.
Fornece funções para avaliar e visualizar performance de modelos.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe para avaliar modelos de regressão."""
    
    def __init__(self):
        self.predictions = None
        self.actuals = None
        self.metrics = {}
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de avaliação para modelos de regressão.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            Dicionário com métricas calculadas
        """
        logger.info("Calculando métricas de avaliação")
        
        # Métricas básicas
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE - tratando divisão por zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf
        
        # Métricas adicionais
        # Erro médio
        mean_error = np.mean(y_pred - y_true)
        
        # Erro padrão
        std_error = np.std(y_pred - y_true)
        
        # Percentis de erro absoluto
        abs_errors = np.abs(y_pred - y_true)
        percentile_50 = np.percentile(abs_errors, 50)
        percentile_75 = np.percentile(abs_errors, 75)
        percentile_90 = np.percentile(abs_errors, 90)
        
        # Acurácia para tolerância (ex: predições dentro de ±0.5 da fase real)
        tolerance = 0.5
        accuracy_within_tolerance = np.mean(abs_errors <= tolerance) * 100
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'mean_error': mean_error,
            'std_error': std_error,
            'median_abs_error': percentile_50,
            'p75_abs_error': percentile_75,
            'p90_abs_error': percentile_90,
            'accuracy_within_tolerance': accuracy_within_tolerance
        }
        
        self.predictions = y_pred
        self.actuals = y_true
        
        return self.metrics
    
    def print_metrics(self) -> None:
        """Imprime métricas de forma formatada."""
        if not self.metrics:
            logger.warning("Nenhuma métrica calculada. Execute calculate_metrics() primeiro.")
            return
        
        print("\n" + "=" * 60)
        print("MÉTRICAS DE AVALIAÇÃO DO MODELO")
        print("=" * 60)
        print(f"MSE (Mean Squared Error):              {self.metrics['mse']:.4f}")
        print(f"RMSE (Root Mean Squared Error):        {self.metrics['rmse']:.4f}")
        print(f"MAE (Mean Absolute Error):             {self.metrics['mae']:.4f}")
        print(f"R² Score:                              {self.metrics['r2']:.4f}")
        print(f"MAPE (Mean Absolute % Error):          {self.metrics['mape']:.2f}%")
        print(f"\nErro Médio:                            {self.metrics['mean_error']:.4f}")
        print(f"Desvio Padrão do Erro:                 {self.metrics['std_error']:.4f}")
        print(f"\nMediana do Erro Absoluto:              {self.metrics['median_abs_error']:.4f}")
        print(f"75º Percentil do Erro Absoluto:        {self.metrics['p75_abs_error']:.4f}")
        print(f"90º Percentil do Erro Absoluto:        {self.metrics['p90_abs_error']:.4f}")
        print(f"\nAcurácia (tolerância ±0.5):            {self.metrics['accuracy_within_tolerance']:.2f}%")
        print("=" * 60 + "\n")
    
    def analyze_predictions_by_class(self, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Analisa predições agrupadas por classe de defasagem.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            DataFrame com análise por classe
        """
        logger.info("Analisando predições por classe de defasagem")
        
        df_analysis = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'error': y_pred - y_true,
            'abs_error': np.abs(y_pred - y_true)
        })
        
        # Criar categorias de defasagem
        df_analysis['defasagem_categoria'] = pd.cut(
            df_analysis['actual'],
            bins=[-np.inf, -2, -1, 0, 1, np.inf],
            labels=['Muito Atrasado (<-2)', 'Atrasado (-2 a -1)', 
                   'Adequado (0)', 'Adiantado (1)', 'Muito Adiantado (>1)']
        )
        
        # Análise por categoria
        analysis_by_class = df_analysis.groupby('defasagem_categoria').agg({
            'actual': 'count',
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'median', 'max']
        }).round(4)
        
        analysis_by_class.columns = ['count', 'mean_error', 'std_error', 
                                     'mean_abs_error', 'median_abs_error', 'max_abs_error']
        
        print("\n" + "=" * 80)
        print("ANÁLISE DE PREDIÇÕES POR CLASSE DE DEFASAGEM")
        print("=" * 80)
        print(analysis_by_class)
        print("=" * 80 + "\n")
        
        return analysis_by_class
    
    def get_model_confidence_message(self) -> str:
        """
        Gera mensagem sobre a confiabilidade do modelo.
        
        Returns:
            String com avaliação da confiabilidade
        """
        if not self.metrics:
            return "Métricas não calculadas."
        
        r2 = self.metrics['r2']
        mae = self.metrics['mae']
        accuracy = self.metrics['accuracy_within_tolerance']
        
        confidence_level = ""
        reasoning = []
        
        # Avaliar R²
        if r2 >= 0.8:
            reasoning.append(f"R² de {r2:.3f} indica excelente capacidade explicativa")
        elif r2 >= 0.6:
            reasoning.append(f"R² de {r2:.3f} indica boa capacidade explicativa")
        elif r2 >= 0.4:
            reasoning.append(f"R² de {r2:.3f} indica capacidade explicativa moderada")
        else:
            reasoning.append(f"R² de {r2:.3f} indica capacidade explicativa limitada")
        
        # Avaliar MAE
        if mae <= 0.3:
            reasoning.append(f"MAE de {mae:.3f} indica alta precisão nas predições")
        elif mae <= 0.5:
            reasoning.append(f"MAE de {mae:.3f} indica precisão aceitável")
        else:
            reasoning.append(f"MAE de {mae:.3f} sugere espaço para melhorias")
        
        # Avaliar acurácia com tolerância
        if accuracy >= 80:
            reasoning.append(f"{accuracy:.1f}% das predições estão dentro da tolerância")
        elif accuracy >= 60:
            reasoning.append(f"{accuracy:.1f}% das predições estão dentro da tolerância (aceitável)")
        else:
            reasoning.append(f"Apenas {accuracy:.1f}% das predições estão dentro da tolerância")
        
        # Determinar nível de confiança geral
        if r2 >= 0.7 and mae <= 0.4 and accuracy >= 70:
            confidence_level = "ALTA CONFIANÇA"
        elif r2 >= 0.5 and mae <= 0.6 and accuracy >= 50:
            confidence_level = "CONFIANÇA MODERADA"
        else:
            confidence_level = "REQUER MELHORIAS"
        
        message = f"""
{'=' * 80}
AVALIAÇÃO DE CONFIABILIDADE DO MODELO: {confidence_level}
{'=' * 80}

Análise:
{chr(10).join(f'  • {r}' for r in reasoning)}

Recomendação para Produção:
"""
        
        if confidence_level == "ALTA CONFIANÇA":
            message += """  ✓ O modelo está PRONTO para produção
  ✓ Métricas indicam alto desempenho e confiabilidade
  ✓ Recomenda-se monitoramento contínuo para detectar degradação
"""
        elif confidence_level == "CONFIANÇA MODERADA":
            message += """  ⚠ O modelo pode ser colocado em produção com CAUTELA
  ⚠ Recomenda-se:
    - Validação adicional com especialistas do domínio
    - Monitoramento rigoroso das predições
    - Coleta de feedback para retreinamento
"""
        else:
            message += """  ✗ O modelo NÃO é recomendado para produção
  ✗ Ações necessárias:
    - Revisão da engenharia de features
    - Teste de modelos alternativos
    - Coleta de mais dados ou features relevantes
    - Investigação de overfitting/underfitting
"""
        
        message += "=" * 80 + "\n"
        
        return message


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  show_plots: bool = False) -> Tuple[Dict[str, float], str]:
    """
    Função auxiliar para avaliar um modelo.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        show_plots: Se deve gerar gráficos (requer ambiente gráfico)
        
    Returns:
        Tupla (métricas, mensagem de confiança)
    """
    evaluator = ModelEvaluator()
    
    # Fazer predições
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    
    # Imprimir métricas
    evaluator.print_metrics()
    
    # Análise por classe
    evaluator.analyze_predictions_by_class(y_test, y_pred)
    
    # Mensagem de confiança
    confidence_msg = evaluator.get_model_confidence_message()
    print(confidence_msg)
    
    return metrics, confidence_msg


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de avaliação - Use através de train.py ou como import")
