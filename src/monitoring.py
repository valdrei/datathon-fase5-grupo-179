"""
Módulo de monitoramento e logging.
Implementa logging de predições e detecção de drift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionLogger:
    """Classe para registrar predições e métricas."""
    
    def __init__(self, log_dir: str = "../logs", max_records: int = 10000):
        """
        Inicializa o logger de predições.
        
        Args:
            log_dir: Diretório onde salvar os logs
            max_records: Número máximo de registros a manter (0 = sem limite)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_file = self.log_dir / "predictions.jsonl"
        self.metrics_file = self.log_dir / "metrics.json"
        self.max_records = max_records
        self._write_count = 0
        self._truncate_interval = 100  # verifica a cada N escritas
        
    def log_prediction(self, input_data: Dict, prediction: float, 
                      confidence: float, risk: str, timestamp: Optional[datetime] = None) -> None:
        """
        Registra uma predição.
        
        Args:
            input_data: Dados de entrada
            prediction: Valor predito
            confidence: Confiança da predição
            risk: Nível de risco
            timestamp: Timestamp opcional (usa now() se não fornecido)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'risk': risk,
            'input_data': input_data  # Logar todos os dados de entrada para detecção de drift
        }
        
        # Append ao arquivo JSONL
        with open(self.predictions_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Verificar e aplicar retenção periodicamente
        self._write_count += 1
        if self.max_records > 0 and self._write_count >= self._truncate_interval:
            self._write_count = 0
            self._enforce_retention()
        
        logger.info(f"Predição registrada: {prediction:.2f} (risco: {risk})")
    
    def _enforce_retention(self) -> None:
        """
        Mantém apenas os últimos max_records registros no arquivo.
        Remove os mais antigos quando o limite é ultrapassado.
        """
        if not self.predictions_file.exists():
            return
        
        with open(self.predictions_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= self.max_records:
            return
        
        # Manter apenas os mais recentes
        keep_lines = lines[-self.max_records:]
        with open(self.predictions_file, 'w') as f:
            f.writelines(keep_lines)
        
        removed = len(lines) - self.max_records
        logger.info(
            f"Retenção aplicada: {removed} registros antigos removidos, "
            f"{self.max_records} mantidos"
        )
    
    def get_prediction_statistics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtém estatísticas das últimas predições.
        
        Args:
            last_n: Número de últimas predições a considerar (None = todas)
            
        Returns:
            Dicionário com estatísticas
        """
        if not self.predictions_file.exists():
            return {}
        
        # Ler predições
        predictions = []
        with open(self.predictions_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if not predictions:
            return {}
        
        # Filtrar últimas N
        if last_n:
            predictions = predictions[-last_n:]
        
        # Calcular estatísticas
        pred_values = [p['prediction'] for p in predictions]
        confidence_values = [p['confidence'] for p in predictions]
        risk_counts = {}
        for p in predictions:
            risk = p['risk']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        stats_dict = {
            'total_predictions': len(predictions),
            'mean_prediction': np.mean(pred_values),
            'std_prediction': np.std(pred_values),
            'min_prediction': np.min(pred_values),
            'max_prediction': np.max(pred_values),
            'mean_confidence': np.mean(confidence_values),
            'risk_distribution': risk_counts,
            'last_prediction_time': predictions[-1]['timestamp']
        }
        
        return stats_dict
    
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Salva métricas do modelo.
        
        Args:
            metrics: Dicionário com métricas
        """
        metrics_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=4)
        
        logger.info(f"Métricas salvas: {self.metrics_file}")


class DriftDetector:
    """Classe para detectar drift nos dados."""
    
    def __init__(self, reference_file: Optional[str] = None, 
                 threshold: float = 0.05):
        """
        Inicializa o detector de drift.
        
        Args:
            reference_file: Arquivo com dados de referência
            threshold: Threshold de p-value para teste de Kolmogorov-Smirnov
        """
        self.threshold = threshold
        self.reference_data = None
        
        if reference_file and Path(reference_file).exists():
            self.reference_data = pd.read_csv(reference_file)
            logger.info(f"Dados de referência carregados: {len(self.reference_data)} registros")
        else:
            logger.warning("Dados de referência não encontrados. Drift detection desabilitado.")
    
    def detect_drift(self, new_data: pd.DataFrame) -> bool:
        """
        Detecta drift comparando novos dados com dados de referência.
        
        Args:
            new_data: Novos dados para comparar
            
        Returns:
            True se drift foi detectado
        """
        if self.reference_data is None:
            return False
        
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        drift_detected = False
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
            # Teste de Kolmogorov-Smirnov
            try:
                ref_col = self.reference_data[col].dropna()
                new_col = new_data[col].dropna()
                
                if len(ref_col) > 0 and len(new_col) > 0:
                    statistic, p_value = stats.ks_2samp(ref_col, new_col)
                    
                    if p_value < self.threshold:
                        logger.warning(
                            f"Drift detectado em '{col}': "
                            f"KS statistic={statistic:.4f}, p-value={p_value:.4f}"
                        )
                        drift_detected = True
            except Exception as e:
                logger.error(f"Erro ao testar drift em '{col}': {str(e)}")
        
        return drift_detected
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                     buckets: int = 10) -> float:
        """
        Calcula Population Stability Index (PSI).
        
        Args:
            expected: Distribuição esperada (referência)
            actual: Distribuição atual
            buckets: Número de buckets para discretização
            
        Returns:
            Valor de PSI
        """
        try:
            # Criar buckets
            breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
            breakpoints = np.unique(breakpoints)
            
            if len(breakpoints) <= 1:
                return 0.0
            
            # Calcular distribuições
            expected_percents = pd.cut(expected, bins=breakpoints, 
                                      include_lowest=True).value_counts() / len(expected)
            actual_percents = pd.cut(actual, bins=breakpoints, 
                                    include_lowest=True).value_counts() / len(actual)
            
            # Alinhar índices
            expected_percents, actual_percents = expected_percents.align(
                actual_percents, fill_value=0.0001
            )
            
            # Calcular PSI
            psi = np.sum((actual_percents - expected_percents) * 
                        np.log(actual_percents / expected_percents))
            
            return psi
        except Exception as e:
            logger.error(f"Erro ao calcular PSI: {str(e)}")
            return 0.0
    
    def monitor_psi(self, new_data: pd.DataFrame) -> Dict[str, float]:
        """
        Monitora PSI para múltiplas features.
        
        Args:
            new_data: Novos dados
            
        Returns:
            Dicionário com PSI por feature
        """
        if self.reference_data is None:
            return {}
        
        psi_values = {}
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
            try:
                expected = self.reference_data[col].dropna()
                actual = new_data[col].dropna()
                
                if len(expected) > 0 and len(actual) > 0:
                    psi = self.calculate_psi(expected, actual)
                    psi_values[col] = psi
                    
                    # PSI thresholds: <0.1 (sem mudança), 0.1-0.25 (mudança moderada), >0.25 (mudança significativa)
                    if psi > 0.25:
                        logger.warning(f"PSI alto para '{col}': {psi:.4f} (mudança significativa)")
                    elif psi > 0.1:
                        logger.info(f"PSI moderado para '{col}': {psi:.4f}")
            except Exception as e:
                logger.error(f"Erro ao calcular PSI para '{col}': {str(e)}")
        
        return psi_values


class ModelMonitor:
    """Classe para monitorar performance do modelo em produção."""
    
    def __init__(self, log_dir: str = "../logs"):
        """
        Inicializa o monitor de modelo.
        
        Args:
            log_dir: Diretório de logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.performance_file = self.log_dir / "model_performance.jsonl"
    
    def log_performance(self, metrics: Dict[str, float], metadata: Optional[Dict] = None) -> None:
        """
        Registra métricas de performance.
        
        Args:
            metrics: Métricas calculadas
            metadata: Metadados adicionais
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        with open(self.performance_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info("Performance registrada")
    
    def check_degradation(self, current_metrics: Dict[str, float],
                         baseline_metrics: Dict[str, float],
                         threshold: float = 0.1) -> bool:
        """
        Verifica se houve degradação nas métricas.
        
        Args:
            current_metrics: Métricas atuais
            baseline_metrics: Métricas baseline
            threshold: Threshold de degradação aceitável
            
        Returns:
            True se degradação foi detectada
        """
        degradation_detected = False
        
        for metric in ['r2', 'mae', 'rmse']:
            if metric not in current_metrics or metric not in baseline_metrics:
                continue
            
            current = current_metrics[metric]
            baseline = baseline_metrics[metric]
            
            # Para R², menor é pior
            if metric == 'r2':
                change = (baseline - current) / baseline
                if change > threshold:
                    logger.warning(
                        f"Degradação detectada em {metric}: "
                        f"baseline={baseline:.4f}, current={current:.4f}, "
                        f"change={change*100:.2f}%"
                    )
                    degradation_detected = True
            
            # Para MAE e RMSE, maior é pior
            else:
                change = (current - baseline) / baseline
                if change > threshold:
                    logger.warning(
                        f"Degradação detectada em {metric}: "
                        f"baseline={baseline:.4f}, current={current:.4f}, "
                        f"change={change*100:.2f}%"
                    )
                    degradation_detected = True
        
        return degradation_detected


def create_monitoring_dashboard(log_dir: str = "../logs", 
                               output_file: str = "../logs/dashboard.html") -> None:
    """
    Cria um dashboard HTML simples com estatísticas de monitoramento.
    
    Args:
        log_dir: Diretório com logs
        output_file: Arquivo HTML de saída
    """
    logger.info("Gerando dashboard de monitoramento")
    
    prediction_logger = PredictionLogger(log_dir)
    stats = prediction_logger.get_prediction_statistics(last_n=100)
    
    if not stats:
        logger.warning("Sem dados para dashboard")
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Passos Mágicos - Dashboard de Monitoramento</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .metric-box {{
                display: inline-block;
                width: 200px;
                padding: 15px;
                margin: 10px;
                background-color: #f9f9f9;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
            }}
            .metric-title {{
                font-size: 14px;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .risk-distribution {{
                margin-top: 30px;
            }}
            .risk-bar {{
                margin: 10px 0;
            }}
            .risk-label {{
                display: inline-block;
                width: 100px;
                font-weight: bold;
            }}
            .bar {{
                display: inline-block;
                height: 30px;
                background-color: #4CAF50;
                color: white;
                text-align: center;
                line-height: 30px;
                border-radius: 4px;
            }}
            .timestamp {{
                color: #999;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Passos Mágicos - Dashboard de Monitoramento</h1>
            
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-title">Total de Predições</div>
                    <div class="metric-value">{stats.get('total_predictions', 0)}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Defasagem Média</div>
                    <div class="metric-value">{stats.get('mean_prediction', 0):.2f}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Confiança Média</div>
                    <div class="metric-value">{stats.get('mean_confidence', 0):.2f}</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Desvio Padrão</div>
                    <div class="metric-value">{stats.get('std_prediction', 0):.2f}</div>
                </div>
            </div>
            
            <div class="risk-distribution">
                <h2>Distribuição de Risco</h2>
                {_generate_risk_bars(stats.get('risk_distribution', {}))}
            </div>
            
            <div class="timestamp">
                Última atualização: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
    </body>
    </html>
    """
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Dashboard gerado: {output_file}")


def _generate_risk_bars(risk_dist: Dict[str, int]) -> str:
    """Gera HTML para barras de distribuição de risco."""
    if not risk_dist:
        return "<p>Sem dados</p>"
    
    total = sum(risk_dist.values())
    colors = {
        'Baixo': '#4CAF50',
        'Moderado': '#FFC107',
        'Alto': '#FF9800',
        'Crítico': '#F44336'
    }
    
    html = ""
    for risk, count in risk_dist.items():
        percentage = (count / total) * 100
        width = int((count / total) * 500)
        color = colors.get(risk, '#999')
        
        html += f"""
        <div class="risk-bar">
            <span class="risk-label">{risk}:</span>
            <div class="bar" style="width: {width}px; background-color: {color};">
                {count} ({percentage:.1f}%)
            </div>
        </div>
        """
    
    return html


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de monitoramento - Use através de api.py ou como import")
    
    # Criar dashboard de exemplo
    create_monitoring_dashboard()
