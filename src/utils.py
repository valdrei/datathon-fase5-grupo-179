"""
Funções auxiliares e utilitárias para o projeto.
Inclui funções de logging, validação, e transformações comuns.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import json


def setup_logging(log_dir: str = "logs", log_file: str = "app.log", level=logging.INFO) -> logging.Logger:
    """
    Configura o sistema de logging para a aplicação.
    
    Args:
        log_dir: Diretório para salvar logs
        log_file: Nome do arquivo de log
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar handlers
    handlers = [
        logging.FileHandler(log_path / log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    # Configurar logging básico
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Sistema de logging configurado. Logs salvos em: {log_path / log_file}")
    
    return logger


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Salva dicionário como arquivo JSON.
    
    Args:
        data: Dados para salvar
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    
    logging.info(f"Dados salvos em: {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Carrega dados de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dados carregados
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"Dados carregados de: {filepath}")
    return data


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Valida se DataFrame contém colunas necessárias.
    
    Args:
        df: DataFrame para validar
        required_columns: Lista de colunas obrigatórias
        
    Returns:
        True se válido, False caso contrário
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logging.error(f"Colunas faltando no DataFrame: {missing_columns}")
        return False
    
    logging.info("DataFrame validado com sucesso")
    return True


def convert_comma_to_dot(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Converte vírgulas em pontos para valores numéricos.
    
    Args:
        df: DataFrame
        columns: Colunas específicas (None = todas as colunas object)
        
    Returns:
        DataFrame com valores convertidos
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns:
            try:
                # Tentar converter de string com vírgula para float
                df[col] = df[col].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else x)
            except (ValueError, AttributeError):
                # Ignorar colunas que não podem ser convertidas
                pass
    
    return df


def calculate_percentiles(series: pd.Series, percentiles: List[float] = [25, 50, 75]) -> Dict[str, float]:
    """
    Calcula percentis de uma série.
    
    Args:
        series: Série de dados
        percentiles: Lista de percentis a calcular
        
    Returns:
        Dicionário com percentis
    """
    result = {}
    for p in percentiles:
        result[f'p{int(p)}'] = series.quantile(p / 100)
    
    return result


def format_number(value: float, decimal_places: int = 2) -> str:
    """
    Formata número com casas decimais especificadas.
    
    Args:
        value: Valor a formatar
        decimal_places: Número de casas decimais
        
    Returns:
        Valor formatado como string
    """
    return f"{value:.{decimal_places}f}"


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Retorna timestamp formatado.
    
    Args:
        format_str: Formato do timestamp
        
    Returns:
        Timestamp formatado
    """
    return datetime.now().strftime(format_str)


def create_directories(*paths: Union[str, Path]) -> None:
    """
    Cria múltiplos diretórios se não existirem.
    
    Args:
        *paths: Caminhos dos diretórios a criar
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Diretórios criados/verificados: {len(paths)} diretórios")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Realiza divisão segura, evitando divisão por zero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se denominador for zero
        
    Returns:
        Resultado da divisão ou valor padrão
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def extract_phase_number(phase_str: str) -> int:
    """
    Extrai número da fase de strings como "Fase 7 (Ensino Médio)".
    
    Args:
        phase_str: String contendo a fase
        
    Returns:
        Número da fase
    """
    import re
    
    if pd.isna(phase_str):
        return 0
    
    # Procurar por número na string
    match = re.search(r'\d+', str(phase_str))
    if match:
        return int(match.group())
    
    return 0


def get_memory_usage(df: pd.DataFrame, unit: str = 'MB') -> float:
    """
    Calcula uso de memória de um DataFrame.
    
    Args:
        df: DataFrame
        unit: Unidade de medida ('B', 'KB', 'MB', 'GB')
        
    Returns:
        Uso de memória na unidade especificada
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    conversions = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }
    
    return memory_bytes / conversions.get(unit, 1)


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando método IQR (Interquartile Range).
    
    Args:
        series: Série de dados
        multiplier: Multiplicador para definir limites (padrão: 1.5)
        
    Returns:
        Série booleana indicando outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def normalize_string(text: str) -> str:
    """
    Normaliza string removendo acentos e convertendo para minúsculas.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    import unicodedata
    
    if pd.isna(text):
        return ""
    
    # Remover acentos
    text = unicodedata.normalize('NFKD', str(text))
    text = text.encode('ASCII', 'ignore').decode('ASCII')
    
    # Converter para minúsculas e remover espaços extras
    text = text.lower().strip()
    
    return text


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula intervalo de confiança para os dados.
    
    Args:
        data: Array de dados
        confidence: Nível de confiança (0-1)
        
    Returns:
        Tupla (limite_inferior, limite_superior)
    """
    from scipy import stats
    
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - margin, mean + margin)


def print_section_header(title: str, width: int = 80, char: str = "=") -> None:
    """
    Imprime cabeçalho formatado para seções.
    
    Args:
        title: Título da seção
        width: Largura total
        char: Caractere para bordas
    """
    border = char * width
    padding = (width - len(title) - 2) // 2
    
    print(f"\n{border}")
    print(f"{char}{' ' * padding}{title}{' ' * padding}{char}")
    print(f"{border}\n")


class DataValidator:
    """Classe para validação de dados."""
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float) -> bool:
        """Valida se valor está dentro do range."""
        return min_val <= value <= max_val
    
    @staticmethod
    def validate_not_null(value: Any) -> bool:
        """Valida se valor não é nulo."""
        return value is not None and pd.notna(value)
    
    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Valida se valor é do tipo esperado."""
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_in_list(value: Any, valid_values: List[Any]) -> bool:
        """Valida se valor está em lista de valores válidos."""
        return value in valid_values


# Constantes úteis
PEDRAS_ORDER = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
RISK_LEVELS = ['Baixo', 'Moderado', 'Alto', 'Crítico']
BOOLEAN_VALUES = ['Sim', 'Não']

# Mapping de valores
PEDRAS_MAPPING = {
    'Quartzo': 1,
    'Ágata': 2,
    'Ametista': 3,
    'Topázio': 4
}

BOOLEAN_MAPPING = {
    'Sim': 1,
    'sim': 1,
    'SIM': 1,
    'Não': 0,
    'não': 0,
    'NÃO': 0,
    'nao': 0,
    'NAO': 0
}


if __name__ == "__main__":
    # Testes básicos
    logger = setup_logging()
    print("Módulo utils.py carregado com sucesso!")
    
    # Teste de timestamp
    print(f"Timestamp: {get_timestamp()}")
    
    # Teste de formatação
    print(f"Número formatado: {format_number(3.14159, 2)}")
    
    # Teste de extração de fase
    print(f"Fase extraída: {extract_phase_number('Fase 7 (Ensino Médio)')}")
