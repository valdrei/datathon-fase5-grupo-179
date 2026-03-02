"""
Script para gerar dados de teste e popular os logs com 50 cenários.
"""

import random
import numpy as np
from datetime import datetime, timedelta
from src.monitoring import PredictionLogger

def generate_student_data() -> dict:
    """
    Gera dados realistas de um estudante.
    
    Returns:
        Dicionário com dados de entrada simulados
    """
    # Definir pedras
    pedras = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    
    # Definir fases
    fases = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Definir recomendações
    recomendacoes = [
        'Promovido',
        'Bolsa Integral',
        'Bolsa Parcial',
        'Acompanhamento',
        'Requer Atenção'
    ]
    
    # Gerar dados aleatórios mas realistas
    student = {
        'ID': random.randint(1000, 9999),
        'Nome': f'Aluno_{random.randint(1, 1000)}',
        'Fase': random.choice(fases),
        'Idade 22': random.randint(10, 18),
        'Ano ingresso': random.randint(2018, 2022),
        
        # Indicadores (0-10)
        'INDE 22': round(random.uniform(4, 10), 2),
        'IAA': round(random.uniform(3, 10), 2),
        'IEG': round(random.uniform(3, 10), 2),
        'IPS': round(random.uniform(3, 10), 2),
        'IDA': round(random.uniform(3, 10), 2),
        'IPV': round(random.uniform(3, 10), 2),
        'IAN': round(random.uniform(3, 10), 2),
        
        # Notas (0-10)
        'Matem': round(random.uniform(3, 10), 2),
        'Portug': round(random.uniform(3, 10), 2),
        'Inglês': round(random.uniform(3, 10), 2),
        
        # Rankings
        'Cg': random.randint(1, 500),  # Ranking geral
        'Cf': random.randint(1, 100),  # Ranking fase
        'Ct': random.randint(1, 50),   # Ranking turma
        
        # Pedras
        'Pedra 20': random.choice(pedras),
        'Pedra 21': random.choice(pedras),
        'Pedra 22': random.choice(pedras),
        
        # Recomendações
        'Rec Av1': random.choice(recomendacoes),
        'Rec Av2': random.choice(recomendacoes),
        'Rec Av3': random.choice(recomendacoes),
        'Rec Av4': random.choice(recomendacoes),
        'Rec Psicologia': random.choice(['Normal', 'Requer Atenção']) 
    }
    
    return student


def generate_risk_level(inde: float) -> str:
    """
    Gera nível de risco baseado em INDE.
    
    Args:
        inde: Valor do INDE 22
        
    Returns:
        Nível de risco
    """
    if inde >= 8:
        return 'Baixo'
    elif inde >= 6:
        return 'Moderado'
    elif inde >= 4:
        return 'Alto'
    else:
        return 'Crítico'


def generate_confidence(inde: float) -> float:
    """
    Gera confiança baseada em INDE.
    
    Args:
        inde: Valor do INDE 22
        
    Returns:
        Confiança (0-1)
    """
    # Quanto maior INDE, maior a confiança
    confidence = (inde / 10) * 0.9 + 0.1  # Entre 0.1 e 1.0
    return round(confidence, 2)


def main():
    """Gera logs distribuídos ao longo de 20 dias (10 logs/dia até hoje)."""
    
    logger = PredictionLogger(log_dir="logs", max_records=10000)
    
    print("Gerando 200 cenários distribuídos em 20 dias (10 logs/dia)...")
    print("Data: -20 dias até hoje\n")
    
    now = datetime.now()
    total_logs = 0
    
    # Iterar 20 dias para trás
    for day_offset in range(20, -1, -1):  # 20 dias atrás até hoje (21 dias no total = 210 logs)
        target_date = now - timedelta(days=day_offset)
        
        print(f"Dia {20 - day_offset + 1}/20 ({target_date.strftime('%Y-%m-%d')}):", end=" ")
        
        # Gerar 10 logs para este dia
        for log_num in range(10):
            # Timestamp espalhado ao longo do dia
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = target_date.replace(hour=hour, minute=minute, second=second)
            
            # Gerar dados do estudante
            student_data = generate_student_data()
            
            # Simular predição
            inde = student_data['INDE 22']
            prediction = round(inde + random.uniform(-1, 1), 2)
            
            # Gerar risco e confiança
            risk = generate_risk_level(prediction)
            confidence = generate_confidence(prediction)
            
            # Registrar predição com timestamp personalizado
            logger.log_prediction(
                input_data=student_data,
                prediction=prediction,
                confidence=confidence,
                risk=risk,
                timestamp=timestamp  # Usar timestamp customizado
            )
            
            total_logs += 1
        
        print(f"✓ 10 logs")
    
    print(f"\n✅ {total_logs} logs gerados com sucesso em logs/predictions.jsonl")
    
    # Mostrar estatísticas
    stats = logger.get_prediction_statistics()
    print(f"\nEstatísticas:")
    print(f"  Total de predições: {stats.get('total_predictions')}")
    print(f"  Média de predições: {stats.get('mean_prediction'):.2f}")
    print(f"  Confiança média: {stats.get('mean_confidence'):.2f}")
    print(f"  Distribuição de risco: {stats.get('risk_distribution')}")


if __name__ == '__main__':
    main()
