"""
Script para testar a API localmente.
Envia requisições de exemplo para o endpoint /predict.
"""

import requests
import json
from typing import Dict, Any

# URL da API (ajuste se necessário)
API_URL = "http://localhost:8000"


def test_health_check():
    """Testa o endpoint de health check."""
    print("=" * 60)
    print("Testando Health Check...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Testa o endpoint de informações do modelo."""
    print("=" * 60)
    print("Testando Model Info...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_prediction(student_data: Dict[str, Any]):
    """
    Testa o endpoint de predição.
    
    Args:
        student_data: Dados do estudante
    """
    print("=" * 60)
    print("Testando Predição...")
    print("=" * 60)
    
    response = requests.post(
        f"{API_URL}/predict",
        json=student_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 RESULTADO DA PREDIÇÃO:")
        print(f"  Defasagem Prevista: {result['defasagem_prevista']}")
        print(f"  Nível de Risco: {result['risco']}")
        print(f"  Confiança: {result['confianca']}")
        print(f"  Recomendação: {result['recomendacao']}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"Erro: {response.text}")
    
    print()


def get_sample_student_data() -> Dict[str, Any]:
    """Retorna dados de exemplo de um estudante."""
    return {
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
    }


def get_sample_student_2() -> Dict[str, Any]:
    """Retorna dados de outro estudante (bom desempenho)."""
    return {
        "Fase": 6,
        "Turma": "A",
        "Idade 22": 17,
        "Gênero": "Menina",
        "Ano ingresso": 2019,
        "Instituição de ensino": "Rede Decisão",
        "Pedra 20": "Topázio",
        "Pedra 21": "Topázio",
        "Pedra 22": "Topázio",
        "INDE 22": 8.843,
        "Cg": 9.0,
        "Cf": 2,
        "Ct": 2,
        "Nº Av": 4,
        "Avaliador1": "Avaliador-5",
        "Rec Av1": "Promovido de Fase",
        "Avaliador2": "Avaliador-27",
        "Rec Av2": "Promovido de Fase + Bolsa",
        "Avaliador3": "Avaliador-28",
        "Rec Av3": "Promovido de Fase + Bolsa",
        "Avaliador4": "Avaliador-31",
        "Rec Av4": "Promovido de Fase + Bolsa",
        "IAA": 10.0,
        "IEG": 9.5,
        "IPS": 9.4,
        "Rec Psicologia": "Sem limitações",
        "IDA": 8.0,
        "Matem": 9.0,
        "Portug": 5.7,
        "Inglês": 9.3,
        "Indicado": "Não",
        "Atingiu PV": "Sim",
        "IPV": 10.0,
        "IAN": 5.0,
        "Fase ideal": "Fase 7 (3º EM)",
        "Destaque IEG": "Destaque: A sua boa entrega das lições de casa.",
        "Destaque IDA": "Destaque: As suas boas notas na Passos Mágicos.",
        "Destaque IPV": "Destaque: A sua boa integração aos Princípios Passos Mágicos."
    }


def main():
    """Função principal para executar os testes."""
    print("\n" + "=" * 60)
    print("🎓 TESTE DA API PASSOS MÁGICOS")
    print("=" * 60 + "\n")
    
    try:
        # Teste 1: Health Check
        test_health_check()
        
        # Teste 2: Model Info
        test_model_info()
        
        # Teste 3: Predição - Aluno com dificuldades
        print("📝 CASO 1: Aluno com indicadores de risco")
        test_prediction(get_sample_student_data())
        
        # Teste 4: Predição - Aluno com bom desempenho
        print("📝 CASO 2: Aluno com bom desempenho")
        test_prediction(get_sample_student_2())
        
        print("=" * 60)
        print("✅ Testes concluídos!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ ERRO: Não foi possível conectar à API.")
        print("Certifique-se de que a API está rodando em http://localhost:8000")
        print("\nPara iniciar a API, execute:")
        print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ ERRO: {str(e)}")


if __name__ == "__main__":
    main()
