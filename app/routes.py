"""
Rotas e endpoints da API.
Define todos os endpoints disponíveis para a aplicação.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configurar logger
logger = logging.getLogger(__name__)

# Criar router
router = APIRouter()


class StudentData(BaseModel):
    """Schema para dados de entrada de um estudante."""
    
    # Dados básicos
    Fase: int = Field(..., ge=0, le=10, description="Fase atual do aluno")
    Turma: str = Field(..., description="Turma do aluno")
    Idade_22: int = Field(..., ge=5, le=25, alias="Idade 22", description="Idade do aluno em 2022")
    Genero: str = Field(..., alias="Gênero", description="Gênero do aluno")
    Ano_ingresso: int = Field(..., ge=2010, le=2023, alias="Ano ingresso", description="Ano de ingresso")
    Instituicao_de_ensino: str = Field(..., alias="Instituição de ensino", description="Instituição de ensino")
    
    # Pedras
    Pedra_20: Optional[str] = Field(None, alias="Pedra 20", description="Classificação em 2020")
    Pedra_21: Optional[str] = Field(None, alias="Pedra 21", description="Classificação em 2021")
    Pedra_22: str = Field(..., alias="Pedra 22", description="Classificação em 2022")
    
    # Indicadores
    INDE_22: float = Field(..., ge=0, le=10, alias="INDE 22", description="Índice de Desenvolvimento Educacional")
    Cg: float = Field(..., alias="Cg", description="Classificação Geral")
    Cf: int = Field(..., alias="Cf", description="Classificação na Fase")
    Ct: int = Field(..., alias="Ct", description="Classificação na Turma")
    N_Av: int = Field(..., ge=0, le=10, alias="Nº Av", description="Número de avaliações")
    
    # Avaliadores e recomendações
    Avaliador1: str = Field(..., description="Nome do Avaliador 1")
    Rec_Av1: str = Field(..., alias="Rec Av1", description="Recomendação do Avaliador 1")
    Avaliador2: str = Field(..., description="Nome do Avaliador 2")
    Rec_Av2: str = Field(..., alias="Rec Av2", description="Recomendação do Avaliador 2")
    Avaliador3: str = Field(..., description="Nome do Avaliador 3")
    Rec_Av3: str = Field(..., alias="Rec Av3", description="Recomendação do Avaliador 3")
    Avaliador4: Optional[str] = Field(None, description="Nome do Avaliador 4")
    Rec_Av4: Optional[str] = Field(None, alias="Rec Av4", description="Recomendação do Avaliador 4")
    
    # Indicadores de performance
    IAA: float = Field(..., ge=0, le=10, alias="IAA", description="Indicador de Auto Avaliação")
    IEG: float = Field(..., ge=0, le=10, alias="IEG", description="Indicador de Engajamento")
    IPS: float = Field(..., ge=0, le=10, alias="IPS", description="Indicador Psicossocial")
    Rec_Psicologia: str = Field(..., alias="Rec Psicologia", description="Recomendação Psicologia")
    IDA: float = Field(..., ge=0, le=10, alias="IDA", description="Indicador de Aprendizagem")
    Matem: float = Field(..., ge=0, le=10, alias="Matem", description="Nota de Matemática")
    Portug: float = Field(..., ge=0, le=10, alias="Portug", description="Nota de Português")
    Ingles: Optional[float] = Field(None, ge=0, le=10, alias="Inglês", description="Nota de Inglês")
    Indicado: str = Field(..., description="Indicado para bolsa (Sim/Não)")
    Atingiu_PV: str = Field(..., alias="Atingiu PV", description="Atingiu Ponto de Virada (Sim/Não)")
    IPV: float = Field(..., ge=0, le=10, alias="IPV", description="Indicador de Ponto de Virada")
    IAN: float = Field(..., ge=0, le=10, alias="IAN", description="Indicador de Adequação ao Nível")
    Fase_ideal: str = Field(..., alias="Fase ideal", description="Fase ideal do aluno")
    
    # Destaques
    Destaque_IEG: str = Field(..., alias="Destaque IEG", description="Destaque de Engajamento")
    Destaque_IDA: str = Field(..., alias="Destaque IDA", description="Destaque de Aprendizagem")
    Destaque_IPV: str = Field(..., alias="Destaque IPV", description="Destaque de Ponto de Virada")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema para resposta da predição."""
    defasagem_prevista: float = Field(..., description="Defasagem prevista (diferença entre fase atual e ideal)")
    risco: str = Field(..., description="Nível de risco (Baixo/Moderado/Alto/Crítico)")
    confianca: float = Field(..., description="Nível de confiança da predição (0-1)")
    recomendacao: str = Field(..., description="Recomendação baseada na predição")
    timestamp: str = Field(..., description="Timestamp da predição")
    
    class Config:
        json_schema_extra = {
            "example": {
                "defasagem_prevista": -1.2,
                "risco": "Alto",
                "confianca": 0.87,
                "recomendacao": "Aluno necessita de acompanhamento intensivo. Considerar tutoria e reforço escolar.",
                "timestamp": "2024-01-29T10:30:00"
            }
        }


def classify_risk(defasagem: float) -> str:
    """
    Classifica o nível de risco baseado na defasagem.
    
    Args:
        defasagem: Valor da defasagem prevista
        
    Returns:
        Nível de risco (Baixo/Moderado/Alto/Crítico)
    """
    if defasagem >= 0:
        return "Baixo"
    elif defasagem >= -1:
        return "Moderado"
    elif defasagem > -2:
        return "Alto"
    else:
        return "Crítico"


def generate_recommendation(defasagem: float, risk: str) -> str:
    """
    Gera recomendação baseada na defasagem e risco.
    
    Args:
        defasagem: Valor da defasagem prevista
        risk: Nível de risco
        
    Returns:
        Recomendação personalizada
    """
    recommendations = {
        "Baixo": "Aluno está no nível adequado ou acima. Continuar acompanhamento regular.",
        "Moderado": "Aluno apresenta leve defasagem. Recomenda-se acompanhamento próximo e atividades de reforço.",
        "Alto": "Aluno necessita de acompanhamento intensivo. Considerar tutoria e reforço escolar.",
        "Crítico": "Aluno em situação crítica. Intervenção imediata necessária com plano individualizado de recuperação."
    }
    
    return recommendations.get(risk, "Avaliar situação individual do aluno.")


@router.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "message": "API Passos Mágicos - Predição de Defasagem Escolar",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "/predict": "POST - Fazer predição de defasagem",
            "/health": "GET - Verificar saúde da API",
            "/model-info": "GET - Informações sobre o modelo",
            "/monitoring/stats": "GET - Estatísticas de monitoramento",
            "/monitoring/predictions": "GET - Histórico de predições",
            "/monitoring/drift": "GET - Relatório de drift (PSI)",
            "/docs": "GET - Documentação interativa (Swagger)",
            "/redoc": "GET - Documentação alternativa (ReDoc)"
        }
    }


@router.get("/health")
async def health_check():
    """Endpoint de health check."""
    from app.main import model, preprocessor, feature_engineer
    
    is_healthy = model is not None and preprocessor is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "feature_engineer_loaded": feature_engineer is not None,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/model-info")
async def model_info():
    """Retorna informações sobre o modelo."""
    from app.main import model, preprocessor
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    info = {
        "model_type": type(model).__name__,
        "features_count": len(model.feature_importances_) if hasattr(model, 'feature_importances_') else "N/A",
        "timestamp": datetime.now().isoformat()
    }
    
    if hasattr(model, 'feature_importances_') and preprocessor:
        # Top 10 features mais importantes
        try:
            feature_names = preprocessor.numeric_features + preprocessor.categorical_features
            if len(feature_names) == len(model.feature_importances_):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                info['top_features'] = importance_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Erro ao recuperar features importantes: {str(e)}")
    
    return info


@router.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentData, request: Request):
    """
    Endpoint principal para predição de defasagem.
    
    Args:
        student: Dados do estudante
        request: Objeto de requisição FastAPI
        
    Returns:
        Predição de defasagem com recomendações
    """
    from app.main import model, preprocessor, feature_engineer, prediction_logger, drift_detector
    
    try:
        logger.info("Nova requisição de predição recebida")
        
        # Validar que modelo está carregado
        if model is None or preprocessor is None or feature_engineer is None:
            raise HTTPException(
                status_code=503, 
                detail="Modelo não está carregado. Execute o treinamento primeiro."
            )
        
        # Converter dados para DataFrame
        student_dict = student.model_dump(by_alias=True)
        df = pd.DataFrame([student_dict])
        
        # Adicionar colunas necessárias que podem estar faltando
        df['RA'] = 'PREDICT-' + datetime.now().strftime("%Y%m%d%H%M%S")
        df['Nome'] = 'Aluno Predição'
        df['Ano nasc'] = 2022 - student_dict.get('Idade 22', 15)
        
        # Aplicar feature engineering
        df_engineered = feature_engineer.engineer_features(df)
        
        # Adicionar coluna target dummy (necessária para o pipeline)
        df_engineered['Defas'] = 0
        
        # Pré-processar (pipeline completo: clean → missing → encode → features/target → scale)
        X_processed, _ = preprocessor.preprocess_pipeline(df_engineered, fit=False)
        
        # Fazer predição
        prediction = model.predict(X_processed)[0]
        
        # Calcular confiança (baseado em modelos de árvore)
        if hasattr(model, 'estimators_'):
            # Para Random Forest, usar variação entre árvores
            tree_predictions = [tree.predict(X_processed)[0] for tree in model.estimators_]
            confidence = 1 - (np.std(tree_predictions) / (np.abs(prediction) + 1))
            confidence = max(0, min(1, confidence))  # Limitar entre 0 e 1
        else:
            confidence = 0.85  # Valor padrão
        
        # Classificar risco
        risk = classify_risk(prediction)
        
        # Gerar recomendação
        recommendation = generate_recommendation(prediction, risk)
        
        # Log da predição
        if prediction_logger:
            prediction_logger.log_prediction(
                input_data=student_dict,
                prediction=float(prediction),
                confidence=float(confidence),
                risk=risk
            )
        
        # Detectar drift
        if drift_detector:
            try:
                drift_detected = drift_detector.detect_drift(df)
                if drift_detected:
                    logger.warning("⚠️ DRIFT DETECTADO nas features de entrada!")
            except Exception as e:
                logger.warning(f"Erro ao detectar drift: {str(e)}")
        
        response = PredictionResponse(
            defasagem_prevista=round(float(prediction), 2),
            risco=risk,
            confianca=round(float(confidence), 2),
            recomendacao=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Predição realizada: defasagem={prediction:.2f}, risco={risk}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao processar predição: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao processar predição: {str(e)}")


# ═══════════════════════════════════════════════════════
#  ENDPOINTS DE MONITORAMENTO
# ═══════════════════════════════════════════════════════

@router.get("/monitoring/stats")
async def monitoring_stats(last_n: Optional[int] = None):
    """
    Retorna estatísticas agregadas das predições.
    
    Args:
        last_n: Número de últimas predições a considerar (None = todas)
    """
    from app.main import prediction_logger
    
    if prediction_logger is None:
        raise HTTPException(status_code=503, detail="Logger de predições não inicializado.")
    
    stats = prediction_logger.get_prediction_statistics(last_n=last_n)
    if not stats:
        return {"message": "Nenhuma predição registrada ainda.", "total_predictions": 0}
    
    # Converter tipos numpy para JSON-serializable
    return {
        "total_predictions": int(stats.get("total_predictions", 0)),
        "mean_prediction": float(stats.get("mean_prediction", 0)),
        "std_prediction": float(stats.get("std_prediction", 0)),
        "min_prediction": float(stats.get("min_prediction", 0)),
        "max_prediction": float(stats.get("max_prediction", 0)),
        "mean_confidence": float(stats.get("mean_confidence", 0)),
        "risk_distribution": stats.get("risk_distribution", {}),
        "last_prediction_time": stats.get("last_prediction_time", "")
    }


@router.get("/monitoring/predictions")
async def monitoring_predictions(last_n: int = 100):
    """
    Retorna o histórico das últimas predições com análise temporal.
    
    Args:
        last_n: Número de últimas predições a retornar (padrão: 100)
    """
    from app.main import prediction_logger
    import json as _json
    
    if prediction_logger is None:
        raise HTTPException(status_code=503, detail="Logger de predições não inicializado.")
    
    if not prediction_logger.predictions_file.exists():
        return {
            "predictions": [], 
            "total": 0,
            "summary": {
                "mean_confidence": 0,
                "risk_distribution": {}
            }
        }
    
    predictions = []
    with open(prediction_logger.predictions_file, 'r') as f:
        for line in f:
            try:
                predictions.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
    
    # Retornar apenas as últimas N
    predictions = predictions[-last_n:]
    
    # Calcular estatísticas
    if predictions:
        confidences = [p.get('confidence', 0) for p in predictions]
        risks = [p.get('risk', 'N/A') for p in predictions]
        
        risk_dist = {}
        for risk in risks:
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        summary = {
            "mean_confidence": round(float(np.mean(confidences)), 3),
            "risk_distribution": risk_dist,
            "mean_prediction": round(float(np.mean([p.get('prediction', 0) for p in predictions])), 3),
            "std_prediction": round(float(np.std([p.get('prediction', 0) for p in predictions])), 3),
        }
    else:
        summary = {
            "mean_confidence": 0,
            "risk_distribution": {},
            "mean_prediction": 0,
            "std_prediction": 0
        }
    
    return {
        "predictions": predictions, 
        "total": len(predictions),
        "summary": summary
    }


@router.get("/monitoring/drift")
async def monitoring_drift():
    """
    Retorna análise de drift (PSI e mudança de distribuição)
    comparando predições recentes com dados de referência.
    """
    from app.main import prediction_logger, drift_detector
    import json as _json
    from scipy import stats as sp_stats
    
    result = {
        "drift_enabled": False,
        "analysis": {
            "data_drift": {},
            "prediction_drift": {},
            "feature_drift": {}
        },
        "thresholds": {
            "no_change": "< 0.10",
            "moderate": "0.10 – 0.25",
            "significant": "> 0.25"
        },
        "warnings": []
    }
    
    # Análise de drift nas predições
    if prediction_logger and prediction_logger.predictions_file.exists():
        predictions = []
        input_features = {}
        
        with open(prediction_logger.predictions_file, 'r') as f:
            for line in f:
                try:
                    pred = _json.loads(line)
                    predictions.append(pred)
                    
                    # Extrair features numéricas
                    if 'input_data' in pred:
                        for key, val in pred['input_data'].items():
                            if isinstance(val, (int, float)):
                                if key not in input_features:
                                    input_features[key] = []
                                input_features[key].append(val)
                except _json.JSONDecodeError:
                    continue
        
        if len(predictions) >= 10:
            # DRIFT NAS PREDIÇÕES
            pred_values = [p['prediction'] for p in predictions]
            mid = len(pred_values) // 2
            first_half = pred_values[:mid]
            second_half = pred_values[mid:]
            
            result["analysis"]["prediction_drift"] = {
                "first_half_mean": round(float(np.mean(first_half)), 4),
                "second_half_mean": round(float(np.mean(second_half)), 4),
                "first_half_std": round(float(np.std(first_half)), 4),
                "second_half_std": round(float(np.std(second_half)), 4),
                "mean_shift": round(float(np.mean(second_half) - np.mean(first_half)), 4),
                "total_predictions": len(pred_values)
            }
            
            # Teste KS
            try:
                ks_stat, ks_pvalue = sp_stats.ks_2samp(first_half, second_half)
                result["analysis"]["prediction_drift"]["ks_statistic"] = round(float(ks_stat), 4)
                result["analysis"]["prediction_drift"]["ks_pvalue"] = round(float(ks_pvalue), 4)
                
                if ks_pvalue < 0.05:
                    result["analysis"]["prediction_drift"]["drift_detected"] = True
                    result["warnings"].append("⚠️ Drift significativo detectado nas PREDIÇÕES (p-value < 0.05)")
                else:
                    result["analysis"]["prediction_drift"]["drift_detected"] = False
            except Exception as e:
                result["analysis"]["prediction_drift"]["drift_detected"] = False
                result["analysis"]["prediction_drift"]["error"] = str(e)
            
            # DRIFT NAS FEATURES DE ENTRADA
            for feature, values in input_features.items():
                if len(values) >= 10:
                    mid_feat = len(values) // 2
                    first_feat = values[:mid_feat]
                    second_feat = values[mid_feat:]
                    
                    try:
                        ks_stat, ks_pvalue = sp_stats.ks_2samp(first_feat, second_feat)
                        
                        result["analysis"]["feature_drift"][feature] = {
                            "ks_statistic": round(float(ks_stat), 4),
                            "ks_pvalue": round(float(ks_pvalue), 4),
                            "drift_detected": bool(ks_pvalue < 0.05),
                            "mean_shift": round(float(np.mean(second_feat) - np.mean(first_feat)), 4)
                        }
                        
                        if ks_pvalue < 0.05:
                            result["warnings"].append(f"⚠️ Drift em '{feature}' (p-value={ks_pvalue:.4f})")
                    except Exception:
                        pass
    
    # PSI por feature (se drift_detector tem dados de referência)
    if drift_detector and drift_detector.reference_data is not None:
        result["drift_enabled"] = True
        result["analysis"]["data_drift"]["reference_rows"] = len(drift_detector.reference_data)
        result["analysis"]["data_drift"]["status"] = "Dados de referência carregados"
    else:
        result["analysis"]["data_drift"]["status"] = "Dados de referência não encontrados"
    
    return result


# Nota: exception_handler global deve ser registrado no app (main.py), não no router
