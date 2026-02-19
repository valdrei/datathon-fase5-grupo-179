"""
Arquivo principal da API FastAPI.
Inicializa a aplicação e carrega os modelos necessários.
"""

from fastapi import FastAPI
from pathlib import Path
import joblib
import logging
import sys

# Adicionar diretórios ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.monitoring import PredictionLogger, DriftDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Passos Mágicos - API de Predição de Defasagem",
    description="API para prever o risco de defasagem escolar de estudantes do Instituto Passos Mágicos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Variáveis globais para modelo e processadores
model = None
preprocessor = None
feature_engineer = None
prediction_logger = None
drift_detector = None


@app.on_event("startup")
async def load_models():
    """Carrega os modelos e processadores na inicialização da API."""
    global model, preprocessor, feature_engineer, prediction_logger, drift_detector
    
    try:
        model_dir = Path(__file__).parent / "model"
        
        # Carregar modelo
        model_path = model_dir / "model_random_forest_latest.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"✅ Modelo carregado: {model_path}")
        else:
            logger.warning(f"⚠️ Modelo não encontrado: {model_path}")
        
        # Carregar preprocessador
        preprocessor_path = model_dir / "preprocessor_latest.pkl"
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✅ Preprocessador carregado: {preprocessor_path}")
        else:
            logger.warning(f"⚠️ Preprocessador não encontrado: {preprocessor_path}")
        
        # Carregar feature engineer
        feature_engineer_path = model_dir / "feature_engineer_latest.pkl"
        if feature_engineer_path.exists():
            feature_engineer = joblib.load(feature_engineer_path)
            logger.info(f"✅ Feature Engineer carregado: {feature_engineer_path}")
        else:
            logger.warning(f"⚠️ Feature Engineer não encontrado: {feature_engineer_path}")
        
        # Inicializar logger de predições
        prediction_logger = PredictionLogger(log_dir="logs")
        logger.info("✅ Prediction Logger inicializado")
        
        # Inicializar detector de drift
        drift_detector = DriftDetector()
        logger.info("✅ Drift Detector inicializado")
        
        logger.info("🚀 API inicializada com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelos: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Salva logs e estatísticas antes de desligar a API."""
    global prediction_logger
    
    try:
        if prediction_logger:
            stats = prediction_logger.get_prediction_statistics()
            logger.info(f"📊 Estatísticas da sessão: {stats}")
        logger.info("👋 API encerrada")
    except Exception as e:
        logger.error(f"❌ Erro no shutdown: {str(e)}")


# Importar rotas
from app.routes import router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
