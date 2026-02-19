"""
Módulo de treinamento do modelo.
Implementa a pipeline completa de treinamento, validação e salvamento do modelo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import json
from datetime import datetime

from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Classe para treinar e validar modelos de Machine Learning."""
    
    def __init__(self, model_name: str = 'random_forest'):
        """
        Inicializa o treinador de modelos.
        
        Args:
            model_name: Nome do modelo a ser usado ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
        """
        self.model_name = model_name
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}
        
    def get_model(self) -> Any:
        """
        Retorna o modelo de acordo com o nome especificado.
        
        Returns:
            Modelo de ML inicializado
        """
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=1.0, random_state=42)
        }
        
        if self.model_name not in models:
            logger.warning(f"Modelo {self.model_name} não encontrado. Usando Random Forest.")
            self.model_name = 'random_forest'
        
        return models[self.model_name]
    
    def get_param_grid(self) -> Dict:
        """
        Retorna o grid de hiperparâmetros para busca.
        
        Returns:
            Dicionário com grid de parâmetros
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        return param_grids.get(self.model_name, {})
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   tune_hyperparameters: bool = True) -> None:
        """
        Treina o modelo com os dados fornecidos.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            tune_hyperparameters: Se True, realiza busca de hiperparâmetros
        """
        logger.info(f"=== Iniciando treinamento do modelo: {self.model_name} ===")
        logger.info(f"Shape dos dados de treino: X={X_train.shape}, y={y_train.shape}")
        
        if tune_hyperparameters:
            logger.info("Realizando busca de hiperparâmetros com GridSearchCV")
            base_model = self.get_model()
            param_grid = self.get_param_grid()
            
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
                
                logger.info(f"Melhores parâmetros: {self.best_params}")
                logger.info(f"Melhor score (neg_MSE): {grid_search.best_score_:.4f}")
            else:
                logger.info("Sem grid de parâmetros definido. Treinando com parâmetros padrão.")
                self.model = base_model
                self.model.fit(X_train, y_train)
        else:
            logger.info("Treinando com parâmetros padrão (sem tuning)")
            self.model = self.get_model()
            self.model.fit(X_train, y_train)
        
        # Calcular feature importance para modelos tree-based
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 features mais importantes:")
            logger.info(f"\n{self.feature_importance.head(10)}")
        
        logger.info("=== Treinamento concluído ===")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Avalia o modelo nos dados de teste.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info("=== Avaliando modelo ===")
        
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda. Execute train_model() primeiro.")
        
        # Fazer predições
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calcular MAPE (Mean Absolute Percentage Error)
        # Evitar divisão por zero
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.inf
        
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"MSE (Mean Squared Error): {mse:.4f}")
        logger.info(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        logger.info(f"MAE (Mean Absolute Error): {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        
        # Análise de resíduos
        residuals = y_test - y_pred
        logger.info(f"\nAnálise de Resíduos:")
        logger.info(f"Média dos resíduos: {residuals.mean():.4f}")
        logger.info(f"Std dos resíduos: {residuals.std():.4f}")
        logger.info(f"Min resíduo: {residuals.min():.4f}")
        logger.info(f"Max resíduo: {residuals.max():.4f}")
        
        logger.info("=== Avaliação concluída ===")
        return self.metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Realiza validação cruzada.
        
        Args:
            X: Features completas
            y: Target completo
            cv: Número de folds
            
        Returns:
            Dicionário com métricas de validação cruzada
        """
        logger.info(f"=== Realizando validação cruzada com {cv} folds ===")
        
        if self.model is None:
            self.model = self.get_model()
        
        # Validação cruzada com diferentes métricas
        cv_mse = -cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_mae = -cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(self.model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        
        cv_metrics = {
            'cv_mse_mean': cv_mse.mean(),
            'cv_mse_std': cv_mse.std(),
            'cv_rmse_mean': np.sqrt(cv_mse.mean()),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
        
        logger.info(f"CV MSE: {cv_metrics['cv_mse_mean']:.4f} (+/- {cv_metrics['cv_mse_std']:.4f})")
        logger.info(f"CV RMSE: {cv_metrics['cv_rmse_mean']:.4f}")
        logger.info(f"CV MAE: {cv_metrics['cv_mae_mean']:.4f} (+/- {cv_metrics['cv_mae_std']:.4f})")
        logger.info(f"CV R²: {cv_metrics['cv_r2_mean']:.4f} (+/- {cv_metrics['cv_r2_std']:.4f})")
        
        logger.info("=== Validação cruzada concluída ===")
        return cv_metrics
    
    def save_model(self, output_dir: str = 'models') -> None:
        """
        Salva o modelo treinado e metadados.
        
        Args:
            output_dir: Diretório onde salvar o modelo
        """
        logger.info(f"=== Salvando modelo em {output_dir} ===")
        
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelo
        model_filename = output_path / f'model_{self.model_name}_{timestamp}.pkl'
        joblib.dump(self.model, model_filename)
        logger.info(f"Modelo salvo: {model_filename}")
        
        # Salvar modelo como "latest" também
        latest_model_filename = output_path / f'model_{self.model_name}_latest.pkl'
        joblib.dump(self.model, latest_model_filename)
        logger.info(f"Modelo salvo (latest): {latest_model_filename}")
        
        # Salvar metadados
        metadata = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'model_filename': str(model_filename.name)
        }
        
        metadata_filename = output_path / f'metadata_{self.model_name}_{timestamp}.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadados salvos: {metadata_filename}")
        
        # Salvar feature importance se disponível
        if self.feature_importance is not None:
            importance_filename = output_path / f'feature_importance_{self.model_name}_{timestamp}.csv'
            self.feature_importance.to_csv(importance_filename, index=False)
            logger.info(f"Feature importance salva: {importance_filename}")
        
        logger.info("=== Salvamento concluído ===")
    
    def load_model(self, model_path: str) -> None:
        """
        Carrega um modelo salvo.
        
        Args:
            model_path: Caminho para o arquivo do modelo
        """
        logger.info(f"Carregando modelo de {model_path}")
        self.model = joblib.load(model_path)
        logger.info("Modelo carregado com sucesso")


def train_pipeline(data_path: str, 
                   model_name: str = 'random_forest',
                   test_size: float = 0.2,
                   tune_hyperparameters: bool = True,
                   output_dir: str = 'models') -> Tuple[ModelTrainer, DataPreprocessor, FeatureEngineer]:
    """
    Pipeline completo de treinamento.
    
    Args:
        data_path: Caminho para o arquivo de dados
        model_name: Nome do modelo a usar
        test_size: Proporção dos dados para teste
        tune_hyperparameters: Se deve realizar busca de hiperparâmetros
        output_dir: Diretório para salvar o modelo
        
    Returns:
        Tupla (ModelTrainer, DataPreprocessor, FeatureEngineer)
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO")
    logger.info("=" * 80)
    
    # 1. Pré-processamento
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    
    # 2. Feature Engineering (antes do split para evitar leakage)
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    # 3. Processamento final
    X, y = preprocessor.preprocess_pipeline(df_engineered, fit=True)
    
    # 4. Split treino/teste
    logger.info(f"\nDividindo dados: {test_size*100}% para teste")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 5. Treinamento
    trainer = ModelTrainer(model_name=model_name)
    trainer.train_model(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
    
    # 6. Avaliação
    trainer.evaluate_model(X_test, y_test)
    
    # 7. Validação cruzada
    trainer.cross_validate(X, y, cv=5)
    
    # 8. Salvar modelo
    trainer.save_model(output_dir=output_dir)
    
    # 9. Salvar preprocessor e feature engineer
    output_path = Path(output_dir)
    joblib.dump(preprocessor, output_path / 'preprocessor_latest.pkl')
    joblib.dump(engineer, output_path / 'feature_engineer_latest.pkl')
    logger.info("Preprocessor e Feature Engineer salvos")
    
    logger.info("=" * 80)
    logger.info("PIPELINE DE TREINAMENTO CONCLUÍDO")
    logger.info("=" * 80)
    
    return trainer, preprocessor, engineer


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "../data/PEDE2022.csv"
    
    train_pipeline(
        data_path=data_path,
        model_name='random_forest',
        test_size=0.2,
        tune_hyperparameters=True,
        output_dir='app/model'  # Modelos salvos em app/model/
    )
