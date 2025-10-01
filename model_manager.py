"""
Gestor de modelos con XGBoost y búsqueda de hiperparámetros
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, log_loss
import xgboost as xgb
import joblib
import logging
from typing import Dict, Tuple, List
import os

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        feature_cols = [col for col in df.columns if col.startswith('Feature_')]
        return feature_cols
    
    def train_goals_ft_model(self, df: pd.DataFrame, params: Dict = None, optimize_params: bool = False):
        logger.info("Entrenando modelo de goles FT (90min)...")
        
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].fillna(0)
        y = df[['Target_FTHG', 'Target_FTAG']].fillna(0)
        
        if params is None:
            params = {
                'objective': 'count:poisson',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if optimize_params:
            params = self._optimize_hyperparameters(X, y, base_params=params)
        
        scaler = StandardScaler()
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        
        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Goals FT MAE: {mae:.4f}")
        
        model_path = os.path.join(self.models_dir, 'goals_ft_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        return pipeline
    
    def train_goals_ht_model(self, df: pd.DataFrame, params: Dict = None, optimize_params: bool = False):
        logger.info("Entrenando modelo de goles HT (45min)...")
        
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].fillna(0)
        y = df[['Target_HTHG', 'Target_HTAG']].fillna(0)
        
        if params is None:
            params = {
                'objective': 'count:poisson',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if optimize_params:
            params = self._optimize_hyperparameters(X, y, base_params=params)
        
        scaler = StandardScaler()
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        
        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Goals HT MAE: {mae:.4f}")
        
        model_path = os.path.join(self.models_dir, 'goals_ht_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        return pipeline
    
    def train_corners_model(self, df: pd.DataFrame, params: Dict = None, optimize_params: bool = False):
        logger.info("Entrenando modelo de córners...")
        
        df_clean = df[df['Target_HC'].notna() & df['Target_AC'].notna()].copy()
        
        feature_cols = self.get_feature_columns(df_clean)
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[['Target_HC', 'Target_AC']].fillna(0)
        
        if params is None:
            params = {
                'objective': 'count:poisson',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if optimize_params:
            params = self._optimize_hyperparameters(X, y, base_params=params)
        
        scaler = StandardScaler()
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        
        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Corners MAE: {mae:.4f}")
        
        model_path = os.path.join(self.models_dir, 'corners_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        return pipeline
    
    def train_cards_model(self, df: pd.DataFrame, params: Dict = None, optimize_params: bool = False):
        logger.info("Entrenando modelo de tarjetas...")
        
        df_clean = df[df['Target_HY'].notna() & df['Target_AY'].notna()].copy()
        
        feature_cols = self.get_feature_columns(df_clean)
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[['Target_HY', 'Target_AY']].fillna(0)
        
        if params is None:
            params = {
                'objective': 'count:poisson',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if optimize_params:
            params = self._optimize_hyperparameters(X, y, base_params=params)
        
        scaler = StandardScaler()
        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        
        y_pred = pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        logger.info(f"Cards MAE: {mae:.4f}")
        
        model_path = os.path.join(self.models_dir, 'cards_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        return pipeline
    
    def train_results_model(self, df: pd.DataFrame, params: Dict = None, optimize_params: bool = False):
        logger.info("Entrenando modelo clasificador de resultados...")
        
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].fillna(0)
        y = df['Target_FTR'].fillna(1)
        
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        scaler = StandardScaler()
        model = xgb.XGBClassifier(**params)
        
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        
        y_pred_proba = pipeline.predict_proba(X)
        logloss = log_loss(y, y_pred_proba)
        logger.info(f"Results LogLoss: {logloss:.4f}")
        
        model_path = os.path.join(self.models_dir, 'results_model.joblib')
        joblib.dump(pipeline, model_path)
        logger.info(f"Modelo guardado: {model_path}")
        
        return pipeline
    
    def _optimize_hyperparameters(self, X, y, base_params: Dict) -> Dict:
        logger.info("Optimizando hiperparámetros con TimeSeriesSplit...")
        
        param_grid = {
            'max_depth': [4, 5, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [150, 200]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = xgb.XGBRegressor(**base_params)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        if len(y.shape) > 1:
            grid_search.fit(X, y.iloc[:, 0])
        else:
            grid_search.fit(X, y)
        
        best_params = {**base_params, **grid_search.best_params_}
        logger.info(f"Mejores parámetros: {best_params}")
        
        return best_params
    
    def load_models(self) -> Dict:
        logger.info("Cargando modelos entrenados...")
        
        models = {}
        model_files = {
            'goals_ft': 'goals_ft_model.joblib',
            'goals_ht': 'goals_ht_model.joblib',
            'corners': 'corners_model.joblib',
            'cards': 'cards_model.joblib',
            'results': 'results_model.joblib'
        }
        
        for key, filename in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                models[key] = joblib.load(path)
                logger.info(f"Cargado: {key}")
            else:
                logger.warning(f"Modelo no encontrado: {filename}")
        
        return models
