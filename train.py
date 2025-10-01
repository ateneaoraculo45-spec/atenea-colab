"""
Script de entrenamiento optimizado para Google Colab
Con TimeSeriesSplit y métricas completas
"""

import sys
import os
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("INICIANDO ENTRENAMIENTO DE ATENEA")
    logger.info("=" * 80)
    
    data_path = 'data/processed/historico_completo.csv'
    
    if not os.path.exists(data_path):
        logger.error(f"Archivo no encontrado: {data_path}")
        logger.info("Ejecuta primero: python sync_data.py")
        sys.exit(1)
    
    logger.info(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['Datetime'])
    logger.info(f"Datos cargados: {len(df)} partidos")
    
    processor = DataProcessor()
    df = processor.clean_data(df)
    logger.info(f"Datos limpios: {len(df)} partidos")
    
    df = processor.prepare_targets(df)
    logger.info("Targets preparados")
    
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    logger.info("Features creadas")
    
    train_df, test_df = processor.split_data(df, train_ratio=0.8)
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    manager = ModelManager()
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENANDO MODELO: GOLES FT (90min)")
    logger.info("=" * 80)
    manager.train_goals_ft_model(train_df, optimize_params=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENANDO MODELO: GOLES HT (45min)")
    logger.info("=" * 80)
    manager.train_goals_ht_model(train_df, optimize_params=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENANDO MODELO: CÓRNERS")
    logger.info("=" * 80)
    manager.train_corners_model(train_df, optimize_params=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENANDO MODELO: TARJETAS")
    logger.info("=" * 80)
    manager.train_cards_model(train_df, optimize_params=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENANDO MODELO: RESULTADOS (FTR)")
    logger.info("=" * 80)
    manager.train_results_model(train_df, optimize_params=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 80)
    logger.info("\nModelos guardados en: models/")
    logger.info("  - goals_ft_model.joblib")
    logger.info("  - goals_ht_model.joblib")
    logger.info("  - corners_model.joblib")
    logger.info("  - cards_model.joblib")
    logger.info("  - results_model.joblib")
    
    logger.info("\nPARA USAR EN REPLIT:")
    logger.info("1. Descarga todos los archivos .joblib de la carpeta models/")
    logger.info("2. Sube los archivos a la carpeta models/ en Replit")
    logger.info("3. Ejecuta: python main.py")

if __name__ == '__main__':
    main()
