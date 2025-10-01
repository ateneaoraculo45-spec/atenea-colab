"""
Procesador de datos con 100+ targets optimizados
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.over_under_thresholds = {
            'goals': [0.5, 1.5, 2.5, 3.5, 4.5],
            'goals_ht': [0.5, 1.5],
            'corners': [7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
            'corners_ht': [3.5, 4.5, 5.5],
            'cards': [2.5, 3.5, 4.5, 5.5],
            'cards_ht': [1.5, 2.5]
        }
    
    def prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preparando 100+ targets...")
        
        df = df.copy()
        
        df = self._add_base_targets(df)
        df = self._add_over_under_targets(df)
        df = self._add_combined_targets(df)
        df = self._add_half_time_targets(df)
        df = self._add_market_targets(df)
        
        logger.info(f"Total de targets creados: {len([col for col in df.columns if col.startswith('Target_')])}")
        return df
    
    def _add_base_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Target_FTHG'] = df['FTHG']
        df['Target_FTAG'] = df['FTAG']
        df['Target_HTHG'] = df['HTHG']
        df['Target_HTAG'] = df['HTAG']
        df['Target_HC'] = df['HC']
        df['Target_AC'] = df['AC']
        df['Target_HY'] = df['HY']
        df['Target_AY'] = df['AY']
        
        df['Target_FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        df['Target_HTR'] = df['HTR'].map({'H': 0, 'D': 1, 'A': 2})
        
        return df
    
    def _add_over_under_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        for threshold in self.over_under_thresholds['goals']:
            total_goals = df['FTHG'] + df['FTAG']
            df[f'Target_Over_{threshold}_Goals'] = (total_goals > threshold).astype(int)
            df[f'Target_Under_{threshold}_Goals'] = (total_goals < threshold).astype(int)
        
        for threshold in self.over_under_thresholds['goals_ht']:
            ht_goals = df['HTHG'] + df['HTAG']
            df[f'Target_Over_{threshold}_Goals_HT'] = (ht_goals > threshold).astype(int)
        
        if 'HC' in df.columns and 'AC' in df.columns:
            for threshold in self.over_under_thresholds['corners']:
                total_corners = df['HC'] + df['AC']
                df[f'Target_Over_{threshold}_Corners'] = (total_corners > threshold).astype(int)
                df[f'Target_Under_{threshold}_Corners'] = (total_corners < threshold).astype(int)
            
            for threshold in self.over_under_thresholds['corners_ht']:
                ht_corners = (df['HC'] + df['AC']) / 2
                df[f'Target_Over_{threshold}_Corners_HT'] = (ht_corners > threshold).astype(int)
        
        if 'HY' in df.columns and 'AY' in df.columns:
            for threshold in self.over_under_thresholds['cards']:
                total_cards = df['HY'] + df['AY']
                df[f'Target_Over_{threshold}_Cards'] = (total_cards > threshold).astype(int)
                df[f'Target_Under_{threshold}_Cards'] = (total_cards < threshold).astype(int)
            
            for threshold in self.over_under_thresholds['cards_ht']:
                ht_cards = (df['HY'] + df['AY']) / 2
                df[f'Target_Over_{threshold}_Cards_HT'] = (ht_cards > threshold).astype(int)
        
        return df
    
    def _add_combined_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Target_BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        df['Target_BTTS_HT'] = ((df['HTHG'] > 0) & (df['HTAG'] > 0)).astype(int)
        
        df['Target_HomeWin'] = (df['FTR'] == 'H').astype(int)
        df['Target_Draw'] = (df['FTR'] == 'D').astype(int)
        df['Target_AwayWin'] = (df['FTR'] == 'A').astype(int)
        
        df['Target_DoubleChance_1X'] = df['FTR'].isin(['H', 'D']).astype(int)
        df['Target_DoubleChance_12'] = df['FTR'].isin(['H', 'A']).astype(int)
        df['Target_DoubleChance_X2'] = df['FTR'].isin(['D', 'A']).astype(int)
        
        total_goals = df['FTHG'] + df['FTAG']
        df['Target_Over_2.5_BTTS'] = ((total_goals > 2.5) & (df['Target_BTTS'] == 1)).astype(int)
        
        df['Target_Home_CleanSheet'] = (df['FTAG'] == 0).astype(int)
        df['Target_Away_CleanSheet'] = (df['FTHG'] == 0).astype(int)
        
        df['Target_Home_ToScore'] = (df['FTHG'] > 0).astype(int)
        df['Target_Away_ToScore'] = (df['FTAG'] > 0).astype(int)
        
        return df
    
    def _add_half_time_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'HC' in df.columns and 'AC' in df.columns:
            df['Target_Home_Corners_HT'] = df['HC'] / 2
            df['Target_Away_Corners_HT'] = df['AC'] / 2
        
        if 'HY' in df.columns and 'AY' in df.columns:
            df['Target_Home_Cards_HT'] = df['HY'] / 2
            df['Target_Away_Cards_HT'] = df['AY'] / 2
        
        df['Target_2H_Goals'] = (df['FTHG'] - df['HTHG']) + (df['FTAG'] - df['HTAG'])
        
        for threshold in [0.5, 1.5, 2.5]:
            df[f'Target_2H_Over_{threshold}_Goals'] = (df['Target_2H_Goals'] > threshold).astype(int)
        
        return df
    
    def _add_market_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        total_goals = df['FTHG'] + df['FTAG']
        
        df['Target_HomeWin_Over_2.5'] = ((df['FTR'] == 'H') & (total_goals > 2.5)).astype(int)
        df['Target_AwayWin_Over_2.5'] = ((df['FTR'] == 'A') & (total_goals > 2.5)).astype(int)
        df['Target_Draw_Under_2.5'] = ((df['FTR'] == 'D') & (total_goals < 2.5)).astype(int)
        
        if 'HC' in df.columns and 'AC' in df.columns:
            total_corners = df['HC'] + df['AC']
            df['Target_HomeWin_Over_9.5_Corners'] = ((df['FTR'] == 'H') & (total_corners > 9.5)).astype(int)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Limpiando datos...")
        
        df = df[df['Datetime'].notna()].copy()
        df = df[df['HomeTeam'].notna()].copy()
        df = df[df['AwayTeam'].notna()].copy()
        
        essential_cols = ['FTHG', 'FTAG', 'FTR']
        df = df.dropna(subset=essential_cols)
        
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        logger.info(f"Datos limpios: {len(df)} partidos")
        return df
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8):
        df = df.sort_values('Datetime')
        
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df
