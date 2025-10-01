"""
Ingeniería de características avanzada con ratings segregadas, EWMA y SES
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, ses_windows=[5, 10], ses_alpha=0.3, ewma_span=5):
        self.ses_windows = ses_windows
        self.ses_alpha = ses_alpha
        self.ewma_span = ewma_span
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creando features avanzadas...")
        
        df = df.copy()
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        df = self.calculate_strength_ratings(df)
        df = self.calculate_ses_features(df)
        df = self.calculate_form_features(df)
        df = self.calculate_head_to_head(df)
        df = self.calculate_variance_features(df)
        df = self.calculate_streak_features(df)
        
        logger.info(f"Features creadas: {len([col for col in df.columns if col.startswith('Feature_')])}")
        
        return df
    
    def calculate_strength_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando ratings segregadas por condición de campo (sin fuga de datos)...")
        
        df['Feature_Home_OffStrength'] = 1.0
        df['Feature_Home_DefStrength'] = 1.0
        df['Feature_Away_OffStrength'] = 1.0
        df['Feature_Away_DefStrength'] = 1.0
        
        for idx, row in df.iterrows():
            current_date = row['Datetime']
            league = row['Div']
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            historical = df[(df['Datetime'] < current_date) & (df['Div'] == league)]
            
            if len(historical) == 0:
                continue
            
            league_avg_home_goals = historical['FTHG'].mean() if len(historical) > 0 else 1
            league_avg_away_goals = historical['FTAG'].mean() if len(historical) > 0 else 1
            
            home_historical_home = historical[historical['HomeTeam'] == home_team]
            if len(home_historical_home) > 0:
                team_home_scored = home_historical_home['FTHG'].mean()
                team_home_conceded = home_historical_home['FTAG'].mean()
                
                df.at[idx, 'Feature_Home_OffStrength'] = team_home_scored / league_avg_home_goals if league_avg_home_goals > 0 else 1.0
                df.at[idx, 'Feature_Home_DefStrength'] = team_home_conceded / league_avg_away_goals if league_avg_away_goals > 0 else 1.0
            
            away_historical_away = historical[historical['AwayTeam'] == away_team]
            if len(away_historical_away) > 0:
                team_away_scored = away_historical_away['FTAG'].mean()
                team_away_conceded = away_historical_away['FTHG'].mean()
                
                df.at[idx, 'Feature_Away_OffStrength'] = team_away_scored / league_avg_away_goals if league_avg_away_goals > 0 else 1.0
                df.at[idx, 'Feature_Away_DefStrength'] = team_away_conceded / league_avg_home_goals if league_avg_home_goals > 0 else 1.0
        
        return df
    
    def calculate_ses_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando SES features con EWMA...")
        
        for window in self.ses_windows:
            for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
                home_matches = df[df['HomeTeam'] == team].copy()
                away_matches = df[df['AwayTeam'] == team].copy()
                
                if len(home_matches) > 0:
                    df.loc[df['HomeTeam'] == team, f'Feature_Home_SES_{window}_GoalsScored'] = \
                        home_matches['FTHG'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    df.loc[df['HomeTeam'] == team, f'Feature_Home_SES_{window}_GoalsConceded'] = \
                        home_matches['FTAG'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    if 'HC' in df.columns:
                        df.loc[df['HomeTeam'] == team, f'Feature_Home_SES_{window}_Corners'] = \
                            home_matches['HC'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    if 'HY' in df.columns:
                        df.loc[df['HomeTeam'] == team, f'Feature_Home_SES_{window}_Cards'] = \
                            home_matches['HY'].ewm(span=window, adjust=False).mean().shift(1)
                
                if len(away_matches) > 0:
                    df.loc[df['AwayTeam'] == team, f'Feature_Away_SES_{window}_GoalsScored'] = \
                        away_matches['FTAG'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    df.loc[df['AwayTeam'] == team, f'Feature_Away_SES_{window}_GoalsConceded'] = \
                        away_matches['FTHG'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    if 'AC' in df.columns:
                        df.loc[df['AwayTeam'] == team, f'Feature_Away_SES_{window}_Corners'] = \
                            away_matches['AC'].ewm(span=window, adjust=False).mean().shift(1)
                    
                    if 'AY' in df.columns:
                        df.loc[df['AwayTeam'] == team, f'Feature_Away_SES_{window}_Cards'] = \
                            away_matches['AY'].ewm(span=window, adjust=False).mean().shift(1)
        
        ses_cols = [col for col in df.columns if 'SES' in col]
        for col in ses_cols:
            df[col] = df[col].ffill().fillna(0)
        
        return df
    
    def calculate_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando métricas de volatilidad EWMA...")
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            home_matches = df[df['HomeTeam'] == team].copy()
            away_matches = df[df['AwayTeam'] == team].copy()
            
            if len(home_matches) > 0:
                df.loc[df['HomeTeam'] == team, 'Feature_Home_EWMA_Var_Goals'] = \
                    home_matches['FTHG'].ewm(span=self.ewma_span).var().shift(1)
                
                if 'HC' in df.columns:
                    df.loc[df['HomeTeam'] == team, 'Feature_Home_EWMA_Var_Corners'] = \
                        home_matches['HC'].ewm(span=self.ewma_span).var().shift(1)
                
                if 'HY' in df.columns:
                    df.loc[df['HomeTeam'] == team, 'Feature_Home_EWMA_Var_Cards'] = \
                        home_matches['HY'].ewm(span=self.ewma_span).var().shift(1)
            
            if len(away_matches) > 0:
                df.loc[df['AwayTeam'] == team, 'Feature_Away_EWMA_Var_Goals'] = \
                    away_matches['FTAG'].ewm(span=self.ewma_span).var().shift(1)
                
                if 'AC' in df.columns:
                    df.loc[df['AwayTeam'] == team, 'Feature_Away_EWMA_Var_Corners'] = \
                        away_matches['AC'].ewm(span=self.ewma_span).var().shift(1)
                
                if 'AY' in df.columns:
                    df.loc[df['AwayTeam'] == team, 'Feature_Away_EWMA_Var_Cards'] = \
                        away_matches['AY'].ewm(span=self.ewma_span).var().shift(1)
        
        var_cols = [col for col in df.columns if 'EWMA_Var' in col]
        df[var_cols] = df[var_cols].fillna(0)
        
        return df
    
    def calculate_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando features de forma...")
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            
            points = []
            for idx, row in team_matches.iterrows():
                if row['HomeTeam'] == team:
                    if row['FTR'] == 'H':
                        points.append(3)
                    elif row['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if row['FTR'] == 'A':
                        points.append(3)
                    elif row['FTR'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
            
            team_matches['points'] = points
            
            team_matches['Feature_Form_L5'] = team_matches['points'].rolling(5, min_periods=1).sum().shift(1)
            team_matches['Feature_Form_L10'] = team_matches['points'].rolling(10, min_periods=1).sum().shift(1)
            
            for idx, row in team_matches.iterrows():
                if row['HomeTeam'] == team:
                    df.loc[idx, 'Feature_Home_Form_L5'] = row['Feature_Form_L5']
                    df.loc[idx, 'Feature_Home_Form_L10'] = row['Feature_Form_L10']
                else:
                    df.loc[idx, 'Feature_Away_Form_L5'] = row['Feature_Form_L5']
                    df.loc[idx, 'Feature_Away_Form_L10'] = row['Feature_Form_L10']
        
        df['Feature_Home_Form_L5'] = df['Feature_Home_Form_L5'].fillna(0)
        df['Feature_Home_Form_L10'] = df['Feature_Home_Form_L10'].fillna(0)
        df['Feature_Away_Form_L5'] = df['Feature_Away_Form_L5'].fillna(0)
        df['Feature_Away_Form_L10'] = df['Feature_Away_Form_L10'].fillna(0)
        
        return df
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando head to head...")
        
        df['Feature_H2H_HomeWins'] = 0
        df['Feature_H2H_Draws'] = 0
        df['Feature_H2H_AwayWins'] = 0
        
        for idx, row in df.iterrows():
            h2h = df[
                ((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam'])) |
                ((df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam']))
            ]
            h2h = h2h[h2h['Datetime'] < row['Datetime']]
            
            if len(h2h) > 0:
                home_wins = len(h2h[
                    ((h2h['HomeTeam'] == row['HomeTeam']) & (h2h['FTR'] == 'H')) |
                    ((h2h['AwayTeam'] == row['HomeTeam']) & (h2h['FTR'] == 'A'))
                ])
                draws = len(h2h[h2h['FTR'] == 'D'])
                away_wins = len(h2h[
                    ((h2h['HomeTeam'] == row['AwayTeam']) & (h2h['FTR'] == 'H')) |
                    ((h2h['AwayTeam'] == row['AwayTeam']) & (h2h['FTR'] == 'A'))
                ])
                
                df.at[idx, 'Feature_H2H_HomeWins'] = home_wins
                df.at[idx, 'Feature_H2H_Draws'] = draws
                df.at[idx, 'Feature_H2H_AwayWins'] = away_wins
        
        return df
    
    def calculate_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando rachas...")
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            
            for idx, row in team_matches.iterrows():
                recent = team_matches[team_matches['Datetime'] < row['Datetime']].tail(5)
                
                if len(recent) > 0:
                    wins = 0
                    for _, match in recent.iterrows():
                        if (match['HomeTeam'] == team and match['FTR'] == 'H') or \
                           (match['AwayTeam'] == team and match['FTR'] == 'A'):
                            wins += 1
                    
                    if row['HomeTeam'] == team:
                        df.at[idx, 'Feature_Home_WinStreak'] = wins
                    else:
                        df.at[idx, 'Feature_Away_WinStreak'] = wins
        
        df['Feature_Home_WinStreak'] = df['Feature_Home_WinStreak'].fillna(0)
        df['Feature_Away_WinStreak'] = df['Feature_Away_WinStreak'].fillna(0)
        
        return df
