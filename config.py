"""
Configuración centralizada del sistema Atenea
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

POPULAR_START_SEASON = "0001"
POPULAR_END_SEASON = "2526"

EXTRAS_COUNTRIES = [
    'ARG', 'AUT', 'BRA', 'CHN', 'DNK', 'FIN', 
    'IRL', 'JPN', 'MEX', 'NOR', 'POL', 'ROU', 
    'RUS', 'SWE', 'SWZ', 'USA'
]

CANONICAL_COLUMNS = [
    'Div', 'Country', 'League', 'Season', 'Date', 'Time', 'Datetime',
    'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    'Referee', 'B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA',
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'BFEH', 'BFED', 'BFEA'
]

COLUMN_MAPPING = {
    'home': 'HomeTeam',
    'away': 'AwayTeam',
    'hg': 'FTHG',
    'ag': 'FTAG',
    'res': 'FTR',
    'psch': 'PSH',
    'pscd': 'PSD',
    'psca': 'PSA',
    'maxch': 'MaxH',
    'maxcd': 'MaxD',
    'maxca': 'MaxA',
    'avgch': 'AvgH',
    'avgcd': 'AvgD',
    'avgca': 'AvgA'
}

NUMERIC_COLUMNS = [
    'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
    'B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA',
    'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
    'BFEH', 'BFED', 'BFEA'
]

XGBOOST_PARAMS_GOALS_FT = {
    'objective': 'count:poisson',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS_GOALS_HT = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS_CORNERS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS_CARDS = {
    'objective': 'count:poisson',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS_RESULTS = {
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

SES_WINDOWS = [5, 10]
SES_ALPHA = 0.3
EWMA_SPAN = 5

OVER_UNDER_THRESHOLDS = {
    'goals': [0.5, 1.5, 2.5, 3.5, 4.5],
    'goals_ht': [0.5, 1.5],
    'corners': [7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
    'corners_ht': [3.5, 4.5, 5.5],
    'cards': [2.5, 3.5, 4.5, 5.5],
    'cards_ht': [1.5, 2.5]
}

TENSION_RULES_PATH = os.path.join(DATA_DIR, 'tension_rules.json')
