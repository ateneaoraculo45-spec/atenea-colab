# 🎯 ATENEA - Sistema de Predicción de Fútbol

## 🚀 Uso en Google Colab

### 1. Clonar repositorio
```python
!git clone https://github.com/ateneaoraculo45-spec/atenea-colab.git
%cd atenea-colab
```

### 2. Instalar dependencias
```python
!pip install -r requirements.txt
```

### 3. Entrenar modelos
```python
!python train.py
```

### 4. Descargar modelos entrenados
```python
from google.colab import files

# Descargar todos los modelos
models = ['goals_ft_model.joblib', 'goals_ht_model.joblib', 'corners_model.joblib', 'cards_model.joblib', 'results_model.joblib']
for model in models:
    files.download(f'models/{model}')
```

## 📊 Archivos Incluidos

- `historico_completo.csv` - Base de datos histórica (2000-2026)
- `train.py` - Script de entrenamiento con 5 modelos XGBoost
- `config.py`, `data_processor.py`, `feature_engineer.py`, `model_manager.py` - Módulos del sistema
- `requirements.txt` - Dependencias necesarias

## ⚡ Características

- **5 modelos XGBoost** (goles FT, goles HT, córners, tarjetas, resultados)
- **100+ targets** de predicción
- **Distribución Binomial Negativa** para probabilidades
- **Expected Value (EV)** para picks inteligentes

¡Entrena en Colab y obtén modelos precisos! 🚀⚽
