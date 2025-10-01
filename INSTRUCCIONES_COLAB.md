# 🎯 ATENEA - Instrucciones para Google Colab

## 📋 Resumen del Sistema

Atenea es un sistema avanzado de predicción de fútbol que utiliza:
- **5 modelos XGBoost** (goles FT, goles HT, córners, tarjetas, resultados)
- **100+ targets** de predicción
- **Distribución Binomial Negativa** para probabilidades
- **Expected Value (EV)** para picks inteligentes
- **Firmas de tensión** para detección de patrones

## 🚀 Paso 1: Preparar Datos en Replit

El workflow "Sync Data" está descargando automáticamente toda la base de datos histórica (2000-2026).

Una vez completado, tendrás:
- `data/processed/historico_completo.csv` - Base de datos maestra con todos los partidos

## 📤 Paso 2: Subir a Google Colab

### Archivos que DEBES subir a Colab:

1. **Base de datos histórica:**
   - `data/processed/historico_completo.csv`

2. **Scripts de entrenamiento:**
   - `train.py`
   
3. **Módulos del sistema:**
   - `src/config.py`
   - `src/data_processor.py`
   - `src/feature_engineer.py`
   - `src/model_manager.py`

4. **requirements.txt**

## 💻 Paso 3: Código para Google Colab

### Celda 1: Instalar dependencias
```python
!pip install numpy pandas scikit-learn xgboost joblib pyarrow statsmodels optuna
```

### Celda 2: Subir archivos
```python
from google.colab import files
import os

# Crear estructura de carpetas
os.makedirs('data/processed', exist_ok=True)
os.makedirs('src', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Subir archivos (usa el botón de upload de Colab)
print("Sube los siguientes archivos:")
print("1. historico_completo.csv -> data/processed/")
print("2. train.py -> raíz")
print("3. Todos los archivos de src/ -> src/")
```

### Celda 3: Entrenar modelos
```python
# Ejecutar entrenamiento
!python train.py
```

### Celda 4: Descargar modelos entrenados
```python
from google.colab import files

# Descargar todos los modelos
files.download('models/goals_ft_model.joblib')
files.download('models/goals_ht_model.joblib')
files.download('models/corners_model.joblib')
files.download('models/cards_model.joblib')
files.download('models/results_model.joblib')
```

## 📥 Paso 4: Volver a Replit

1. **Sube los modelos** a la carpeta `models/` en Replit:
   - `goals_ft_model.joblib`
   - `goals_ht_model.joblib`
   - `corners_model.joblib`
   - `cards_model.joblib`
   - `results_model.joblib`

2. **Ejecuta el sistema de predicción:**
   ```bash
   python main.py
   ```

## 🔍 Estructura del Sistema

### Archivos Creados en Replit:

```
📁 Proyecto Atenea
├── 📄 sync_data.py          # Descarga datos históricos
├── 📄 train.py              # Entrenamiento (para Colab)
├── 📄 main.py               # Sistema de predicción
├── 📄 requirements.txt
│
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── 📂 populares/    # Ligas 2000-2026
│   │   ├── 📂 extras/       # Ligas extras (16 países)
│   │   └── 📂 fixtures/     # Partidos futuros
│   ├── 📂 processed/
│   │   ├── historico_populares.csv
│   │   ├── historico_extras.csv
│   │   └── historico_completo.csv  # ⭐ BASE DE DATOS MAESTRA
│   └── tension_rules.json
│
├── 📂 src/
│   ├── config.py            # Configuración
│   ├── data_processor.py    # 100+ targets
│   ├── feature_engineer.py  # Features avanzadas
│   ├── model_manager.py     # 5 modelos XGBoost
│   ├── prediction_engine.py # Motor probabilístico
│   └── output_formatter.py  # Formateo de salida
│
└── 📂 models/               # Modelos entrenados (subir desde Colab)
    ├── goals_ft_model.joblib
    ├── goals_ht_model.joblib
    ├── corners_model.joblib
    ├── cards_model.joblib
    └── results_model.joblib
```

## 📊 Características del Sistema

### 100+ Targets Implementados:
- ✅ Goles FT/HT (Casa/Visitante)
- ✅ Over/Under (0.5, 1.5, 2.5, 3.5, 4.5)
- ✅ Córners (Over/Under 7.5-12.5)
- ✅ Tarjetas (Over/Under 2.5-5.5)
- ✅ BTTS (Ambos equipos marcan)
- ✅ 1X2, Doble Oportunidad
- ✅ Mercados combinados (HomeWin + Over 2.5)

### Features Avanzadas:
- ✅ Ratings segregadas por condición (Local/Visitante)
- ✅ SES (Simple Exponential Smoothing) - ventanas 5 y 10
- ✅ EWMA Variance (volatilidad)
- ✅ Goles concedidos (forma defensiva)
- ✅ Head-to-Head histórico
- ✅ Rachas de victorias

### Motor Probabilístico:
- ✅ Distribución Binomial Negativa (más precisa que Poisson)
- ✅ Predicción segregada HT/FT
- ✅ Detección de firmas de tensión
- ✅ Filtro histórico dinámico
- ✅ Expected Value (EV) >= 5%

## 🎯 Uso del Sistema

Una vez que hayas subido los modelos desde Colab:

```bash
# Ver predicciones en consola
python main.py
```

El sistema generará:
- Predicciones en JSON: `predictions_*.json`
- Predicciones en CSV: `predictions_*.csv`
- Salida formateada en consola

## ⚡ Optimización para Colab

Si quieres optimizar hiperparámetros en Colab:

```python
# En train.py, cambia optimize_params a True
manager.train_goals_ft_model(train_df, optimize_params=True)
```

Esto ejecutará GridSearch con TimeSeriesSplit, pero tomará más tiempo.

## 🔄 Resincronizar Datos

Para actualizar la base de datos:

```bash
# En Replit
python sync_data.py

# Forzar descarga completa
python sync_data.py --force
```

## 📈 Expected Value (EV)

El sistema calcula automáticamente el EV comparando:
- Probabilidad del modelo
- Cuotas del mercado

Prioriza picks con **EV >= 5%** y los clasifica por confianza.

## 🎓 Próximos Pasos

1. ✅ Espera a que termine la descarga de datos (workflow "Sync Data")
2. ✅ Sube `historico_completo.csv` y archivos necesarios a Colab
3. ✅ Entrena los 5 modelos en Colab
4. ✅ Descarga los archivos .joblib
5. ✅ Sube los modelos a Replit
6. ✅ Ejecuta `python main.py` para ver predicciones

---

**¡Atenea está lista para predecir con precisión exponencial! 🚀⚽**
