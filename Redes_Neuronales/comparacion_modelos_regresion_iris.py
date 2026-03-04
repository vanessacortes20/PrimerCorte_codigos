# ============================================================
# COMPARACIÓN DE MODELOS:
# REGRESIÓN LINEAL MÚLTIPLE vs RED NEURONAL
# Dataset: Iris
# ============================================================
# Objetivo:
# Comparar el desempeño predictivo de dos enfoques:
#   1. Modelo estadístico clásico (Regresión Lineal)
#   2. Modelo de Machine Learning no lineal (Red Neuronal)
#
# Se evalúan mediante métricas:
#   - MSE (Error Cuadrático Medio)
#   - MAE (Error Absoluto Medio)
#   - R² (Coeficiente de determinación)
# ============================================================

# ------------------------------------------------------------
# Importación de librerías
# ------------------------------------------------------------
# pandas: para estructurar las métricas en DataFrame.
# sklearn.metrics: métricas de evaluación.
# LinearRegression: modelo estadístico clásico.
# train_test_split: separación entrenamiento/prueba.
# StandardScaler: escalamiento necesario para redes neuronales.
# seaborn: carga del dataset Iris.
# numpy: soporte numérico.
# keras + layers: construcción de red neuronal.
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Carga y preparación de datos
# ------------------------------------------------------------
# Variable objetivo: sepal_length (continua)
# Variables predictoras:
#   - sepal_width
#   - petal_length
#   - petal_width
iris = sns.load_dataset("iris")
X = iris[["sepal_width", "petal_length", "petal_width"]].values
y = iris["sepal_length"].values

# División 80% entrenamiento, 20% prueba
# random_state garantiza reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 1. Regresión Lineal Múltiple
# =========================
# Modelo que asume relación lineal:
# y = β0 + β1X1 + β2X2 + β3X3
# Minimiza el Error Cuadrático Medio.
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicciones sobre el conjunto de prueba
y_pred_lr = lr.predict(X_test)

# =========================
# 2. Red Neuronal
# =========================
# Se requiere escalamiento previo para estabilidad del entrenamiento.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Arquitectura:
# - 16 neuronas ReLU
# - 8 neuronas ReLU
# - 1 neurona salida (regresión → sin activación)
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

# Compilación:
# - Adam: optimizador adaptativo
# - MSE: función de pérdida para regresión
# - MAE: métrica complementaria
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Entrenamiento
model.fit(X_train_s, y_train, epochs=100, batch_size=16, verbose=0)

# Predicciones del modelo neuronal
y_pred_nn = model.predict(X_test_s).flatten()

# =========================
# Métricas de Evaluación
# =========================
# MSE: penaliza más los errores grandes.
# MAE: error promedio absoluto en unidades reales.
# R2: proporción de varianza explicada por el modelo.
metricas = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Red Neuronal"],
    "MSE": [
        mean_squared_error(y_test, y_pred_lr),
        mean_squared_error(y_test, y_pred_nn)
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_nn)
    ],
    "R2": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_nn)
    ]
})

# Impresión comparativa final
print(metricas)