# ============================================================
# RED NEURONAL PARA REGRESIÓN - DATASET IRIS
# ============================================================
# Objetivo:
# Construir un modelo de red neuronal artificial (ANN)
# para predecir la variable continua sepal_length
# a partir de:
#   - sepal_width
#   - petal_length
#   - petal_width
#
# A diferencia de la regresión lineal, la red neuronal
# puede capturar relaciones no lineales entre variables.
# ============================================================

# ------------------------------------------------------------
# Importación de librerías
# ------------------------------------------------------------
# numpy: operaciones numéricas.
# seaborn: carga del dataset Iris.
# train_test_split: separación entrenamiento/prueba.
# StandardScaler: normalización de variables.
# keras + layers: construcción de la red neuronal.
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Carga del dataset
# ------------------------------------------------------------
iris = sns.load_dataset("iris")

# ------------------------------------------------------------
# Definición de variables
# ------------------------------------------------------------
# X → Variables independientes (predictoras)
# y → Variable dependiente (objetivo continuo)
X = iris[["sepal_width", "petal_length", "petal_width"]].values
y = iris["sepal_length"].values

# ------------------------------------------------------------
# División entrenamiento / prueba
# ------------------------------------------------------------
# 80% entrenamiento, 20% prueba.
# random_state=42 garantiza reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Escalamiento de variables
# ------------------------------------------------------------
# Las redes neuronales requieren datos escalados porque:
# - Mejoran estabilidad numérica
# - Aceleran convergencia del gradiente
# - Evitan que una variable domine por su escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# Construcción de la arquitectura de la red
# ------------------------------------------------------------
# Capa 1:
# 16 neuronas con activación ReLU
# input_shape=(3,) indica que hay 3 variables de entrada.
#
# Capa 2:
# 8 neuronas con ReLU.
#
# Capa final:
# 1 neurona SIN activación porque es un problema
# de regresión (salida continua).
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

# ------------------------------------------------------------
# Compilación del modelo
# ------------------------------------------------------------
# optimizer="adam":
# Algoritmo de optimización adaptativo que combina
# Momentum y RMSProp.
#
# loss="mse":
# Error Cuadrático Medio, adecuado para regresión.
#
# metrics=["mae"]:
# Error Absoluto Medio, más interpretable en unidades reales.
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ------------------------------------------------------------
# Entrenamiento del modelo
# ------------------------------------------------------------
# epochs=100:
# Número de veces que el modelo ve todo el dataset.
#
# batch_size=16:
# Tamaño de subconjuntos para actualización de pesos.
#
# validation_split=0.2:
# 20% del entrenamiento se usa para validación interna
# y detectar posible sobreajuste.
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ------------------------------------------------------------
# Evaluación en conjunto de prueba
# ------------------------------------------------------------
# Aquí medimos capacidad de generalización.
loss, mae = model.evaluate(X_test, y_test, verbose=0)

# ------------------------------------------------------------
# Resultados finales
# ------------------------------------------------------------
# Test MSE → penaliza errores grandes.
# Test MAE → error promedio absoluto en unidades reales.
print("Test MSE:", loss)
print("Test MAE:", mae)