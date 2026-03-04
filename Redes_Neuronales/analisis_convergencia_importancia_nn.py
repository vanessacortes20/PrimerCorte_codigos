# ============================================================
# ANÁLISIS DE RED NEURONAL:
# CONVERGENCIA + CONTRIBUCIÓN DE VARIABLES
# Dataset: Iris
# ============================================================
# Objetivo:
# 1. Analizar la convergencia del entrenamiento del modelo
#    mediante el comportamiento del error (MSE).
# 2. Estimar la contribución relativa de cada variable
#    a través de los pesos de la primera capa.
#
# Este archivo no solo entrena el modelo, sino que
# profundiza en su comportamiento interno.
# ============================================================

# ------------------------------------------------------------
# Importación de librerías
# ------------------------------------------------------------
# numpy: cálculos numéricos.
# pandas: estructuración tabular.
# seaborn: carga del dataset Iris.
# matplotlib: visualización gráfica.
# train_test_split: separación entrenamiento/prueba.
# StandardScaler: normalización de datos.
# keras + layers: construcción de la red neuronal.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# Datos
# =========================
# Se carga el dataset Iris.
# Variable objetivo (continua): sepal_length.
# Variables predictoras:
#   - sepal_width
#   - petal_length
#   - petal_width
iris = sns.load_dataset("iris")
X = iris[["sepal_width", "petal_length", "petal_width"]].values
y = iris["sepal_length"].values

# Se guardan los nombres para visualización posterior.
feature_names = ["sepal_width", "petal_length", "petal_width"]

# División 80% entrenamiento / 20% prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalamiento de variables.
# Fundamental en redes neuronales para:
# - Mejorar estabilidad del gradiente
# - Acelerar convergencia
# - Evitar sesgo por escala
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =========================
# Modelo
# =========================
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

# Entrenamiento:
# epochs=150 permite observar mejor la convergencia.
# validation_split=0.2 permite detectar posible sobreajuste.
history = model.fit(
    X_train_s,
    y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# =========================
# 1. Gráfico de Convergencia
# =========================
# Se analiza la evolución del error:
# - loss → error en entrenamiento
# - val_loss → error en validación
#
# Interpretación en sustentación:
# - Si ambas curvas bajan y se estabilizan → buena convergencia.
# - Si val_loss sube mientras loss baja → sobreajuste (overfitting).
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Convergencia del Modelo")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend(["Train", "Validation"])
plt.show()

# =========================
# 2. Contribución de Variables (Primera Capa)
# =========================
# Se extraen los pesos de la primera capa.
# Dimensión: (n_variables, n_neuronas)
weights = model.layers[0].get_weights()[0]

# Se calcula la media del valor absoluto de los pesos
# por variable como aproximación de importancia relativa.
# (No es una técnica exacta de interpretabilidad,
# pero da una estimación básica de influencia).
contribucion = np.mean(np.abs(weights), axis=1)

# Se organiza en DataFrame para visualización.
df_contrib = pd.DataFrame({
    "Variable": feature_names,
    "Importancia": contribucion
})

# Visualización de importancia relativa.
plt.figure()
plt.bar(df_contrib["Variable"], df_contrib["Importancia"])
plt.title("Contribución de Variables a la Primera Capa")
plt.ylabel("Importancia Promedio |Peso|")
plt.show()