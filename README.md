# Primer Corte: Análisis de Código 🧠💻

Este repositorio forma parte de la entrega del primer corte. El objetivo principal de este trabajo es **entender a profundidad el funcionamiento y la lógica** detrás de distintos modelos de Machine Learning y procesamiento de datos.

Por esta razón, cada archivo `.py` en este repositorio ha sido **meticulosamente analizado y comentado línea por línea** para explicar qué hace cada instrucción, por qué se utiliza y cómo contribuye al objetivo final del script.

---

## 📂 Estructura del Repositorio

El repositorio está dividido en las siguientes carpetas principales:

- **`Redes_Neuronales/`**: Contiene los scripts en Python enfocados en modelos predictivos (Regresión Lineal, Redes Neuronales) y manejo de datos con PySpark.
- **`SQL/`**: Carpeta destinada para futuros análisis y consultas en lenguaje estructurado.

---

## 🧠 Scripts Analizados (`Redes_Neuronales/`)

A continuación se detalla el propósito de cada código documentado en este corte:

### 1. Modelos Clásicos vs Redes Neuronales
- **`comparacion_modelos_regresion_iris.py`**: Compara el desempeño de un modelo estadístico clásico (Regresión Lineal Múltiple) contra un modelo no lineal (Red Neuronal) utilizando el famoso dataset de la flor Iris. Se evalúan y explican métricas como el MSE, MAE y R².
- **`red_neuronal_regresion_iris.py`**: Construye paso a paso una Red Neuronal Artificial (ANN) con Keras y TensorFlow para predecir la longitud del sépalo de las flores Iris. Explica la importancia del escalamiento de datos y la arquitectura de capas ocultas.

### 2. Análisis Profundo de Modelos
- **`analisis_convergencia_importancia_nn.py`**: Va más allá de la predicción y entra al "corazón" del modelo. Analiza cómo converge el error (MSE) durante el entrenamiento (épocas) y extrae los pesos de la primera capa para estimar qué variable de entrada tiene más peso (importancia) en la decisión de la red.
- **`analisis_nn_iris.py`**: *Script base inicial.*

### 3. Procesamiento a Gran Escala (PySpark)
- **`analisis_exploratorio_flights_spark.py`**: Demuestra el uso de **Spark SQL** para el análisis exploratorio de datos. A través de consultas SQL declarativas estructuradas sobre DataFrames de Spark, analiza tendencias de vuelo (dataset Flights) usando agrupaciones y funciones de ventana complejas.
- **`regresion_multiple_iris.py`**: Implementa el ecosistema **PySpark ML** (Machine Learning distribuido). Explica cómo ensamblar características en vectores (`VectorAssembler`) y cómo entrenar un modelo de regresión lineal escalable utilizando la arquitectura de Spark.

---

> **Nota:** Todos los códigos están diseñados para ser educativos y autodescriptivos. Se recomienda leer los comentarios internos de cada `.py` para comprender las decisiones de diseño, desde la importación de librerías hasta la evaluación de resultados.