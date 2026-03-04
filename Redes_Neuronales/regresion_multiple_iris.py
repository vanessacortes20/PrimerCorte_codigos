# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE CON PYSPARK
# Dataset: Iris
# ============================================================
# Objetivo:
# Implementar un modelo de regresión lineal múltiple utilizando
# Spark ML para predecir la variable sepal_length (longitud del sépalo)
# a partir de:
#   - sepal_width
#   - petal_length
#   - petal_width
#
# Se usa PySpark para demostrar procesamiento estructurado
# y modelado bajo un entorno distribuido.
# ============================================================

# ------------------------------------------------------------
# Importación de librerías necesarias
# ------------------------------------------------------------
# SparkSession: punto de entrada para trabajar con Spark.
# VectorAssembler: convierte múltiples columnas en un solo vector
#                  requerido por Spark ML.
# LinearRegression: modelo de regresión lineal en Spark ML.
# RegressionEvaluator: permite evaluar métricas como R².
# seaborn: se utiliza para cargar el dataset Iris.
# pandas: soporte para manipulación tabular previa a Spark.
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import seaborn as sns
import pandas as pd

# ------------------------------------------------------------
# Creación de la sesión Spark
# ------------------------------------------------------------
# Inicializa el entorno de ejecución distribuido.
# El nombre facilita identificar el proceso en el Spark UI.
spark = SparkSession.builder.appName("RegresionMultipleIris").getOrCreate()

# ------------------------------------------------------------
# Carga del dataset Iris
# ------------------------------------------------------------
# Se carga desde seaborn como DataFrame de pandas.
iris = sns.load_dataset("iris")

# Selección únicamente de variables numéricas relevantes.
# sepal_length será la variable dependiente (y).
pdf = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# ------------------------------------------------------------
# Conversión de pandas a Spark DataFrame
# ------------------------------------------------------------
# Esto permite trabajar con el dataset dentro del ecosistema Spark.
df = spark.createDataFrame(pdf)

# ------------------------------------------------------------
# Creación de vista temporal SQL
# ------------------------------------------------------------
# Permite usar consultas SQL sobre el DataFrame.
df.createOrReplaceTempView("iris_table")

# Consulta SQL donde:
# - sepal_length se define como variable objetivo (y)
# - Las demás variables serán predictoras
df_sql = spark.sql("""
    SELECT
        sepal_length AS y,
        sepal_width,
        petal_length,
        petal_width
    FROM iris_table
""")

# ------------------------------------------------------------
# Ensamblado de variables independientes
# ------------------------------------------------------------
# Spark ML requiere que las variables predictoras estén
# agrupadas en una sola columna tipo vector llamada "features".
assembler = VectorAssembler(
    inputCols=["sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)

# Se genera el dataset final con:
# - features (vector)
# - y (variable dependiente)
data_modelo = assembler.transform(df_sql).select("features", "y")

# ------------------------------------------------------------
# Definición del modelo de regresión lineal
# ------------------------------------------------------------
# El modelo estima la ecuación:
# y = β0 + β1X1 + β2X2 + β3X3
# minimizando el Error Cuadrático Medio (MSE).
lr = LinearRegression(featuresCol="features", labelCol="y")

# Entrenamiento del modelo
modelo = lr.fit(data_modelo)

# ------------------------------------------------------------
# Generación de predicciones
# ------------------------------------------------------------
# Se aplica el modelo sobre el mismo conjunto de datos
# (nota: aquí no se divide en train/test).
predicciones = modelo.transform(data_modelo)

# ------------------------------------------------------------
# Evaluación del modelo
# ------------------------------------------------------------
# Se utiliza R² (coeficiente de determinación).
# R² indica qué proporción de la varianza de y
# es explicada por las variables independientes.
evaluador = RegressionEvaluator(
    labelCol="y",
    predictionCol="prediction",
    metricName="r2"
)

# ------------------------------------------------------------
# Resultados del modelo
# ------------------------------------------------------------
# Intercepto: valor esperado de y cuando todas las X son 0.
# Coeficientes: impacto marginal de cada variable predictora.
# R²: calidad global del ajuste del modelo.
print("Intercepto:", modelo.intercept)
print("Coeficientes:", modelo.coefficients)
print("R2:", evaluador.evaluate(predicciones))

# ------------------------------------------------------------
# Cierre de sesión Spark
# ------------------------------------------------------------
# Libera recursos del entorno distribuido.
spark.stop()