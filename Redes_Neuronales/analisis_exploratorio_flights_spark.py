# ============================================================
# ANÁLISIS EXPLORATORIO DE DATOS CON SPARK SQL
# Dataset: Flights (Seaborn)
# ============================================================
# Objetivo:
# Realizar un análisis exploratorio utilizando Spark SQL
# sobre el dataset "flights", el cual contiene:
#   - year (año)
#   - month (mes)
#   - passengers (cantidad de pasajeros)
#
# Se aplican:
#   - Funciones de agregación
#   - Agrupaciones
#   - Funciones ventana (Window Functions)
#
# Esto demuestra manejo de datos estructurados en Spark
# bajo un enfoque declarativo con SQL.
# ============================================================

# ------------------------------------------------------------
# Importación de librerías
# ------------------------------------------------------------
# SparkSession: punto de entrada al entorno Spark.
# pyspark.sql.functions: funciones agregadas y auxiliares.
# Window: soporte para funciones ventana.
# seaborn: carga del dataset.
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import seaborn as sns

# ------------------------------------------------------------
# Creación de sesión Spark
# ------------------------------------------------------------
# Inicializa el entorno distribuido.
spark = SparkSession.builder.appName("Base_Default_Flights").getOrCreate()

# =========================
# Base de datos buena por default
# =========================
# Se carga el dataset flights desde seaborn.
# Representa el número de pasajeros por mes y año.
flights_pd = sns.load_dataset("flights")

# Conversión de pandas DataFrame a Spark DataFrame.
flights = spark.createDataFrame(flights_pd)

# Creación de vista temporal para consultas SQL.
flights.createOrReplaceTempView("flights")

# =========================
# Vista general
# =========================
# Muestra las primeras 10 filas para inspección inicial.
# Permite validar estructura y tipos de datos.
spark.sql("SELECT * FROM flights LIMIT 10").show()

# =========================
# 1. Estadísticas básicas
# =========================
# Se calculan métricas descriptivas:
# - AVG: promedio de pasajeros
# - MIN: valor mínimo
# - MAX: valor máximo
#
# Esto permite entender rango y tendencia central.
spark.sql("""
    SELECT
        AVG(passengers) AS promedio,
        MIN(passengers) AS minimo,
        MAX(passengers) AS maximo
    FROM flights
""").show()

# =========================
# 2. Crecimiento anual
# =========================
# Se agrupa por año y se calcula el total anual de pasajeros.
# SUM(passengers) permite observar tendencia de crecimiento.
#
# ORDER BY year garantiza orden cronológico.
spark.sql("""
    SELECT
        year,
        SUM(passengers) AS total_pasajeros
    FROM flights
    GROUP BY year
    ORDER BY year
""").show()

# =========================
# 3. Mes con más pasajeros por año
# =========================
# Se utiliza una función ventana:
# RANK() OVER (PARTITION BY year ORDER BY passengers DESC)
#
# - PARTITION BY year → reinicia ranking cada año.
# - ORDER BY passengers DESC → mayor número primero.
#
# Luego se filtra rnk = 1 para obtener el mes
# con mayor tráfico por cada año.
spark.sql("""
    SELECT year, month, passengers
    FROM (
        SELECT *,
               RANK() OVER (PARTITION BY year ORDER BY passengers DESC) AS rnk
        FROM flights
    ) t
    WHERE rnk = 1
""").show()

# =========================
# 4. Promedio mensual histórico
# =========================
# Se agrupa por mes considerando todos los años.
# Permite identificar estacionalidad histórica.
#
# ORDER BY promedio_mes DESC muestra los meses
# con mayor promedio de pasajeros.
spark.sql("""
    SELECT
        month,
        AVG(passengers) AS promedio_mes
    FROM flights
    GROUP BY month
    ORDER BY promedio_mes DESC
""").show()

# =========================
# 5. Comparación contra promedio anual
# =========================
# Se usa una función ventana para calcular
# el promedio anual dinámicamente:
#
# AVG(f.passengers) OVER (PARTITION BY f.year)
#
# Luego se calcula la desviación respecto a ese promedio.
#
# Esto permite identificar:
# - Meses por encima del promedio anual
# - Meses por debajo del promedio anual
spark.sql("""
    SELECT
        f.year,
        f.month,
        f.passengers,
        f.passengers - AVG(f.passengers) OVER (PARTITION BY f.year) AS desviacion_anual
    FROM flights f
    ORDER BY f.year, f.month
""").show()

# ------------------------------------------------------------
# Cierre de sesión Spark
# ------------------------------------------------------------
# Libera recursos del entorno distribuido.
spark.stop()