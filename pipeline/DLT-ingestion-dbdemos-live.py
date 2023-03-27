# Databricks notebook source
# MAGIC %pip install mlflow

# COMMAND ----------

# MAGIC %md 
# MAGIC # DLT ingestion pipeline
# MAGIC 
# MAGIC Simple model to validate pseudonym input - use a predefined list to simplify the demo.

# COMMAND ----------

import mlflow
model_name = "dbdemos_live_firstname"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/Production", result_type='prediction boolean, time long', env_manager='local')

# COMMAND ----------

import dlt
from pyspark.sql.functions import col, from_json, current_timestamp, split, explode    


@dlt.view(comment="raw source")
def kinesis_bronze():
  return (
    spark.readStream.format("kinesis")
    .option("streamName", "dbdemos-live")
    .option("region", "us-east-2")
    .load()
)

@dlt.view(comment="real schema for kinesis payload")
@dlt.expect("Correct pseudo and choice", "pseudo is not null and choice is not null")
@dlt.expect("Correct choice", "choice in ('unity-catalog', 'delta-live-table', 'model-serving', 'serverless-warehouse')")
def kinesis_silver():
  return (        
    dlt.read_stream("kinesis_bronze")
    .withColumn(
        "dataStruct",
        from_json(col("data").cast("string"), "pseudo STRING, choice STRING, lambda_time long"),
    )
    .select(col("dataStruct.*"))
    .withColumn('choice', explode(split(col("choice"),",")))
    .withColumn('dlt_time', current_timestamp())
    .withColumn("lambda_time", (col("lambda_time")/1000).cast("timestamp"))
  )
  

@dlt.table(comment="gold table from the kinesis payload")
def dbdemos_choice():
  df =  dlt.read_stream("kinesis_silver")
  df = df.withColumn('predictions', loaded_model(col("pseudo"))).withColumn('inference_time', (col("predictions.time")).cast("timestamp")).withColumn('predictions', col("predictions.prediction"))
  return df

