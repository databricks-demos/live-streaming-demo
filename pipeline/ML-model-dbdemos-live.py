# Databricks notebook source
# MAGIC %md 
# MAGIC # ML Firstname model
# MAGIC 
# MAGIC Simple model to validate pseudonym input - use a predefined list to simplify the demo.

# COMMAND ----------

import mlflow
import pandas as pd
 
  
# define a custom model
class FirstnameModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import requests
        import os.path
        #Download the list of valid firstnames
        firstname_local = '/tmp/firstnames.snappy.parquet'
        if not os.path.isfile(firstname_local):
          with requests.get("https://github.com/databricks-demos/live-streaming-demo/raw/main/pipeline/firstnames.snappy.parquet", stream=True) as r:
            r.raise_for_status()
            with open(firstname_local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        df = pd.read_parquet(firstname_local)
        self.firstnames = set(df['_c0'])

    def strip_accents(self, s):
        import unicodedata
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
      
    def validate(self, pseudo):
        return self.strip_accents(pseudo).upper() in self.firstnames

    def predict(self, context, model_input):
        import time
        pseudo = model_input.iloc[:, 0]
        ts = time.time()
        return [{"prediction":c,"time":ts} for c in pseudo.apply(self.validate)]
    

# save the model
firstname_model = FirstnameModel()


with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=firstname_model)

# COMMAND ----------

model_name = "dbdemos_live_firstname"
model_registered = mlflow.register_model(f"runs:/{ run.info.run_id }/model", model_name)

#Move the model in production
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(model_name, model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

#Test:
test = False
if test:
  import mlflow
  from pyspark.sql.functions import struct, col

  # Load model as a Spark UDF.
  #model_name = "dbdemos_live_firstname"
  loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/Production", result_type='prediction boolean, time long', env_manager='local')
  df = spark.read.table("dbdemos_live.dbdemos_choice")
  # Predict on a Spark DataFrame.
  df = df.withColumn('predictions', loaded_model(col("pseudo")))
  display(df)
