# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ray Setup
# MAGIC setup testing

# COMMAND ----------

import mlflow
from ray import serve
from starlette.requests import Request
from typing import Dict
import os
import json

# COMMAND ----------

browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

@serve.deployment(
  ray_actor_options={"num_cpus": 4},
  num_replicas=3,
  user_config=json.dumps{
      {
          "DATABRICKS_HOST": db_host,
          "DATABRICKS_TOKEN": db_token,
      }}
)
class BoostingModel:
    def __init__(self, model_path: str, db_host: str, db_token: str):

        # We need to set these vars so that new python thread can find the 
        #os.environ['DATABRICKS_HOST'] = db_host
        #os.environ['DATABRICKS_TOKEN'] = db_token
        # path = "runs:/327e476d760e4723a7f0b8efeb676931/model"
        self.model = mlflow.sklearn.load_model(model_path)

        self.label_list = ['setosa', 'versicolor', 'virginica']

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}
      
# COMMAND ----------

# "https://e2-demo-west.cloud.databricks.com/",
# "dapi3bd2c7a30da1b9c1bac89f81816e7787"
        

boosting_model = BoostingModel.bind("runs:/327e476d760e4723a7f0b8efeb676931/model", 
                                    )