# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # S Setup Serve Deployment
# MAGIC Basic sentence-transformers example

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
  ray_actor_options={"num_gpus": 1},
  num_replicas=2,
)
class SentenceTransformer:
    def __init__(self, model_path: str, db_host: str, db_token: str):

        # We need to set these vars so that new python thread can find the 
        os.environ['DATABRICKS_HOST'] = db_host
        os.environ['DATABRICKS_TOKEN'] = db_token
        #path = "runs:/b38a0b53f8884608a4fde549a0b4e48d/embedding_model"
        self.model = mlflow.sentence_transformers.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:

        payload = await starlette_request.json()
        print("Received request: {}".format(payload))

        # we assume that the payload json has a field text
        # we assume that the text field is a list of strings
        # This may require refactoring
        input_vector = payload['text']

        # it can be advantages to setup a 
        result = self.model.encode(input_vector)

        return {"embeddings": result}

# COMMAND ----------

# hardcoding for now
instruct_xl_model = SentenceTransformer.bind("runs:/b38a0b53f8884608a4fde549a0b4e48d/embedding_model",
                                             db_host,
                                             db_token)

# COMMAND ----------

import ray
ray.init('auto',ignore_reinit_error=True)

serve.run(instruct_xl_model, name='embedding_model')

# COMMAND ----------

# getting list of running apps
#serve.status()

# deleting a deployment (since we didn't set any names it is called default)
#serve.delete('embedding_model')