# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Testing Endpoints
# MAGIC Testing running application

# COMMAND ----------

# testing things

import requests
from dbruntime.databricks_repl_context import get_context

ctx = get_context()

port = "10101"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}/"

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {"User-Agent": "Test Client",
               "Content-Type": "application/json",
               "Authorization": "Bearer "+token}

sample_request_input = {
    "sepal length": 2.1,
    "sepal width": 10.0,
    "petal length": 1.1,
    "petal width": 0.9,
}
response = requests.get(driver_proxy_api, headers=headers, json=sample_request_input)
print(response.text)
