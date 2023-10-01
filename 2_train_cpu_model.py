# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Train Model for serving
# MAGIC This is a boring standard sklearn model that we will serve

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
import mlflow
import numpy as np
import os

# COMMAND ----------

# DBTITLE 1,Setup Dataset
iris_dataset = load_iris()
data, target, target_names = (
    iris_dataset["data"],
    iris_dataset["target"],
    iris_dataset["target_names"],
)

np.random.shuffle(data), np.random.shuffle(target)
train_x, train_y = data[:100], target[:100]
val_x, val_y = data[100:], target[100:]

# COMMAND ----------
# Note we need the target names in our serve deployment code
target_names

# COMMAND ----------

# DBTITLE 1,Train and Log Model
mlflow.sklearn.autolog()

with mlflow.start_run():
  model = GradientBoostingClassifier()
  
  model.fit(train_x, train_y)
  print("MSE:", mean_squared_error(model.predict(val_x), val_y))
