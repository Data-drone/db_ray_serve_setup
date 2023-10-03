# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup Serve Deployment
# MAGIC Basic ML example

# COMMAND ----------

# MAGIC %pip install --upgrade git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb

# COMMAND ----------

dbutils.library.restartPython()
# COMMAND ----------

from sentence_transformers import SentenceTransformer
import mlflow

# COMMAND ----------

# DBTITLE 1,Setting Up a Model
model_name='hkunlp/instructor-xl'
artifact_path = 'embedding_model'

embedding_model = SentenceTransformer(model_name)

# COMMAND ----------

# DBTITLE 1,Setting MLflow Additions
# Lets create a signature example
example_sentences = ["welcome to sentence transformers", 
                    "This model is for embedding sentences"]


embedding_signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=embedding_model.encode(example_sentences)
)

extra_pip_requirements  = ["git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb"]

# COMMAND ----------

# DBTITLE 1,Logging to MLflow
with mlflow.start_run() as run:
    mlflow.sentence_transformers.log_model(embedding_model,
                                  artifact_path=artifact_path,
                                  signature=embedding_signature,
                                  input_example=example_sentences,
                                  extra_pip_requirements=extra_pip_requirements)
