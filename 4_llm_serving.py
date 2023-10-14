# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup VLLM Deployment
# MAGIC Note that this current runs with it's own code

# COMMAND ----------

%pip install vllm==0.2.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# setup configs
import os

os.environ['HF_HOME'] = 'local_disk0/hf'
model_type = 'TheBloke/'

log_path = '/tmp/ray_logs'
dbutils.fs.mkdirs(log_path)
dbfs_log_path = f'/dbfs/{log_path}'

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

try:
    shutdown_ray_cluster()
except RuntimeError:
    pass

# COMMAND ----------

# This setup is intended for g5.48xlarge
setup_ray_cluster(
    num_worker_nodes = 1,
    num_cpus_per_node = 40,
    num_gpus_per_node = 4,
    collectl_log_to_path = dbfs_log_path
)

# COMMAND ----------

%sh

python -m vllm.entrypoints.api_server \
--host 0.0.0.0 \
--port 10101 \
--model 'TheBloke/Llama-2-13B-chat-AWQ' \
--quantization 'awq' \
--tensor-parallel-size 2


