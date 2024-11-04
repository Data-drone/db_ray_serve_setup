# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup VLLM Deployment
# MAGIC 
# MAGIC We can use ray server to make a model available and performant \
# MAGIC Note that this has to be done on MLR 15.4 LTS - MLR 16 Beta has a conflict

# COMMAND ----------

# DBTITLE 1,Install Libs
# MAGIC %pip install vllm==0.6.2
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config and Logging
import os

# In case we need to temporarily stash files
os.environ['HF_HOME'] = 'local_disk0/hf'

# optional can also use a volume instead for the logging
log_path = '/tmp/ray_logs'
dbutils.fs.mkdirs(log_path)
dbfs_log_path = f'/dbfs/{log_path}'

# COMMAND ----------

# DBTITLE 1,Clear Ray Environment
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

try:
    shutdown_ray_cluster()
except RuntimeError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC # Starting Ray Cluster
# MAGIC This needs to be done correctly to ensure that we have access all the necessary resources
# MAGIC Key parameters include:
# MAGIC - max_worker_nodes - set to match DB Cluster Setup
# MAGIC - num_gpus_head_node - set to number of GPUs on driver
# MAGIC - num_cpus_worker_node - set to match the CPU on the worker, leave a few cores out for OS andother overheads
# MAGIC - num_gpus_worker_node - set to the number of gpus per worker node
# MAGIC
# MAGIC **NOTE** it seems like vllm doesn't configure ray placement groups well. \
# MAGIC either that or there are issues with the `setup_ray_cluster` command. So make sure that the driver node has GPUs and usable GPUS at that \
# MAGIC don't create a small T4 GPU driver with beefy A100 Workers - could get out of GPU issue and issues detecting GPU on the workers.

# COMMAND ----------

# DBTITLE 1,Start cluster

# config for 1x A100 Azure node - adjust as needed
setup_ray_cluster(
    max_worker_nodes = 2,
    num_gpus_head_node = 1,
    num_cpus_worker_node = 20,
    num_gpus_worker_node = 1,
    collectl_log_to_path = dbfs_log_path
)

# COMMAND ----------

# MAGIC %sh
# MAGIC #vllm serve /Volumes/brian_ml_dev/hf_models/model_weights/llama_3_1_70b-instruct

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuring the Serving pattern
# MAGIC - tensor-parallel distributes the model across multiple GPU by splitting layers apart
# MAGIC - pipeline parallel distributes the model across multiple GPU by putting whole layers on separate gpu
# MAGIC 
# MAGIC **NOTE** Not all Models can be used in pipeline-parallel mode
# MAGIC **NOTE** tensor-parallel number must evenly divide the layers
# MAGIC Usually it is recommended to go tensor parallel within the gpus on a single node then pipeline parallel across multiple nodes \
# MAGIC Assuming your model is that big ie Llama 3.1 405b \
# MAGIC Always try to fit everyone on one GPU first rather than split.
# MAGIC
# MAGIC **MODEL NOTES**
# MAGIC As discussed here: https://github.com/vllm-project/vllm/issues/8879 \
# MAGIC Setting max-num-seq is important ot not run out of VRAM \
# MAGIC specifically for llama 3.2 set the `--enforce-eager` mode 

# COMMAND ----------

# MAGIC %sh
# MAGIC vllm serve /Volumes/brian_ml_dev/hf_models/model_weights/llama_3_1_70b-instruct \
# MAGIC    --tensor-parallel-size 8 \
# MAGIC    --pipeline_parallel_size 2 \
# MAGIC    --max-num-seq 16 \
# MAGIC    --enforce-eager \
# MAGIC    --distributed-executor-backend ray \
# MAGIC    --port 10101

