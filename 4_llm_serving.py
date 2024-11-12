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

# MAGIC %md
# MAGIC # Single Node Cluster
# MAGIC
# MAGIC The `setup_ray_cluster` integration is for multimodal setups. For single node we can use standard Ray APIs

# COMMAND ----------

import ray

try:
  ray.shutdown()
except RuntimeError:
  pass

ray.init(address="local", 
         num_cpus=20, 
         num_gpus=1,
         dashboard_host="0.0.0.0")

# COMMAND ----------

# MAGIC %md
# MAGIC # Multi Node Cluster
# MAGIC
# MAGIC The `setup_ray_cluster` integration is for multimodal setups. For single node we can use standard Ray APIs

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

# Make sure CPU / GPU etc is configured correctly
setup_ray_cluster(
    max_worker_nodes = 2,
    num_gpus_head_node = 1,
    num_cpus_worker_node = 20,
    num_gpus_worker_node = 1,
    collectl_log_to_path = dbfs_log_path
)

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

# DBTITLE 1,Other Useful Flags
# --tensor-parallel-size 8 \ # for distributing to multiple GPUs on single node
# --pipeline_parallel_size 2 \ # for distributing across nodes

# COMMAND ----------

# DBTITLE 1,Start Ray Serve
# MAGIC %sh
# MAGIC vllm serve /Volumes/brian_ml_dev/hf_models/model_weights/llama_3_2_11_vision_instruct \
# MAGIC    --served-model-name meta-llama/Llama-3.2-11B-Vision-Instruct \
# MAGIC    --max-num-seq 16 \
# MAGIC    --enforce-eager \
# MAGIC    --distributed-executor-backend ray \
# MAGIC    --port 10101

# COMMAND ----------

# MAGIC %md
# MAGIC # Pinging the Server
# MAGIC
# MAGIC **Notes** the uri for the cluster is:
# MAGIC https://<workspace-uri>/driver-proxy-api/o/0/<cluster_id>/<port-we-set>/<rest of openai api spec>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Ping Server via CLI
# MAGIC %sh
# MAGIC curl \
# MAGIC  -H "Authorization: Bearer <insert_token>" \
# MAGIC  -H "Content-Type": "application/json" \
# MAGIC  https://adb-984752964297111.11.azuredatabricks.net/driver-proxy-api/o/0/1015-220646-53zdi0og/7682/v1/models

# COMMAND ----------

# DBTITLE 1,Ping Server via OpenAI API
from openai import OpenAI

client = OpenAI(
  base_url="https://adb-984752964297111.11.azuredatabricks.net/driver-proxy-api/o/0/1015-220646-53zdi0og/7682/v1/",
  api_key="<db_auth_token>"
)

client.chat.completions.create(
  model="meta-llama/Llama-3.2-11B-Vision-Instruct",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "tell me a joke"}
      ],
    }
  ],
  max_tokens=1064,
)

# COMMAND ----------

# MAGIC %sh
# MAGIC # Using with AI Gateway
# MAGIC
# MAGIC A vllm server is compliant with OpenAI API Spec and can be used with that \
# MAGIC Integrating with Databricks AI Gateway will give you a stable URI and also the monitoring and governance structures necessary for production use. \
# MAGIC
# MAGIC For Gateway configure as follows:
# MAGIC - *Provider*: OpenAI / Azure OpenAI
# MAGIC - *OpenAI API type*: OpenAI
# MAGIC - *Secret*: Databricks token of the user who created the vllm server
# MAGIC - *Task*: Depends on model, usually chat
# MAGIC - *Provider model*: the name set in vllm ie the `--served-model-name`` param
# MAGIC
# MAGIC Open the Advanced Configuration Tab
# MAGIC - *OpenAI API Base*: the same uri used in base_uri above ie: https://adb-984752964297111.11.azuredatabricks.net/driver-proxy-api/o/0/1015-220646-53zdi0og/7682/v1
# MAGIC - *OpenAI Organiszation*: leave blank
# MAGIC - *Served Entity Name*: the name set in vllm ie the `--served-model-name`` param