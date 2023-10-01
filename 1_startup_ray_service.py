# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ray Setup
# MAGIC Once the cluster is up we can start Ray On Spark

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# To make sure that if we rerun notebook it'll restart the ray cluster
try:
  shutdown_ray_cluster()
except RuntimeError:
  pass

# COMMAND ----------

tmp_folder = '/tmp/ray'
dbutils.fs.mkdirs(tmp_folder)

# COMMAND ----------

# This setup is for running CPU workloads and assumes a small 16 core workers
setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=10,
  collect_log_to_path=f"/dbfs{tmp_folder}"
)

# GPU Version with 4x A10G
# setup_ray_cluster(
#   num_worker_nodes=2,
#   num_cpus_per_node=10,
#   num_gpus_per_node=1,
#   collect_log_to_path=f"/dbfs{tmp_folder}"
# )

# COMMAND ----------

# MAGIC %md # Ray Serve
# MAGIC Ray Serve allows us to host Apps and run inferencing services \
# MAGIC We set the `proxy_Location`` for ingress to be host only as the access proxy runs on driver \
# MAGIC We need to set host to `0.0.0.0` so that outside traffic is allowed \
# MAGIC Port you can choose just don't clash with existing spark or DB service

# COMMAND ----------

from ray import serve

host = '0.0.0.0'
port = '10101'

serve.start(detached=True,
            proxy_Location='HeadOnly',
            http_options={'host': host, 'port': port})

# COMMAND ----------

# MAGIC %md # Starting Model
# MAGIC It should be possible to launch from notebook but we will use a script for now

# COMMAND ----------
# MAGIC %sh
# MAGIC serve run models/sklearn_boosting_cpu/cpu_model:boosting_model 