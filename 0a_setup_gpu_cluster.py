# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Cluster Launch
# MAGIC We need to setup a cluster with certain libraries installed \
# MAGIC These need to be installed at cluster level so we can use SDK to make this easier \
# MAGIC Running Ray serve also requires that we have a multinode setup \
# MAGIC Ray On Spark does not support Autoscale at this stage

# COMMAND ----------

# DBTITLE 1,Update for sdk to allow single node GPU instances
%pip install -U databricks-sdk >= 0.9.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Workspace Client
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Library, PythonPyPiLibrary, InitScriptInfo
import os

# Get current workspace host and token
browser_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
current_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

w = WorkspaceClient(
  host  = db_host,
  token = db_token
)

# COMMAND ----------

# DBTITLE 1,Helper function to check if cluster exists
import fnmatch

def find_clusters_by_name(cluster_list, target_name):
    matching_clusters = []
    for cluster in cluster_list:
        if fnmatch.fnmatchcase(cluster.cluster_name, target_name):
            matching_clusters.append(cluster)
    return matching_clusters

# COMMAND ----------

target_name = "ray_serve_embedding_cluster"

matching_clusters = find_clusters_by_name(w.clusters.list(), target_name)

if len(matching_clusters) == 0:

    # we must add the embedding init-script
    current_directory, current_filename = os.path.split(current_notebook_path)
    init_script_path = os.path.join(current_directory, 'init-scripts/latest-sentence-transformer.sh')
    update_sentence_transformer = InitScriptInfo().from_dict({'workspace': {'destination': init_script_path}})

    print("Attempting to create cluster. Please wait...")

    c = w.clusters.create_and_wait(
    cluster_name             = 'ray_serve_embedding_cluster',
    spark_version            = '14.0.x-gpu-ml-scala2.12',
    node_type_id             = 'g5.2xlarge',
    autotermination_minutes  = 45,
    num_workers              = 2,
    init_scripts             = [update_sentence_transformer],
    )

    print(f"The cluster is now ready at " \
        f"{w.config.host}#setting/clusters/{c.cluster_id}/configuration\n")
    
    print("installing Libraries")

    # Install Libraries
    # We need ray serve installed and also aiorwlock for the deployment
    ray_lib = Library().from_dict({'pypi': {'package': 'ray[default,serve]==2.7.0'}})
    aiorwlock_lib = Library().from_dict({'pypi': {'package': 'aiorwlock==1.3.0'}})

    w.libraries.install(c.cluster_id, [ray_lib, aiorwlock_lib])

elif len(matching_clusters) == 1:

    cluster = matching_clusters[0].cluster_id

    print("cluster exists already")

    print("current installed python libraries")
    library_status_list = w.libraries.cluster_status(cluster)

    python_packages = []

    for library_status in library_status_list:
        library = library_status.library
        if library and isinstance(library.pypi, PythonPyPiLibrary) and library.pypi.package:
            python_packages.append(library.pypi.package)

    print(python_packages)

# COMMAND ----------
