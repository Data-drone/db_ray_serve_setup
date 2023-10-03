# Setting up Ray Serve on Databricks

There are can advanced applications where Ray Serve can be useful to manually scale ml models

## Setup

The `0_setup_cpu_cluster.py` notebook sets up the cluster definition using the latest databricks-sdk.
- For embedding models, we recommend g5.2xlarge nodes which should comfortably run an instance of `instruct-xl`. See `0a_setup_gpu_cluster.py` there is additional complexity in this one as we have to replace sentence-transformers via init script.

The `1_startup_ray_service.py` will initialise the Ray Service that will host the services.

The models folder provides various examples of deploying models on our Ray Service

The `2_xxxx.py` scripts are there to log models into mlflow for us to use with our model deployments.

The `3_xxxx.py` scripts provide an example of how to query the RESTAPis that we standup.

## LIMITATIONS

These have to run on single-user clusters. That means that for general access, the devices that hit the service need to have tokens created for them by the user assigned to the cluster.

All traffic is routed via the driver-proxy which is not intended for heavy duty scaling so mileage may vary

