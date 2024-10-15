# Databricks notebook source
# MAGIC %md
# MAGIC # Create endpoint
# MAGIC This sets up the model serving endpoint for the model that is going to be used.  You could do this step in the UI if you prefer:
# MAGIC
# MAGIC - Access the Serving UI:
# MAGIC   - Click on Serving in the Databricks sidebar to display the Serving UI.
# MAGIC - Create a Serving Endpoint:
# MAGIC   - Click Create serving endpoint.
# MAGIC - Configure the Endpoint:
# MAGIC   - name: gte-large-en
# MAGIC   - model: system.ai.gte_large_en_v1_5
# MAGIC   - version: 1

# COMMAND ----------

# Define the endpoint name and configuration
from collections import namedtuple

# ModelConfig
ModelConfig = namedtuple("ModelConfig", ["endpoint_name", "feature_spec_name", "entity_version"])
model = ModelConfig("gte-large-en", "system.ai.gte_large_en_v1_5", 1)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Initialize the workspace client
workspace = WorkspaceClient()

# Create the endpoint
status = workspace.serving_endpoints.create_and_wait(
    name=model.endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model.feature_spec_name,
                entity_version=model.entity_version,  # Add the entity version here
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)

# Print the status of the endpoint
print(status)

# Get the status of the endpoint
status = workspace.serving_endpoints.get(name=model.endpoint_name)
print(status)
