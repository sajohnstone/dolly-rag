# Databricks notebook source
# MAGIC %md
# MAGIC # Create serving endpoint
# MAGIC This just sets up the serving endpoint for the model that is going to be used.  You could do this step in the UI if you prefer:
# MAGIC
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

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Initialize the workspace client
workspace = WorkspaceClient()

# Define the endpoint name and configuration
endpoint_name = "bge-small-en"
feature_spec_name = "system.ai.bge_small_en_v1_5"
entity_version = 3

# Create the endpoint
status = workspace.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,
                entity_version=entity_version,  # Add the entity version here
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)

# Print the status of the endpoint
print(status)

# Get the status of the endpoint
status = workspace.serving_endpoints.get(name=endpoint_name)
print(status)
