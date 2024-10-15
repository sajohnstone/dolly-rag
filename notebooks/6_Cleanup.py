# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup
# MAGIC Run this to clean up everthing that was created

# COMMAND ----------

catalog_name = "stu_sandbox"
schema_name = "rag_model"
volume_name = "pdf_data"
model_endpoint_name = "gte-large-en"
articles_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"

raw_table_name = f"{catalog_name}.{schema_name}.bronze_pdfs_raw"
raw_checkpoints_path = f"/Volumes/{catalog_name}/{schema_name}/checkpoints/bronze_pdfs_raw/"

encoded_table_name = f"{catalog_name}.{schema_name}.silver_pdfs_chunked_encoded"
encoded_checkpoints_path = f"/Volumes/{catalog_name}/{schema_name}/checkpoints/silver_pdfs_chunked_encoded/"

vector_search_endpoint_name = "stu-test-vector"

encoded_table_name = f"{catalog_name}.{schema_name}.silver_pdfs_chunked_encoded"
encoded_table_name_vs_index = f"{encoded_table_name}_vs_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

vsc.delete_delta_sync_index(name=encoded_table_name_vs_index)
vsc.delete_endpoint(name=vector_search_endpoint_name)

# COMMAND ----------

spark.sql(f"""
    use catalog {catalog_name}
""")

# Create managed volume for pdf data
spark.sql(f"""
    delete volume if exists {catalog_name}.{schema_name}.pdf_data
""")

# Create managed volume for checkpoints
spark.sql(f"""
    delete volume if exists {catalog_name}.{schema_name}.checkpoints
""")

# Create the schema
spark.sql(f"""
    delete schema if exists {schema_name}
""")

# Create the catalog
spark.sql(f"""
    delete catalog if exists {catalog_name}
""")







