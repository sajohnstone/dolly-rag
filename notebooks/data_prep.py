# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# Define your attributes
username = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
print(f"Username: {username}")
catalog_name = "stu_sandbox"
schema_name = "rag_test"
volume_name = "pdf_data"

# COMMAND ----------

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
articles_path = "/Volumes/${catalog_name}/${schema_name}/${volume_name}/"
table_name = f"{catalog_name}.{schema_name}.pdf_raw_text"
# read pdf files
df = (
    spark.read.format("binaryfile")
    .option("recursiveFileLookup", "true")
    .load(articles_path)
)
df.write.mode("overwrite").saveAsTable(table_name)
display(df)
