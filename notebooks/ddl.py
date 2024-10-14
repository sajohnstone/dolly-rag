# Define variables
catalog_name = 'stu_sandbox'
schema_name = 'rag_model'
volume_name = 'pdf_data'

# Create the catalog
spark.sql(f"""
    create catalog if not exists {catalog_name}
""")

# Use the newly created catalog
spark.sql(f"""
    use catalog {catalog_name}
""")

# Create the schema
spark.sql(f"""
    create schema if not exists {schema_name}
""")

# Create managed volume
spark.sql(f"""
    create volume if not exists {catalog_name}.{schema_name}.{volume_name}
""")