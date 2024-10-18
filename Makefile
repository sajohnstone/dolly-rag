# Define variables
DATABRICKS_NOTEBOOK_PATH = /Workspace/path/to/notebook
LOCAL_NOTEBOOK_PATH = ./notebook/import.py
LOCAL_DATA_PATH = ./data
MANAGED_VOLUME_PATH = dbfs:/Volumes/stu_sandbox/rag_model/pdf_data
DATABRICKS_PROFILE = STUTEST  # Change this to your desired Databricks profile

.PHONY: upload_files
upload_files:
	@echo "Uploading PDF files to managed volume..."
	@for file in $(LOCAL_DATA_PATH)/*.md; do \
		if [ -f "$$file" ]; then \
			echo "Uploading $$file to $(MANAGED_VOLUME_PATH)"; \
			databricks --profile $(DATABRICKS_PROFILE) fs cp "$$file" "$(MANAGED_VOLUME_PATH)/"; \
		else \
			echo "No PDF files found in $(LOCAL_DATA_PATH)"; \
		fi; \
	done
	@echo "Upload completed."

.PHONY: upload
upload:
	databricks --profile $(DATABRICKS_PROFILE) workspace import $(LOCAL_NOTEBOOK_PATH) $(DATABRICKS_NOTEBOOK_PATH)

.PHONY: run
run:
	databricks --profile $(DATABRICKS_PROFILE) jobs run-now --job-id <job-id>

.PHONY: deploy-run
deploy-run: upload run