# Variables
PROFILE ?= STUTEST
DATABRICKS_DIR = /Workspace/Users/stu.johnstone@mantelgroup.com.au/databricks_apps/stu-test_2024_10_17-06_59/streamlit-chatbot-app
APP_NAME = stu-test

# Define help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  sync-from     Sync files from Databricks to local directory"
	@echo "  sync-to       Sync changes back to Databricks"
	@echo "  sync-to-watch Watch local directory and sync changes back to Databricks"
	@echo "  run-python    Run the app locally using Python"
	@echo "  run-streamlit Run the app locally using Streamlit"
	@echo "  deploy        Deploy the app to Databricks Apps"
	@echo "  all           Run all steps: sync-export, sync-watch, run-python, and deploy"
	@echo "  help          Show this help message"

# Sync files from Databricks to local
sync-from:
	@echo "Syncing files from Databricks to local directory..."
	databricks workspace export-dir $(DATABRICKS_DIR) . --profile $(PROFILE)

# Watch and sync files from local to Databricks
sync-to:
	@echo "Syncing changes to Databricks..."
	databricks sync . $(DATABRICKS_DIR) --profile $(PROFILE)

# Watch and sync files from local to Databricks
sync-to-watch:
	@echo "Watching local directory and syncing changes to Databricks..."
	databricks sync --watch . $(DATABRICKS_DIR) --profile $(PROFILE)

# Run the app locally using Python
run-python:
	@echo "Running the app locally using Python..."
	python app.py

# Run the app locally using Streamlit
run-streamlit:
	@echo "Running the app locally using Streamlit..."
	streamlit run app.py

# Deploy app to Databricks
deploy:
	@echo "Deploying the app to Databricks..."
	databricks apps deploy $(APP_NAME) --source-code-path $(DATABRICKS_DIR) --profile $(PROFILE)

# Default target: run the sync, watch for changes, and deploy
all: sync-export sync-watch run-python deploy
	@echo "Completed all steps: sync, watch, run, and deploy."