# Define variables
PYTHON=python3
VENV=venv
DRIVER_PATH=/opt/homebrew/bin/chromedriver
OUTPUT_DIR=./../data
REQUIREMENTS=requirements.txt
SCRAPER=scraper.py

# Create a virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

# Run the scraper, with parameters passed from the command line
scrape: setup
	$(VENV)/bin/python $(SCRAPER) $(url) $(start_page) $(DRIVER_PATH) $(OUTPUT_DIR)

# Clean up the environment (optional)
clean:
	rm -rf $(VENV) $(OUTPUT_DIR)

# Help target that shows usage and example
help:
	@echo "Usage: make scrape url='<confluence_base_url>' start_page='<confluence_start_page>'"
	@echo "Example: make scrape url=\"https://mantelgroup.atlassian.net\" start_page=\"wiki/spaces/DTD/overview\""
	@echo ""
	@echo "Steps:"
	@echo " 1. Ensure you have ChromeDriver installed and update the DRIVER_PATH variable."
	@echo " 2. Run 'make scrape' with the necessary parameters to scrape and convert Confluence pages to PDF."