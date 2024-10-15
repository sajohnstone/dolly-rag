# RAG Setup with Databricks (Scraping Confluence)

## Purpose

This project sets up a Retrieval-Augmented Generation (RAG) pipeline using Databricks. The data source in this example is Confluence, which is scraped and saved as PDFs. While there are more straightforward ways to fetch data, such as using APIs, those weren't available for this project. However, the scraper can easily be adapted for any PDF-based data source.

## How It Works

1. **Scraping Confluence**: 
    - To start the scraping process, run the following command:
      ```bash
      make scrape url="https://mantelgroup.atlassian.net" start_page="wiki/spaces/DTD/overview"
      ```
    - This will scrape the content from the specified Confluence page and save it as PDFs.

2. **Setting up Databricks DDL**:
    - After scraping some data, run the DDL notebook to create the necessary catalogs and database objects:
      - `1_setup_ddl.py` - This sets up the Databricks environment by creating catalogs, schemas, and other required structures.

3. **Uploading PDFs to Databricks**:
    - Once the PDFs are scraped, upload them to the managed volume in Databricks:
      ```bash
      make upload_pdfs
      ```
    - This will place the PDFs into your Databricks environment, making them available for processing.

4. **Running the Processing Notebooks**:
    - After the PDFs are uploaded, you can run the additional notebooks for processing, analysis, or any other operations related to your RAG pipeline.

## Customization

- The scraping process is currently designed for Confluence, but the scraper can be adapted to work with any data source that outputs PDFs.
  
## Requirements

- **Databricks**: Ensure your Databricks environment is set up and that you have the appropriate permissions.
- **Confluence Access**: Adjust scraping if using different Confluence instances or similar web-based data sources.

## Links
 - Databricks LLMS: https://notebooks.databricks.com/demos/llm-rag-chatbot/index.html
