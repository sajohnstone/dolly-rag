from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import pdfkit
import argparse

def create_driver(chrome_driver_path):
    # Get the current user's home directory
    home_directory = os.path.expanduser("~")
    
    # Construct the path to the Chrome profile
    chrome_profile_path = os.path.join(home_directory, "Library", "Application Support", "Google", "Chrome", "Default")

    print(chrome_profile_path)

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument(f"--user-data-dir={chrome_profile_path}")
    chrome_options.add_argument("--profile-directory=Default")

    # Create a Chrome driver service
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Define scraping function
def scrape_all_pages(start_page, base_url, driver_path, output_folder):
    driver = create_driver(driver_path)
    driver.get(f"{base_url}/{start_page}")

    visited_pages = set()
    visited_pages.add(f"{base_url}/{start_page}")

    # Example page scraping logic - can be modified based on your needs
    while True:
        try:
            # Scrape page content here...
            time.sleep(2)  # Simulate some delay for page loading

            # After scraping, save to PDF
            save_as_pdf(driver.page_source, os.path.join(output_folder, 'page.pdf'))

            # Logic for following links and scraping more pages
            # Add more to visited_pages, navigate, etc.

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    driver.quit()

# Define a function to save the page content to PDF
def save_as_pdf(html_content, output_path):
    options = {
        'enable-local-file-access': None
    }
    pdfkit.from_string(html_content, output_path, options=options)

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Scrape Confluence pages and export to PDF.")
    parser.add_argument("base_url", help="The base URL of the Confluence instance.")
    parser.add_argument("start_page", help="The starting page to scrape.")
    parser.add_argument("driver_path", help="The path to the ChromeDriver executable.")
    parser.add_argument("output_folder", help="The folder to store the output PDFs.")
    args = parser.parse_args()

    scrape_all_pages(args.start_page, args.base_url, args.driver_path, args.output_folder)

if __name__ == "__main__":
    main()