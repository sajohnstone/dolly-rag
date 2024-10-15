from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import re
import pdfkit
import argparse
from bs4 import BeautifulSoup

def create_driver(chrome_driver_path):
    # Get the current user's home directory
    home_directory = os.path.expanduser("~")
    
    # Construct the path to the Chrome profile
    chrome_profile_path = os.path.join(home_directory, "Library", "Application Support", "Google", "Chrome", "Default")

    print(f"Using Chrome profile path: {chrome_profile_path}")

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

    # Pause for manual login
    input("Please log in and then press Enter to continue...")

    visited_pages = set()
    to_visit = {f"{base_url}/{start_page}"}

    while to_visit:
        current_page = to_visit.pop()
        if current_page in visited_pages:
            continue
        
        print(f"Visiting: {current_page}")
        driver.get(current_page)

        # Pause for page loading
        time.sleep(2)

        # Save the page content to PDF
        cleaned_filename = re.sub(r'\W+', '', current_page.split("/")[-1]) + '.pdf'
        save_as_pdf(driver.page_source, os.path.join(output_folder, cleaned_filename))

        # Mark the current page as visited
        visited_pages.add(current_page)

        # Find all anchor links on the current page
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            # Check if the link is a valid subpage and not already visited
            if href and href.startswith(base_url) and href not in visited_pages:
                to_visit.add(href)

    driver.quit()


def extract_cleaned_html(html_content):
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted tags like <script>, <style>, <img>, <link> (CSS links), etc.
    for tag in soup(['script', 'style', 'img', 'link']):
        tag.decompose()

    # Remove the sidebar using its class and attributes
    sidebar = soup.find('div', class_='cc-10tn8d', attrs={'data-testid': 'grid-left-sidebar'})
    if sidebar:
        sidebar.decompose()

    # Remove the top bar using its class and attributes
    sidebar = soup.find('div', class_="cc-1rx1hm4", attrs={'data-skip-link-wrapper': 'true'})
    if sidebar:
        sidebar.decompose()

    # Remove anchor tags but keep their text
    for a in soup.find_all('a'):
        a.replace_with(a.get_text())

    # Clean the remaining content but keep important tags like headings, bold text, etc.
    allowed_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'strong', 'em', 'ul', 'li', 'br']

    # Create a new BeautifulSoup object with just the desired tags
    cleaned_content = ''
    for tag in soup.find_all(True):
        if tag.name in allowed_tags:
            cleaned_content += str(tag)

    # Remove excess whitespace and return the cleaned HTML
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
    return cleaned_content

def save_as_pdf(html_content, output_path):
    # Use BeautifulSoup to parse the HTML and extract text
    text_content = extract_cleaned_html(html_content)

    # Reduce multiple consecutive whitespace (spaces, newlines) into a single space
    cleaned_text = text_content

    # Compare the length of the cleaned text with the length of the reference string
    if len(cleaned_text) > 250:
        # Wrap the cleaned text in basic HTML so pdfkit can still process it
        html_with_cleaned_text = f"<html><body><pre>{cleaned_text}</pre></body></html>"

        # Save the cleaned text to PDF
        options = {
            'enable-local-file-access': None
        }
        print(f"*** Saving content to {output_path} ***")
        pdfkit.from_string(html_with_cleaned_text, output_path, options=options)
    else:
        print(f"*** Skipping file - content too short for {output_path} ***")

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
