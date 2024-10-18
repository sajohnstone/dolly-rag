from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import re
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

        # Save the page content to markdown instead of PDF
        cleaned_filename = re.sub(r'\W+', '', current_page.split("/")[-1]) + '.md'
        save_as_markdown(driver.page_source, os.path.join(output_folder, cleaned_filename))

        # Mark the current page as visited
        visited_pages.add(current_page)

        # Find all anchor links on the current page
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            if is_valid_link(href, base_url, visited_pages):
                to_visit.add(href)

    driver.quit()

def is_valid_link(href, base_url, visited_pages):
    """Check if the link is valid based on defined criteria."""
    print(f"******{href}")
    return (
        href is not None and
        href.startswith(base_url) and
        href not in visited_pages and
        '/wiki/people/' not in href and
        '_fcontenttree.md' not in href and
        '/wiki/display/' not in href
    )

def extract_cleaned_html(html_content):
    import hashlib
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted tags like <script>, <style>, <img>, <link> (CSS links), etc.
    for tag in soup(['script', 'style', 'img', 'link']):
        tag.decompose()

    # Remove specific Confluence elements like sidebars, top bars, footers, etc.
    sidebar = soup.find('div', class_='cc-10tn8d', attrs={'data-testid': 'grid-left-sidebar'})
    if sidebar:
        sidebar.decompose()

    top_bar = soup.find('div', class_="cc-1rx1hm4", attrs={'data-skip-link-wrapper': 'true'})
    if top_bar:
        top_bar.decompose()

    footer = soup.find('div', class_="footer")
    if footer:
        footer.decompose()

    # Dictionary to store hashes of seen blocks of text
    seen_text_blocks = set()

    def is_duplicate(text):
        """Check if the text block is a duplicate."""
        # Compute a hash of the text to detect duplicates
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in seen_text_blocks:
            return True
        seen_text_blocks.add(text_hash)
        return False

    # Iterate over text-containing elements
    cleaned_content = ''
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'ul', 'ol', 'li']):
        text = tag.get_text(strip=True)

        # Skip empty or duplicate text blocks
        if text and not is_duplicate(text):
            cleaned_content += str(tag)

    # Clean excess whitespace
    cleaned_content = ' '.join(cleaned_content.split()).strip()

    return cleaned_content

def save_as_markdown(html_content, output_path):
    # Use BeautifulSoup to parse the HTML and extract text
    cleaned_html = extract_cleaned_html(html_content).strip()

    # Convert HTML to basic Markdown format (headings, lists, etc.)
    markdown_content = convert_html_to_markdown(cleaned_html)

    # Check if the cleaned content length is less than 500 characters
    if len(markdown_content) < 500:
        print(f"*** Skipping file - content too short ({len(cleaned_html)} chars) ***")
        return

    # Save the markdown content to a file
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    print(f"*** Saved content to {output_path} ***")

def convert_html_to_markdown(html_content):
    # Use BeautifulSoup to convert HTML tags to Markdown
    soup = BeautifulSoup(html_content, 'html.parser')
    

    # Conversion rules
    tag_to_md = {
        'h1': lambda tag: f"# {tag.get_text()}\n",
        'h2': lambda tag: f"## {tag.get_text()}\n",
        'h3': lambda tag: f"### {tag.get_text()}\n",
        'h4': lambda tag: f"#### {tag.get_text()}\n",
        'p': lambda tag: f"{tag.get_text()}\n",
        'strong': lambda tag: f"**{tag.get_text()}**",
        'em': lambda tag: f"*{tag.get_text()}*",
        'ul': lambda tag: '\n'.join([f"- {li.get_text()}" for li in tag.find_all('li')]) + '\n',
        'ol': lambda tag: '\n'.join([f"1. {li.get_text()}" for li in tag.find_all('li')]) + '\n',
        'br': lambda tag: '\n'
    }

    # Apply rules to convert HTML to Markdown
    markdown_content = ''
    for tag in soup.find_all(True):
        if tag.name in tag_to_md:
            markdown_content += tag_to_md[tag.name](tag)

    return markdown_content.strip()

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Scrape Confluence pages and export to Markdown.")
    parser.add_argument("base_url", help="The base URL of the Confluence instance.")
    parser.add_argument("start_page", help="The starting page to scrape.")
    parser.add_argument("driver_path", help="The path to the ChromeDriver executable.")
    parser.add_argument("output_folder", help="The folder to store the output Markdown files.")
    args = parser.parse_args()

    scrape_all_pages(args.start_page, args.base_url, args.driver_path, args.output_folder)

if __name__ == "__main__":
    main()