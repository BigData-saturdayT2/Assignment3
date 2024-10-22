import os
import requests
import boto3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# S3 Configuration
s3_bucket_name = os.getenv("S3_BUCKET_NAME")  # From .env: S3_BUCKET_NAME=bdiaassignment3
s3_pdfs_folder = os.getenv("S3_PDFS_FOLDER")  # From .env: S3_PDFS_FOLDER=pdfs1/
s3_images_folder = os.getenv("S3_IMAGES_FOLDER")  # From .env: S3_IMAGES_FOLDER=images1/

# AWS Credentials (configured via environment variables)
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Function to upload file content to S3 directly
def upload_to_s3(file_content, s3_bucket, s3_key):
    try:
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=file_content)
        print(f"Uploaded to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")

# Function to scrape all publication links by clicking the "Next" button until it is no longer available
def scrape_all_publication_links_with_clicking(landing_url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(landing_url)
    
    # Initialize an empty list to store all publication links
    all_publication_links = []
    
    while True:
        # Wait and find publication links on the current page
        try:
            elements = WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[href*="/research/foundation"]'))
            )
            for elem in elements:
                link = elem.get_attribute('href')
                if link and link not in all_publication_links:
                    all_publication_links.append(link)
        except Exception as e:
            print(f"Error finding publication links: {e}")
        
        # Attempt to click the "Next" button
        try:
            # Scroll to the bottom of the page to ensure the "Next" button is visible
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)  # Allow time for scrolling

            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//li[@aria-label="Next"]'))
            )
            print("Clicking the 'Next' button...")
            driver.execute_script("arguments[0].click();", next_button)  # Use JavaScript to click
            time.sleep(3)  # Wait for the next page to load

        except (NoSuchElementException, TimeoutException) as e:
            print("No more pages or failed to click 'Next'. Exiting loop.")
            break
    
    driver.quit()
    return all_publication_links

# Function to visit each publication page, download PDFs and images, and upload them directly to S3
def download_content_and_upload_to_s3(publication_links):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    for link in publication_links:
        try:
            driver.get(link)
            time.sleep(3)  # Allow time for the page to load

            # Extract the title
            title = "No title found."
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, 'h1.spotlight-hero__title.spotlight-max-width-item')
                title = title_element.text.strip()
                print(f"Title: {title}")
            except NoSuchElementException:
                pass
            
            # Locate and check for the PDF element
            try:
                pdf_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a.content-asset--primary'))
                )
                pdf_url = pdf_element.get_attribute('href')

                if pdf_url and pdf_url.endswith('.pdf'):
                    pdf_name = pdf_url.split("/")[-1]

                    # Download the PDF directly to memory
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        # Upload PDF content directly to S3
                        s3_key = f"{s3_pdfs_folder}{pdf_name}"
                        upload_to_s3(response.content, s3_bucket_name, s3_key)
                    else:
                        print(f"Failed to download PDF: {pdf_name}")
                else:
                    print("No valid PDF found, skipping download.")
            
            except TimeoutException:
                print("PDF link not found or not a valid PDF, skipping download.")
            
            # Locate and download the image
            try:
                image_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'section.book__cover-image img'))
                )
                image_url = image_element.get_attribute('src')
                image_name = image_url.split("/")[-1].split("?")[0]

                # Download the image directly to memory
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    # Upload Image content directly to S3
                    s3_key = f"{s3_images_folder}{image_name}"
                    upload_to_s3(img_response.content, s3_bucket_name, s3_key)
                else:
                    print(f"Failed to download Image: {image_name}")
            except TimeoutException:
                print("Image not found, skipping download.")
            
        except Exception as e:
            print(f"Error accessing or downloading content on {link}: {e}")
    
    driver.quit()

# Example usage
landing_page_url = "https://rpc.cfainstitute.org/en/research-foundation/publications#sort=%40officialz32xdate%20descending&f:SeriesContent=[Research%20Foundation]"
publication_links = scrape_all_publication_links_with_clicking(landing_page_url)

# Download PDFs and images, and upload directly to S3 bucket if available
download_content_and_upload_to_s3(publication_links)
