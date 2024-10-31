import os
import streamlit as st
import requests
import uuid  # For generating unique IDs for each document
import datetime
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from io import BytesIO
import boto3
import logging
from processor import get_pdf_documents
from utils import set_environment_variables

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# S3 configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)
BUCKET_NAME = os.getenv('BUCKET_NAME')
S3_PDFS_FOLDER = os.getenv("S3_PDFS_FOLDER")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

# Initialize NVIDIA Settings
def initialize_nvidia_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")

# Initialize NVIDIA settings once
initialize_nvidia_settings()

# Function to register a new user
def register_user(username, password):
    response = requests.post(f"{FASTAPI_URL}/signup?username={username}&password={password}")
    return response.json()

# Function to login and retrieve JWT token
def login_user(username, password):
    response = requests.post(f"{FASTAPI_URL}/login?username={username}&password={password}")
    return response.json()

# Function to check if the session is expired
def is_session_expired():
    if "token_expiration" not in st.session_state:
        return True
    current_time = datetime.datetime.utcnow()
    return current_time >= st.session_state["token_expiration"]

# Function to fetch publications
def fetch_publications():
    response = requests.get(f"{FASTAPI_URL}/publications")
    return response.json().get("publications", [])

# Function to list PDFs from S3
def list_pdfs_from_s3():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PDFS_FOLDER)
    return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]

# Function to process PDF
def show_process_pdf_page():
    # Check if the user is authenticated
    if "access_token" not in st.session_state or is_session_expired():
        st.warning("Please log in to access this page.")
        return

    st.subheader("Process PDF Document")
    
    # List PDFs in S3 bucket
    pdf_files = list_pdfs_from_s3()
    if not pdf_files:
        st.warning("No PDFs available in the S3 bucket.")
        return

    # Dropdown for selecting a PDF
    selected_pdf = st.selectbox("Select a PDF Document to Process", pdf_files, key="pdf_selectbox")
    if selected_pdf:
        st.write("Selected file:", selected_pdf)

        if st.button("Process PDF"):
            with st.spinner("Downloading and Processing..."):
                logging.info("Downloading PDF from S3...")
                pdf_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=selected_pdf)
                pdf_content = pdf_obj["Body"].read()
                logging.info("Downloaded PDF from S3.")

                pdf_file = BytesIO(pdf_content)
                pdf_file.name = selected_pdf.split("/")[-1]  # Use just the file name

                logging.info("Processing PDF...")
                documents = get_pdf_documents(pdf_file)
                logging.info("Finished processing PDF.")

            # Display processed data from documents
            for doc in documents:
                doc_type = doc.metadata['type']
                
                if doc_type == "text":
                    # Display text blocks
                    st.write("**Text Block:**", doc.text)

                elif doc_type == "table":
                    # Display tables and their images with captions
                    st.write("**Table Data:**", doc.text)
                    st.image(doc.metadata["image"], caption=doc.metadata["caption"])

                elif doc_type == "image":
                    # Display images with descriptions and captions
                    st.write("**Image Description:**", doc.text)
                    st.image(doc.metadata["image"], caption=doc.metadata["caption"])

# Main Streamlit App
def main():
    st.title("Document Exploration App")

    if "access_token" in st.session_state and not is_session_expired():
        publications = fetch_publications()
    else:
        publications = []

    # Menu options based on login status
    if "access_token" not in st.session_state or is_session_expired():
        menu_options = ["Login", "Signup"]
    else:
        menu_options = ["View Profile", "Update Password", "Explore Documents", "Process PDF", "Chat with Document", "Logout"]

    with st.sidebar:
        choice = option_menu(
            "Menu",
            menu_options,
            icons=["box-arrow-in-right", "person-plus"] if "access_token" not in st.session_state else ["person-circle", "lock", "folder", "file-earmark", "chat", "box-arrow-right"],
            key="main_menu_option"
        )

    if choice == "Signup":
        show_signup_page()
    elif choice == "Login":
        show_login_page()
    elif choice == "View Profile":
        show_profile_page()
    elif choice == "Update Password":
        show_update_password_page()
    elif choice == "Explore Documents":
        if "access_token" in st.session_state:
            show_explore_documents(publications)
        else:
            st.warning("Log in to explore documents.")
    elif choice == "Process PDF":
        show_process_pdf_page()  # Only accessible if logged in
    elif choice == "Chat with Document":
        show_chat_page()
    elif choice == "Logout":
        handle_logout()

# Show the signup page
def show_signup_page():
    st.subheader("Signup Page")
    new_username = st.text_input("Create a Username")
    new_password = st.text_input("Create a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if new_password == confirm_password:
            result = register_user(new_username, new_password)
            st.success(result.get("msg", "Signup successful"))
        else:
            st.warning("Passwords do not match!")

# Show the login page
def show_login_page():
    st.subheader("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = login_user(username, password)
        if "access_token" in result:
            st.success(f"Login Successful! Welcome, {username}!")
            st.session_state["access_token"] = result["access_token"]
            st.session_state["username"] = username
            st.session_state["token_expiration"] = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
            st.experimental_rerun()
        else:
            st.error(result.get("detail", "Login Failed"))

# Handle user logout
def handle_logout():
    st.session_state.clear()
    st.success("You have been logged out successfully!")
    st.experimental_rerun()

if __name__ == "__main__":
    main()

