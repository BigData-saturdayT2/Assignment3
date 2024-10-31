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

# Load environment variables
load_dotenv()
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

# Function to view profile
def view_profile():
    return {
        "username": st.session_state.get("username"),
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }

# Function to list PDFs from S3
def list_pdfs_from_s3():
    response = requests.get(f"{FASTAPI_URL}/s3/pdfs")
    return response.json().get("files", [])

# Function to process PDF
def process_pdf(pdf_key):
    response = requests.post(f"{FASTAPI_URL}/process-pdf-from-s3/", json={"pdf_key": pdf_key})
    return response.json() if response.status_code == 200 else None

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
        show_process_pdf_page()
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
            st.rerun()
        else:
            st.error(result.get("detail", "Login Failed"))

# Show profile page
def show_profile_page():
    st.subheader("User Profile")
    if "access_token" in st.session_state:
        profile_data = view_profile()
        if "username" in profile_data:
            st.write(f"Username: {profile_data['username']}")
            st.write(f"Created At: {profile_data['created_at']}")
        else:
            st.error("Could not retrieve profile.")
    else:
        st.warning("Log in first.")

# Show the update password page
def show_update_password_page():
    st.subheader("Update Password")
    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")

    if st.button("Update Password"):
        st.success("Password updated successfully!") if new_password else st.error("Update failed")

# Show Explore Documents section
def show_explore_documents(publications):
    st.header("Explore Documents")
    for pub in publications:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(pub["image_link"], width=100)
            st.markdown(f"[Open PDF]({pub['pdf_link']})", unsafe_allow_html=True)
        with col2:
            st.write(f"**{pub['title']}**")
            overview = pub["brief_summary"]
            if st.session_state.get(f"show_full_overview_{pub['title']}", False):
                st.write(overview)
                if st.button("Read Less", key=f"read_less_{pub['title']}"):
                    st.session_state[f"show_full_overview_{pub['title']}"] = False
            else:
                st.write(overview[:100] + "...")
                if st.button("Read More", key=f"read_more_{pub['title']}"):
                    st.session_state[f"show_full_overview_{pub['title']}"] = True
        st.write("---")

# Process PDF page
def show_process_pdf_page():
    st.subheader("Process PDF Document")
    pdf_files = list_pdfs_from_s3()
    if not pdf_files:
        st.warning("No PDFs available in the S3 bucket.")
        return

    selected_pdf = st.selectbox("Select a PDF Document to Process", pdf_files)

    if selected_pdf and st.button("Process PDF"):
        response = process_pdf(selected_pdf)
        if response:
            st.success("PDF processed successfully!")
            st.write(response)

import uuid

def show_chat_page():
    st.subheader("Chat with Document")
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Prepare chat history texts as Node objects with unique IDs
    chat_history_texts = [{"id_": str(uuid.uuid4()), "text": message["content"]} for message in st.session_state["history"]]
    
    # Convert each text message to a Node
    nodes = [TextNode(id_=doc["id_"], text=doc["text"]) for doc in chat_history_texts]

    # Initialize VectorStoreIndex with nodes
    storage_context = StorageContext.from_defaults()
    query_engine = VectorStoreIndex(nodes, storage_context=storage_context).as_query_engine(similarity_top_k=5, streaming=True)

    user_input = st.text_input("Enter your query:")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state["history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["history"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            response = query_engine.query(user_input)
            full_response = "".join(response.response_gen)
            st.markdown(full_response)
            st.session_state["history"].append({"role": "assistant", "content": full_response})


# Handle user logout
def handle_logout():
    st.session_state.clear()
    st.success("You have been logged out successfully!")
    st.rerun()

if __name__ == "__main__":
    main()
