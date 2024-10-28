import streamlit as st
import requests
import datetime
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# FastAPI backend URL from .env file
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")  # Default to localhost if not found

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
        return True  # No expiration time set
    current_time = datetime.datetime.utcnow()
    return current_time >= st.session_state["token_expiration"]

# Function to fetch publications
def fetch_publications():
    response = requests.get(f"{FASTAPI_URL}/publications")
    return response.json().get("publications", [])

# Main Streamlit App
def main():
    st.title("Document Exploration App")

    # Check if the user is logged in and fetch publications
    if "access_token" in st.session_state and not is_session_expired():
        publications = fetch_publications()
    else:
        publications = []

    # Menu options based on the user's login status
    if "access_token" not in st.session_state or is_session_expired():
        menu_options = ["Login", "Signup"]
    else:
        menu_options = ["View Profile", "Update Password", "Explore Documents", "Logout"]

    # Sidebar menu
    with st.sidebar:
        choice = option_menu(
            "Menu",
            menu_options,
            icons=["box-arrow-in-right", "person-plus"] if "access_token" not in st.session_state else ["person-circle", "lock", "folder", "box-arrow-right"],
            key="main_menu_option"
        )

    # Menu Logic
    if choice == "Signup":
        show_signup_page()
    elif choice == "Login":
        show_login_page()
    elif choice == "View Profile":
        show_profile_page()
    elif choice == "Update Password":
        show_update_password_page()
    elif choice == "Explore Documents":
        if "access_token" in st.session_state:  # Check if user is logged in
            show_explore_documents(publications)
        else:
            st.warning("You need to log in to explore documents.")
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
            # Store access token and expiration time in session state
            st.session_state["access_token"] = result["access_token"]
            st.session_state["username"] = username  # Store the username for later use
            st.session_state["token_expiration"] = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
            st.session_state["logged_in"] = True  # Set logged in state
            st.rerun()  # Refresh the app
        else:
            st.error(result.get("detail", "Login Failed"))

# Show the user profile page
def show_profile_page():
    st.subheader("User Profile")
    if "access_token" in st.session_state:
        profile_data = view_profile(st.session_state["access_token"])
        if "username" in profile_data:
            st.write(f"Username: {profile_data['username']}")
            st.write(f"Created At: {profile_data['created_at']}")
        else:
            st.error(profile_data.get("detail", "Could not retrieve profile"))
    else:
        st.warning("You need to login first.")

# Show the update password page
def show_update_password_page():
    st.subheader("Update Password")
    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")

    if st.button("Update Password"):
        result = update_password(old_password, new_password, st.session_state["access_token"])
        if "msg" in result:
            st.success(result["msg"])
        else:
            st.error(result.get("detail", "Password update failed"))

# Show Explore Documents section
def show_explore_documents(publications):
    st.header("Explore Documents")

    # Display Documents in a Card Layout
    for pub in publications:
        if pub["image_link"] and pub["pdf_link"]:  # Ensure both image and PDF link exist
            col1, col2 = st.columns([1, 3])  # Adjust the column sizes as needed
            with col1:
                st.image(pub["image_link"], width=100)  # Display the image
                st.markdown(f"[Open PDF]({pub['pdf_link']})", unsafe_allow_html=True)  # PDF link
            with col2:
                st.write(f"**{pub['title']}**")  # Title
                
                # Display overview with read more/read less functionality
                overview = pub["brief_summary"]
                if st.session_state.get(f"show_full_overview_{pub['title']}", False):
                    st.write(overview)  # Show full overview
                    if st.button("Read Less", key=f"read_less_{pub['title']}"):
                        st.session_state[f"show_full_overview_{pub['title']}"] = False  # Hide full overview
                else:
                    st.write(overview[:100] + "...")  # Show truncated overview
                    if st.button("Read More", key=f"read_more_{pub['title']}"):
                        st.session_state[f"show_full_overview_{pub['title']}"] = True  # Show full overview
            st.write("---")  # Separator for clarity

# Handle user logout
def handle_logout():
    st.session_state.clear()
    st.success("You have been logged out successfully!")
    st.session_state["logged_in"] = False  # Reset logged in state
    st.rerun()  # Rerun to refresh the app

if __name__ == "__main__":
    main()
