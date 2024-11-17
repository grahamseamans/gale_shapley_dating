import streamlit as st
import os
import random
import sqlite3
from collections import defaultdict

# Path to the directory containing images
IMAGE_DIR = "lfw"

# Path to the SQLite database
DB_FILE = "preferences.db"

# Ensure the image directory exists
if not os.path.exists(IMAGE_DIR):
    st.error(f"The directory '{IMAGE_DIR}' does not exist.")
    st.stop()


# Function to recursively collect image file paths
def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    return image_files


# Collect all image paths
image_files = get_image_files(IMAGE_DIR)
if len(image_files) < 2:
    st.error("Not enough images in the directory to compare.")
    st.stop()


# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Create tables for users and preferences
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            profile_picture TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT,
            preferred TEXT,
            not_preferred TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """
    )
    conn.commit()
    return conn


# Function to register a new user
def register_user(conn, user_id):
    cursor = conn.cursor()
    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    if cursor.fetchone():
        st.warning("User ID already exists!")
        return None

    # Assign a random profile picture
    profile_picture = random.choice(image_files)
    cursor.execute(
        "INSERT INTO users (user_id, profile_picture) VALUES (?, ?)",
        (user_id, profile_picture),
    )
    conn.commit()
    st.success(f"User '{user_id}' registered successfully!")
    st.image(
        profile_picture,
        caption=f"Profile Picture for {user_id}",
        use_container_width=True,
    )


# Function to display user profile
def display_user_profile(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT profile_picture FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    if result:
        profile_picture = result[0]
        st.sidebar.write(f"Currently selecting for user: **{user_id}**")
        st.sidebar.image(profile_picture, use_container_width=True)


# Function to record user preference
def record_preference(conn, user_id, preferred, not_preferred):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO preferences (user_id, preferred, not_preferred) VALUES (?, ?, ?)",
        (user_id, preferred, not_preferred),
    )
    conn.commit()
    st.success("Preference recorded!")


# Function to display two random images for pairwise comparison
def display_images(conn, user_id):
    img1, img2 = random.sample(image_files, 2)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, use_container_width=True)
        if st.button("Select", key="left"):
            record_preference(conn, user_id, img1, img2)

    with col2:
        st.image(img2, use_container_width=True)
        if st.button("Select", key="right"):
            record_preference(conn, user_id, img2, img1)


# Function to view all preferences
def view_preferences(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM preferences")
    data = cursor.fetchall()
    if data:
        st.write("User Preferences:")
        for row in data:
            st.write(f"User: {row[0]}, Preferred: {row[1]}, Not Preferred: {row[2]}")
    else:
        st.write("No preferences recorded yet.")


# Main App
st.title("Image Preference Selector with User Profiles")
st.sidebar.title("User Options")

# Initialize the database
conn = init_db()

# Register a new user
st.sidebar.write("### Register a New User")
user_id = st.sidebar.text_input("Enter a new user ID:")
if st.sidebar.button("Submit User ID"):
    if user_id:
        register_user(conn, user_id)

# Select a user
cursor = conn.cursor()
cursor.execute("SELECT user_id FROM users")
users = [row[0] for row in cursor.fetchall()]

if users:
    selected_user = st.sidebar.selectbox("Select User", users)
    if selected_user:
        display_user_profile(conn, selected_user)
        st.write(f"Pairwise selection for user: **{selected_user}**")
        display_images(conn, selected_user)
else:
    st.sidebar.write("No users registered yet.")

# View preferences
if st.sidebar.button("View Preferences"):
    view_preferences(conn)
