# import streamlit as st
# import random
# import os
# from utils.database import (
#     init_db,
#     register_user,
#     get_all_users,
#     get_user_profile,
#     update_user_description,
#     record_preference,
#     remove_uncertain_pair,
#     get_uncertain_pairs,
#     get_preferences,
# )
# from utils.image_ops import get_image_files


# IMAGE_DIR = "lfw"

# if not os.path.exists(IMAGE_DIR):
#     st.error(f"The directory '{IMAGE_DIR}' does not exist.")
#     st.stop()

# image_files = get_image_files(IMAGE_DIR)
# if len(image_files) < 2:
#     st.error("Not enough images in the directory to compare.")
#     st.stop()


# if __name__ == "__main__":
#     st.title("Image Preference Selector with User Profiles")
#     st.sidebar.title("User Options")

#     conn = init_db()

#     # Register a new user
#     st.sidebar.write("### Register a New User")
#     new_user_id = st.sidebar.text_input("Enter a new user ID:", key="register_user_id")
#     description = st.sidebar.text_area(
#         "Enter a description for the user:", key="register_description"
#     )

#     if st.sidebar.button("Submit User ID", key="register_button"):
#         if new_user_id:
#             profile_picture = random.choice(image_files)
#             success = register_user(conn, new_user_id, description, profile_picture)
#             if success:
#                 st.sidebar.success(f"User '{new_user_id}' registered successfully!")
#                 st.sidebar.image(
#                     profile_picture,
#                     caption=f"Profile Picture for {new_user_id}",
#                     use_container_width=True,
#                 )
#             else:
#                 st.sidebar.warning("User ID already exists!")
#         else:
#             st.warning("Please enter a user ID!")

#     users = get_all_users(conn)

#     if users:
#         selected_user = st.sidebar.selectbox("Select User", users, key="select_user")
#         if selected_user:
#             profile_picture, user_description = get_user_profile(conn, selected_user)
#             st.sidebar.write(f"Currently selecting for user: **{selected_user}**")
#             st.sidebar.image(profile_picture, use_container_width=True)
#             st.sidebar.write("### Edit Description")
#             new_description = st.sidebar.text_area(
#                 "User Description:",
#                 value=user_description if user_description else "",
#                 key=f"description_{selected_user}",
#             )
#             if st.sidebar.button(
#                 "Save Description", key=f"save_description_{selected_user}"
#             ):
#                 update_user_description(conn, selected_user, new_description)
#                 st.sidebar.success("Description updated successfully!")

#             st.write(f"Pairwise selection for user: **{selected_user}**")
#             pairs = get_uncertain_pairs(conn, selected_user)

#             if not pairs:
#                 st.write(
#                     "No uncertain pairs available. Please run the training and uncertain pair generation outside this app."
#                 )
#             else:
#                 img1_path, img2_path = pairs[0]
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.image(img1_path, use_container_width=True)
#                     if st.button(
#                         "Select Left Image", key=f"select_left_{img1_path}_{img2_path}"
#                     ):
#                         record_preference(conn, selected_user, img1_path, img2_path)
#                         remove_uncertain_pair(conn, selected_user, img1_path, img2_path)
#                         st.experimental_rerun()
#                 with col2:
#                     st.image(img2_path, use_container_width=True)
#                     if st.button(
#                         "Select Right Image",
#                         key=f"select_right_{img1_path}_{img2_path}",
#                     ):
#                         record_preference(conn, selected_user, img2_path, img1_path)
#                         remove_uncertain_pair(conn, selected_user, img1_path, img2_path)
#                         st.experimental_rerun()

#     else:
#         st.sidebar.write("No users registered yet.")

#     if st.sidebar.button("View Preferences", key="view_preferences"):
#         prefs = get_preferences(conn)
#         if prefs:
#             st.write("User Preferences:")
#             for row in prefs:
#                 st.write(
#                     f"User: {row[0]}, Preferred: {row[1]}, Not Preferred: {row[2]}"
#                 )
#         else:
#             st.write("No preferences recorded yet.")

import streamlit as st
import random
import os
from utils.database import (
    init_db,
    register_user,
    get_all_users,
    get_user_profile,
    update_user_description,
    record_preference,
    remove_uncertain_pair,
    get_uncertain_pairs,
    get_preferences,
    # New function for counting labeled pairs
    get_label_count,
)
from utils.image_ops import get_image_files

IMAGE_DIR = "lfw"

if not os.path.exists(IMAGE_DIR):
    st.error(f"The directory '{IMAGE_DIR}' does not exist.")
    st.stop()

image_files = get_image_files(IMAGE_DIR)
if len(image_files) < 2:
    st.error("Not enough images in the directory to compare.")
    st.stop()

if __name__ == "__main__":
    st.title("Image Preference Selector with User Profiles")
    st.sidebar.title("User Options")

    conn = init_db()

    # Register a new user
    st.sidebar.write("### Register a New User")
    new_user_id = st.sidebar.text_input("Enter a new user ID:", key="register_user_id")
    description = st.sidebar.text_area(
        "Enter a description for the user:", key="register_description"
    )

    if st.sidebar.button("Submit User ID", key="register_button"):
        if new_user_id:
            profile_picture = random.choice(image_files)
            success = register_user(conn, new_user_id, description, profile_picture)
            if success:
                st.sidebar.success(f"User '{new_user_id}' registered successfully!")
                st.sidebar.image(
                    profile_picture,
                    caption=f"Profile Picture for {new_user_id}",
                    use_container_width=True,
                )
            else:
                st.sidebar.warning("User ID already exists!")
        else:
            st.warning("Please enter a user ID!")

    users = get_all_users(conn)

    if users:
        selected_user = st.sidebar.selectbox("Select User", users, key="select_user")
        if selected_user:
            # Get user’s profile and description
            profile_picture, user_description = get_user_profile(conn, selected_user)
            st.sidebar.write(f"**Currently selecting for user:** {selected_user}")
            st.sidebar.image(profile_picture, use_container_width=True)

            # ----- SHOW LABEL COUNT -----
            label_count = get_label_count(conn, selected_user)
            st.sidebar.write(f"Labeled pairs: **{label_count}**")

            st.sidebar.write("### Edit Description")
            new_description = st.sidebar.text_area(
                "User Description:",
                value=user_description if user_description else "",
                key=f"description_{selected_user}",
            )
            if st.sidebar.button(
                "Save Description", key=f"save_description_{selected_user}"
            ):
                update_user_description(conn, selected_user, new_description)
                st.sidebar.success("Description updated successfully!")

            # --- TOGGLE FOR RANDOM LABELING MODE ---
            random_label_mode = st.sidebar.checkbox("Random Labeling Mode", value=False)

            if random_label_mode:
                st.subheader("Random Pair Labeling")
                img1_path, img2_path = random.sample(image_files, 2)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img1_path, use_container_width=True)
                    if st.button(
                        f"Select Left Image",
                        key=f"select_left_random_{img1_path}_{img2_path}",
                    ):
                        record_preference(conn, selected_user, img1_path, img2_path)
                        st.experimental_rerun()
                with col2:
                    st.image(img2_path, use_container_width=True)
                    if st.button(
                        f"Select Right Image",
                        key=f"select_right_random_{img1_path}_{img2_path}",
                    ):
                        record_preference(conn, selected_user, img2_path, img1_path)
                        st.experimental_rerun()

            else:
                st.subheader("Uncertain Pair Labeling")
                pairs = get_uncertain_pairs(conn, selected_user)
                if not pairs:
                    st.write(
                        "No uncertain pairs available. Please run the training and uncertain pair generation outside this app."
                    )
                else:
                    img1_path, img2_path = pairs[0]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img1_path, use_container_width=True)
                        if st.button(
                            "Select Left Image",
                            key=f"select_left_{img1_path}_{img2_path}",
                        ):
                            record_preference(conn, selected_user, img1_path, img2_path)
                            remove_uncertain_pair(
                                conn, selected_user, img1_path, img2_path
                            )
                            st.experimental_rerun()
                    with col2:
                        st.image(img2_path, use_container_width=True)
                        if st.button(
                            "Select Right Image",
                            key=f"select_right_{img1_path}_{img2_path}",
                        ):
                            record_preference(conn, selected_user, img2_path, img1_path)
                            remove_uncertain_pair(
                                conn, selected_user, img1_path, img2_path
                            )
                            st.experimental_rerun()

    else:
        st.sidebar.write("No users registered yet.")

    # Button to show all preferences
    if st.sidebar.button("View Preferences", key="view_preferences"):
        prefs = get_preferences(conn)
        if prefs:
            st.write("User Preferences:")
            for row in prefs:
                st.write(
                    f"User: {row[0]}, Preferred: {row[1]}, Not Preferred: {row[2]}"
                )
        else:
            st.write("No preferences recorded yet.")

# # app.py

# import streamlit as st
# import os
# import random
# import sqlite3
# import torch
# import itertools
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from asldf.model import ProbabilisticResNet, BayesianPairwiseRankingLoss, sample_embeddings

# # ---- Configuration ---- #
# # Path to the directory containing images
# IMAGE_DIR = "lfw"

# # Path to the SQLite database
# DB_FILE = "preferences.db"

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize the model globally
# model = ProbabilisticResNet(embedding_dim=64)

# # Initialize optimizer and loss function globally
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = BayesianPairwiseRankingLoss()

# # ---- Helper Functions ---- #

# # Ensure the image directory exists
# if not os.path.exists(IMAGE_DIR):
#     st.error(f"The directory '{IMAGE_DIR}' does not exist.")
#     st.stop()


# # Function to recursively collect image file paths
# def get_image_files(directory):
#     image_files = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_files.append(os.path.join(root, file))
#     return image_files


# # Collect all image paths
# image_files = get_image_files(IMAGE_DIR)
# if len(image_files) < 2:
#     st.error("Not enough images in the directory to compare.")
#     st.stop()


# # Initialize SQLite Database
# def init_db():
#     conn = sqlite3.connect(DB_FILE)
#     cursor = conn.cursor()
#     # Create tables for users and preferences
#     cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS users (
#             user_id TEXT PRIMARY KEY,
#             profile_picture TEXT,
#             description TEXT
#         )
#         """
#     )
#     cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS preferences (
#             user_id TEXT,
#             preferred TEXT,
#             not_preferred TEXT,
#             FOREIGN KEY (user_id) REFERENCES users (user_id)
#         )
#         """
#     )
#     cursor.execute(
#         """
#         CREATE TABLE IF NOT EXISTS uncertain_pairs (
#             user_id TEXT,
#             img1 TEXT,
#             img2 TEXT,
#             UNIQUE(user_id, img1, img2),
#             FOREIGN KEY (user_id) REFERENCES users (user_id)
#         )
#         """
#     )
#     conn.commit()
#     return conn


# # Function to register a new user
# def register_user(conn, user_id, description):
#     cursor = conn.cursor()
#     # Check if user already exists
#     cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
#     if cursor.fetchone():
#         st.warning("User ID already exists!")
#         return None

#     # Assign a random profile picture
#     profile_picture = random.choice(image_files)
#     cursor.execute(
#         "INSERT INTO users (user_id, profile_picture, description) VALUES (?, ?, ?)",
#         (user_id, profile_picture, description),
#     )
#     conn.commit()
#     st.success(f"User '{user_id}' registered successfully!")
#     st.image(
#         profile_picture,
#         caption=f"Profile Picture for {user_id}",
#         use_container_width=True,
#     )


# # Function to display and edit user profile
# def display_and_edit_user_profile(conn, user_id):
#     cursor = conn.cursor()
#     cursor.execute(
#         "SELECT profile_picture, description FROM users WHERE user_id = ?", (user_id,)
#     )
#     result = cursor.fetchone()
#     if result:
#         profile_picture, description = result
#         st.sidebar.write(f"Currently selecting for user: **{user_id}**")
#         st.sidebar.image(profile_picture, use_container_width=True)

#         # Display and edit description
#         st.sidebar.write("### Edit Description")
#         new_description = st.sidebar.text_area(
#             "User Description:",
#             value=description if description else "",
#             key=f"description_{user_id}",
#         )
#         if st.sidebar.button("Save Description", key=f"save_description_{user_id}"):
#             cursor.execute(
#                 "UPDATE users SET description = ? WHERE user_id = ?",
#                 (new_description, user_id),
#             )
#             conn.commit()
#             st.sidebar.success("Description updated successfully!")


# # Function to record user preference
# def record_preference(conn, user_id, preferred, not_preferred):
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO preferences (user_id, preferred, not_preferred) VALUES (?, ?, ?)",
#         (user_id, preferred, not_preferred),
#     )
#     conn.commit()
#     st.success("Preference recorded!")


# # Function to remove uncertain pair after recording preference
# def remove_uncertain_pair(conn, user_id, img1, img2):
#     cursor = conn.cursor()
#     cursor.execute(
#         "DELETE FROM uncertain_pairs WHERE user_id = ? AND img1 = ? AND img2 = ?",
#         (user_id, img1, img2),
#     )
#     conn.commit()


# # Function to view all preferences
# def view_preferences(conn):
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM preferences")
#     data = cursor.fetchall()
#     if data:
#         st.write("User Preferences:")
#         for row in data:
#             st.write(f"User: {row[0]}, Preferred: {row[1]}, Not Preferred: {row[2]}")
#     else:
#         st.write("No preferences recorded yet.")


# # Load and preprocess images
# def load_and_preprocess_image(image_path):
#     transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image)
#     return image  # Return tensor without adding batch dimension


# # Get candidate images for a user
# def get_candidate_images_for_user(user_id, conn):
#     cursor = conn.cursor()
#     # Exclude the user's own profile picture
#     cursor.execute("SELECT profile_picture FROM users WHERE user_id = ?", (user_id,))
#     result = cursor.fetchone()
#     if result:
#         user_profile_image = result[0]
#         candidate_images = [img for img in image_files if img != user_profile_image]
#         return candidate_images
#     else:
#         return image_files


# # Compute score differences
# def compute_score_diff(
#     anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
# ):
#     # Sample embeddings
#     anchor_samples = sample_embeddings(anchor_mu, anchor_sigma, num_samples=10)
#     img1_samples = sample_embeddings(img1_mu, img1_sigma, num_samples=10)
#     img2_samples = sample_embeddings(img2_mu, img2_sigma, num_samples=10)

#     # Compute scores
#     img1_scores = torch.sum(anchor_samples * img1_samples, dim=1)
#     img2_scores = torch.sum(anchor_samples * img2_samples, dim=1)

#     # Score differences
#     score_diffs = img1_scores - img2_scores  # Tensor of size [num_samples]
#     return score_diffs


# # Compute uncertainty
# def compute_uncertainty(score_diffs):
#     # Compute variance as uncertainty measure
#     uncertainty = torch.var(score_diffs).item()
#     return uncertainty


# # Identify uncertain pairs for a user
# def identify_uncertain_pairs(model, user_id, user_data, conn, top_k=10):
#     # Get all candidates
#     candidate_images = get_candidate_images_for_user(user_id, conn)
#     # Sample a subset if too many combinations
#     candidate_pairs = list(itertools.combinations(candidate_images, 2))
#     if len(candidate_pairs) > 500:
#         candidate_pairs = random.sample(candidate_pairs, 500)

#     uncertainties = []
#     # Load anchor image once
#     anchor_image = user_data["profile_image"].to(device).unsqueeze(0)
#     with torch.no_grad():
#         anchor_mu, anchor_sigma = model(anchor_image)

#     for img1_path, img2_path in candidate_pairs:
#         # Load images and preprocess
#         img1 = load_and_preprocess_image(img1_path).to(device).unsqueeze(0)
#         img2 = load_and_preprocess_image(img2_path).to(device).unsqueeze(0)
#         # Get embeddings
#         with torch.no_grad():
#             img1_mu, img1_sigma = model(img1)
#             img2_mu, img2_sigma = model(img2)
#         # Compute uncertainty
#         score_diff = compute_score_diff(
#             anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
#         )
#         uncertainty = compute_uncertainty(score_diff)
#         uncertainties.append((uncertainty, img1_path, img2_path))

#     # Sort by uncertainty
#     uncertainties.sort(reverse=True)  # Higher uncertainty first
#     # Get top_k uncertain pairs
#     top_uncertain_pairs = uncertainties[:top_k]
#     return top_uncertain_pairs


# # Load user data
# def load_user_data(conn, user_id):
#     cursor = conn.cursor()
#     cursor.execute("SELECT profile_picture FROM users WHERE user_id = ?", (user_id,))
#     result = cursor.fetchone()
#     if result:
#         profile_image_path = result[0]
#         # Load and preprocess the profile image
#         profile_image = load_and_preprocess_image(profile_image_path)
#         return {"profile_image": profile_image}
#     else:
#         return None


# # Train the model on all users' data
# def train_model(model, conn):
#     # Fetch preferences for all users
#     cursor = conn.cursor()
#     cursor.execute("SELECT user_id, preferred, not_preferred FROM preferences")
#     data = cursor.fetchall()
#     if not data:
#         st.warning("No preferences available for training.")
#         return

#     # Create a dataset and dataloader
#     class PreferenceDataset(Dataset):
#         def __init__(self, data):
#             self.data = data

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             user_id, preferred_path, not_preferred_path = self.data[idx]
#             # Load and preprocess images
#             preferred_image = load_and_preprocess_image(preferred_path)
#             not_preferred_image = load_and_preprocess_image(not_preferred_path)
#             # Use the user's profile image as the anchor
#             cursor.execute(
#                 "SELECT profile_picture FROM users WHERE user_id = ?", (user_id,)
#             )
#             result = cursor.fetchone()
#             if result:
#                 profile_image_path = result[0]
#                 anchor_image = load_and_preprocess_image(profile_image_path)
#             else:
#                 # Use a random image if no profile picture is found
#                 anchor_image = torch.zeros_like(preferred_image)
#             return anchor_image, preferred_image, not_preferred_image

#     dataset = PreferenceDataset(data)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     # Training loop
#     model.train()
#     for epoch in range(1):  # Adjust the number of epochs as needed
#         total_loss = 0
#         for anchor_image, preferred_image, not_preferred_image in dataloader:
#             anchor_image = anchor_image.to(device)
#             preferred_image = preferred_image.to(device)
#             not_preferred_image = not_preferred_image.to(device)

#             # Get embeddings
#             anchor_mu, anchor_sigma = model(anchor_image)
#             preferred_mu, preferred_sigma = model(preferred_image)
#             not_preferred_mu, not_preferred_sigma = model(not_preferred_image)

#             # Compute loss
#             loss = loss_fn(
#                 anchor_mu,
#                 anchor_sigma,
#                 preferred_mu,
#                 preferred_sigma,
#                 not_preferred_mu,
#                 not_preferred_sigma,
#             )

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         st.write(f"Training Loss: {total_loss / len(dataloader):.4f}")

#     # Save the model after training
#     save_model(model)


# # Save the model
# def save_model(model):
#     torch.save(model.state_dict(), "model.pth")


# # Load the model
# def load_model(model):
#     model_path = "model.pth"
#     if os.path.exists(model_path):
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.to(device)
#         model.eval()
#     else:
#         st.warning("No saved model found. Training a new model.")


# # Display images for user to select
# def display_images(conn, user_id):
#     cursor = conn.cursor()
#     # Fetch uncertain pairs for the user
#     cursor.execute(
#         "SELECT img1, img2 FROM uncertain_pairs WHERE user_id = ?", (user_id,)
#     )
#     pairs = cursor.fetchall()

#     if not pairs:
#         st.write(
#             "No uncertain pairs available. Please train the model or add more data."
#         )
#         return

#     # Select the first pair
#     img1_path, img2_path = pairs[0]

#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(img1_path, use_container_width=True)
#         if st.button("Select Left Image", key=f"select_left_{img1_path}_{img2_path}"):
#             record_preference(conn, user_id, img1_path, img2_path)
#             # Remove the pair from uncertain_pairs
#             remove_uncertain_pair(conn, user_id, img1_path, img2_path)
#             # Refresh the page to show the next pair
#             st.experimental_rerun()
#     with col2:
#         st.image(img2_path, use_container_width=True)
#         if st.button("Select Right Image", key=f"select_right_{img1_path}_{img2_path}"):
#             record_preference(conn, user_id, img2_path, img1_path)
#             # Remove the pair from uncertain_pairs
#             remove_uncertain_pair(conn, user_id, img1_path, img2_path)
#             # Refresh the page to show the next pair
#             st.experimental_rerun()


# def store_uncertain_pairs(conn, user_id, uncertain_pairs):
#     cursor = conn.cursor()
#     for _, img1_path, img2_path in uncertain_pairs:
#         try:
#             cursor.execute(
#                 "INSERT OR IGNORE INTO uncertain_pairs (user_id, img1, img2) VALUES (?, ?, ?)",
#                 (user_id, img1_path, img2_path),
#             )
#         except sqlite3.IntegrityError:
#             # Pair already exists
#             continue
#     conn.commit()


# # ---- Main App ---- #
# def main():
#     st.title("Image Preference Selector with User Profiles")
#     st.sidebar.title("User Options")

#     # Initialize the database
#     conn = init_db()

#     # Load the model
#     load_model(model)

#     # Register a new user
#     st.sidebar.write("### Register a New User")
#     user_id = st.sidebar.text_input("Enter a new user ID:", key="register_user_id")
#     description = st.sidebar.text_area(
#         "Enter a description for the user:", key="register_description"
#     )

#     if st.sidebar.button("Submit User ID", key="register_button"):
#         if user_id:
#             register_user(conn, user_id, description)
#         else:
#             st.warning("Please enter a user ID!")

#     # Select a user
#     cursor = conn.cursor()
#     cursor.execute("SELECT user_id FROM users")
#     users = [row[0] for row in cursor.fetchall()]

#     if users:
#         selected_user = st.sidebar.selectbox("Select User", users, key="select_user")
#         if selected_user:
#             display_and_edit_user_profile(conn, selected_user)
#             st.write(f"Pairwise selection for user: **{selected_user}**")
#             display_images(conn, selected_user)
#     else:
#         st.sidebar.write("No users registered yet.")

#     # Train the model and generate uncertain pairs
#     if st.sidebar.button("Train Model and Generate Uncertain Pairs", key="train_model"):
#         st.write("Training model and generating uncertain pairs...")
#         # Train the model on all users' data
#         train_model(model, conn)
#         # For each user, identify uncertain pairs
#         for user_id in users:
#             user_data = load_user_data(conn, user_id)
#             if user_data:
#                 uncertain_pairs = identify_uncertain_pairs(
#                     model, user_id, user_data, conn, top_k=10
#                 )
#                 # Store uncertain pairs
#                 store_uncertain_pairs(conn, user_id, uncertain_pairs)
#         st.success("Model trained and uncertain pairs generated for all users!")

#     # View preferences
#     if st.sidebar.button("View Preferences", key="view_preferences"):
#         view_preferences(conn)


# if __name__ == "__main__":
#     main()
