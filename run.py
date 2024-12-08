import os
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import (
    get_state_dict,
    load_state_dict,
    safe_save,
    safe_load,
    get_parameters,
)
from model.model import (
    ProbabilisticResNet,
    BayesianPairwiseRankingLoss,
    sample_embeddings,
)
from utils.database import (
    init_db,
    get_all_users,
    get_user_profile,
    store_uncertain_pairs,
)
from utils.selection import identify_uncertain_pairs

# Configuration
IMAGE_DIR = "lfw"
DB_FILE = "preferences.db"

# Initialize model, optimizer, and loss function
model = ProbabilisticResNet(embedding_dim=64)
optimizer = Adam(get_parameters(model), lr=1e-3)
loss_fn = BayesianPairwiseRankingLoss()


def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    return image_files


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((64, 64))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    return Tensor(image)


def load_user_data(conn, user_id):
    result = get_user_profile(conn, user_id)
    if result:
        profile_image_path, _ = result
        profile_image = load_and_preprocess_image(profile_image_path)
        return {"profile_image": profile_image, "loader": load_and_preprocess_image}
    else:
        return None


def get_candidate_images_for_user(user_id, conn, image_files):
    profile = get_user_profile(conn, user_id)
    if profile:
        user_profile_image = profile[0]
        candidate_images = [img for img in image_files if img != user_profile_image]
        return candidate_images
    else:
        return image_files


def train_model(model, conn, image_files):
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, preferred, not_preferred FROM preferences")
    data = cursor.fetchall()

    if not data:
        print("No preferences available for training.")
        return

    # Prepare training data
    training_data = []
    for user_id, preferred_path, not_preferred_path in data:
        preferred_image = load_and_preprocess_image(preferred_path)
        not_preferred_image = load_and_preprocess_image(not_preferred_path)
        cursor.execute(
            "SELECT profile_picture FROM users WHERE user_id = ?", (user_id,)
        )
        result = cursor.fetchone()
        if result:
            profile_image_path = result[0]
            anchor_image = load_and_preprocess_image(profile_image_path)
        else:
            anchor_image = Tensor(np.zeros_like(preferred_image.numpy()))
        training_data.append((anchor_image, preferred_image, not_preferred_image))

    # Training loop
    for epoch in range(1):
        total_loss = 0
        for anchor_image, preferred_image, not_preferred_image in training_data:
            # Forward pass
            anchor_mu, anchor_sigma = model(anchor_image)
            preferred_mu, preferred_sigma = model(preferred_image)
            not_preferred_mu, not_preferred_sigma = model(not_preferred_image)

            # Compute loss
            loss = loss_fn(
                anchor_mu,
                anchor_sigma,
                preferred_mu,
                preferred_sigma,
                not_preferred_mu,
                not_preferred_sigma,
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.numpy()
        print(f"Training Loss: {total_loss / len(training_data):.4f}")

    # Save model
    state_dict = get_state_dict(model)
    safe_save(state_dict, "model.safetensors")


def load_model(model):
    model_path = "model.safetensors"
    if os.path.exists(model_path):
        state_dict = safe_load(model_path)
        load_state_dict(model, state_dict)
    else:
        print("No saved model found. Please train the model first.")


if __name__ == "__main__":
    conn = init_db()
    users = get_all_users(conn)
    image_files = get_image_files(IMAGE_DIR)

    # Train model
    train_model(model, conn, image_files)
    load_model(model)

    # Generate uncertain pairs
    for user_id in users:
        user_data = load_user_data(conn, user_id)
        if user_data:
            candidate_images = get_candidate_images_for_user(user_id, conn, image_files)
            uncertain_pairs = identify_uncertain_pairs(
                model, user_id, user_data, candidate_images, top_k=10
            )
            store_uncertain_pairs(conn, user_id, uncertain_pairs)
    print("Model trained and uncertain pairs generated for all users!")

# from tinygrad.nn.optim import Adam

# from model.model import (
#     ProbabilisticResNet,
#     BayesianPairwiseRankingLoss,
#     sample_embeddings,
# )
# from utils.database import (
#     init_db,
#     get_all_users,
#     get_user_profile,
#     store_uncertain_pairs,
# )
# from utils.image_ops import get_image_files, load_and_preprocess_image
# from utils.selection import identify_uncertain_pairs
# from torch.utils.data import DataLoader, Dataset

# # Configuration
# IMAGE_DIR = "lfw"
# DB_FILE = "preferences.db"

# model = ProbabilisticResNet(embedding_dim=64)
# optimizer = Adam([model.l1, model.l2], lr=1e-3)
# loss_fn = BayesianPairwiseRankingLoss()


# def load_user_data(conn, user_id):
#     result = get_user_profile(conn, user_id)
#     if result:
#         profile_image_path, _ = result
#         profile_image = load_and_preprocess_image(profile_image_path)
#         return {"profile_image": profile_image, "loader": load_and_preprocess_image}
#     else:
#         return None


# def get_candidate_images_for_user(user_id, conn, image_files):
#     profile = get_user_profile(conn, user_id)
#     if profile:
#         user_profile_image = profile[0]
#         candidate_images = [img for img in image_files if img != user_profile_image]
#         return candidate_images
#     else:
#         return image_files


# def train_model(model, conn, image_files):
#     cursor = conn.cursor()
#     cursor.execute("SELECT user_id, preferred, not_preferred FROM preferences")
#     data = cursor.fetchall()

#     if not data:
#         print("No preferences available for training.")
#         return

#     class PreferenceDataset(Dataset):
#         def __init__(self, data):
#             self.data = data

#         def __len__(self):
#             return len(self.data)

#         def __getitem__(self, idx):
#             user_id, preferred_path, not_preferred_path = self.data[idx]
#             preferred_image = load_and_preprocess_image(preferred_path)
#             not_preferred_image = load_and_preprocess_image(not_preferred_path)
#             cursor.execute(
#                 "SELECT profile_picture FROM users WHERE user_id = ?", (user_id,)
#             )
#             result = cursor.fetchone()
#             if result:
#                 profile_image_path = result[0]
#                 anchor_image = load_and_preprocess_image(profile_image_path)
#             else:
#                 anchor_image = torch.zeros_like(preferred_image)
#             return anchor_image, preferred_image, not_preferred_image

#     dataset = PreferenceDataset(data)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#     from tinygrad.tensor import Tensor


# # Example training loop
# for epoch in range(num_epochs):
#     total_loss = 0
#     for anchor_image, preferred_image, not_preferred_image in dataloader:
#         # Convert images to Tinygrad tensors
#         anchor_image = Tensor(anchor_image)
#         preferred_image = Tensor(preferred_image)
#         not_preferred_image = Tensor(not_preferred_image)

#         # Forward pass
#         anchor_mu, anchor_sigma = model(anchor_image)
#         preferred_mu, preferred_sigma = model(preferred_image)
#         not_preferred_mu, not_preferred_sigma = model(not_preferred_image)

#         # Compute loss
#         loss = loss_fn(
#             anchor_mu,
#             anchor_sigma,
#             preferred_mu,
#             preferred_sigma,
#             not_preferred_mu,
#             not_preferred_sigma,
#         )

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.numpy()
#     print(f"Training Loss: {total_loss / len(dataloader):.4f}")


# def load_model(model):
#     model_path = "model.pth"

#     if os.path.exists(model_path):
#         return model.base_resnet.load_state_dict()
#     else:
#         print("No saved model found. Please train the model first.")


# if __name__ == "__main__":
#     import os

#     conn = init_db()
#     users = get_all_users(conn)
#     image_files = get_image_files(IMAGE_DIR)

#     # Train model
#     train_model(model, conn, image_files)
#     load_model(model)

#     # Generate uncertain pairs
#     for user_id in users:
#         user_data = load_user_data(conn, user_id)
#         if user_data:
#             candidate_images = get_candidate_images_for_user(user_id, conn, image_files)
#             uncertain_pairs = identify_uncertain_pairs(
#                 model, user_id, user_data, candidate_images, top_k=10
#             )
#             store_uncertain_pairs(conn, user_id, uncertain_pairs)
#     print("Model trained and uncertain pairs generated for all users!")
