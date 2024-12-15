import os
import random
from PIL import Image
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import (
    load_state_dict,
    safe_load,
    get_parameters,
)
from model.model import (
    ProbabilisticResNet,
    baysean_pairwise_ranking_loss,
)
from utils.database import (
    init_db,
    get_all_users,
    get_user_profile,
    store_uncertain_pairs,
)
from utils.selection import identify_uncertain_pairs
import sqlite3

import matplotlib.pyplot as plt

# Configuration
IMAGE_DIR = "lfw"
DB_FILE = "preferences.db"

# Initialize model, optimizer, and loss function
model = ProbabilisticResNet(embedding_dim=16)
optimizer = Adam(get_parameters(model), lr=1e-3)


def display_batch(user_ids, profile_images, preferred_images, not_preferred_images):
    # Convert tinygrad Tensors to NumPy arrays
    profile_np = profile_images.numpy()
    preferred_np = preferred_images.numpy()
    not_preferred_np = not_preferred_images.numpy()

    batch_size = len(user_ids)
    fig, axs = plt.subplots(nrows=batch_size, ncols=3, figsize=(9, 3 * batch_size))

    if batch_size == 1:
        # If there's only one item in the batch, axs will not be a 2D array
        axs = [axs]

    for i in range(batch_size):
        # profile image
        ax = axs[i][0]
        img = profile_np[i].transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        ax.imshow(img)
        ax.set_title(f"User {user_ids[i]}: Profile")
        ax.axis("off")

        # preferred image
        ax = axs[i][1]
        img = preferred_np[i].transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"User {user_ids[i]}: Preferred")
        ax.axis("off")

        # not preferred image
        ax = axs[i][2]
        img = not_preferred_np[i].transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"User {user_ids[i]}: Not Preferred")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    return image_files


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_data = list(image.getdata())
    image_tensor = Tensor(image_data).div(255.0)
    return image_tensor.reshape(224, 224, 3).permute(2, 0, 1)


def load_user_data(conn, user_id):
    result = get_user_profile(conn, user_id)
    if result:
        profile_image_path, _ = result
        profile_image = load_and_preprocess_image(profile_image_path)
        return {"profile_image": profile_image, "loader": load_and_preprocess_image}
    else:
        return None


def fetch_all_preferences(conn):
    # Fetch all rows from preferences along with user profile pictures
    query = """
    SELECT u.user_id, u.profile_picture, p.preferred, p.not_preferred
    FROM preferences p
    JOIN users u ON u.user_id = p.user_id
    """
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    return (
        rows  # [(user_id, profile_pic_path, preferred_path, not_preferred_path), ...]
    )


def train_test_split(data, test_ratio=0.2, seed=42):
    random.seed(seed)
    data_shuffled = data[:]
    random.shuffle(data_shuffled)
    test_size = int(len(data_shuffled) * test_ratio)
    test_data = data_shuffled[:test_size]
    train_data = data_shuffled[test_size:]
    return train_data, test_data


class Samples:
    def __init__(self, length, shuffle=True):
        self.length = length
        self.sample_idxs = list(range(length))
        if self.shuffle:
            random.shuffle(self.sample_idxs)

    def idxs(self, batch_size):
        assert batch_size <= self.length
        if len(self.sample_idxs) < batch_size:
            self.sample_idxs = list(range(self.length))
            if self.shuffle:
                random.shuffle(self.sample_idxs)
        ret = self.sample_idxs[:batch_size]
        self.sample_idxs = self.sample_idxs[batch_size:]
        return ret

    def shuffle(self):
        random.shuffle(self.sample_idxs)


def make_batch(data, idxs):
    user_ids = [data[i][0] for i in idxs]
    profile_images = [data[i][1] for i in idxs]
    preferred_images = [data[i][2] for i in idxs]
    not_preferred_images = [data[i][3] for i in idxs]

    for i in range(len(profile_images)):
        profile_images[i] = load_and_preprocess_image(profile_images[i])
        preferred_images[i] = load_and_preprocess_image(preferred_images[i])
        not_preferred_images[i] = load_and_preprocess_image(not_preferred_images[i])

    profile_images = Tensor.stack(*profile_images)
    preferred_images = Tensor.stack(*preferred_images)
    not_preferred_images = Tensor.stack(*not_preferred_images)

    return user_ids, profile_images, preferred_images, not_preferred_images


def train_model(model, batch_size=8, num_steps=10_000, eval_interval=1):

    def step(user_profile_batch, preferred_batch, not_preferred_batch):
        Tensor.training = True

        optimizer.zero_grad()
        anchor_mu, anchor_sigma = model(user_profile_batch)
        preferred_mu, preferred_sigma = model(preferred_batch)
        not_preferred_mu, not_preferred_sigma = model(not_preferred_batch)

        current_loss = baysean_pairwise_ranking_loss(
            anchor_mu,
            anchor_sigma,
            preferred_mu,
            preferred_sigma,
            not_preferred_mu,
            not_preferred_sigma,
        )
        current_loss.backward()
        optimizer.step()
        return current_loss

    jit_step = TinyJit(step)

    conn = sqlite3.connect("preferences.db")
    all_data = fetch_all_preferences(conn)
    train_data, test_data = train_test_split(all_data, test_ratio=0.2, seed=42)
    train_samples = Samples(len(train_data))
    test_samples = Samples(len(test_data))

    for i in range(num_steps // batch_size):

        user_ids, profile_batch, preferred_batch, not_preferred_batch = make_batch(
            train_data, train_samples.idxs(batch_size)
        )
        loss = jit_step(profile_batch, preferred_batch, not_preferred_batch)

        if i % eval_interval == 0:
            # Switch off training mode for evaluation
            Tensor.training = False
            anchor_mu, anchor_sigma = model(profile_batch)
            preferred_mu, preferred_sigma = model(preferred_batch)
            not_preferred_mu, not_preferred_sigma = model(not_preferred_batch)

            acc = baysean_pairwise_ranking_loss(
                anchor_mu,
                anchor_sigma,
                preferred_mu,
                preferred_sigma,
                not_preferred_mu,
                not_preferred_sigma,
            )
            print(f"step {i:4d}, loss {loss.item():.4f}, acc {acc.item():.2f}%")

    # Save the trained model state
    # state_dict = get_state_dict(model)
    # safe_save(state_dict, "model.safetensors")
    # print("Model saved to model.safetensors")


if __name__ == "__main__":

    train_model(
        model,
    )


def load_model(model):
    model_path = "model.safetensors"
    if os.path.exists(model_path):
        state_dict = safe_load(model_path)
        load_state_dict(model, state_dict)
    else:
        print("No saved model found. Please train the model first.")


# def get_candidate_images_for_user(user_id, conn, image_files):
#     profile = get_user_profile(conn, user_id)
#     if profile:
#         user_profile_image = profile[0]
#         candidate_images = [img for img in image_files if img != user_profile_image]
#         return candidate_images
#     else:
#         return image_files


# def get_batch(conn, user_id, batch_size):
#     cursor = conn.cursor()

#     # Fetch the user's profile picture
#     cursor.execute("SELECT profile_picture FROM users WHERE user_id = ?", (user_id,))
#     profile_picture_path = cursor.fetchone()
#     if not profile_picture_path:
#         raise ValueError(f"Profile picture not found for user_id: {user_id}")
#     profile_picture_path = profile_picture_path[0]

#     # Fetch preferences for the user
#     cursor.execute(
#         """
#         SELECT preferred, not_preferred
#         FROM preferences
#         WHERE user_id = ?
#         LIMIT ?
#         """,
#         (user_id, batch_size),
#     )
#     preferences = cursor.fetchall()
#     if not preferences:
#         raise ValueError(f"No preferences found for user_id: {user_id}")

#     # Load the profile picture and normalize it
#     profile_image = load_and_preprocess_image(profile_picture_path)

#     # Prepare the batches
#     preferred_batch = []
#     not_preferred_batch = []
#     for preferred_path, not_preferred_path in preferences:
#         preferred_batch.append(load_and_preprocess_image(preferred_path))
#         not_preferred_batch.append(load_and_preprocess_image(not_preferred_path))

#     user_profile_batch = profile_image.expand(batch_size, -1, -1, -1)
#     preferred_batch = Tensor.stack(*preferred_batch)
#     not_preferred_batch = Tensor.stack(*not_preferred_batch)

#     return user_profile_batch, preferred_batch, not_preferred_batch


# class PreferenceDataset:
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         user_id, profile_pic, preferred_path, not_preferred_path = self.data[index]
#         profile_image = load_and_preprocess_image(profile_pic)
#         preferred_image = load_and_preprocess_image(preferred_path)
#         not_preferred_image = load_and_preprocess_image(not_preferred_path)
#         return user_id, profile_image, preferred_image, not_preferred_image


# def generate_batches(dataset, batch_size=32, shuffle=False):
#     indices = list(range(len(dataset)))
#     if shuffle:
#         random.shuffle(indices)

#     for i in range(0, len(indices), batch_size):
#         batch_indices = indices[i : i + batch_size]
#         batch_data = [dataset[j] for j in batch_indices]

#         # Extract data into arrays
#         user_ids = [x[0] for x in batch_data]
#         profile_images = np.stack(
#             [x[1] for x in batch_data], axis=0
#         )  # shape [B, C, H, W]
#         preferred_images = np.stack([x[2] for x in batch_data], axis=0)
#         not_preferred_images = np.stack([x[3] for x in batch_data], axis=0)

#         # Convert NumPy arrays to tinygrad Tensors
#         profile_images = Tensor(profile_images)
#         preferred_images = Tensor(preferred_images)
#         not_preferred_images = Tensor(not_preferred_images)

#         yield user_ids, profile_images, preferred_images, not_preferred_images
