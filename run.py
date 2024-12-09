import os
from PIL import Image
from tinygrad import TinyJit
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
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")

    # Resize the image to 224x224
    image = image.resize((224, 224))

    # Convert the image to raw pixel data
    image_data = list(image.getdata())

    # Create a Tinygrad Tensor from the raw pixel data and normalize
    image_tensor = Tensor(image_data).div(255.0)

    # Reshape and transpose to match CHW format (channels, height, width)
    return image_tensor.reshape(224, 224, 3).permute(2, 0, 1)


# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((224, 224))
#     image = np.array(image).astype(np.float32) / 255.0
#     image = np.transpose(image, (2, 0, 1))  # Change to CHW format
#     return Tensor(image)


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


def get_batch(conn, user_id, batch_size):
    """
    Get a batch of data for a given user.

    Args:
        conn: SQLite connection object.
        user_id: ID of the user for whom data is being fetched.
        batch_size: Number of preferences to include in the batch.

    Returns:
        A tuple of Tensors: (user_profile_batch, preferred_batch, not_preferred_batch).
    """
    cursor = conn.cursor()

    # Fetch the user's profile picture
    cursor.execute("SELECT profile_picture FROM users WHERE user_id = ?", (user_id,))
    profile_picture_path = cursor.fetchone()
    if not profile_picture_path:
        raise ValueError(f"Profile picture not found for user_id: {user_id}")
    profile_picture_path = profile_picture_path[0]

    # Fetch preferences for the user
    cursor.execute(
        """
        SELECT preferred, not_preferred
        FROM preferences
        WHERE user_id = ?
        LIMIT ?
        """,
        (user_id, batch_size),
    )
    preferences = cursor.fetchall()
    if not preferences:
        raise ValueError(f"No preferences found for user_id: {user_id}")

    # Load the profile picture and normalize it
    profile_image = load_and_preprocess_image(profile_picture_path)

    # Prepare the batches
    preferred_batch = []
    not_preferred_batch = []
    for preferred_path, not_preferred_path in preferences:
        preferred_batch.append(load_and_preprocess_image(preferred_path))
        not_preferred_batch.append(load_and_preprocess_image(not_preferred_path))

    user_profile_batch = profile_image.expand(batch_size, -1, -1, -1)
    preferred_batch = Tensor.stack(*preferred_batch)
    not_preferred_batch = Tensor.stack(*not_preferred_batch)

    return user_profile_batch, preferred_batch, not_preferred_batch


# def train_model(model, conn, image_files):
#     cursor = conn.cursor()
#     cursor.execute("SELECT user_id, preferred, not_preferred FROM preferences")
#     data = cursor.fetchall()

#     if not data:
#         print("No preferences available for training.")
#         return

#     # Prepare training data
#     training_data = []
#     for user_id, preferred_path, not_preferred_path in data:
#         preferred_image = load_and_preprocess_image(preferred_path)
#         not_preferred_image = load_and_preprocess_image(not_preferred_path)
#         cursor.execute(
#             "SELECT profile_picture FROM users WHERE user_id = ?", (user_id,)
#         )
#         result = cursor.fetchone()
#         if result:
#             profile_image_path = result[0]
#             anchor_image = load_and_preprocess_image(profile_image_path)
#         else:
#             anchor_image = Tensor(np.zeros_like(preferred_image.numpy()))
#         training_data.append((anchor_image, preferred_image, not_preferred_image))

#     # Training loop
#     for epoch in range(1):
#         total_loss = 0
#         for anchor_image, preferred_image, not_preferred_image in training_data:
#             # Forward pass
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

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.numpy()
#         print(f"Training Loss: {total_loss / len(training_data):.4f}")

#     # Save model
#     state_dict = get_state_dict(model)
#     safe_save(state_dict, "model.safetensors")


def train_model(model, conn, batch_size, num_steps=7000, eval_interval=100):
    """
    Train the model using preferences stored in the database, similar style to the tutorial.

    Args:
        model: The Tinygrad model to train.
        conn: SQLite connection object.
        batch_size: The number of examples per batch.
        num_steps: Total number of training steps.
        eval_interval: How often to print accuracy/loss during training.
    """
    users = get_all_users(conn)
    if not users:
        print("No users found in the database.")
        return

    # Make sure model is on training mode
    Tensor.training = True

    # Define the step function that fetches data and does forward/backward
    def step():
        # Randomly pick a user from the list
        # (If your dataset access strategy differs, adapt accordingly)
        import random

        user_id = random.choice(users)

        user_profile_batch, preferred_batch, not_preferred_batch = get_batch(
            conn, user_id, batch_size
        )

        optimizer.zero_grad()
        anchor_mu, anchor_sigma = model(user_profile_batch)
        preferred_mu, preferred_sigma = model(preferred_batch)
        not_preferred_mu, not_preferred_sigma = model(not_preferred_batch)

        current_loss = loss_fn(
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

    # JIT compile the step function for speed
    jit_step = TinyJit(step)

    for i in range(num_steps):
        loss = jit_step()

        if i % eval_interval == 0:
            # Switch off training mode for evaluation
            Tensor.training = False
            acc = (model(X_test).argmax(axis=1) == Y_test).mean()
            print(f"step {i:4d}, loss {loss.item():.4f}, acc {acc:.2f}%")
            # Turn training mode back on
            Tensor.training = True

    # Save the trained model state
    state_dict = get_state_dict(model)
    safe_save(state_dict, "model.safetensors")
    print("Model saved to model.safetensors")


# def train_model(model, conn, batch_size, num_epochs=1):
#     """
#     Train the model using preferences stored in the database.

#     Args:
#         model: The Tinygrad model to train.
#         conn: SQLite connection object.
#         batch_size: The number of examples per batch.
#         num_epochs: The number of training epochs.
#     """
#     users = get_all_users(conn)
#     if not users:
#         print("No users found in the database.")
#         return

#     with Tensor.train():
#         # Training loop
#         for epoch in range(num_epochs):
#             batch_count = 0
#             total_loss = 0

#             for user_id in users:
#                 try:
#                     # Fetch a batch for the current user
#                     user_profile_batch, preferred_batch, not_preferred_batch = (
#                         get_batch(conn, user_id, batch_size)
#                     )

#                     # Forward pass
#                     anchor_mu, anchor_sigma = model(user_profile_batch)
#                     preferred_mu, preferred_sigma = model(preferred_batch)
#                     not_preferred_mu, not_preferred_sigma = model(not_preferred_batch)

#                     # Compute loss
#                     loss = loss_fn(
#                         anchor_mu,
#                         anchor_sigma,
#                         preferred_mu,
#                         preferred_sigma,
#                         not_preferred_mu,
#                         not_preferred_sigma,
#                     )

#                     # Backward pass and optimization
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                     print(loss.shape)
#                     total_loss += loss.realize()
#                     batch_count += 1
#                 except ValueError as e:
#                     # Handle cases where a user has no preferences or profile picture
#                     print(f"Skipping user {user_id}: {e}")
#                     continue

#             if batch_count > 0:
#                 print(
#                     f"Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {total_loss / batch_count:.4f}"
#                 )
#             else:
#                 print(f"Epoch {epoch + 1}/{num_epochs} - No valid training data.")

#     # Save the trained model
#     state_dict = get_state_dict(model)
#     safe_save(state_dict, "model.safetensors")
#     print("Model saved to model.safetensors")


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

    # Train the model
    if users:
        print("Starting training...")
        train_model(
            model, conn, batch_size=16, num_epochs=5
        )  # Adjust batch size and epochs as needed
        print("Training complete.")

    assert False

    # Load the trained model
    load_model(model)

    # Generate uncertain pairs
    for user_id in users:
        try:
            # Use the data loader to fetch the batch
            user_profile_batch, preferred_batch, not_preferred_batch = get_batch(
                conn, user_id, batch_size=10  # Adjust batch size if needed
            )

            # Combine preferred and not_preferred batches for candidates
            candidate_images = Tensor.cat([preferred_batch, not_preferred_batch], dim=0)

            # Generate uncertain pairs
            uncertain_pairs = identify_uncertain_pairs(
                model, user_profile_batch, candidate_images, top_k=10
            )

            # Store uncertain pairs in the database
            store_uncertain_pairs(conn, user_id, uncertain_pairs)
            print(f"Uncertain pairs generated and stored for user {user_id}.")
        except ValueError as e:
            print(f"Skipping uncertain pair generation for user {user_id}: {e}")

    print("All users processed!")
# if __name__ == "__main__":
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
