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
    BayesianPairwiseRankingLoss,
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


def get_candidate_images_for_user(user_id, conn, image_files):
    profile = get_user_profile(conn, user_id)
    if profile:
        user_profile_image = profile[0]
        candidate_images = [img for img in image_files if img != user_profile_image]
        return candidate_images
    else:
        return image_files


def get_batch(conn, user_id, batch_size):
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


def load_model(model):
    model_path = "model.safetensors"
    if os.path.exists(model_path):
        state_dict = safe_load(model_path)
        load_state_dict(model, state_dict)
    else:
        print("No saved model found. Please train the model first.")


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

    # Define the step function that fetches data and does forward/backward
    def step(user_profile_batch, preferred_batch, not_preferred_batch):
        Tensor.training = True
        # Randomly pick a user from the list
        # (If your dataset access strategy differs, adapt accordingly)

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
    user_id = random.choice(users)

    user_profile_batch, preferred_batch, not_preferred_batch = get_batch(
        conn, user_id, batch_size
    )
    jit_step = TinyJit(step)

    for i in range(num_steps):
        loss = jit_step(user_profile_batch, preferred_batch, not_preferred_batch)

        if i % eval_interval == 0:
            # Switch off training mode for evaluation
            Tensor.training = False
            # acc = (model(X_test).argmax(axis=1) == Y_test).mean()
            anchor_mu, anchor_sigma = model(user_profile_batch)
            preferred_mu, preferred_sigma = model(preferred_batch)
            not_preferred_mu, not_preferred_sigma = model(not_preferred_batch)

            acc = loss_fn(
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
    conn = init_db()
    users = get_all_users(conn)

    # Train the model
    if users:
        print("Starting training...")
        train_model(
            model,
            conn,
            batch_size=16,
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
