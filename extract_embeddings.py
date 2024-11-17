import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np

# Path to the directory containing images
IMAGE_DIR = "img_align_celeba"

# Path to the JSON file to store embeddings
EMBEDDINGS_FILE = "embeddings.json"

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Remove the final classification layer
modules = list(resnet.children())[:-1]
resnet = torch.nn.Sequential(*modules)

# Transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Function to extract embedding
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(image).squeeze().numpy()
    return embedding


# Extract embeddings for all images
embeddings = {}
for img_file in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_file)
    embeddings[img_file] = get_embedding(img_path).tolist()

# Save embeddings to JSON file
with open(EMBEDDINGS_FILE, "w") as f:
    json.dump(embeddings, f)
