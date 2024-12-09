# import os
# from PIL import Image
# import numpy as np


# def get_image_files(directory):
#     image_files = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_files.append(os.path.join(root, file))
#     return image_files


# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((224, 224))
#     image = np.array(image).astype(np.float32) / 255.0
#     # Rearrange dimensions to match Tinygrad's expected input (channels, height, width)
#     image = np.transpose(image, (2, 0, 1))
#     return image
