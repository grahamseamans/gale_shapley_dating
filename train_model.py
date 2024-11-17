import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

# Load embeddings and preferences
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

with open("preferences.json", "r") as f:
    preferences = json.load(f)

# Convert embeddings to tensor
embedding_tensor = {k: torch.tensor(v) for k, v in embeddings.items()}


# Define the model
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Initialize model, loss function, and optimizer
input_dim = len(next(iter(embedding_tensor.values())))
output_dim = 128  # Dimension of the embedding space
model = EmbeddingNet(input_dim, output_dim)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare triplets
triplets = []
for anchor, positive in preferences.items():
    for negative in embedding_tensor.keys():
        if negative != anchor and negative != positive:
            triplets.append((anchor, positive, negative))

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for anchor, positive, negative in triplets:
        anchor_emb = embedding_tensor[anchor]
        positive
