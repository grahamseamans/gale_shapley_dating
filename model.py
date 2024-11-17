import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# ---- Small ResNet Variant ---- #
class ProbabilisticResNet(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ProbabilisticResNet, self).__init__()
        # Use a lightweight ResNet
        base_resnet = resnet18(pretrained=True)
        base_resnet.fc = nn.Linear(base_resnet.fc.in_features, embedding_dim * 2)
        self.base_resnet = base_resnet

    def forward(self, x):
        out = self.base_resnet(x)  # Output will be 2 * embedding_dim
        mu, log_sigma = torch.chunk(out, 2, dim=1)  # Split into mean and log variance
        sigma = torch.exp(log_sigma)  # Convert log variance to standard deviation
        return mu, sigma


# ---- Sampling from Gaussian Embeddings ---- #
def sample_embeddings(mu, sigma, num_samples=1):
    return mu + sigma * torch.randn_like(sigma)  # Sampling using reparameterization


# ---- Pairwise Bayesian Ranking Loss ---- #
class BayesianPairwiseRankingLoss(nn.Module):
    def forward(self, anchor_mu, anchor_sigma, pos_mu, pos_sigma, neg_mu, neg_sigma):
        # Sample embeddings
        anchor_samples = sample_embeddings(anchor_mu, anchor_sigma)
        pos_samples = sample_embeddings(pos_mu, pos_sigma)
        neg_samples = sample_embeddings(neg_mu, neg_sigma)

        # Compute scores
        pos_scores = torch.sum(anchor_samples * pos_samples, dim=1)
        neg_scores = torch.sum(anchor_samples * neg_samples, dim=1)

        # Score differences
        score_diffs = pos_scores - neg_scores

        # Bayesian loss
        loss = -torch.mean(F.logsigmoid(score_diffs))
        return loss


# ---- Toy Dataset ---- #
class PairwiseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels  # Use labels to simulate user preferences

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        # Create a pair with a "positive" and "negative" example
        pos_index = (self.labels == label).nonzero(as_tuple=True)[0][0].item()
        neg_index = (self.labels != label).nonzero(as_tuple=True)[0][0].item()
        return img, self.data[pos_index], self.data[neg_index]

    def __len__(self):
        return len(self.data)


# ---- Training Loop ---- #
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device),
        )

        # Forward pass
        anchor_mu, anchor_sigma = model(anchor)
        pos_mu, pos_sigma = model(positive)
        neg_mu, neg_sigma = model(negative)

        # Compute loss
        loss = loss_fn(anchor_mu, anchor_sigma, pos_mu, pos_sigma, neg_mu, neg_sigma)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# ---- Main ---- #
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
    dataset = datasets.FakeData(
        transform=transform
    )  # Replace with MNIST/CIFAR-10 if needed
    pairwise_dataset = PairwiseDataset(dataset.data, dataset.targets)
    dataloader = DataLoader(pairwise_dataset, batch_size=32, shuffle=True)

    # Model, loss, optimizer
    model = ProbabilisticResNet(embedding_dim=64).to(device)
    loss_fn = BayesianPairwiseRankingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        avg_loss = train(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
