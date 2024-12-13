# from tinygrad import Tensor, nn
# from .resnet18_tinygrad import load_resnet


# class ProbabilisticResNet:
#     def __init__(self, embedding_dim=64):
#         self.base_resnet = load_resnet()
#         self.base_resnet.fc = nn.Linear(512, embedding_dim * 2)

#     def __call__(self, x: Tensor):
#         out = self.base_resnet(x)
#         mu, log_sigma = out.chunk(2, dim=1)
#         sigma = log_sigma.exp()
#         return mu, sigma


# def sample_embeddings(mu: Tensor, sigma: Tensor, num_samples=1):
#     return mu + sigma * Tensor.randn_like(sigma)


# class BayesianPairwiseRankingLoss:
#     def __call__(self, anchor_mu, anchor_sigma, pos_mu, pos_sigma, neg_mu, neg_sigma):
#         anchor_samples = sample_embeddings(anchor_mu, anchor_sigma)
#         pos_samples = sample_embeddings(pos_mu, pos_sigma)
#         neg_samples = sample_embeddings(neg_mu, neg_sigma)

#         pos_scores = (anchor_samples * pos_samples).sum(axis=1)
#         neg_scores = (anchor_samples * neg_samples).sum(axis=1)

#         score_diffs = pos_scores - neg_scores
#         loss = -Tensor.logsigmoid(score_diffs).mean()
#         return loss


from tinygrad import Tensor, nn
from model.resnet18_tinygrad import load_resnet

# import model


# ---- Tinygrad Probabilistic ResNet Model ---- #
class ProbabilisticResNet:
    def __init__(self, embedding_dim=64):
        # Use ResNet as the backbone
        # self.base_resnet = ResNet(Block, [2, 2, 2, 2], num_classes=embedding_dim * 2)
        self.base_resnet = load_resnet()
        self.base_resnet.fc = nn.Linear(512, embedding_dim * 2)

    def __call__(self, x: Tensor):
        # Forward pass through ResNet
        out = self.base_resnet(x)  # Output shape: (batch_size, 2 * embedding_dim)
        # mu, log_sigma = out.chunk(2, axis=1)  # Split into mean and log variance
        mu, log_sigma = out.chunk(2, dim=1)  # Split into mean and log variance
        sigma = log_sigma.exp()  # Convert log variance to standard deviation
        return mu, sigma


def sample_embeddings(mu: Tensor, sigma: Tensor, num_samples=1):
    random_samples = Tensor.randn(*sigma.shape)
    return mu + sigma * random_samples


# CHAT GPT LOSS FUNCTION!!! CHECK THIS OUT
def baysean_pairwise_ranking_loss(
    anchor_mu, anchor_sigma, pos_mu, pos_sigma, neg_mu, neg_sigma
):
    # Sample embeddings
    anchor_samples = sample_embeddings(anchor_mu, anchor_sigma)
    pos_samples = sample_embeddings(pos_mu, pos_sigma)
    neg_samples = sample_embeddings(neg_mu, neg_sigma)

    # Compute scores
    pos_scores = (anchor_samples * pos_samples).sum(axis=1)
    neg_scores = (anchor_samples * neg_samples).sum(axis=1)

    # Score differences
    score_diffs = pos_scores - neg_scores

    def log_sigmoid(x: Tensor):
        # log_sigmoid(x) = -softplus(-x) = x - softplus(x)
        return x - x.softplus()

    # Bayesian loss
    loss = -log_sigmoid(score_diffs).mean()
    return loss


# # ---- Bayesian Pairwise Ranking Loss ---- #
# class BayesianPairwiseRankingLoss:
#     def __call__(self, anchor_mu, anchor_sigma, pos_mu, pos_sigma, neg_mu, neg_sigma):
#         # Sample embeddings
#         anchor_samples = sample_embeddings(anchor_mu, anchor_sigma)
#         pos_samples = sample_embeddings(pos_mu, pos_sigma)
#         neg_samples = sample_embeddings(neg_mu, neg_sigma)

#         # Compute scores
#         pos_scores = (anchor_samples * pos_samples).sum(axis=1)
#         neg_scores = (anchor_samples * neg_samples).sum(axis=1)

#         # Score differences
#         score_diffs = pos_scores - neg_scores

#         # Bayesian loss
#         loss = -Tensor.logsigmoid(score_diffs).mean()
#         return loss


# ---- Example Usage ---- #
if __name__ == "__main__":
    # Load the probabilistic ResNet model
    model = ProbabilisticResNet(embedding_dim=64)

    # Create dummy input
    dummy_input = Tensor.uniform(1, 3, 224, 224)

    # Perform forward pass
    mu, sigma = model(dummy_input)
    print("Mean:", mu)
    print("Sigma:", sigma)
