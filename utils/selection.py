import itertools
import random
from model.model import sample_embeddings


def compute_score_diff(
    anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
):
    """
    Compute score differences between img1 and img2 relative to anchor.
    """
    # Sample embeddings for anchor, img1, and img2
    anchor_samples = sample_embeddings(anchor_mu, anchor_sigma, num_samples=10)
    img1_samples = sample_embeddings(img1_mu, img1_sigma, num_samples=10)
    img2_samples = sample_embeddings(img2_mu, img2_sigma, num_samples=10)

    # Compute scores
    img1_scores = (anchor_samples * img1_samples).sum(axis=1)
    img2_scores = (anchor_samples * img2_samples).sum(axis=1)
    score_diffs = img1_scores - img2_scores
    return score_diffs


def compute_uncertainty(score_diffs):
    """
    Compute variance of score differences as a measure of uncertainty.
    """
    uncertainty = score_diffs.var().numpy()
    return uncertainty


def identify_uncertain_pairs(model, user_profile_batch, candidate_images, top_k=10):
    """
    Identify the top uncertain pairs of candidate images for active learning.
    """
    candidate_pairs = list(itertools.combinations(candidate_images, 2))
    if len(candidate_pairs) > 500:
        candidate_pairs = random.sample(candidate_pairs, 500)

    uncertainties = []

    # Perform forward pass for the anchor image batch
    anchor_mu, anchor_sigma = model(user_profile_batch)

    for img1, img2 in candidate_pairs:
        # Forward pass for candidate images
        img1_mu, img1_sigma = model(img1.unsqueeze(0))  # Add batch dimension
        img2_mu, img2_sigma = model(img2.unsqueeze(0))  # Add batch dimension

        # Compute uncertainty for the pair
        score_diff = compute_score_diff(
            anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
        )
        uncertainty = compute_uncertainty(score_diff)
        uncertainties.append((uncertainty, img1, img2))

    # Sort by uncertainty and select top_k pairs
    uncertainties.sort(reverse=True, key=lambda x: x[0])
    top_uncertain_pairs = uncertainties[:top_k]
    return top_uncertain_pairs
