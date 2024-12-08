import itertools
import random
import torch
from model.model import sample_embeddings


def compute_score_diff(
    anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
):
    anchor_samples = sample_embeddings(anchor_mu, anchor_sigma, num_samples=10)
    img1_samples = sample_embeddings(img1_mu, img1_sigma, num_samples=10)
    img2_samples = sample_embeddings(img2_mu, img2_sigma, num_samples=10)

    img1_scores = torch.sum(anchor_samples * img1_samples, dim=1)
    img2_scores = torch.sum(anchor_samples * img2_samples, dim=1)
    score_diffs = img1_scores - img2_scores
    return score_diffs


def compute_uncertainty(score_diffs):
    uncertainty = torch.var(score_diffs).item()
    return uncertainty


def identify_uncertain_pairs(model, user_id, user_data, candidate_images, top_k=10):
    candidate_pairs = list(itertools.combinations(candidate_images, 2))
    if len(candidate_pairs) > 500:
        candidate_pairs = random.sample(candidate_pairs, 500)

    uncertainties = []
    anchor_image = (
        user_data["profile_image"].to("cpu").unsqueeze(0)
    )  # or device if needed
    with torch.no_grad():
        anchor_mu, anchor_sigma = model(anchor_image)

    for img1_path, img2_path in candidate_pairs:
        img1 = user_data["loader"](img1_path).to("cpu").unsqueeze(0)
        img2 = user_data["loader"](img2_path).to("cpu").unsqueeze(0)

        with torch.no_grad():
            img1_mu, img1_sigma = model(img1)
            img2_mu, img2_sigma = model(img2)

        score_diff = compute_score_diff(
            anchor_mu, anchor_sigma, img1_mu, img1_sigma, img2_mu, img2_sigma
        )
        uncertainty = compute_uncertainty(score_diff)
        uncertainties.append((uncertainty, img1_path, img2_path))

    uncertainties.sort(reverse=True)
    top_uncertain_pairs = uncertainties[:top_k]
    return top_uncertain_pairs
