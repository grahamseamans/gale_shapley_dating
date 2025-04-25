# Gale-Shapley Approximate Matching with User Embeddings

## Overview

This project implements a scalable, user-centric matching system inspired by the Gale-Shapley stable marriage problem, but adapted for large populations and realistic data collection. The core workflow is:

- **Users provide pairwise comparisons** (which of two other users they prefer).
- **A neural network embedding** is learned from these comparisons, mapping each user to a point in a latent space. The relationship between two users' embeddings (e.g., via dot product or distance) determines how much one user is predicted to prefer the other. The embedding itself does not represent a user's preferences in isolation, but only in relation to others.
- **Embeddings are generated for all users** in a one-off batch operation.
- **Approximate stable matching** is performed by repeatedly sampling sub-batches of users, computing local pairwise scores, and coalescing the best matches—without ever computing the full O(n²) score matrix.

## Why Pairwise Comparisons?

- **User-centric:** The system is designed to learn each individual user's preferences, not a global ranking. Every pairwise comparison is about what *one user* prefers between two others.
- **Practicality:** Full ranked lists are unrealistic for large populations. Pairwise comparisons are simple, intuitive, and scalable for users.
- **Direct signal:** Each comparison gives a clear, interpretable datapoint about a user's preferences, which is used to build their personalized embedding.

## How Pairwise Comparisons Are Used

- Users interact with a web interface, selecting their preferred user from a pair.
- Each comparison is stored as a (user, preferred, not_preferred) tuple.
- The system can present either random pairs or pairs where the model is most uncertain (active learning).

## Embedding: Learning and Usage

- **Learning the embedding network:**  
  The neural network is trained continually as new pairwise data arrives. This is a standard continual learning problem, and is the only computationally intensive part of the system (scales with n, the number of users).
- **Making the embeddings:**  
  Once the network is trained, all users are passed through it in a one-off batch to generate their embeddings. This is done before any matching.
- **Key point:**  
  After embeddings are made, *no neural network inference or training is needed during matching*—all matching is done with simple, fast vector arithmetic.

- **Probabilistic embeddings and variance:**  
  Each user's embedding is not just a point, but a distribution (mean and variance). The variance reflects the model's uncertainty about that user's preferences.  
  Ideally, this variance can be used to identify which users (or types of users) the model is least certain about, so the system can prioritize collecting more pairwise comparisons from those users. This targeted data collection helps the model learn more efficiently and improves overall matching quality.

## Matching: Scalable Approximate Gale-Shapley

- **Batch sampling:**  
  Instead of computing all O(n²) user-user scores, the system samples multiple random sub-batches of users (each much smaller than n).
- **Local matching:**  
  Within each sub-batch, all pairwise scores are computed using the precomputed embeddings (e.g., via dot product or distance), and the top-K matches per user are selected.
- **Coalescing:**  
  The best matches from all sub-batches are merged and sorted by mutual score. A greedy algorithm constructs the final matching, ensuring each user does not exceed the allowed number of matches.
- **No O(n²) bottleneck:**  
  At no point is the full n × n score matrix computed or stored. The process is efficient and parallelizable.

## Evaluation

- **Blocking pairs:**  
  After matching, the system counts "blocking pairs"—pairs of users not matched to each other who would both prefer to be matched together over their current matches. Fewer blocking pairs means a more stable matching.
- **Match statistics:**  
  The system reports statistics such as the minimum, maximum, and mean number of matches per user, and provides a histogram of match counts.

## Architectural Advantages

- **Separation of concerns:**  
  - The hard computation (neural network training) is done up front and scales with n.
  - The matching stage, which would naively be O(n²), is reduced to fast, simple arithmetic on precomputed embeddings, thanks to the batch sampling and coalescing approach.
- **User-only embeddings:**  
  Only users are embedded; there is no separate set of items.
- **Flexibility:**  
  The batch sampling and matching strategy can be adapted or extended as needed.

## Notes on Locational/Batch Sampling

- The current implementation demonstrates batch sampling: random sub-batches of users are selected, matched locally, and results are merged.
- This is a scalable, practical way to approximate stable matching for large populations.
- The sampling strategy can be modified (e.g., to sample based on attributes or embedding proximity) as needed for different applications.

---

This README reflects the actual implementation and design philosophy of the project. For further details, see the code in `match_testing.py` and related modules.
