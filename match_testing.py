import numpy as np
from collections import defaultdict, Counter

# -----------------------------
# 1) Setup and Preference Creation
# -----------------------------
N = 3000
users = np.arange(N)

# We'll use a NumPy 2D array for scores, of shape (N,N).
score = np.zeros((N, N), dtype=float)

for u in range(N):
    # Create a list/array of all other users
    others = np.delete(users, u)  # remove 'u' from the list of users
    # Shuffle them in place (so user 'u' has a random ranking over these others)
    np.random.shuffle(others)

    # We'll assign a linearly decreasing score from 1.0 down to 0.0 (exclusive).
    # e.g., if (N-1) = 1999, we get 1999 values from 1.0 down to ~0.0 in small steps
    # endpoint=False means we'll never hit 0 exactly, just something close.
    pref_values = np.linspace(1.0, 0.0, num=N, endpoint=False, dtype=float)

    # Now assign these preference values in the order of 'others' array:
    # rank=0 => top preference near 1.0
    # rank=N-2 => near 0.0
    for rank, v in enumerate(others):
        score[u, v] = pref_values[rank]

# -----------------------------
# 2) Sampling and Local Top-K Picks
# -----------------------------
SAMPLES = 30
sample_size = 1000
all_edges = []

for _ in range(SAMPLES):
    # Randomly choose 'sample_size' distinct users
    subset = np.random.choice(users, sample_size, replace=False)

    # We'll store edges in a Python list for convenience
    edges = []
    # Compute mutual scores among this subset
    # (We've just a simple double loop for clarity)
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            u = subset[i]
            v = subset[j]
            mutual = min(score[u, v], score[v, u])
            edges.append((mutual, int(u), int(v)))

    # Sort these edges locally by descending mutual score
    edges.sort(reverse=True, key=lambda e: e[0])

    # For each user in 'subset', pick up to K edges
    K = 2
    picks_for_user = {int(u): 0 for u in subset}

    best_edges = []
    for val, u, v in edges:
        # If either user can still accept more edges in this local pass, pick it
        if picks_for_user[u] < K or picks_for_user[v] < K:
            best_edges.append((val, u, v))
            picks_for_user[u] += 1
            picks_for_user[v] += 1

    all_edges.extend(best_edges)

# -----------------------------
# 3) Consolidate Final Matches
# -----------------------------
# Sort all edges from all samples by descending mutual score
all_edges.sort(reverse=True, key=lambda e: e[0])

min_matches = 1  # absolute max final matches each user can have

used = np.zeros(N, dtype=int)
final_matches = []

for val, u, v in all_edges:
    if used[u] < min_matches and used[v] < min_matches:
        final_matches.append((val, u, v))
        used[u] += 1
        used[v] += 1

print("Final matching has {} edges".format(len(final_matches)))

# -----------------------------
# 4) Evaluate Approximate Stability
# -----------------------------
matched_with = defaultdict(list)
for val, u, v in final_matches:
    matched_with[u].append(v)
    matched_with[v].append(u)

blocking_count = 0

for u in range(N):
    # We'll only check v > u to avoid double-counting
    for v in range(u + 1, N):
        if v not in matched_with[u]:
            # Check if they prefer each other over their current matches
            mutual_uv = min(score[u, v], score[v, u])
            # If there's a match x for u with mutual < mutual_uv, u would prefer v
            # If there's a match y for v with mutual < mutual_uv, v would prefer u
            better_for_u = any(
                min(score[u, x], score[x, u]) < mutual_uv for x in matched_with[u]
            )
            better_for_v = any(
                min(score[v, y], score[y, v]) < mutual_uv for y in matched_with[v]
            )
            if better_for_u and better_for_v:
                blocking_count += 1

print("Final matching has {} edges".format(len(final_matches)))
print("Blocking pairs found:", blocking_count)

# Number of matches for each user
num_matches = np.array([len(matched_with[u]) for u in range(N)], dtype=int)

min_matches = num_matches.min()
max_matches = num_matches.max()
mean_matches = num_matches.mean()

print("\nMatch Count Statistics:")
print("  Minimum:", min_matches)
print("  Maximum:", max_matches)
print("  Mean:   ", mean_matches)

# Create a simple histogram of match counts
counts = Counter(num_matches)
print("\nHistogram of match counts (count how many users have x matches):")
for match_count in sorted(counts.keys()):
    print(f"  {match_count} match(es): {counts[match_count]} user(s)")
