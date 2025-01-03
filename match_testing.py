import random
from collections import defaultdict
import numpy as np
from collections import Counter

N = 2_000  # population size
users = list(range(N))

score = [[0.0] * N for _ in range(N)]
for u in range(N):
    # Make a list of all other users
    others = [x for x in range(N) if x != u]
    # Shuffle them (random ranking)
    random.shuffle(others)

    # We'll assign a linearly decreasing score from 1.0 down to 0.0 (exclusive).
    # Step size = 1.0/(N-1). So top user gets ~1.0, next gets ~1 - step, etc.
    step = 1.0 / (N - 1)

    # We could do descending:
    #   1.0, 1.0-step, 1.0-2*step, ... => top preference ~1.0, worst ~0.0
    # This is effectively a random total order for user u.

    for rank, v in enumerate(others):
        # rank=0 is top preference, rank=N-2 is least preference
        preference_value = 1.0 - rank * step
        score[u][v] = preference_value

# Step 2: Possibly do local sampling
# (Or skip and do a single global pass for simplicity)
SAMPLES = 100
sample_size = 200
all_edges = []

for _ in range(SAMPLES):
    subset = random.sample(users, sample_size)
    edges = []
    # compute mutual scores within the subset
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            u = subset[i]
            v = subset[j]
            mutual = min(score[u][v], score[v][u])
            edges.append((mutual, u, v))
    # sort edges locally
    edges.sort(reverse=True, key=lambda e: e[0])
    # pick top K per user
    K = 4
    picks_for_user = {u: 0 for u in subset}
    best_edges = []
    for val, u, v in edges:
        if picks_for_user[u] < K or picks_for_user[v] < K:
            best_edges.append((val, u, v))
            picks_for_user[u] += 1
            picks_for_user[v] += 1
    # add these best edges to global list
    all_edges.extend(best_edges)

# Step 3: Consolidate final
all_edges.sort(reverse=True, key=lambda e: e[0])
K_final = 1
used = [0] * N
final_matches = []
for val, u, v in all_edges:
    if used[u] < K_final or used[v] < K_final:
        final_matches.append((val, u, v))
        used[u] += 1
        used[v] += 1

# Step 4: Evaluate approximate stability
blocking_count = 0
# build a map: matched[u] -> who they ended up with

matched_with = defaultdict(list)
for val, u, v in final_matches:
    matched_with[u].append(v)
    matched_with[v].append(u)

# check for blocking pairs
for u in range(N):
    for v in range(u + 1, N):
        # if they're not matched together
        if v not in matched_with[u]:
            # check if they'd prefer each other over their current matches
            mutual_uv = min(score[u][v], score[v][u])
            # see if u has a match x with mutual < mutual_uv
            # see if v has a match y with mutual < mutual_uv
            # if so, it's blocking
            better_for_u = any(
                min(score[u][x], score[x][u]) < mutual_uv for x in matched_with[u]
            )
            better_for_v = any(
                min(score[v][y], score[y][v]) < mutual_uv for y in matched_with[v]
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
