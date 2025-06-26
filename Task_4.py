import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv("customer_loan_data.csv")

# Sort data by FICO score
df = df.sort_values("fico_score").reset_index(drop=True)

# Extract relevant columns
fico_scores = df["fico_score"].values
defaults = df["default"].values

# -----------------------------
# Method 1: MSE-Based Bucketing
# -----------------------------

def mse_buckets(fico_scores, k):
    """
    Simple method to bucket FICO scores minimizing intra-bucket MSE.
    """
    sorted_scores = np.sort(fico_scores)
    buckets = np.array_split(sorted_scores, k)
    boundaries = [bucket[0] for bucket in buckets]
    boundaries.append(sorted_scores[-1] + 1)  # upper bound
    return boundaries

# -----------------------------
# Method 2: Log-Likelihood Bucketing
# -----------------------------

def log_likelihood(k, scores, labels):
    """
    Dynamic programming based log-likelihood optimization
    for finding bucket boundaries.
    """
    n = len(scores)
    dp = np.full((k + 1, n + 1), -np.inf)
    split = np.zeros((k + 1, n + 1), dtype=int)
    dp[0][0] = 0

    # Precompute cumulative sums
    cum_k = np.cumsum(labels)
    cum_n = np.arange(1, n + 1)

    def segment_ll(start, end):
        ki = np.sum(labels[start:end])
        ni = end - start
        if ki == 0 or ki == ni:
            return 0  # log(0) safe-guard
        pi = ki / ni
        return ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

    for b in range(1, k + 1):
        for i in range(1, n + 1):
            for j in range(b - 1, i):
                ll = dp[b - 1][j] + segment_ll(j, i)
                if ll > dp[b][i]:
                    dp[b][i] = ll
                    split[b][i] = j

    # Recover bucket boundaries
    boundaries = []
    idx = n
    for b in range(k, 0, -1):
        idx = split[b][idx]
        boundaries.append(scores[idx])
    boundaries = sorted(boundaries)
    boundaries.append(scores[-1] + 1)  # max boundary
    return boundaries

# -----------------------------
# Bucket Evaluation and Mapping
# -----------------------------

def assign_buckets(scores, boundaries):
    """
    Map FICO scores to bucket categories using boundaries.
    Lower bucket ID means better credit score.
    """
    return np.digitize(scores, boundaries) - 1

def evaluate_mse(scores, buckets):
    mse = 0
    for b in np.unique(buckets):
        group = scores[buckets == b]
        mse += np.sum((group - group.mean())**2)
    return mse / len(scores)

# -----------------------------
# Run Both Methods and Compare
# -----------------------------

k = 5  # Number of buckets

# Method 1: MSE Bucketing
mse_bounds = mse_buckets(fico_scores, k)
mse_buckets_labels = assign_buckets(fico_scores, mse_bounds)
mse_loss = evaluate_mse(fico_scores, mse_buckets_labels)

# Method 2: Log-Likelihood Bucketing
ll_bounds = log_likelihood(k, fico_scores, defaults)
ll_buckets_labels = assign_buckets(fico_scores, ll_bounds)

# Print Results
print("=== MSE-Based Buckets ===")
print("Boundaries:", mse_bounds)
print("MSE:", round(mse_loss, 2))

print("\n=== Log-Likelihood-Based Buckets ===")
print("Boundaries:", ll_bounds)

# Optional: Add to DataFrame for export
df["mse_bucket"] = assign_buckets(df["fico_score"], mse_bounds)
df["ll_bucket"] = assign_buckets(df["fico_score"], ll_bounds)

# Save result
df.to_csv("bucketed_fico_output.csv", index=False)
print("\nResult saved to bucketed_fico_output.csv")
