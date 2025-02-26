from config.config import positional_penalty_alpha


def compute_contextual_weighted_similarity(v1, v2, w1, w2, alpha=positional_penalty_alpha):
    """
    Compute contextual weighted similarity for two feature vectors.
    Stricter penalties are applied for dissimilar features.
    """
    if len(v1) != len(v2):
        print(f"[ERROR] Vector length mismatch: {len(v1)} vs {len(v2)}")
        return 0.0  # Avoid invalid similarity computation

    print(f"[DEBUG] Comparing vectors: {v1} vs {v2}")

    n, m = len(v1), len(v2)
    length_penalty = min(n, m) / max(n, m)
    normalization_factor = max(sum(w1), sum(w2))

    weighted_sum = 0
    for i in range(n):
        for j in range(m):
            # Apply a stricter penalty for larger differences
            sim = 1 - (abs(v1[i] - v2[j]) ** 2)  # Squaring amplifies dissimilarity
            sim = max(sim, 0)  # Ensure similarity does not go negative
            positional_penalty = alpha if i != j else 1
            weight = (w1[i] + w2[j]) / 2
            weighted_sum += sim * positional_penalty * weight

    similarity = (weighted_sum / normalization_factor) * length_penalty
    similarity = min(max(similarity, 0.0), 1.0)
    return similarity