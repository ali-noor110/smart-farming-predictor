import heapq
import numpy as np
CROP_CONDITIONS = {
    "rice":        [80, 40, 40, 23, 82, 6.5, 200],
    "maize":       [80, 45, 40, 22, 65, 6.0, 75],
    "chickpea":    [45, 70, 80, 18, 17, 7.3, 75],
    "kidneybeans": [20, 65, 20, 20, 22, 5.7, 105],
    "pigeonpeas":  [20, 65, 20, 28, 49, 5.8, 145],
    "mothbeans":   [21, 48, 20, 28, 53, 6.9, 50],
    "mungbean":    [20, 47, 20, 28, 85, 6.7, 45],
    "blackgram":   [40, 67, 19, 30, 66, 7.1, 67],
    "lentil":      [18, 65, 19, 24, 65, 6.9, 45],
    "pomegranate": [18, 18, 40, 22, 90, 6.0, 105],
    "banana":      [100, 82, 50, 27, 80, 6.0, 105],
    "mango":       [20, 27, 30, 31, 50, 6.3, 95],
    "grapes":      [23, 133, 200, 24, 82, 6.5, 65],
    "watermelon":  [99, 17, 50, 25, 85, 6.5, 50],
    "muskmelon":   [100, 17, 50, 28, 92, 6.5, 25],
    "apple":       [20, 134, 199, 22, 92, 6.0, 112],
    "orange":      [20, 16, 10, 23, 92, 7.0, 110],
    "papaya":      [49, 59, 50, 34, 92, 6.7, 145],
    "coconut":     [23, 17, 30, 27, 94, 6.5, 150],
    "cotton":      [118, 46, 43, 24, 80, 6.7, 80],
    "jute":        [78, 46, 40, 25, 80, 6.5, 175],
    "coffee":      [101, 28, 29, 25, 58, 6.5, 155],
}
FEATURE_RANGES = {
    "N":           (0, 140),
    "P":           (5, 145),
    "K":           (5, 205),
    "temperature": (5, 45),
    "humidity":    (10, 100),
    "ph":          (3.5, 9.5),
    "rainfall":    (20, 300),
}

def normalize(values: list) -> np.ndarray:
    """Min-max normalize a 7-feature vector."""
    ranges = list(FEATURE_RANGES.values())
    norm = []
    for v, (lo, hi) in zip(values, ranges):
        norm.append((v - lo) / (hi - lo + 1e-9))
    return np.array(norm)


def astar_recommend(input_features: dict, top_k: int = 5) -> list:
    required = set(FEATURE_RANGES)
    missing = required - set(input_features)
    if missing:
        raise ValueError(f"Missing input features: {', '.join(sorted(missing))}")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    query = normalize([
        input_features["N"],
        input_features["P"],
        input_features["K"],
        input_features["temperature"],
        input_features["humidity"],
        input_features["ph"],
        input_features["rainfall"],
    ])
    open_list = []
    visited = set()
    results = []
    nodes_expanded = 0

    for crop, ideal in CROP_CONDITIONS.items():
        ideal_norm = normalize(ideal)
        h = float(np.linalg.norm(query - ideal_norm))
        g = 0.0
        f = g + h
        heapq.heappush(open_list, (f, g, crop))

    # Expand nodes until we have top_k
    while open_list and len(results) < top_k:
        f, g, crop = heapq.heappop(open_list)
        if crop in visited:
            continue
        visited.add(crop)
        nodes_expanded += 1
        results.append({
            "crop":      crop,
            "score":     round(1 / (1 + f), 4),
            "heuristic": round(f, 4),
            "rank":      len(results) + 1,
            "nodes_expanded": nodes_expanded,
        })

    return results
