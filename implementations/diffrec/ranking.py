"""Full-catalog ranking with the authors' macro averages and history masks."""

import numpy as np

from presets import METRICS

CUTOFFS = (10, 20)
DISCOUNTS = 1 / np.log2(np.arange(max(CUTOFFS)) + 2)
IDEAL = np.r_[0, np.cumsum(DISCOUNTS)]


def metric_sums(hits, target_counts):
    recalls, ndcgs = [], []
    for k in CUTOFFS:
        top_hits = hits[:, :k]
        recalls.append(np.sum(top_hits.sum(axis=1) / np.maximum(target_counts, 1)))
        ndcgs.append(np.sum((top_hits * DISCOUNTS[:top_hits.shape[1]]).sum(axis=1)
                            / IDEAL[np.clip(target_counts, 1, k)]))
    return np.array(recalls + ndcgs)


def evaluate(predict, inputs, targets, mask, batch_size):
    n_users, n_items = targets.shape
    counts = targets.getnnz(axis=1)
    totals = np.zeros(len(METRICS))
    for start in range(0, n_users, batch_size):
        stop = min(start + batch_size, n_users)
        scores = predict(inputs[start:stop])
        scores[mask[start:stop].nonzero()] = -float("inf")
        values, indices = scores.topk(min(max(CUTOFFS), n_items), dim=1)
        users = np.broadcast_to(np.arange(start, stop)[:, None], indices.shape)
        ranked = indices.cpu().numpy()
        hits = np.asarray(targets[users.ravel(), ranked.ravel()]).reshape(ranked.shape) > 0
        hits &= values.isfinite().cpu().numpy()
        totals += metric_sums(hits, counts[start:stop])
    # Upstream rounds to four decimals before selecting the best epoch.
    return dict(zip(METRICS, (round(float(value) / n_users, 4) for value in totals)))
