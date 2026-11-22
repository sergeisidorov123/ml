def intersection_over_union(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def summarize(results, threshold=0.5):
    tp = fp = fn = 0
    best_ious = []
    for result in results:
        truth = result["truth"]
        predicted = result["boxes"]
        best_ious.extend(max((intersection_over_union(a, b) for b in predicted), default=0.0) for a in truth)
        pairs = sorted(
            ((intersection_over_union(a, b), i, j)
             for i, a in enumerate(truth) for j, b in enumerate(predicted)),
            reverse=True,
        )
        matched_truth, matched_prediction = set(), set()
        for overlap, i, j in pairs:
            if overlap < threshold:
                break
            if i not in matched_truth and j not in matched_prediction:
                matched_truth.add(i)
                matched_prediction.add(j)
        tp += len(matched_truth)
        fp += len(predicted) - len(matched_prediction)
        fn += len(truth) - len(matched_truth)
    return {
        "mean_best_iou": sum(best_ious) / max(1, len(best_ious)),
        "precision_at_iou50": tp / max(1, tp + fp),
        "recall_at_iou50": tp / max(1, tp + fn),
        "f1_at_iou50": 2 * tp / max(1, 2 * tp + fp + fn),
        "tp": tp, "fp": fp, "fn": fn, "ground_truth": len(best_ious), "photos": len(results),
    }
