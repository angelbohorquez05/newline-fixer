"""Evaluation metrics over gap labels.

SPACE is ~95% of the gaps, so accuracy says little: a system that never changes
anything scores ~95%. Per-label precision/recall/F1 and macro-F1 do not hide that.
"""

from collections import Counter

from newline_fixer.labels import Label


def evaluate(gold: list[list[int]], pred: list[list[int]]) -> dict:
    """Per-label precision/recall/F1, macro-F1 over the four labels, and document exact match."""
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = 0
    for gold_doc, pred_doc in zip(gold, pred, strict=True):
        exact += list(gold_doc) == list(pred_doc)
        for g, p in zip(gold_doc, pred_doc, strict=True):
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

    labels = {}
    for label in Label:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        labels[label.name] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "labels": labels,
        "macro_f1": sum(m["f1"] for m in labels.values()) / len(labels),
        "exact_match": exact / len(gold),
    }
