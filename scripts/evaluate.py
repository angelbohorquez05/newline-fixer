"""Compare systems on a dataset split: per-label precision/recall/F1, macro-F1, exact match, latency.

Usage: python scripts/evaluate.py [--data data/generated] [--split test]
       python scripts/evaluate.py --split real    # hand-corrected PDF excerpts in data/real/
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

from newline_fixer.baselines import heuristic, identity, vocabulary
from newline_fixer.labels import Label, align, decode
from newline_fixer.metrics import evaluate


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_real(folder: Path) -> list[dict]:
    """Pairs NN.input.txt (as pasted from the PDF) / NN.target.txt (corrected by hand)."""
    samples = []
    for broken in sorted(folder.glob("*.input.txt")):
        fixed = broken.with_name(broken.name.replace(".input.txt", ".target.txt"))
        try:
            words, breaks, labels = align(broken.read_text("utf-8-sig"), fixed.read_text("utf-8-sig"))
        except ValueError as e:
            raise ValueError(f"{broken.name}: {e}") from None
        samples.append({"words": words, "breaks": breaks, "labels": labels})
    return samples


def run(system, samples: list[dict]) -> tuple[list, list[float]]:
    """Predictions and per-document latency in milliseconds."""
    preds, times = [], []
    for s in samples:
        start = time.perf_counter()
        preds.append(system(s["words"], s["breaks"]))
        times.append((time.perf_counter() - start) * 1000)
    return preds, times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if args.split == "real":
        samples = load_real(Path("data/real"))
    else:
        samples = load(args.data / f"{args.split}.jsonl")
    vocab = vocabulary(decode(s["words"], s["labels"]) for s in load(args.data / "train.jsonl"))
    systems = {"identity": identity, "heuristic": partial(heuristic, vocab=vocab)}

    names = [l.name for l in Label]
    print(f"{args.split}: {len(samples)} documents\n")
    print("| System | " + " | ".join(f"{n} P / R / F1" for n in names) + " | Macro-F1 | Exact match | p50 ms | p95 ms |")
    print("|---" * (len(names) + 5) + "|")
    for name, system in systems.items():
        preds, times = run(system, samples)
        result = evaluate([s["labels"] for s in samples], preds)
        per_label = " | ".join(
            "{precision:.2f} / {recall:.2f} / {f1:.2f}".format(**result["labels"][n]) for n in names
        )
        p50, p95 = (statistics.quantiles(times, n=100)[q] for q in (49, 94))
        print(
            f"| {name} | {per_label} | {result['macro_f1']:.3f} | {result['exact_match']:.1%} "
            f"| {p50:.2f} | {p95:.2f} |"
        )


if __name__ == "__main__":
    main()
