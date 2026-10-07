"""Build the synthetic train/val/test sets from Cosmopedia (see DECISIONS.md, D5).

Each document is cleaned, cut into chunks of 150-300 words and corrupted.
Splits are made per document, so chunks of one document never land in two splits.

Usage: python scripts/build_dataset.py [--docs-per-subset 2000] [--out data/generated]
"""

import argparse
import json
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from datasets import load_dataset

from newline_fixer.clean import clean
from newline_fixer.corrupt import corrupt
from newline_fixer.labels import Label, decode, encode

SUBSETS = ["stanford", "openstax", "wikihow", "web_samples_v2"]
CHUNK_WORDS = (150, 300)  # stays under the 512-token limit of the encoder
SPLITS = ("train", "val", "test")


def split_of(doc_index: int) -> str:
    """90 / 5 / 5 split by document."""
    return {0: "test", 1: "val"}.get(doc_index % 20, "train")


def chunks(text: str, rng: random.Random):
    """Consecutive chunks of random length; a tail shorter than the minimum is dropped."""
    words, labels = encode(text)
    start = 0
    while start + CHUNK_WORDS[0] <= len(words):
        end = start + rng.randint(*CHUNK_WORDS)
        yield decode(words[start:end], labels[start : end - 1])
        start = end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-per-subset", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    samples, labels, skipped = Counter(), {s: Counter() for s in SPLITS}, 0

    with ExitStack() as stack:
        files = {s: stack.enter_context(open(args.out / f"{s}.jsonl", "w", encoding="utf-8")) for s in SPLITS}
        for subset in SUBSETS:
            docs = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
            for i, doc in enumerate(docs.take(args.docs_per_subset)):
                if "```" in doc["text"] or "|" in doc["text"]:  # code blocks and tables are out of scope
                    skipped += 1
                    continue
                split = split_of(i)
                for chunk in chunks(clean(doc["text"]), rng):
                    sample = corrupt(chunk, rng)
                    files[split].write(json.dumps({"source": subset, **sample}) + "\n")
                    samples[split] += 1
                    labels[split].update(Label(l).name for l in sample["labels"])

    print(f"skipped documents (code or tables): {skipped}")
    for s in SPLITS:
        total = sum(labels[s].values())
        dist = ", ".join(f"{l.name} {labels[s][l.name] / total:.2%}" for l in Label)
        print(f"{s:5}  {samples[s]:6} samples  |  {dist}")


if __name__ == "__main__":
    main()
