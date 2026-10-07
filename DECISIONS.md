# Decision log

Key technical decisions, newest last. Each entry: context, decision, alternatives considered, reason.
Progress over time is tracked by git history (`git log`).

---

### D1 — Frame the task as gap classification, not text generation
**2026-10-06 15:05**

- **Context:** Only whitespace is wrong in the input; the words themselves are correct.
- **Decision:** Split the text into words and predict, for every gap between two consecutive words, one of
  `SPACE`, `JOIN` (no separator, a word broken across lines), `NEWLINE` or `PARAGRAPH` (blank line).
- **Alternatives:** seq2seq / LLM rewriting the whole text; pure rule-based heuristics.
- **Reason:** The model cannot alter or hallucinate words, inference is a single forward pass (fast),
  and per-label precision/recall gives clear metrics. `JOIN` is only allowed where the input had a line break.

### D2 — Build a synthetic training set
**2026-10-06 15:05**

- **Context:** No dataset is provided.
- **Decision:** Take well-formatted English text and corrupt it to mimic PDF extraction
  (hard wrapping at random widths, words split across lines, merged paragraphs, headings and bullets glued to text).
  The original text is the target.
- **Alternatives:** Hand-labelling real PDF text (slow, small).
- **Reason:** Unlimited labelled data for free. A small hand-made set from real PDFs is kept for evaluation only,
  to check the model generalises beyond the synthetic distribution.

### D3 — Fine-tune a small pretrained encoder for token classification
**2026-10-06 15:05**

- **Context:** The service must answer efficiently; training on free GPUs (Colab).
- **Decision:** Fine-tune a small transformer encoder with a token-classification head.
- **Alternatives:** Large LLM via prompting (slow, costly, may rewrite text); training from scratch (needs far more data).
- **Reason:** Good accuracy/latency trade-off and runs on CPU in production.
  Related prior work: punctuation restoration as token classification (`fullstop-punctuation-multilang`)
  and newline-based text segmentation (`wtpsplit` / SaT).

### D4 — Project layout and dependencies
**2026-10-06 15:05**

- **Decision:** `src/` layout; runtime dependencies in `requirements.txt`, development ones in `requirements-dev.txt`;
  model weights stored on Hugging Face Hub, not in git.
- **Reason:** Smaller Docker image, tests import the package as a user would, and the repository stays light.

### D5 — Use Cosmopedia as the single source of clean text
**2026-10-07 07:00**

- **Context:** The corruption (D2) needs well-formatted English text whose line and paragraph breaks are meaningful.
- **Evidence:** 100-300 documents sampled per candidate source.

  | Source | Finding |
  |---|---|
  | FineWeb-Edu | Every break is a single `\n` (0% `PARAGRAPH`): cannot teach paragraphs |
  | peS2o | Does not load with `datasets` 5.x (legacy loading script) |
  | arXiv (Common Pile) | Closest to papers, but Markdown hard-wrapped at ~80 characters; lines would need rebuilding |
  | Cosmopedia | Clean paragraphs, headings and lists |

  Cosmopedia subsets, 200 documents each:

  | Subset | NEWLINE | PARAGRAPH | Docs with lists | Docs with headings | Docs with Markdown |
  |---|---|---|---|---|---|
  | `stanford` | 0.80% | 2.08% | 89 | 43 | 59 |
  | `openstax` | 0.55% | 2.38% | 70 | 44 | 112 |
  | `wikihow` | 2.14% | 2.85% | 148 | 56 | 100 |
  | `web_samples_v2` | 0.52% | 1.58% | 63 | 12 | 25 |

- **Decision:** Mix the four subsets in equal parts: textbooks bring headings and document structure,
  `wikihow` brings lists, `web_samples_v2` brings plain prose. Markdown is stripped (`clean.py`):
  heading marks, bold markers and rules removed, bullets turned into `•`. Documents with code blocks or tables are skipped.
  Texts are cut into chunks of 150-300 words (below the 512-token limit) and split 90/5/5 by document, so no
  document leaks across splits.
- **Alternatives:** Mixing several corpora (FineWeb-Edu + arXiv).
- **Reason:** One source keeps the pipeline simple, and Cosmopedia is the only candidate with reliable paragraph
  structure. If the real-PDF test set shows weak results on papers, arXiv will be added as a documented iteration.

### D6 — Metrics and baselines
**2026-10-07 07:30**

- **Context:** `SPACE` is ~95% of the gaps, so accuracy is misleading: returning the input unchanged already scores ~91%.
- **Decision:** Report precision, recall and F1 per label, macro-F1 over the four labels (main metric),
  document exact match and per-document latency (p50 / p95). Two baselines set the bar:
  - `identity`: returns the input unchanged.
  - `heuristic`: hand-written rules (bullets start a line; a break inside a word known from the train set is a `JOIN`;
    a line shorter than 70% of the text width ended on purpose). The 70% threshold was tuned on val
    (macro-F1 0.654 at 50%, 0.675 at 70%, 0.621 at 90%).
- **Results (synthetic test, 992 documents):**

  | System | JOIN F1 | NEWLINE F1 | PARAGRAPH F1 | Macro-F1 | Exact match |
  |---|---|---|---|---|---|
  | identity | 0.00 | 0.07 | 0.00 | 0.254 | 0.0% |
  | heuristic | 0.76 | 0.54 | 0.43 | 0.682 | 0.7% |

- **Reason:** A learned model is only worth its cost if it clearly beats simple rules. The heuristic shows where the
  difficulty is: `JOIN` is precise but misses splits whose two pieces are real words, and real breaks inside
  full-width lines or glued to the next word (`Model The Transformer`) cannot be detected from layout alone.
