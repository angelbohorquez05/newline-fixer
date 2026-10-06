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
