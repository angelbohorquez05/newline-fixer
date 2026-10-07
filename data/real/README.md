# Real PDF test set

Hand-made, used **only for evaluation**, to check that a model trained on synthetic data works on real PDF text.

Each excerpt is a pair of files:

- `NN.input.txt`: the text exactly as it is pasted after copying it from a PDF viewer.
- `NN.target.txt`: a copy of the input where **only whitespace was changed** (spaces, line breaks,
  blank lines between paragraphs, and words split across lines joined back). Every other character is kept,
  so the target can be aligned with the input (`newline_fixer.labels.align`).

Words split with a hyphen (`atten-` / `tion`) are joined keeping the hyphen (`atten-tion`), since the model
never changes characters (a known limitation, see the report).

| NN | Source | Type |
|---|---|---|
