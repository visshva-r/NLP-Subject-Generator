# Email Subject Line Generator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Generate short email subject lines from message bodies with zero-shot **DistilGPT-2**, then rank candidates with a keyword-overlap **best-of-N** filter. Results are evaluated on the [AESLC](https://huggingface.co/datasets/aeslc) test split using ROUGE, alongside a simple extractive baseline.

On a fixed sample of 100 AESLC emails (seed 42), the extractive baseline still outscores DistilGPT-2. That comparison is part of the evaluation, not a claim of state-of-the-art performance.

## Features

- DistilGPT-2 subject generation with cleanup and best-of-N ranking
- Extractive baseline (first content words, title-cased)
- AESLC loading, ROUGE metrics, and saved benchmark outputs
- Gradio demo, CLI scripts, pytest suite, and GitHub Actions CI

## How it works

1. Load AESLC test emails and truncate each body to the first 50 words.
2. **Naive baseline:** title-case the first five content words.
3. **Model:** sample short subject lines with DistilGPT-2; keep candidates with the highest keyword overlap with the body.
4. Score both against the human `subject_line` with ROUGE-1 / ROUGE-2 / ROUGE-L.

## Results (n=100, seed=42)

Full metrics: [`outputs/metrics.json`](outputs/metrics.json).

| Approach | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Naive extractive (first 5 content words) | 0.0991 | 0.0263 | 0.0963 |
| DistilGPT-2, one sample | 0.0488 | 0.0155 | 0.0482 |
| DistilGPT-2 + best-of-3 | 0.0526 | 0.0131 | 0.0518 |

Best-of-3 improves over a single sample. Extractive still leads on this split: AESLC subjects are short and often reuse opening words, and DistilGPT-2 is a small decoder rather than a task-specific summarizer.

| Email (truncated) | Human | Naive | Best-of-N |
|-------------------|-------|-------|-----------|
| Last week was the hardest week… people who were laid off… | Layoffs | Last Week Was Hardest Week | The Next Day When You |
| As you know, the SEC is conducting an informal inquiry… | SEC Inquiry | As Know Sec Conducting Informal | As Know Sec Conducting Informal |
| Dear Medicine Bow Shipper: …FERC approval for the fourth tie-in… | Medicine Bow Tie-In Capacity | Dear Medicine Bow Shipper Morning | Dr Sarah Siegel From Mdh |
| We are in the process of confirming that all traders are set up… | **IMPORTANT** Stack Manager Users | Process Confirming All Traders Set | Process Confirming All Traders Set |
| Suzanne: Could you please coordinate… a 30 minute meeting… | Meeting | Suzanne Could Please Coordinate Sara | Suzanne Could Please Coordinate Sara |

Human labels are often abstract (for example "Layoffs"). ROUGE rewards overlap with the opening text. The model can still hallucinate (row 3).

## Limitations

- Zero-shot DistilGPT-2 only; not fine-tuned for subject generation.
- Evaluation uses 100 test emails (seed 42), not the full AESLC test set.
- Keyword overlap is a heuristic, not a learned reranker.

## Setup

```bash
git clone https://github.com/visshva-r/NLP-Email-Subject-Generator.git
cd NLP-Email-Subject-Generator
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

| Task | Command |
|------|---------|
| Gradio demo | `python app.py` |
| Single email | `python scripts/demo.py "Your email body"` |
| Full benchmark | `python scripts/run_benchmark.py --samples 100` |
| Tests | `pip install -r requirements-dev.txt && pytest -q` |
| CI-style smoke (no GPU) | `python scripts/run_benchmark.py --samples 5 --skip-model --output-dir outputs/smoke` |
| Notebook | `notebooks/Email Subject Generator.ipynb` (run from repo root) |

GitHub Actions runs pytest and a 5-row naive smoke on CPU. It does not download DistilGPT-2.

Optional hosting:

```bash
gradio deploy
# or
docker build -t email-subject-generator .
docker run -p 7860:7860 email-subject-generator
```

## Project layout

```
app.py          Gradio demo
src/            generation, cleanup, AESLC load, ROUGE
scripts/        CLI demo and benchmark
tests/          unit tests (mocked generator; no GPU required)
notebooks/      demo notebook that imports the package
outputs/        metrics.json and generated CSV / JSON
```

## Future work

- Fine-tune T5 or BART on AESLC train for stronger ROUGE
- Replace keyword overlap with a learned reranker
- Evaluate on the full AESLC test split

## Author

Visshva R · [MIT License](LICENSE)
