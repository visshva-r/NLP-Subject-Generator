# Subject lines from email bodies

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

I built this to draft short email subjects from the body and score them on [AESLC](https://huggingface.co/datasets/aeslc). DistilGPT-2 proposes a few lines; a simple keyword-overlap filter picks the best. A five-word extractive baseline still wins ROUGE on the test split I used, and I left that in on purpose.

Run locally: `pip install -r requirements.txt`, then `python app.py`.

The demo puts extractive and model output side by side. Temperature and sampling live under *Decoding (optional)*.

## How it works

Take AESLC test emails (first 50 words of the body), clean them, then:

- Naive: title-case the first five content words.
- Model: sample a few short lines with DistilGPT-2, keep the ones that overlap most with the body.

Score both against the human `subject_line` with ROUGE.

## Results (n=100, seed=42)

Same 100 emails every time. Numbers: [`outputs/metrics.json`](outputs/metrics.json).

| Approach | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Naive extractive (first 5 content words) | 0.0991 | 0.0263 | 0.0963 |
| DistilGPT-2, one sample | 0.0488 | 0.0155 | 0.0482 |
| DistilGPT-2 + best-of-3 | 0.0526 | 0.0131 | 0.0518 |

Best-of-3 beats a single sample, but extractive still leads. AESLC subjects are short and often reuse words from the opening. DistilGPT-2 is a small decoder, not a summarizer trained for this task.

| Email (truncated) | Human | Naive | Best-of-N |
|-------------------|-------|-------|-----------|
| Last week was the hardest week… people who were laid off… | Layoffs | Last Week Was Hardest Week | The Next Day When You |
| As you know, the SEC is conducting an informal inquiry… | SEC Inquiry | As Know Sec Conducting Informal | As Know Sec Conducting Informal |
| Dear Medicine Bow Shipper: …FERC approval for the fourth tie-in… | Medicine Bow Tie-In Capacity | Dear Medicine Bow Shipper Morning | Dr Sarah Siegel From Mdh |
| We are in the process of confirming that all traders are set up… | **IMPORTANT** Stack Manager Users | Process Confirming All Traders Set | Process Confirming All Traders Set |
| Suzanne: Could you please coordinate… a 30 minute meeting… | Meeting | Suzanne Could Please Coordinate Sara | Suzanne Could Please Coordinate Sara |

Human labels are often one word ("Layoffs", "Meeting"). ROUGE rewards overlap with the opening sentence. The model still makes things up sometimes (row 3).

## Limitations

- Zero-shot DistilGPT-2 only. Not fine-tuned for summarization.
- 100 test emails (seed 42), not the full AESLC test set.
- Keyword overlap helps a bit. It is not a learned reranker.

## Run it

```bash
git clone https://github.com/visshva-r/NLP-Subject-Generator.git
cd NLP-Subject-Generator
python -m venv .venv && .venv\Scripts\activate   # Unix: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

| Task | Command |
|------|---------|
| One email | `python scripts/demo.py "Your email body"` |
| Full eval | `python scripts/run_benchmark.py --samples 100` |
| Tests | `pip install -r requirements-dev.txt && pytest -q` |
| CI smoke (no GPU) | `python scripts/run_benchmark.py --samples 5 --skip-model --output-dir outputs/smoke` |
| Notebook | `notebooks/Email Subject Generator.ipynb` from the repo root |

GitHub Actions runs pytest and a 5-row naive smoke. It does not download DistilGPT-2.

To host: `gradio deploy`, or `docker build -t email-subject-generator . && docker run -p 7860:7860 email-subject-generator`.

```
app.py          Gradio demo
src/            generation, cleanup, AESLC load, ROUGE
scripts/        CLI demo + benchmark
tests/          mocked generator (no GPU)
notebooks/      thin demo that imports src/
outputs/        metrics.json + generated CSV
```

## What I would do next

Fine-tune T5 or BART on AESLC train if I wanted to beat extractive ROUGE for real. After that, a learned reranker and eval on the full test split.

Visshva R · [MIT](LICENSE)
