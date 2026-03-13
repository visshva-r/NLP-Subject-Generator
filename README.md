# NLP Email Subject Line Generator (AESLC Benchmark)

This repository contains a research-oriented Jupyter Notebook that evaluates a pretrained **DistilGPT-2** model on the **AESLC (Annotated Email Subject Line Corpus)** benchmark using Hugging Face `transformers` and `datasets`.

## Project Architecture
1. **Dataset Integration:** Programmatically fetches the AESLC testing split via Hugging Face.
2. **Heuristic Decoding Algorithm:** Instead of relying on raw zero-shot outputs, this pipeline utilizes a custom `best_of_n` sampling algorithm. It generates multiple candidates and scores them based on keyword-overlap with the source text, severely reducing hallucinations.
3. **Quantitative Evaluation:** Evaluates the generated subjects against the human-written ground truth using **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metrics (ROUGE-1, ROUGE-2, ROUGE-L).

## How to Run
1. Create/activate a Python 3.10+ environment
2. Install dependencies: `pip install -r requirements.txt` (or `pip install transformers torch pandas matplotlib datasets evaluate rouge_score tqdm`)
3. Run the notebook cells sequentially from the project directory

## Author
Visshva R
