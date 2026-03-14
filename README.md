# NLP Email Subject Line Generator (AESLC Benchmark)

This repository contains a research-oriented Jupyter Notebook that evaluates a pretrained **DistilGPT-2** model on the **AESLC (Annotated Email Subject Line Corpus)** benchmark using Hugging Face `transformers` and `datasets`. The full pipeline runs without errors from setup through evaluation and output.

## Project Architecture
1. **Dataset Integration:** Programmatically fetches the AESLC testing split via Hugging Face.
2. **Heuristic Decoding Algorithm:** Instead of relying on raw zero-shot outputs, this pipeline utilizes a custom `best_of_n` sampling algorithm. It generates multiple candidates and scores them based on keyword-overlap with the source text, severely reducing hallucinations.
3. **Quantitative Evaluation:** Evaluates the generated subjects against the human-written ground truth using **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metrics (ROUGE-1, ROUGE-2, ROUGE-L).

## How to Run
1. Clone the repo and open a terminal in the project directory.
2. Create/activate a Python 3.10+ environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Open **Email Subject Generator.ipynb** in Jupyter and run all cells in order (or start Jupyter from the project directory so paths resolve correctly).

Generated CSV and JSON results are written to the `outputs/` folder.

## Author
Visshva R
