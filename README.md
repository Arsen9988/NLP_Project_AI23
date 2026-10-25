# NLP_Project_AI23

Sentiment analysis experiments using IMDb movie reviews. The project explores traditional bag-of-words methods, transformer models, and GPT-based approaches, with notebooks for training, evaluation, predictions, and an interactive DistilBERT demo.

## Project structure

- `code/` — notebooks and shared code (these are separate experiments, not one end-to-end pipeline):
  - `bagofwords.ipynb` — compares unigram and bigram text representations using Keras `TextVectorization` with a 20,000-token vocabulary and TF-IDF output. It evaluates saved models against the prepared IMDb train/test split; the bigram representation also captures adjacent-word patterns.
  - `distilbert_model untuned.ipynb` — evaluates the pretrained DistilBERT sentiment pipeline `distilbert-base-uncased-finetuned-sst-2-english` on IMDb reviews. It maps the model's positive/negative labels to the dataset labels and examines both original and truncated/padded review text; it does not fine-tune the model on this dataset.
  - `imdb_model.ipynb` — evaluates the pretrained `sarahai/movie-sentiment-analysis` sequence-classification model against the IMDb review labels, including tokenization and truncation/padding.
  - `pytorch_finetuning.ipynb` — fine-tunes a DistilBERT sequence classifier with Hugging Face's PyTorch `Trainer` using the prepared, truncated/padded train/test data. It computes accuracy and saves trained checkpoints for later inference.
  - `Gpt_2_model.ipynb` — tests a pretrained GPT-2-medium sentiment classifier (`michelecafagna26/gpt2-medium-finetuned-sst2-sentiment`) on a stratified 70/30 IMDb split. The notebook limits the dataset sample size and reports classification performance.
  - `Gpt_pipeline.ipynb` — compares sentiment-analysis experiments using pretrained DistilBERT and OpenAI GPT models (GPT-3.5 Turbo and GPT-4o mini). It also includes an exploratory text-summarization pipeline; summarization is separate from the sentiment evaluation.
  - `predictions.ipynb` — loads the locally saved `fine_tuned_model_2000` checkpoint and its tokenizer to run sentiment predictions on new text.
  - `gradio_with_distilbert.ipynb` — wraps a pretrained DistilBERT sentiment-analysis pipeline in a Gradio interface so a user can submit text and view its predicted sentiment.
  - `utils.py` — shared Python utilities used by the notebooks.
- `data/` — IMDb review data, sentiment results, and prepared train/test splits.
- `results/` — evaluation CSV files and confusion-matrix images.
- `AI-23_Deep_Learning_LLM_project.pptx` — project presentation.

## Getting started

Use Python 3.12 (as specified in the Pipfile) and install the project dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment, then run:

```bash
python -m pip install -r requirements.txt
```

Open the notebooks in `code/` with Jupyter or VS Code. Start with the notebook for the approach you want to explore; some notebooks may download pretrained models or require access to external services.

## Data and credentials

The notebooks use files under `data/`, including the IMDb review dataset and precomputed train/test splits. Keep any service credentials in local environment variables or another untracked local configuration. Do not commit API keys or other secrets.
