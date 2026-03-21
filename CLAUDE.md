# Project Overview

NLP Sentiment Analysis on employee reviews — a portfolio project for Aaron England's personal website (aaron-england.com). This project is specifically designed to demonstrate skills relevant to a **Staff Data Scientist role at Paylocity**, an HCM/HR software company whose AI Assist product does tone and sentiment analysis on employee feedback and performance reviews.

## Why This Project Exists

Paylocity's job description lists four algorithm families: **regression, random forests, gradient boosting, and neural networks**. This single project demonstrates all four on an NLP task that directly mirrors Paylocity's product work.

## S3 Bucket

`npl-sentiment-analysis-demo`

All data is read from and written to this bucket. Raw data goes in `00_data_collection/`, preprocessed splits in `02_preprocessing/`.

## Dataset

Kaggle Employee Reviews (Glassdoor) — ~67K reviews from Google, Amazon, Apple, Facebook, Microsoft, Netflix. Each review has star ratings (1-5), pros, cons, summary text. Target is 3-class sentiment: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars).

## Project Structure

```
nlp_sentiment_analysis/
├── 00_data_collection/notebook.ipynb     # Download from Kaggle, clean, upload to S3
├── 01_eda/notebook.ipynb                 # Text EDA: distributions, word frequency, sentiment by company
├── 02_preprocessing/notebook.ipynb       # Text cleaning, lemmatization, stratified 60/20/20 split
├── 03_tfidf_logreg/notebook.ipynb        # TF-IDF + Logistic Regression (regression)
├── 04_tfidf_random_forest/notebook.ipynb # TF-IDF + Random Forest (random forests)
├── 05_tfidf_xgboost/notebook.ipynb       # TF-IDF + XGBoost (gradient boosting)
├── 06_neural_network/notebook.ipynb      # PyTorch LSTM (neural networks)
├── 07_comparison/notebook.ipynb          # Head-to-head comparison of all 4 models
├── requirements.txt
└── README.md
```

## Execution Order

Run notebooks sequentially: 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07. Each notebook reads from S3 (outputs of prior steps) and writes its own outputs to `./output/`.

## Key Technical Details

- **All models use Bayesian hyperparameter tuning via Optuna** (20 trials each, TPESampler) — except the LSTM which uses early stopping + ReduceLROnPlateau
- **TF-IDF config**: 10,000 max features, unigrams + bigrams, sublinear TF
- **LSTM**: Pure PyTorch (no TensorFlow/Keras) — Embedding → LSTM → Dense. Custom vocabulary building, manual training loop
- **Evaluation**: Accuracy, F1-macro, F1-weighted, ROC AUC (macro), per-class precision/recall/F1, confusion matrices, ROC curves
- **Comparison notebook** loads all 4 serialized models and evaluates on the same test set

## Coding Conventions

- **Classes** encapsulate all functionality (e.g., `TfidfLogregModel`, `LSTMSentimentModel`)
- **Hungarian notation**: `str_`, `int_`, `flt_`, `cls_`, `df_`, `list_`, `dict_`, `bool_`, `arr_`
- **Constants section** at top of each notebook: `str_bucket`, `str_task`, `str_dirname_output`
- **Output dirs**: Created with `os.mkdir` / `os.makedirs`
- **Plots**: Saved to `./output/` with `dpi=150`, `bbox_inches='tight'`
- **S3 loading**: Try S3 first, fall back to local path
- **Progress bars**: `tqdm`

## Model Artifacts (saved to each notebook's output/)

- `03_tfidf_logreg/output/`: `tfidf_vectorizer.pkl`, `logreg_model.pkl`
- `04_tfidf_random_forest/output/`: `tfidf_vectorizer.pkl`, `rf_model.pkl`
- `05_tfidf_xgboost/output/`: `tfidf_vectorizer.pkl`, `xgboost_model.pkl`
- `06_neural_network/output/`: `lstm_model.pt`, `vocabulary.pkl`, `model_config.pkl`

## What Comes After This Repo

Once all notebooks are run and outputs are generated, the results (serialized models, plots, metrics) will be integrated into Aaron's personal website (aaron-england.com) as a dedicated `/sentiment-analysis` page with an interactive demo and full methodology write-up — similar to the existing `/credit-risk` page.
