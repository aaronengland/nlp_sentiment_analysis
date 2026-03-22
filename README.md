# NLP Sentiment Analysis: Employee Review Classification

## Overview

A comprehensive sentiment analysis project for classifying employee reviews from major tech companies (Google, Amazon, Apple, Facebook, Microsoft, Netflix). This project demonstrates end-to-end NLP pipeline development including data collection, exploratory analysis, text preprocessing, feature engineering, model training across **four distinct algorithm families**, and comparative evaluation.

**Business Context**: This project showcases expertise relevant to Paylocity's AI/ML teams, specifically for sentiment and tone analysis features in HR/HCM software that processes employee feedback and performance reviews.

**Algorithms Demonstrated**:
- **Regression**: Logistic Regression (linear baseline)
- **Random Forests**: TF-IDF + Random Forest (ensemble tree method)
- **Gradient Boosting**: TF-IDF + XGBoost (boosted trees)
- **Neural Networks**: Keras LSTM (deep learning sequence model)

**Target Metrics**:
- 3-class classification: Negative (1-2 stars), Neutral (3 stars), Positive (4-5 stars)
- Expected accuracy: 75-82% on test set
- Expected F1-macro: 0.72-0.78

---

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `00_data_collection/` | Download, clean, prepare raw data |
| `01_eda/` | Exploratory Data Analysis with visualizations |
| `02_preprocessing/` | Text cleaning, preprocessing, train/val/test splits |
| `03_tfidf_logreg/` | TF-IDF + Logistic Regression (regression) |
| `04_tfidf_random_forest/` | TF-IDF + Random Forest (random forests) |
| `05_tfidf_xgboost/` | TF-IDF + XGBoost (gradient boosting) |
| `06_neural_network/` | Keras LSTM (neural networks) |
| `07_comparison/` | Side-by-side evaluation of all four models |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | Kaggle: Employee Reviews (Glassdoor) |
| **URL** | https://www.kaggle.com/datasets/fireball684/hackerearthericsson |
| **Size** | ~67,000 reviews |
| **Companies** | Google, Amazon, Apple, Facebook, Microsoft, Netflix |
| **Features** | overall_rating (1-5), pros, cons, summary, advice_to_management |
| **Target** | Multi-class sentiment (3-class) |
| **Class Distribution** | Imbalanced (skewed toward positive: ~70% 4-5 stars) |

### Sentiment Labels

**3-Class**: Negative (1-2 stars, ~15%), Neutral (3 stars, ~15%), Positive (4-5 stars, ~70%)

---

## Methodology

### 00 - Data Collection & Cleaning

Download Kaggle dataset, clean, create sentiment labels, upload to S3.

**Process**:
1. Download employee reviews from Kaggle
2. Drop rows with missing text fields
3. Combine pros + cons into review_text
4. Create sentiment_3class column (0=Negative, 1=Neutral, 2=Positive)
5. Save cleaned CSV to S3

**Output**: s3://nlp-sentiment-analysis-demo/00_data_collection/employee_reviews_clean.csv

---

### 01 - Exploratory Data Analysis

Understand dataset composition, text characteristics, label distributions via visualization.

**Class**: TextEDA

**Outputs**: 7 PNG visualizations — rating distribution, sentiment distribution, review length, word frequency, word frequency by sentiment, review length by sentiment, company sentiment.

---

### 02 - Text Preprocessing & Data Splitting

Clean text and create stratified train/val/test splits (60/20/20).

**Class**: TextPreprocessor

**Text Pipeline**:
1. Lowercase
2. Remove URLs
3. Remove punctuation
4. Remove stopwords (NLTK English)
5. Lemmatize (WordNetLemmatizer)
6. Min token length: 3 chars

**Data Splitting**: Stratified by sentiment_3class to preserve class distribution.

**Outputs**: 3 CSV files to S3 with review_text_clean column.

---

### 03 - TF-IDF + Logistic Regression

TF-IDF feature extraction + Logistic Regression with Optuna hyperparameter tuning.

**Class**: TfidfLogregModel

**Vectorizer**: Max 10,000 features, unigrams + bigrams, sublinear TF.

**Tuning**: Optuna (50 trials) over C, solver, class_weight. Metric: F1-macro.

**Outputs**: Confusion matrix, ROC curves, feature importance (coefficients), serialized model.

---

### 04 - TF-IDF + Random Forest

Same TF-IDF vectorizer + Random Forest classifier with Optuna tuning.

**Class**: TfidfRandomForestModel

**Tuning**: Optuna (50 trials) over n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight. Metric: F1-macro.

**Outputs**: Confusion matrix, ROC curves, SHAP feature importance, serialized model.

---

### 05 - TF-IDF + XGBoost

Same TF-IDF vectorizer + XGBoost gradient boosting classifier with Optuna tuning.

**Class**: TfidfXgboostModel

**Tuning**: Optuna (50 trials) over n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda, class weighting. Metric: F1-macro. Early stopping on validation set.

**Outputs**: Confusion matrix, ROC curves, gain-based feature importance, serialized model.

---

### 06 - LSTM Neural Network

Deep learning approach using a PyTorch LSTM for sequence-based sentiment classification. Implements a manual training loop with custom vocabulary building and sequence padding — no high-level wrappers.

**Class**: LSTMSentimentModel (with nested SentimentLSTM nn.Module)

**Architecture**:
- Embedding(20000, 128, padding_idx=0) → SpatialDropout(0.3)
- LSTM(64, batch_first=True, dropout=0.3)
- Linear(64→32) → ReLU → Dropout(0.3)
- Linear(32→3)

**Training**: Adam optimizer, CrossEntropyLoss, EarlyStopping (patience=3), ReduceLROnPlateau. Max 20 epochs with batch size 64. CUDA auto-detection.

**Outputs**: Training history (loss/accuracy curves), confusion matrix, ROC curves, serialized PyTorch model (.pt) + vocabulary + config.

---

### 07 - Model Comparison

Head-to-head comparison of all four models on the same test set.

**Class**: ModelComparison

**Analysis**:
- Side-by-side metrics table (Accuracy, F1-Macro, F1-Weighted, ROC-AUC)
- Overlaid ROC curves by class (3 subplots × 4 models)
- 2×2 confusion matrix grid
- Per-class F1 comparison
- Model disagreement analysis

**Outputs**: 4 PNG visualizations + printed metrics tables.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 4 algorithm families | Demonstrates breadth across regression, ensemble trees, gradient boosting, and deep learning |
| TF-IDF + bigrams | Captures phrases ("great culture", "poor management") without deep learning overhead |
| LSTM for neural network | Sequence modeling captures word order, a step beyond bag-of-words approaches |
| Stratified splits | Preserves class imbalance for realistic evaluation |
| Optuna tuning | Bayesian optimization > grid search for efficiency |
| F1-macro metric | Fair metric for imbalanced 3-class data |
| Early stopping (XGBoost + LSTM) | Prevents overfitting on validation set |
| SHAP + Random Forest | Stakeholder-friendly model interpretability |
| S3 storage | Simulates production AWS/SageMaker workflow |

---

## Getting Started

### Installation

```bash
cd nlp_sentiment_analysis

pip install -r requirements.txt

python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

### Run Pipeline

Execute notebooks in order:

```bash
00_data_collection/notebook.ipynb
01_eda/notebook.ipynb
02_preprocessing/notebook.ipynb
03_tfidf_logreg/notebook.ipynb
04_tfidf_random_forest/notebook.ipynb
05_tfidf_xgboost/notebook.ipynb
06_neural_network/notebook.ipynb
07_comparison/notebook.ipynb
```

All outputs (PNG visualizations, model pickles) saved to each step's `output/` directory.

---

## Expected Results

### Test Set Performance (3-Class)

| Model | Accuracy | F1-Macro | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.76 | 0.72 | 0.85 |
| Random Forest | 0.79 | 0.75 | 0.87 |
| XGBoost | 0.80 | 0.76 | 0.88 |
| LSTM Neural Network | 0.78 | 0.74 | 0.86 |

### Key Insights

1. **Gradient boosting (XGBoost)** expected to edge out other models on overall metrics due to its ability to handle sparse TF-IDF features and class imbalance
2. **LSTM** competitive despite different feature pipeline; captures sequential patterns TF-IDF misses
3. **Logistic Regression** provides strong interpretable baseline — coefficients directly map to discriminative words
4. **Random Forest** balances performance and interpretability via SHAP
5. **Neutral class** remains hardest for all models — mixed sentiment in 3-star reviews

---

## Paylocity Relevance

This project demonstrates expertise for Paylocity's product:

1. **Employee Feedback Analysis**: Classifies tone in surveys and reviews
2. **Performance Review Insights**: Identifies sentiment themes in free-text feedback
3. **Algorithm Breadth**: Covers all four ML algorithm families from the job description
4. **Production Ready**: S3 integration, serialized models, real-time inference capable
5. **Interpretability**: SHAP, coefficients, and feature importance for HR stakeholders
6. **Class Imbalance Handling**: Realistic distribution in HR feedback data
7. **Deep Learning**: LSTM demonstrates neural network expertise beyond traditional ML

---

## Technical Stack

| Component | Library |
|-----------|---------|
| Data | pandas, numpy |
| ML | scikit-learn (LogReg, RF), XGBoost |
| Deep Learning | PyTorch (LSTM) |
| Tuning | Optuna |
| Features | TfidfVectorizer, PyTorch Tokenizer |
| NLP | NLTK |
| Interpret | SHAP |
| Viz | matplotlib, seaborn |
| Persist | joblib |
| Cloud | boto3 (S3) |
| Progress | tqdm |

---

## Author

Aaron England — Staff Data Scientist Portfolio Project

Built to demonstrate NLP, ML, and deep learning expertise for Paylocity.
