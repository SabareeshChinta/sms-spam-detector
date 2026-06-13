# SMS Spam Detector using SVM

A machine learning project that classifies SMS messages as **Spam** or **Ham** using a Support Vector Machine (LinearSVC) with TF-IDF text vectorization.

## Results

| Metric | Ham | Spam |
|---|---|---|
| Precision | 98.6% | 99.3% |
| Recall | 99.9% | 90.6% |
| F1-Score | 99.2% | 94.7% |
| **Accuracy** | **98.65%** | |

## Dataset

Download `spam.csv` from [UCI SMS Spam Collection on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project root.

- 5,572 SMS messages
- 4,825 Ham (86.6%) / 747 Spam (13.4%)

## Project Structure

```
sms-spam-detector/
├── sms_spam_detector.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Pipeline

```
Raw Text → Preprocess → TF-IDF Vectorization → LinearSVC → Prediction
```

1. **Preprocess** — lowercase, remove digits and punctuation
2. **TF-IDF** — convert text to 5000-dim numerical vectors (unigrams + bigrams)
3. **LinearSVC** — find the optimal hyperplane separating spam and ham
4. **Evaluate** — accuracy, precision, recall, F1, confusion matrix

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python sms_spam_detector.py
```

## Tech Stack

- Python 3
- scikit-learn
- pandas
