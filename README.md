# SMS Spam Detection

A beginner-friendly natural-language binary-classification project. It turns raw
SMS text into TF-IDF features and trains a multinomial naive-Bayes classifier to
tell **spam** from legitimate **ham** messages.

This project is educational only. It demonstrates a standard text-classification
pipeline and is not a production spam filter.

## Dataset

Download Kaggle's [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
and place `spam.csv` at:

```text
data/raw/spam.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download -d uciml/sms-spam-collection-dataset -p data/raw --unzip
```

The raw file is 5,572 rows in `latin-1` encoding with two real columns (`v1` the
label, `v2` the message) plus three empty trailing columns that are dropped. After
removing 403 exact duplicate messages, 5,169 unique rows remain. About 12.6% are
spam, so the classes are imbalanced — accuracy alone would be misleading.

## Workflow

1. Load the raw CSV in `latin-1` and keep only the label and message columns.
2. Map `ham -> 0`, `spam -> 1` and drop exact duplicate messages.
3. Make a stratified 80/20 train/test split.
4. Vectorize text with TF-IDF (lower-cased, English stop-words, unigrams+bigrams).
5. Train multinomial naive Bayes in a leakage-safe pipeline.
6. Compare against an always-predict-ham dummy model.
7. Evaluate with accuracy, precision, recall, F1, a confusion matrix, and
   five-fold stratified cross-validation scored on F1.

## Results

| Measurement | Result |
|---|---:|
| Dummy test accuracy (always ham) | 87.3% |
| Naive Bayes test accuracy | 96.9% |
| Naive Bayes test precision | 1.00 |
| Naive Bayes test recall | 0.76 |
| CV mean F1 | 0.86 |

Precision of 1.00 means no legitimate message is wrongly flagged; recall of 0.76
means roughly a quarter of spam still slips through. That trade-off is the natural
starting point for this model — raising recall (e.g. logistic regression or tuned
thresholds) is a good next experiment.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The confusion matrix is written to `reports/figures/confusion_matrix.png`.

## Layout

```text
data/raw/         downloaded dataset (git-ignored — see Dataset above)
src/preprocess.py loading, cleaning, TF-IDF vectorizer
src/train.py      pipeline, training loop, evaluation, figure
src/evaluate.py   classification metric helpers
main.py           entry point
```

## Why this project

Inspired by classmate spam/text-classification projects (e.g. Srijansarkar17's
`SpamSMSDetection`, adityaxdubey's `fake-news-detection`). It is the natural NLP
companion to the tabular classifiers in the sibling `ml-*` folders.
