# Dataset:
# https://www.kaggle.com/crowdflower/twitter-airline-sentiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import pipeline

print("CUDA available:", torch.cuda.is_available())

device = 0 if torch.cuda.is_available() else -1
print("Using device:", device)


classifier = pipeline("sentiment-analysis", device=device)

print(classifier("this is such a great movie!"))

print(
    classifier(
        [
            "this is such a great movie!",
            "this course is just what I need",
        ]
    )
)


df_ = pd.read_csv("Tweets.csv")

df = df_[["airline_sentiment", "text"]]

df = df[df["airline_sentiment"] != "neutral"]


target_map = {"positive": 1, "negative": 0}
df["target"] = df["airline_sentiment"].map(target_map)


texts = df["text"].tolist()
predictions = classifier(texts)


probs = []
preds = []

for d in predictions:
    if d["label"].startswith("POS"):
        probs.append(d["score"])
        preds.append(1)
    else:
        probs.append(1 - d["score"])
        preds.append(0)

probs = np.array(probs)
preds = np.array(preds)

y_true = df["target"].values

print("accuracy: ", accuracy_score(y_true, preds))
print("f1_positive: ", f1_score(y_true, preds))
print("f1_negative: ", f1_score(1 - y_true, 1 - preds))
print("roc_auc_score: ", roc_auc_score(y_true, probs))
