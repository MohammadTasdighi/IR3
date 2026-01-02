from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, embeddings, ratings):
        X = []
        y = []
        for emb, rate in zip(embeddings, ratings):
            if rate == 3: continue
            X.append(emb)
            y.append(1 if rate >= 4 else 0)
        
        self.model.fit(np.array(X), np.array(y))
        return "Model Trained Successfully"

    def predict(self, emb):
        return "مثبت" if self.model.predict([emb])[0] == 1 else "منفی"