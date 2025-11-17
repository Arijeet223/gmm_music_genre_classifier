import numpy as np
import pickle

def predict(models, sample):
    scores = {}
    for genre, model in models.items():
        scores[genre] = model.score(sample.reshape(1, -1))
    return max(scores, key=scores.get)

def evaluate(models, X_test, y_test):
    correct = 0
    
    for x, y_true in zip(X_test, y_test):
        y_pred = predict(models, x)
        if y_pred == y_true:
            correct += 1

    return correct / len(y_test)
    