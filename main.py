import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from src.extract_features import extract_dataset_features
from src.train_gmm import train_gmm_models, save_models
from src.evaluate import predict

# 1. Load Features
print("Extracting MFCC features...")
X, y = extract_dataset_features("data/genres")

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train GMMs
print("\nTraining GMM models...")
models = train_gmm_models(X_train, y_train, n_components=8)

# 4. Evaluation
print("\nEvaluating...")

y_pred = []
for x in X_test:
    y_pred.append(predict(models, x))

y_pred = np.array(y_pred)

# ✔ Accuracy
accuracy = (y_pred == y_test).mean()
print("\n🎯 Accuracy:", accuracy)

# ✔ Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ✔ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
genres = np.unique(y_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=genres, yticklabels=genres)
plt.title("Confusion Matrix (GMM Genre Classifier)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 5. Save models
save_models(models)
