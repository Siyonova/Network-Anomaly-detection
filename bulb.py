import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from lightgbm import early_stopping
import time


# Load your data
df = pd.read_csv('vae_iso_labeled.csv')
X = df.drop('label', axis=1)
y = (df['label'] == -1).astype(int)  # 1 = anomaly, 0 = normal

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

# Parameters tuning for more features and complexity
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 20,  # controls logging frequency, set -1 for silent
    'max_depth': -1
}

bst = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    callbacks=[early_stopping(stopping_rounds=10)],
    num_boost_round=500
)

start = time.time()
lgbm_preds = bst.predict(X, num_iteration=bst.best_iteration)
end = time.time()
print(f"LightGBM prediction time: {end - start:.4f} seconds for {len(X)} samples")



# Predict on validation set
y_pred_proba = bst.predict(X_valid, num_iteration=bst.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

# Metrics calculation
acc = accuracy_score(y_valid, y_pred)
prec = precision_score(y_valid, y_pred)
rec = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_pred_proba)
cm = confusion_matrix(y_valid, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Normal", "Anomaly"])
plt.yticks(tick_marks, ["Normal", "Anomaly"])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_valid, y_pred_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Feature Importance
feature_importances = bst.feature_importance()
feature_names = bst.feature_name()
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_idx])
plt.xticks(range(len(feature_importances)), np.array(feature_names)[sorted_idx], rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
