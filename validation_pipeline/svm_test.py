import pickle
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load validation features ---
print("Loading validation features...")
with open('features_val.pkl', 'rb') as f:
    data = pickle.load(f)

X_val = data['features']
y_val = data['labels']

print(f"Features shape: {X_val.shape}")
print(f"Labels shape: {y_val.shape}")
print(f"Number of classes: {len(np.unique(y_val))}")
print(f"Classes: {sorted(np.unique(y_val))}\n")

# --- Load trained SVM model ---
print("Loading trained SVM model...")
svm = joblib.load('svm_asl_model.pkl')

# --- Evaluate ---
print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)

val_score = svm.score(X_val, y_val)
print(f"\nValidation Accuracy: {val_score*100:.2f}%")

# Predictions
y_pred = svm.predict(X_val)

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_val, y_pred)
labels_sorted = sorted(np.unique(y_val))

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_sorted, yticklabels=labels_sorted,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - ASL Validation (SVM RBF)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_val.png', dpi=150)
plt.show()

# --- Per-class accuracy ---
print("\nPer-class accuracy:")
for label in labels_sorted:
    mask = y_val == label
    if mask.sum() > 0:
        acc = accuracy_score(y_val[mask], y_pred[mask])
        print(f"  {label}: {acc*100:.1f}%")
