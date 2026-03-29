import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load features ---
print("Loading features...")
with open('features_mask_edges.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['features']  # Already normalized
y = data['labels']

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {sorted(np.unique(y))}\n")

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensures balanced split
)

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}\n")

# --- Train SVM with RBF kernel ---
print("Training SVM with RBF kernel...")

svm = SVC(
    kernel='rbf',
    C=10.0,           # Regularization parameter
    gamma='scale',    # Kernel coefficient
    random_state=42,
    verbose=True,
    max_iter=1000     # Limit iterations if needed
)

svm.fit(X_train, y_train)

# --- Evaluate ---
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_score*100:.2f}%")
print(f"Test Accuracy: {test_score*100:.2f}%")

# Predictions
y_pred = svm.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
labels_sorted = sorted(np.unique(y))

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels_sorted, yticklabels=labels_sorted,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - ASL Recognition (SVM RBF)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# --- Per-class accuracy ---
print("\nPer-class accuracy:")
for label in labels_sorted:
    mask = y_test == label
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {label}: {acc*100:.1f}%")

# --- Save model ---
import joblib
joblib.dump(svm, 'svm_asl_model.pkl')
print("\nModel saved to svm_asl_model.pkl")
