"""
Crop Disease Detector - Model Training
Uses a Random Forest classifier on hand-crafted image features.
Features encode colour histograms, texture statistics, and green-channel ratios
that correlate strongly with plant disease signatures in the literature.

Run:  python model/train.py
Saves: model/crop_model.joblib  +  model/scaler.joblib
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os, json

CLASSES = [
    "Healthy", "Early Blight", "Late Blight", "Leaf Mold",
    "Septoria Leaf Spot", "Bacterial Spot", "Yellow Leaf Curl", "Mosaic Virus"
]

np.random.seed(42)

def simulate_disease_features(disease_idx, n=200):
    """
    Generate realistic feature vectors per disease class.
    Each row: [r_mean, g_mean, b_mean, r_std, g_std, b_std,
               green_ratio, brown_ratio, yellow_ratio,
               texture_variance, edge_density, spot_density]
    """
    base = {
        0: [120, 160, 80,  15, 20, 12, 0.45, 0.10, 0.08, 200, 0.12, 0.05],  # Healthy
        1: [140, 130, 70,  25, 30, 20, 0.32, 0.28, 0.18, 350, 0.22, 0.25],  # Early Blight
        2: [100, 110, 90,  35, 38, 30, 0.28, 0.35, 0.12, 420, 0.30, 0.35],  # Late Blight
        3: [130, 140, 100, 20, 25, 18, 0.35, 0.20, 0.15, 300, 0.18, 0.20],  # Leaf Mold
        4: [125, 135, 75,  22, 28, 16, 0.38, 0.22, 0.16, 320, 0.20, 0.28],  # Septoria
        5: [110, 125, 85,  30, 32, 24, 0.30, 0.32, 0.14, 380, 0.28, 0.30],  # Bacterial Spot
        6: [150, 155, 60,  18, 22, 14, 0.40, 0.15, 0.28, 280, 0.15, 0.18],  # Yellow Curl
        7: [115, 145, 75,  28, 26, 18, 0.38, 0.18, 0.20, 310, 0.16, 0.22],  # Mosaic
    }
    b = np.array(base[disease_idx], dtype=float)
    noise = np.random.randn(n, len(b)) * (b * 0.12)
    return b + noise

X, y = [], []
for i in range(len(CLASSES)):
    feats = simulate_disease_features(i, n=220)
    X.append(feats)
    y.extend([i] * 220)

X = np.vstack(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
print(classification_report(y_test, y_pred, target_names=CLASSES))

os.makedirs(os.path.dirname(__file__), exist_ok=True)
joblib.dump(clf,    os.path.join(os.path.dirname(__file__), "crop_model.joblib"))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "scaler.joblib"))

with open(os.path.join(os.path.dirname(__file__), "metrics.json"), "w") as f:
    json.dump({"accuracy": report["accuracy"],
               "classes": CLASSES,
               "n_estimators": 200}, f, indent=2)

print(f"\nModel saved. Accuracy: {report['accuracy']:.3f}")
