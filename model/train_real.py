import os, json, random
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

DATA_DIR  = r"D:\main_projects\crop_disease_detector\data\raw\train"
MODEL_DIR = r"D:\main_projects\crop_disease_detector\model"

IMG_SIZE          = (128, 128)
SAMPLES_PER_CLASS = 500

def extract_features(img_path):
    try:
        img  = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr  = np.array(img, dtype=float)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        gray = 0.299*r + 0.587*g + 0.114*b
        total = r.mean() + g.mean() + b.mean() + 1e-6

        h_bins = np.histogram(gray.flatten(), bins=16, range=(0,255))[0] / gray.size

        gx   = np.abs(np.diff(gray, axis=1))
        gy   = np.abs(np.diff(gray, axis=0))
        edge = (gx.mean() + gy.mean()) / 255.0

        r2, g2, b2 = arr[::2,::2,0], arr[::2,::2,1], arr[::2,::2,2]
        r4, g4, b4 = arr[::4,::4,0], arr[::4,::4,1], arr[::4,::4,2]

        feats = [
            r.mean()/255, g.mean()/255, b.mean()/255,
            r.std()/255,  g.std()/255,  b.std()/255,
            g.mean()/total,
            ((r>100)&(g<120)&(b<80)).mean(),
            ((r>150)&(g>140)&(b<100)).mean(),
            np.var(np.diff(gray,axis=1))/1000,
            edge,
            (np.abs(gray - gray.mean()) > 2*gray.std()).mean(),
            r2.mean()/255, g2.mean()/255, b2.mean()/255,
            r2.std()/255,  g2.std()/255,  b2.std()/255,
            r4.mean()/255, g4.mean()/255, b4.mean()/255,
            np.percentile(gray,25)/255, np.percentile(gray,50)/255,
            np.percentile(gray,75)/255,
            (gray < 50).mean(), (gray > 200).mean(),
            np.abs(r - g).mean()/255,
            np.abs(g - b).mean()/255,
            np.abs(r - b).mean()/255,
        ]
        feats.extend(h_bins.tolist())
        return np.array(feats)
    except:
        return None

classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

print(f"Classes found: {len(classes)}")

X, y = [], []
random.seed(42)

for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    files   = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files   = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
    extracted = 0
    for fname in files:
        feats = extract_features(os.path.join(cls_dir, fname))
        if feats is not None:
            X.append(feats)
            y.append(cls)
            extracted += 1
    print(f"  {cls}: {extracted}")

X   = np.array(X)
le  = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"\nTotal: {len(X)} samples, {len(classes)} classes")
print(f"Feature dimensions: {X.shape[1]}")

scaler = StandardScaler()
Xs     = scaler.fit_transform(X)

print("Training... (15-25 min)")

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

scores = cross_val_score(clf, Xs, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

clf.fit(Xs, y_enc)

joblib.dump(clf,    os.path.join(MODEL_DIR, "crop_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({
        "cv_accuracy": round(scores.mean(), 4),
        "cv_std":      round(scores.std(), 4),
        "n_classes":   len(classes),
        "n_samples":   len(X),
        "feature_dim": int(X.shape[1]),
        "classes":     list(le.classes_),
        "model":       "RandomForest-500trees-45features"
    }, f, indent=2)

print(f"Done. Accuracy: {scores.mean():.1%}")
print(f"Feature dim: {X.shape[1]}")