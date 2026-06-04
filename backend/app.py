from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, base64, io, os, joblib
from PIL import Image

app  = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

_clf    = joblib.load(os.path.join(MODEL_DIR, "crop_model.joblib"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
_le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

TREATMENTS = {
    "Healthy":                        "No treatment needed. Continue regular watering and fertiliser schedule.",
    "Apple___Apple_scab":             "Apply fungicide at bud break. Remove fallen leaves. Prune infected branches.",
    "Apple___Black_rot":              "Prune cankers. Apply captan or thiophanate-methyl fungicide.",
    "Apple___Cedar_apple_rust":       "Apply myclobutanil fungicide. Remove nearby juniper hosts if possible.",
    "Apple___healthy":                "Plant is healthy. Maintain regular care routine.",
    "Blueberry___healthy":            "Plant is healthy. Maintain regular care routine.",
    "Cherry___Powdery_mildew":        "Apply sulfur-based fungicide. Improve air circulation. Avoid overhead watering.",
    "Cherry___healthy":               "Plant is healthy. Maintain regular care routine.",
    "Corn___Cercospora_leaf_spot":    "Apply strobilurin fungicide. Rotate crops. Use resistant hybrids.",
    "Corn___Common_rust":             "Apply triazole fungicide. Plant resistant varieties.",
    "Corn___Northern_Leaf_Blight":    "Apply fungicide at early tasseling. Use resistant hybrids.",
    "Corn___healthy":                 "Plant is healthy. Maintain regular care routine.",
    "Grape___Black_rot":              "Apply mancozeb or myclobutanil. Remove mummified berries.",
    "Grape___Esca_(Black_Measles)":   "Prune infected wood. Apply wound sealant. No chemical cure available.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicide. Remove infected leaves.",
    "Grape___healthy":                "Plant is healthy. Maintain regular care routine.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees. Control psyllid vectors. No cure available.",
    "Peach___Bacterial_spot":         "Apply copper sprays in spring. Use resistant varieties.",
    "Peach___healthy":                "Plant is healthy. Maintain regular care routine.",
    "Pepper,_bell___Bacterial_spot":  "Apply copper hydroxide. Avoid overhead irrigation. Use certified seeds.",
    "Pepper,_bell___healthy":         "Plant is healthy. Maintain regular care routine.",
    "Potato___Early_blight":          "Apply chlorothalonil or mancozeb. Remove infected foliage.",
    "Potato___Late_blight":           "Apply mancozeb immediately. Destroy infected plants. Avoid overhead watering.",
    "Potato___healthy":               "Plant is healthy. Maintain regular care routine.",
    "Raspberry___healthy":            "Plant is healthy. Maintain regular care routine.",
    "Soybean___healthy":              "Plant is healthy. Maintain regular care routine.",
    "Squash___Powdery_mildew":        "Apply potassium bicarbonate or sulfur fungicide. Improve airflow.",
    "Strawberry___Leaf_scorch":       "Remove infected leaves. Apply copper fungicide. Avoid wetting foliage.",
    "Strawberry___healthy":           "Plant is healthy. Maintain regular care routine.",
    "Tomato___Bacterial_spot":        "Apply copper-based bactericide. Use disease-free seeds.",
    "Tomato___Early_blight":          "Apply chlorothalonil every 7 days. Remove lower infected leaves.",
    "Tomato___Late_blight":           "Apply mancozeb or chlorothalonil immediately. Destroy infected plants.",
    "Tomato___Leaf_Mold":             "Reduce humidity. Apply copper fungicide. Improve greenhouse ventilation.",
    "Tomato___Septoria_leaf_spot":    "Remove infected leaves. Apply chlorothalonil. Avoid overhead watering.",
    "Tomato___Spider_mites":          "Apply insecticidal soap or neem oil. Increase humidity. Remove webbing.",
    "Tomato___Target_Spot":           "Apply azoxystrobin fungicide. Rotate crops. Remove plant debris.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Use resistant varieties. Remove infected plants.",
    "Tomato___Tomato_mosaic_virus":   "Remove infected plants. Sanitise tools. Control aphid vectors.",
    "Tomato___healthy":               "Plant is healthy. Maintain regular care routine.",
}

def extract_features(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert("RGB").resize((128, 128))
    arr     = np.array(img_rgb, dtype=float)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray    = 0.299*r + 0.587*g + 0.114*b
    total   = r.mean() + g.mean() + b.mean() + 1e-6

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


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        raw     = base64.b64decode(b64.split(",")[-1])
        img     = Image.open(io.BytesIO(raw))
        feats   = extract_features(img).reshape(1, -1)
        feats_s = _scaler.transform(feats)

        proba  = _clf.predict_proba(feats_s)[0]
        top3   = np.argsort(proba)[::-1][:3]

        primary_class = _le.classes_[top3[0]]
        confidence    = round(float(proba[top3[0]]) * 100, 1)

        is_healthy = "healthy" in primary_class.lower()
        severity   = "None" if is_healthy else ("High" if confidence > 80 else "Moderate")
        color      = "#4ade80" if is_healthy else "#f97316"

        return jsonify({
            "success":    True,
            "disease":    primary_class,
            "confidence": confidence,
            "severity":   severity,
            "color":      color,
            "treatment":  TREATMENTS.get(primary_class, "Consult an agronomist for specific treatment advice."),
            "top_predictions": [
                {"disease": _le.classes_[i], "probability": round(float(proba[i]) * 100, 1)}
                for i in top3
            ]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type":   "RandomForestClassifier",
        "n_estimators": 500,
        "classes":      list(_le.classes_),
        "feature_dim":  45
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)