# app.py
from flask import Flask, render_template, request, send_file
import joblib, numpy as np, pandas as pd, os
import lime.lime_tabular
import scipy.sparse as sp

app = Flask(__name__)

# -------------------- Load artifacts --------------------
REQUIRED = [
    "model_pipeline.pkl",
    "feature_names.pkl",
    "class_names.pkl",
    "raw_feature_order.pkl",
    "categorical_cols.pkl",
    "numeric_cols.pkl",
    "lime_training.npy"
]

for f in REQUIRED:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing required file: {f} — place it in the project root.")

pipe = joblib.load("model_pipeline.pkl")

feature_names_trans = joblib.load("feature_names.pkl")
class_names = joblib.load("class_names.pkl")
raw_order = joblib.load("raw_feature_order.pkl")
cat_cols = joblib.load("categorical_cols.pkl")
num_cols = joblib.load("numeric_cols.pkl")

X_tr_tf = np.load("lime_training.npy", allow_pickle=True)
if sp.issparse(X_tr_tf):
    X_tr_tf = X_tr_tf.toarray()
if X_tr_tf.ndim == 1:
    X_tr_tf = X_tr_tf.reshape(-1, len(feature_names_trans))

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_tr_tf,
    feature_names=[str(x) for x in feature_names_trans],
    class_names=[str(c) for c in class_names],
    mode="classification"
)

# -------------------- Predefined categorical options --------------------
CAT_OPTIONS = {
    "Gender": ["Male", "Female"],
    "family_history_with_overweight": ["yes", "no"],
    "FAVC": ["yes", "no"],
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "SMOKE": ["yes", "no"],
    "SCC": ["yes", "no"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
    "MTRANS": ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
}

# -------------------- Helpers --------------------
def generate_explanation(prediction, features):
    """Rule-based explanation similar to your liked file."""
    reasons = []

    if prediction == "Insufficient_Weight":
        if features.get("Weight", 0) < 50:
            reasons.append("low body weight compared to height")
        if features.get("CH2O", 3) < 2:
            reasons.append("low water intake")
        if features.get("FAF", 3) < 2:
            reasons.append("low physical activity")

    elif prediction == "Normal_Weight":
        reasons.append("balanced weight, diet, and lifestyle factors")

    elif prediction == "Overweight_Level_I":
        if features.get("FAVC") in [1, "yes"]:
            reasons.append("frequent consumption of high-calorie food")
        if features.get("FAF", 3) < 2:
            reasons.append("low physical activity")
        if features.get("CAEC") in ["Sometimes", "Frequently"]:
            reasons.append("snacking between meals")

    elif prediction == "Overweight_Level_II":
        reasons.append("higher weight than normal with frequent snacking and low exercise")

    elif prediction == "Obesity_Type_I":
        reasons.append("high weight combined with unhealthy eating and low activity")

    elif prediction == "Obesity_Type_II":
        reasons.append("very high weight and poor lifestyle habits")

    elif prediction == "Obesity_Type_III":
        reasons.append("extremely high weight with multiple risk factors")

    if not reasons:
        return f"The prediction is based on multiple lifestyle and health attributes."
    else:
        return f"Prediction is {prediction} because of " + ", ".join(reasons) + "."

def generate_recommendations(prediction):
    """Rule-based recommendations like your liked file."""
    recs = []
    if prediction == "Insufficient_Weight":
        recs = [
            "Increase calorie intake with nutrient-rich foods.",
            "Stay hydrated and maintain at least 2L water daily.",
            "Do light strength training to build muscle mass."
        ]
    elif prediction == "Normal_Weight":
        recs = [
            "Maintain balanced diet and hydration.",
            "Continue regular physical activity.",
            "Monitor lifestyle to prevent weight changes."
        ]
    elif "Overweight" in prediction:
        recs = [
            "Reduce consumption of high-calorie snacks.",
            "Increase daily physical activity (walking, sports, exercise).",
            "Track calories and maintain portion control."
        ]
    elif "Obesity" in prediction:
        recs = [
            "Consult a healthcare professional for a personalized plan.",
            "Focus on reducing sugary and processed foods.",
            "Adopt consistent exercise habits (cardio + strength training).",
            "Monitor sleep and stress levels."
        ]
    return recs

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        fields=raw_order,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_options=CAT_OPTIONS
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs
    data = {col: request.form.get(col, "").strip() for col in raw_order}

    # Convert numeric fields
    for c in num_cols:
        try:
            data[c] = float(data[c]) if data[c] != "" else np.nan
        except:
            pass

    user_df = pd.DataFrame([data], columns=raw_order)

    # Prediction
    pred = pipe.predict(user_df)[0]
    pred_proba = pipe.predict_proba(user_df)[0]

    # Transform input for LIME
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
    except Exception:
        pre, clf = pipe, pipe

    x_enc = pre.transform(user_df)
    if sp.issparse(x_enc):
        x_enc = x_enc.toarray()

    # LIME explanation
    exp = explainer.explain_instance(
        data_row=x_enc[0],
        predict_fn=clf.predict_proba,
        num_features=8
    )

    html_path = os.path.join("static", "lime_last.html")
    try:
        exp.save_to_file(html_path)
        lime_html_available = True
    except Exception:
        lime_html_available = False
        html_path = None

    # Combine explanations
    features_dict = {k: data[k] for k in raw_order}
    plain_text = generate_explanation(pred, features_dict)

    top_factors = [f"{f} ({w:+.2f})" for f, w in exp.as_list()[:6]]
    recs = generate_recommendations(pred)

    proba_pairs = [(str(cn), float(p)) for cn, p in zip(class_names, pred_proba)]

    return render_template(
        "result.html",
        prediction=str(pred),
        proba=proba_pairs,
        explanation_plain=plain_text,
        top_factors=top_factors,
        recommendations=recs,
        lime_html_path=html_path if lime_html_available else None
    )

@app.route("/lime_view")
def lime_view():
    path = "static/lime_last.html"
    if os.path.exists(path):
        return send_file(path)
    return "No LIME explanation was generated yet. Make a prediction first."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
