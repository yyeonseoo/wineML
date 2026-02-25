"""
Verify Wine Quality model: load model/scaler, run predictions on samples,
and optionally report test-set accuracy. Run from project root: python scripts/verify_model.py
"""
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_order.json") as f:
        feature_order = json.load(f)
    with open(MODELS_DIR / "class_names.json") as f:
        class_names = json.load(f)
    with open(MODELS_DIR / "samples.json") as f:
        samples = json.load(f)

    print("=== Predictions on 5 example wines ===")
    for i, s in enumerate(samples[:5]):
        X = scaler.transform(pd.DataFrame([s], columns=feature_order))
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        assert 0 <= pred <= 2
        assert abs(probs.sum() - 1.0) < 1e-6
        print(f"  Sample {i+1}: {class_names[pred]}, probs sum={probs.sum():.4f}")

    print("\n=== Test set (holdout from UCI Wine Quality) ===")
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.concat([
        pd.read_csv(url_red, sep=";"),
        pd.read_csv(url_white, sep=";"),
    ], ignore_index=True)
    df["y"] = df["quality"].map(lambda q: 0 if q <= 4 else 1 if q <= 6 else 2)
    X = df[feature_order]
    y = df["y"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nModel verification OK.")

if __name__ == "__main__":
    main()
