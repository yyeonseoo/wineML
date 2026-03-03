"""
Verify Wine Quality model (red/white, regression): load regressor/scaler per type,
run predictions on samples, report test-set R²/MAE per type.
Run from project root: python scripts/verify_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    URL_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    for wine_type in ("red", "white"):
        d = MODELS_DIR / wine_type
        if not d.exists():
            print(f"Skip {wine_type} (no {d})")
            continue
        model = joblib.load(d / "model.joblib")
        scaler = joblib.load(d / "scaler.joblib")
        with open(d / "feature_order.json") as f:
            feature_order = json.load(f)
        with open(d / "samples.json") as f:
            samples = json.load(f)

        print(f"\n=== {wine_type.upper()} ===\nPredictions on 5 example wines:")
        for i, s in enumerate(samples[:5]):
            row = {k: v for k, v in s.items() if k in feature_order}
            X = scaler.transform(pd.DataFrame([row], columns=feature_order))
            pred = float(model.predict(X)[0])
            rating_1_5 = round(1 + (pred / 10.0) * 4, 2)
            print(f"  Sample {i+1}: rating={pred:.2f} (Vivino 스타일 {rating_1_5}/5)")

        url = URL_RED if wine_type == "red" else URL_WHITE
        df = pd.read_csv(url, sep=";")
        X = df[feature_order]
        y = df["quality"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
        print(f"\nTest set ({wine_type}): R²={r2_score(y_test, y_pred):.4f}, MAE={mean_absolute_error(y_test, y_pred):.4f}")

    print("\nModel verification OK.")


if __name__ == "__main__":
    main()
