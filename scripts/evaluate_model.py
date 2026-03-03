"""
Wine Quality ML model performance evaluation (red/white, regression).
Loads saved regressor/scaler per type, runs evaluation on same split as training,
outputs MSE, MAE, R². Optionally prints Vivino reference (data/vivino_wine.csv).
Run from project root: python scripts/evaluate_model.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"


def main():
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    URL_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    urls = {"red": URL_RED, "white": URL_WHITE}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for wine_type in ("red", "white"):
        d = MODELS_DIR / wine_type
        if not d.exists():
            print(f"Skip {wine_type} (no {d})")
            continue

        feature_order = json.loads((d / "feature_order.json").read_text())
        model = joblib.load(d / "model.joblib")
        scaler = joblib.load(d / "scaler.joblib")

        df = pd.read_csv(urls[wine_type], sep=";")
        X = df[feature_order]
        y = df["quality"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        mse = float(mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        print(f"\n=== Wine Quality Model Evaluation ({wine_type.upper()}) [Regression] ===\n")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}\n")

        evaluation = {
            "wine_type": wine_type,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
        (REPORTS_DIR / f"evaluation_{wine_type}.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        text_report = [
            f"Wine Quality Model Evaluation ({wine_type}) [Regression]",
            "",
            f"MSE:  {mse:.4f}",
            f"MAE:  {mae:.4f}",
            f"R²:   {r2:.4f}",
        ]
        (REPORTS_DIR / f"evaluation_{wine_type}.txt").write_text(
            "\n".join(text_report), encoding="utf-8"
        )

    print(f"Reports saved to {REPORTS_DIR}/")

    # Vivino reference: summary from data/vivino_wine.csv if present
    vivino_path = DATA_DIR / "vivino_wine.csv"
    if vivino_path.exists():
        try:
            try:
                vivino = pd.read_csv(vivino_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                vivino = pd.read_csv(vivino_path, encoding="latin-1", on_bad_lines="skip")
            if "rating" in vivino.columns and "wine_type" in vivino.columns:
                vivino["wine_type_lower"] = vivino["wine_type"].astype(str).str.strip().str.lower()
                vivino["rating"] = pd.to_numeric(vivino["rating"], errors="coerce")
                vv = vivino.dropna(subset=["rating"]).loc[vivino["wine_type_lower"].isin(["red", "white"])]
                ref = vv.groupby("wine_type_lower")["rating"].agg(["mean", "std", "count"])
                print("\n참고: Vivino 평점 분포 (data/vivino_wine.csv)")
                print(ref.to_string())
        except Exception as e:
            print(f"\nVivino 참고 로드 실패: {e}")


if __name__ == "__main__":
    main()
