"""
Wine Quality ML model performance evaluation.
Loads saved model/scaler, runs evaluation on the same train/test split as training,
outputs accuracy, F1-macro, confusion matrix, and per-class classification report.
Run from project root: python scripts/evaluate_model.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Same as notebook: quality -> 0/1/2
def quality_to_class(q):
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2


def main():
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    feature_order = json.loads((MODELS_DIR / "feature_order.json").read_text())
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.concat(
        [
            pd.read_csv(url_red, sep=";"),
            pd.read_csv(url_white, sep=";"),
        ],
        ignore_index=True,
    )
    df["y"] = df["quality"].map(quality_to_class)
    X = df[feature_order]
    y = df["y"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    # Console output
    print("=== Wine Quality Model Evaluation ===\n")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"F1-macro:  {f1_macro:.4f}\n")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"  labels: {class_names}")
    print(f"  {cm}\n")
    print("Classification report:")
    print(report_str)

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (int, float)):
            return obj
        return obj

    evaluation = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": to_serializable(report_dict),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_report.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text_report = [
        "Wine Quality Model Evaluation",
        "",
        f"Accuracy:  {accuracy:.4f}",
        f"F1-macro: {f1_macro:.4f}",
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        f"  labels: {class_names}",
        str(cm.tolist()),
        "",
        "Classification report:",
        report_str,
    ]
    (REPORTS_DIR / "evaluation_report.txt").write_text(
        "\n".join(text_report), encoding="utf-8"
    )
    print(f"\nReports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
