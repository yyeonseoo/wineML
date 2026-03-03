"""
Vivino 평점 분포 참고: data/vivino_wine.csv를 로드해 wine_type별 rating 요약·시각화.
성능평가와 무관한 참고용. Run from project root: python scripts/vivino_reference.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def main():
    import pandas as pd

    vivino_path = DATA_DIR / "vivino_wine.csv"
    if not vivino_path.exists():
        print(f"Not found: {vivino_path}")
        return

    df = pd.read_csv(vivino_path)
    if "rating" not in df.columns or "wine_type" not in df.columns:
        print("vivino_wine.csv must have columns 'rating' and 'wine_type'")
        return

    df["wine_type_clean"] = df["wine_type"].str.strip().str.lower()
    red_white = df[df["wine_type_clean"].isin(["red", "white"])].copy()

    print("=== Vivino 평점 분포 (참고용) ===\n")
    summary = red_white.groupby("wine_type_clean")["rating"].agg(["count", "mean", "std", "min", "max"])
    print(summary.to_string())
    print()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        for wtype in ["red", "white"]:
            sub = red_white[red_white["wine_type_clean"] == wtype]["rating"]
            if len(sub):
                sub.hist(ax=ax, bins=20, alpha=0.5, label=wtype, edgecolor="black")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Vivino rating distribution (Red vs White, reference)")
        ax.legend()
        plt.tight_layout()
        out = FIG_DIR / "vivino_rating_distribution.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Plot skip: {e}")


if __name__ == "__main__":
    main()
