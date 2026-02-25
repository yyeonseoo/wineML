"""
Benchmark Wine Quality API: POST /predict latency and throughput.
Run with server up: uvicorn api.main:app --host 127.0.0.1 --port 8000
Then: python scripts/benchmark_api.py [--url URL] [--n N]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Map sample keys (from samples.json) to API body keys
SAMPLE_TO_API = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "chlorides": "chlorides",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "density": "density",
    "pH": "ph",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
}


def load_payload():
    samples_path = MODELS_DIR / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"{samples_path} not found")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("samples.json is empty")
    raw = samples[0]
    payload = {}
    for k, api_key in SAMPLE_TO_API.items():
        if k in raw:
            payload[api_key] = raw[k]
    return payload


def percentile(sorted_times_ms, p):
    if not sorted_times_ms:
        return 0.0
    k = (len(sorted_times_ms) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_times_ms) else f
    return sorted_times_ms[f] + (k - f) * (sorted_times_ms[c] - sorted_times_ms[f])


def main():
    parser = argparse.ArgumentParser(description="Benchmark POST /predict")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of sequential requests (default: 100)",
    )
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        raise SystemExit(1)

    payload = load_payload()
    url = args.url.rstrip("/") + "/predict"
    n = args.n

    times_ms = []
    errors = 0
    start_wall = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            r.raise_for_status()
        except Exception as e:
            errors += 1
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
    total_wall_s = time.perf_counter() - start_wall

    print("=== API Benchmark (POST /predict) ===\n")
    print(f"URL:       {url}")
    print(f"Requests:  {n} (success: {len(times_ms)}, errors: {errors})")
    if not times_ms:
        print("No successful requests. Check server and payload.")
        raise SystemExit(1)

    times_ms.sort()
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms = percentile(times_ms, 95)
    p99_ms = percentile(times_ms, 99)
    req_per_s = len(times_ms) / total_wall_s if total_wall_s > 0 else 0

    print(f"\nLatency (ms):  mean={mean_ms:.2f}  median={median_ms:.2f}  p95={p95_ms:.2f}  p99={p99_ms:.2f}")
    print(f"Throughput:    {req_per_s:.1f} req/s")
    print(f"Total time:   {total_wall_s:.2f}s")

    report = {
        "url": url,
        "n_requests": n,
        "n_success": len(times_ms),
        "n_errors": errors,
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "median": round(median_ms, 2),
            "p95": round(p95_ms, 2),
            "p99": round(p99_ms, 2),
        },
        "throughput_req_per_s": round(req_per_s, 2),
        "total_time_s": round(total_wall_s, 2),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "api_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to {REPORTS_DIR}/api_benchmark.json")


if __name__ == "__main__":
    main()
