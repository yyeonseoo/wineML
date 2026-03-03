# Wine Quality ML & Web (wineML)

UCI Wine Quality 데이터셋(약 6,500건)으로 품질 등급(poor / normal / good) 예측 모델을 학습하고, FastAPI + 단일 페이지 웹으로 예측하는 프로젝트입니다.

## 디렉터리 구조

```
JuSeok/
├── data/                    # (선택) CSV 캐시
├── models/                  # 학습 후 생성: model.joblib, scaler.joblib, samples.json 등
├── notebooks/
│   ├── wine_ml_pipeline.ipynb           # 기존 178건 참고용
│   └── wine_quality_ml_pipeline.ipynb   # 대용량 품질 예측 파이프라인
├── api/
│   ├── main.py              # FastAPI 앱 (POST /predict, GET /samples)
│   └── static/
│       └── index.html       # 웹 UI
├── reports/                  # 성능평가 결과 (evaluate_model, benchmark_api 실행 시)
├── scripts/
│   ├── verify_model.py       # 모델 간단 검증
│   ├── evaluate_model.py    # ML 성능평가 (정확도, F1, 혼동행렬, classification report)
│   └── benchmark_api.py     # API 응답시간·처리량 벤치마크
├── requirements.txt
└── README.md
```

## 실행 순서

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습 (노트북 실행)

`notebooks/wine_quality_ml_pipeline.ipynb`를 위에서부터 순서대로 실행합니다.  
실행이 끝나면 프로젝트 루트에 `models/` 폴더가 생기고, 그 안에 `model.joblib`, `scaler.joblib`, `feature_order.json`, `class_names.json`, `samples.json`이 저장됩니다.

- 데이터는 UCI URL에서 직접 로드합니다 (red + white 합침).
- 품질(quality 0~10)을 구간별로 나누어 분류( poor / normal / good )로 학습합니다.

### 3. API 서버 기동

프로젝트 루트(`JuSeok/`)에서:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 웹에서 예측

브라우저에서 `http://localhost:8000` 에 접속합니다.

- 11개 화학 성분 값을 입력한 뒤 **예측** 버튼을 누르면 품질 등급과 확률이 표시됩니다.
- **예시 와인 불러오기** 버튼으로 저장된 샘플 값으로 폼을 채울 수 있습니다.

## API 요약

- **POST /predict**  
  Body: 11개 특징 JSON (예: `fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `ph`, `sulphates`, `alcohol`)  
  응답: `{ "class": "normal", "class_index": 1, "probabilities": { ... } }`

- **GET /samples**  
  응답: 원본 스케일의 예시 와인 목록 (배열).

- **GET /feature_order**  
  응답: 특징 이름 순서 배열.

## 성능평가

- **ML 모델**: `python scripts/evaluate_model.py` — 테스트 세트 기준 정확도, F1-macro, 혼동행렬, 클래스별 리포트를 콘솔에 출력하고 `reports/evaluation_report.json`, `reports/evaluation_report.txt`에 저장합니다.
- **API**: 서버 실행 후 `python scripts/benchmark_api.py [--url http://127.0.0.1:8000] [--n 100]` — POST /predict를 N회 호출해 응답 시간(평균/중앙값/p95/p99)과 처리량(req/s)을 측정하고 `reports/api_benchmark.json`에 저장합니다.
