# Wine Quality ML & Web (wineML)

UCI Wine Quality 데이터셋(약 6,500건)으로 품질 **평점(0–10)** 예측(회귀) 모델을 학습하고, FastAPI + 단일 페이지 웹으로 예측하는 프로젝트입니다.

## 디렉터리 구조

```
JuSeok/
├── data/                    # vivino_wine.csv (Vivino 평점 분포 참고용), 기타 캐시
├── models/                  # 학습 후 생성 (레드·화이트 구분)
│   ├── red/                 # model.joblib, scaler.joblib, feature_order.json, samples.json
│   └── white/
├── notebooks/
│   ├── wine_ml_pipeline.ipynb           # 기존 178건 참고용
│   └── wine_quality_ml_pipeline.ipynb   # 대용량 품질 예측 파이프라인
├── api/
│   ├── main.py              # FastAPI 앱 (POST /predict, GET /samples)
│   └── static/
│       └── index.html       # 웹 UI
├── reports/                  # 성능평가 결과 (evaluate_model, benchmark_api 실행 시)
├── scripts/
│   ├── verify_model.py       # 모델 간단 검증 (회귀: R², MAE)
│   ├── evaluate_model.py    # ML 성능평가 (MSE, MAE, R²) + Vivino 참고
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
실행이 끝나면 `models/red/`와 `models/white/`에 각각 `model.joblib`, `scaler.joblib`, `feature_order.json`, `samples.json`이 저장됩니다.

- 데이터는 UCI에서 레드·화이트를 각각 로드한 뒤, **타입별로 별도 회귀 모델**을 학습합니다.
- 타깃은 **품질 평점(quality 0–10)** 연속값이며, Vivino 스타일 1–5는 API에서 변환해 제공합니다.

### 3. API 서버 기동

프로젝트 루트(`JuSeok/`)에서:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 웹에서 예측

브라우저에서 `http://localhost:8000` 에 접속합니다.

- **레드 / 화이트** 탭으로 와인 종류를 선택한 뒤, 11개 화학 성분을 입력하고 **예측** 버튼을 누르면 해당 타입 모델로 **예측 평점(0–10)** 및 Vivino 스타일(1–5)이 표시됩니다.
- **예시 와인**은 선택한 타입에 맞는 샘플만 표시됩니다.

## API 요약

- **POST /predict**  
  Body: `wine_type` ("red" | "white") + 11개 특징 JSON (`fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `ph`, `sulphates`, `alcohol`)  
  응답: `{ "wine_type": "red", "rating": 6.2, "rating_1_5": 3.48 }` (0–10 평점 및 Vivino 스타일 1–5)

- **GET /samples?type=red** 또는 **GET /samples?type=white**  
  쿼리 생략 시 레드+화이트 전체. 응답: 원본 스케일의 예시 와인 목록 (배열).

- **GET /feature_order**  
  응답: 특징 이름 순서 배열.

## 성능평가

- **ML 모델**: `python scripts/evaluate_model.py` — 레드·화이트 각각 UCI 테스트 세트 기준 **MSE, MAE, R²**를 출력하고 `reports/evaluation_red.json`, `reports/evaluation_white.json` (및 .txt)에 저장합니다. `data/vivino_wine.csv`가 있으면 Vivino 평점 분포 요약을 참고용으로 출력합니다.
- **API**: 서버 실행 후 `python scripts/benchmark_api.py [--url http://127.0.0.1:8000] [--n 100] [--type red|white]` — POST /predict를 N회 호출해 응답 시간·처리량을 측정하고 `reports/api_benchmark.json`에 저장합니다.

## Vivino 참고 데이터

- `data/vivino_wine.csv`: Vivino 소비자 평점 데이터(화학 성분 없음). 스케일·분포 비교용으로만 사용하며, 모델 학습/평가에는 UCI 데이터만 사용합니다.
- `python scripts/vivino_reference.py`: Vivino 평점 분포 요약 및 `reports/figures/vivino_rating_distribution.png` 생성(참고용).

**참고**: `notebooks/wine_quality_visualizations.ipynb`는 이전 3단계 분류 기준으로 작성되었습니다. 평점(회귀) 메트릭은 `scripts/evaluate_model.py`와 리포트 JSON을 사용하세요.
