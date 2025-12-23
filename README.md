# 낙상 감지 시스템 (Fall Detection System)

딥러닝 프로젝트 - 이미지 기반 낙상 감지 시스템 베이스라인

## 📋 프로젝트 개요

본 프로젝트는 이미지에서 사람의 **스켈레톤(Skeleton)을 추출**하고, 추출된 스켈레톤 특징을 사용하여 **낙상 여부를 감지**하는 머신러닝 기반 베이스라인 시스템입니다.

### 주요 특징

- **포즈 추정(Pose Estimation)**: MediaPipe를 활용한 33개의 신체 키포인트 추출
- **특징 추출**: 스켈레톤 좌표 기반 특징 벡터 생성
- **머신러닝 베이스라인**: Random Forest, SVM을 사용한 이진 분류 모델
- **간단한 API**: 학습, 예측, 시각화를 위한 직관적인 인터페이스

## 🏗️ 프로젝트 구조

```
Deeplearning_Project_FallDetection/
├── src/
│   ├── skeleton_extractor.py   # 스켈레톤 추출 모듈
│   ├── baseline_model.py        # 베이스라인 ML 모델
│   ├── data_processor.py        # 데이터 전처리
│   ├── train.py                 # 학습 스크립트
│   └── predict.py               # 예측 스크립트
├── data/
│   ├── raw/                     # 원본 이미지 데이터
│   │   ├── normal/             # 정상 자세 이미지
│   │   └── fall/               # 낙상 자세 이미지
│   └── processed/              # 전처리된 데이터
├── notebooks/                   # Jupyter 노트북
├── requirements.txt            # 의존성 패키지
└── README.md
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Duho0120/Deeplearning_Project_FallDetection.git
cd Deeplearning_Project_FallDetection

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터는 다음과 같은 구조로 준비해야 합니다:

```
data/raw/
├── normal/          # 정상 자세 이미지들
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── fall/           # 낙상 자세 이미지들
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

### 3. 모델 학습

```bash
# 기본 학습 (Random Forest)
python src/train.py --data_dir data/raw --save_model --save_processed

# SVM 모델 사용
python src/train.py --data_dir data/raw --model_type svm --save_model

# 전처리된 데이터 사용
python src/train.py --use_processed --processed_data_path data/processed/dataset.npz --save_model
```

### 4. 예측

```bash
# 단일 이미지 예측
python src/predict.py --image_path path/to/image.jpg --model_path src/models/baseline_model.pkl

# 시각화 포함 예측
python src/predict.py --image_path path/to/image.jpg --visualize --output_path output.jpg
```

## 📊 베이스라인 모델 상세

### 스켈레톤 추출

MediaPipe Pose 모델을 사용하여 다음과 같은 특징을 추출합니다:

- **33개의 키포인트**: 신체의 주요 관절 및 부위
- **4차원 정보**: x, y, z 좌표 + visibility (총 132차원)
- **특징 벡터**: 키포인트 좌표, 신체 중심, 각도 등

### 머신러닝 모델

두 가지 베이스라인 모델을 제공합니다:

1. **Random Forest Classifier**
   - n_estimators: 100
   - max_depth: 10
   - 빠른 학습 속도와 안정적인 성능

2. **Support Vector Machine (SVM)**
   - kernel: RBF
   - 높은 분류 정확도
   - 중소규모 데이터셋에 적합

### 평가 지표

모델은 다음 지표로 평가됩니다:

- **Accuracy (정확도)**: 전체 예측 중 올바른 예측의 비율
- **Precision (정밀도)**: 낙상으로 예측한 것 중 실제 낙상의 비율
- **Recall (재현율)**: 실제 낙상 중 올바르게 탐지한 비율
- **F1 Score**: Precision과 Recall의 조화평균
- **Confusion Matrix**: 예측 결과 상세 분석

## 📈 사용 예제

### Python 코드에서 사용

```python
from src.skeleton_extractor import SkeletonExtractor
from src.baseline_model import BaselineModel

# 스켈레톤 추출
extractor = SkeletonExtractor()
keypoints = extractor.extract_keypoints('image.jpg')
features = extractor.extract_features(keypoints)

# 모델 로드 및 예측
model = BaselineModel()
model.load_model('src/models/baseline_model.pkl')
prediction = model.predict(features.reshape(1, -1))

print("낙상" if prediction[0] == 1 else "정상")
```

## 🔧 커스터마이징

### 새로운 특징 추가

`skeleton_extractor.py`의 `extract_features()` 메서드를 수정하여 새로운 특징을 추가할 수 있습니다:

```python
def extract_features(self, keypoints):
    # 기존 특징
    features = []
    
    # 새로운 특징 추가
    # 예: 신체 각도, 거리 비율 등
    
    return np.array(features)
```

### 모델 하이퍼파라미터 조정

`baseline_model.py`에서 모델 파라미터를 수정할 수 있습니다:

```python
self.model = RandomForestClassifier(
    n_estimators=200,      # 트리 개수 증가
    max_depth=15,          # 깊이 증가
    min_samples_split=5,   # 분할 최소 샘플 수
    random_state=42
)
```

## 📝 향후 개선 방향

- [ ] 시계열 데이터 처리 (비디오 기반 낙상 감지)
- [ ] 딥러닝 모델 적용 (LSTM, CNN 등)
- [ ] 실시간 낙상 감지 시스템 구현
- [ ] 데이터 증강 기법 적용
- [ ] 더 다양한 포즈 특징 추출
- [ ] 앙상블 모델 구현

## 🤝 기여

프로젝트에 기여하고 싶으시다면 Pull Request를 보내주세요!

## 📄 라이선스

MIT License

## 📧 문의

프로젝트에 대한 문의사항은 이슈를 등록해주세요.

---

**Note**: 이 프로젝트는 머신러닝 베이스라인을 제공하며, 실제 운영 환경에서는 더 정교한 딥러닝 모델과 검증이 필요합니다.