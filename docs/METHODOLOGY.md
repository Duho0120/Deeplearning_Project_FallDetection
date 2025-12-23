# 낙상 감지 베이스라인 모델 방법론

## 1. 프로젝트 개요

본 프로젝트는 이미지 기반 낙상 감지 시스템의 **베이스라인 모델**을 구축합니다. 간단한 머신러닝 접근 방식을 통해 낙상 감지 문제의 기본 성능을 평가하고, 향후 딥러닝 모델과의 비교를 위한 기준점을 제공합니다.

## 2. 시스템 아키텍처

### 전체 파이프라인

```
입력 이미지 → 스켈레톤 추출 → 특징 추출 → ML 분류기 → 낙상/정상 판단
```

### 주요 컴포넌트

1. **스켈레톤 추출기 (Skeleton Extractor)**
   - MediaPipe Pose 모델 사용
   - 33개의 신체 키포인트 추출

2. **특징 추출기 (Feature Extractor)**
   - 키포인트 좌표 정규화
   - 신체 각도 및 거리 계산
   - 특징 벡터 생성

3. **베이스라인 분류기 (Baseline Classifier)**
   - Random Forest
   - Support Vector Machine (SVM)

## 3. 스켈레톤 추출 방법

### MediaPipe Pose

Google의 MediaPipe를 활용하여 실시간으로 사람의 포즈를 추정합니다.

#### 추출되는 33개 키포인트

- **얼굴** (5개): 코, 양쪽 눈, 양쪽 귀
- **상체** (6개): 양쪽 어깨, 팔꿈치, 손목
- **몸통** (4개): 양쪽 엉덩이, 가슴 중앙 등
- **하체** (10개): 양쪽 무릎, 발목, 발 등
- **손** (8개): 양쪽 손가락 끝, 손바닥 등

#### 키포인트 정보

각 키포인트는 4가지 정보를 포함합니다:
- `x`: 정규화된 x 좌표 (0.0 ~ 1.0)
- `y`: 정규화된 y 좌표 (0.0 ~ 1.0)
- `z`: 깊이 정보 (상대적 거리)
- `visibility`: 키포인트 가시성 점수 (0.0 ~ 1.0)

### 구현 세부사항

```python
# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,        # 이미지 모드
    model_complexity=1,             # 모델 복잡도 (0, 1, 2)
    min_detection_confidence=0.5    # 최소 탐지 신뢰도
)

# 이미지에서 키포인트 추출
results = pose.process(image_rgb)
keypoints = results.pose_landmarks
```

## 4. 특징 추출

### 기본 특징

1. **정규화된 키포인트 좌표** (66차원)
   - 33개 키포인트의 x, y 좌표
   - 이미지 크기에 무관한 상대 좌표 사용

2. **신체 중심점**
   - 모든 키포인트의 평균 위치
   - 전체적인 신체 위치 파악

### 추가 특징 (확장 가능)

3. **각도 특징**
   - 목-엉덩이-무릎 각도
   - 어깨-팔꿈치-손목 각도
   - 신체 기울기 각도

4. **거리 특징**
   - 머리-발 거리 (신체 높이)
   - 어깨 너비
   - 팔 길이

5. **비율 특징**
   - 신체 높이 대비 중심점 위치
   - 좌우 균형도

## 5. 머신러닝 모델

### Random Forest Classifier

**장점:**
- 빠른 학습 속도
- 과적합 방지
- 특징 중요도 분석 가능
- 하이퍼파라미터 튜닝 불필요

**하이퍼파라미터:**
```python
RandomForestClassifier(
    n_estimators=100,      # 결정 트리 개수
    max_depth=10,          # 최대 깊이
    random_state=42        # 재현성
)
```

### Support Vector Machine (SVM)

**장점:**
- 높은 분류 정확도
- 커널 트릭으로 비선형 분류 가능
- 중소규모 데이터셋에 효과적

**하이퍼파라미터:**
```python
SVC(
    kernel='rbf',          # RBF 커널
    C=1.0,                 # 정규화 파라미터
    random_state=42
)
```

## 6. 학습 프로세스

### 데이터 준비

```
data/raw/
├── normal/     # 정상 자세 이미지
└── fall/       # 낙상 자세 이미지
```

### 학습 단계

1. **데이터 로드**
   - 각 이미지에서 스켈레톤 추출
   - 특징 벡터 생성
   - 레이블 할당 (0: 정상, 1: 낙상)

2. **데이터 분할**
   - 학습 데이터: 80%
   - 테스트 데이터: 20%
   - Stratified split으로 클래스 비율 유지

3. **모델 학습**
   - 데이터 표준화
   - 분류 모델 학습

4. **평가**
   - 테스트 데이터로 성능 평가
   - 혼동 행렬, 정확도, F1 점수 계산

## 7. 평가 지표

### 혼동 행렬 (Confusion Matrix)

```
                예측
              정상  낙상
실제  정상     TN    FP
      낙상     FN    TP
```

### 주요 지표

1. **정확도 (Accuracy)**: 전체 예측 중 올바른 예측의 비율
2. **정밀도 (Precision)**: 낙상으로 예측한 것 중 실제 낙상의 비율
3. **재현율 (Recall)**: 실제 낙상 중 올바르게 탐지한 비율
4. **F1 Score**: Precision과 Recall의 조화평균

### 낙상 감지에서 중요한 지표

**재현율(Recall)이 가장 중요!**
- 실제 낙상을 놓치는 것이 가장 위험
- False Negative를 최소화해야 함

## 8. 예상 성능

### 베이스라인 목표 성능

- **정확도**: 70-85%
- **F1 Score**: 0.65-0.80

### 한계점

1. **단일 이미지 기반** - 시간적 정보 부족
2. **간단한 특징** - 수작업 특징 엔지니어링
3. **데이터 의존성** - 다양한 환경에 대한 일반화 어려움

## 9. 향후 개선 방향

### 딥러닝 모델로 확장

1. **CNN**: 원본 이미지에서 직접 특징 학습
2. **LSTM**: 비디오에서 낙상 과정 학습
3. **Graph Neural Network**: 스켈레톤 그래프 구조 활용

## 10. 참고 문헌

- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose
- Random Forest: Breiman, L. (2001). Random forests. Machine learning
- SVM: Cortes, C., & Vapnik, V. (1995). Support-vector networks
