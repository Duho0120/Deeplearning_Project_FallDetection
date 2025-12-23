#!/bin/bash

# 낙상 감지 시스템 빠른 시작 스크립트

echo "======================================"
echo "낙상 감지 시스템 - 빠른 시작"
echo "======================================"

# 1. 가상환경 확인
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 2. 가상환경 활성화
echo "가상환경 활성화 중..."
source venv/bin/activate

# 3. 패키지 설치
echo "필요한 패키지 설치 중..."
pip install -r requirements.txt

# 4. 데이터 디렉토리 확인
if [ ! -d "data/raw/normal" ]; then
    echo "데이터 디렉토리 생성 중..."
    mkdir -p data/raw/normal
    mkdir -p data/raw/fall
fi

# 5. 안내 메시지
echo ""
echo "======================================"
echo "설치 완료!"
echo "======================================"
echo ""
echo "다음 단계:"
echo "1. 데이터를 준비하세요:"
echo "   - data/raw/normal/ : 정상 자세 이미지"
echo "   - data/raw/fall/   : 낙상 자세 이미지"
echo ""
echo "2. 모델을 학습하세요:"
echo "   python src/train.py --data_dir data/raw --save_model --save_processed"
echo ""
echo "3. 예측을 수행하세요:"
echo "   python src/predict.py --image_path path/to/image.jpg --model_path src/models/baseline_model.pkl"
echo ""
echo "자세한 내용은 README.md를 참고하세요."
