"""
예측 스크립트
학습된 모델을 사용하여 새로운 이미지에 대해 낙상 여부를 예측
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skeleton_extractor import SkeletonExtractor
from baseline_model import BaselineModel


def predict_image(image_path, model_path):
    """
    이미지에서 낙상 여부 예측
    
    Args:
        image_path: 예측할 이미지 경로
        model_path: 학습된 모델 경로
        
    Returns:
        prediction: 예측 결과 (0: 정상, 1: 낙상)
    """
    # 스켈레톤 추출기 초기화
    extractor = SkeletonExtractor()
    
    # 키포인트 추출
    print(f"이미지에서 스켈레톤 추출 중: {image_path}")
    keypoints = extractor.extract_keypoints(image_path)
    
    if keypoints is None:
        print("오류: 스켈레톤을 추출할 수 없습니다.")
        return None
    
    # 특징 추출
    features = extractor.extract_features(keypoints)
    
    if features is None:
        print("오류: 특징을 추출할 수 없습니다.")
        return None
    
    # 모델 로드 및 예측
    print(f"모델 로드 중: {model_path}")
    model = BaselineModel()
    model.load_model(model_path)
    
    # 예측 (배치 형태로 변환)
    features_batch = features.reshape(1, -1)
    prediction = model.predict(features_batch)[0]
    
    return prediction


def main(args):
    """메인 함수"""
    
    print("=" * 50)
    print("낙상 감지 예측")
    print("=" * 50)
    
    # 이미지 경로 확인
    if not os.path.exists(args.image_path):
        print(f"오류: 이미지 파일이 존재하지 않습니다: {args.image_path}")
        return
    
    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일이 존재하지 않습니다: {args.model_path}")
        return
    
    # 예측 수행
    prediction = predict_image(args.image_path, args.model_path)
    
    if prediction is not None:
        result = "낙상" if prediction == 1 else "정상"
        print(f"\n예측 결과: {result}")
        
        # 시각화 옵션
        if args.visualize:
            extractor = SkeletonExtractor()
            output_path = args.output_path if args.output_path else None
            extractor.visualize_skeleton(args.image_path, output_path)
            
            if output_path:
                print(f"시각화 결과 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='낙상 감지 예측')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='예측할 이미지 경로')
    parser.add_argument('--model_path', type=str, default='src/models/baseline_model.pkl',
                        help='학습된 모델 경로')
    parser.add_argument('--visualize', action='store_true',
                        help='스켈레톤 시각화 여부')
    parser.add_argument('--output_path', type=str, default=None,
                        help='시각화 결과 저장 경로')
    
    args = parser.parse_args()
    main(args)
