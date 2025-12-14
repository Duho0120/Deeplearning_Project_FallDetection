"""
학습 및 평가 스크립트
베이스라인 모델을 학습하고 평가하는 메인 스크립트
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# 현재 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from baseline_model import BaselineModel, print_evaluation_results


def main(args):
    """메인 함수"""
    
    print("=" * 50)
    print("낙상 감지 베이스라인 모델 학습")
    print("=" * 50)
    
    # 데이터 프로세서 초기화
    processor = DataProcessor()
    
    # 데이터 로드
    if args.use_processed:
        print("\n전처리된 데이터 로드 중...")
        X, y = processor.load_processed_data(args.processed_data_path)
    else:
        print("\n원본 이미지에서 데이터 로드 중...")
        X, y, failed_images = processor.load_dataset(args.data_dir)
        
        # 전처리된 데이터 저장
        if args.save_processed:
            processor.save_processed_data(X, y, args.processed_data_path)
    
    if len(X) == 0:
        print("오류: 로드된 데이터가 없습니다.")
        return
    
    # 데이터 분할
    print(f"\n데이터 분할 중... (테스트 비율: {args.test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )
    
    print(f"- 학습 데이터: {len(X_train)}개")
    print(f"- 테스트 데이터: {len(X_test)}개")
    
    # 클래스 분포 확인
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n학습 데이터 클래스 분포:")
    for label, count in zip(unique, counts):
        class_name = "정상" if label == 0 else "낙상"
        print(f"  {class_name}: {count}개 ({count/len(y_train)*100:.1f}%)")
    
    # 모델 초기화
    print(f"\n모델 초기화 중... (모델 타입: {args.model_type})")
    model = BaselineModel(model_type=args.model_type)
    
    # 모델 학습
    print("\n모델 학습 중...")
    model.train(X_train, y_train)
    
    # 모델 평가
    print("\n모델 평가 중...")
    
    # 학습 데이터 평가
    print("\n[학습 데이터 성능]")
    train_metrics = model.evaluate(X_train, y_train)
    print_evaluation_results(train_metrics)
    
    # 테스트 데이터 평가
    print("\n[테스트 데이터 성능]")
    test_metrics = model.evaluate(X_test, y_test)
    print_evaluation_results(test_metrics)
    
    # 모델 저장
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save_model(args.model_path)
    
    print("\n학습 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='낙상 감지 베이스라인 모델 학습')
    
    # 데이터 관련 인자
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='원본 데이터 디렉토리 경로')
    parser.add_argument('--use_processed', action='store_true',
                        help='전처리된 데이터 사용 여부')
    parser.add_argument('--processed_data_path', type=str, default='data/processed/dataset.npz',
                        help='전처리된 데이터 파일 경로')
    parser.add_argument('--save_processed', action='store_true',
                        help='전처리된 데이터 저장 여부')
    
    # 학습 관련 인자
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'svm'],
                        help='사용할 모델 타입')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='테스트 데이터 비율')
    
    # 모델 저장 관련 인자
    parser.add_argument('--save_model', action='store_true',
                        help='학습된 모델 저장 여부')
    parser.add_argument('--model_path', type=str, default='src/models/baseline_model.pkl',
                        help='모델 저장 경로')
    
    args = parser.parse_args()
    main(args)
