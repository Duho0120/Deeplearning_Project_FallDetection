"""
데이터 전처리 모듈
이미지 데이터를 로드하고 전처리하여 모델 학습에 사용할 수 있는 형태로 변환
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skeleton_extractor import SkeletonExtractor


class DataProcessor:
    """데이터 로드 및 전처리 클래스"""
    
    def __init__(self):
        """데이터 프로세서 초기화"""
        self.skeleton_extractor = SkeletonExtractor()
    
    def load_dataset(self, data_dir, label_mapping={'normal': 0, 'fall': 1}):
        """
        데이터셋 로드
        
        디렉토리 구조:
        data_dir/
        ├── normal/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── fall/
            ├── img1.jpg
            ├── img2.jpg
            └── ...
        
        Args:
            data_dir: 데이터 디렉토리 경로
            label_mapping: 클래스명-레이블 매핑 딕셔너리
            
        Returns:
            X: 특징 배열 (n_samples, n_features)
            y: 레이블 배열 (n_samples,)
            failed_images: 처리 실패한 이미지 리스트
        """
        X = []
        y = []
        failed_images = []
        
        data_path = Path(data_dir)
        
        # 각 클래스별로 이미지 처리
        for class_name, label in label_mapping.items():
            class_dir = data_path / class_name
            
            if not class_dir.exists():
                print(f"경고: {class_dir} 디렉토리가 존재하지 않습니다.")
                continue
            
            # 이미지 파일 리스트
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            print(f"\n{class_name} 클래스 처리 중... (총 {len(image_files)}개 이미지)")
            
            for image_path in tqdm(image_files):
                try:
                    # 키포인트 추출
                    keypoints = self.skeleton_extractor.extract_keypoints(str(image_path))
                    
                    if keypoints is not None:
                        # 특징 추출
                        features = self.skeleton_extractor.extract_features(keypoints)
                        
                        if features is not None:
                            X.append(features)
                            y.append(label)
                    else:
                        print(f"키포인트 추출 실패: {image_path}")
                        failed_images.append((str(image_path), "keypoint_extraction_failed"))
                        
                except Exception as e:
                    print(f"오류 발생 ({image_path}): {e}")
                    failed_images.append((str(image_path), str(e)))
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n데이터 로드 완료!")
        print(f"- 총 샘플 수: {len(X)}")
        print(f"- 특징 차원: {X.shape[1] if len(X) > 0 else 0}")
        print(f"- 처리 실패 이미지 수: {len(failed_images)}")
        
        return X, y, failed_images
    
    def save_processed_data(self, X, y, output_path):
        """
        전처리된 데이터 저장
        
        Args:
            X: 특징 배열
            y: 레이블 배열
            output_path: 저장할 파일 경로 (.npz)
        """
        np.savez(output_path, X=X, y=y)
        print(f"전처리된 데이터가 저장되었습니다: {output_path}")
    
    def load_processed_data(self, file_path):
        """
        저장된 전처리 데이터 로드
        
        Args:
            file_path: 데이터 파일 경로 (.npz)
            
        Returns:
            X: 특징 배열
            y: 레이블 배열
        """
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        
        print(f"데이터 로드 완료: {file_path}")
        print(f"- 샘플 수: {len(X)}")
        print(f"- 특징 차원: {X.shape[1]}")
        
        return X, y
