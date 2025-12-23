"""
스켈레톤 추출 모듈
이미지에서 포즈 추정(Pose Estimation)을 통해 스켈레톤 키포인트를 추출합니다.
"""

import cv2
import numpy as np
import mediapipe as mp


class SkeletonExtractor:
    """MediaPipe를 사용한 스켈레톤 추출 클래스"""
    
    def __init__(self):
        """MediaPipe Pose 모델 초기화"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_keypoints(self, image_path):
        """
        이미지에서 키포인트 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            keypoints: 33개의 키포인트 좌표 (x, y, z, visibility) numpy array
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 포즈 추정
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 키포인트를 numpy array로 변환
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(keypoints)
        else:
            # 키포인트를 찾지 못한 경우 None 반환
            return None
    
    def visualize_skeleton(self, image_path, output_path=None):
        """
        이미지에 스켈레톤을 시각화
        
        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (None이면 화면에 표시)
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 스켈레톤 그리기
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        if output_path:
            cv2.imwrite(output_path, image)
        else:
            cv2.imshow('Skeleton', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def extract_features(self, keypoints):
        """
        키포인트로부터 특징 벡터 추출
        
        Args:
            keypoints: 스켈레톤 키포인트 (132차원: 33 keypoints × 4)
            
        Returns:
            features: 추출된 특징 벡터
        """
        if keypoints is None:
            return None
            
        # 키포인트 재구성 (33, 4)
        kp_reshaped = keypoints.reshape(33, 4)
        
        # 주요 특징 추출
        features = []
        
        # 1. 전체 키포인트 좌표 (x, y만 사용)
        features.extend(kp_reshaped[:, :2].flatten())
        
        # 2. 신체 중심 계산
        center_x = np.mean(kp_reshaped[:, 0])
        center_y = np.mean(kp_reshaped[:, 1])
        features.extend([center_x, center_y])
        
        return np.array(features)
    
    def __del__(self):
        """리소스 정리"""
        self.pose.close()
