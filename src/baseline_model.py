"""
낙상 감지 베이스라인 모델
간단한 머신러닝 모델(SVM, Random Forest 등)을 사용한 베이스라인
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class BaselineModel:
    """낙상 감지를 위한 베이스라인 머신러닝 모델"""
    
    def __init__(self, model_type='random_forest'):
        """
        모델 초기화
        
        Args:
            model_type: 사용할 모델 타입 ('random_forest' 또는 'svm')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                random_state=42
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    def train(self, X_train, y_train):
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터 특징 (n_samples, n_features)
            y_train: 학습 데이터 레이블 (n_samples,) - 0: 정상, 1: 낙상
        """
        # 데이터 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 모델 학습
        self.model.fit(X_train_scaled, y_train)
        
        print(f"{self.model_type} 모델 학습 완료")
    
    def predict(self, X_test):
        """
        예측 수행
        
        Args:
            X_test: 테스트 데이터 (n_samples, n_features)
            
        Returns:
            predictions: 예측 결과 (n_samples,)
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test: 테스트 데이터
            y_test: 테스트 레이블
            
        Returns:
            metrics: 평가 지표 딕셔너리
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """
        모델 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath):
        """
        모델 로드
        
        Args:
            filepath: 불러올 파일 경로
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        
        print(f"모델이 로드되었습니다: {filepath}")


def print_evaluation_results(metrics):
    """
    평가 결과 출력
    
    Args:
        metrics: evaluate() 함수의 반환값
    """
    print("\n===== 모델 평가 결과 =====")
    print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {metrics['precision']:.4f}")
    print(f"재현율 (Recall): {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\n혼동 행렬 (Confusion Matrix):")
    print(metrics['confusion_matrix'])
    print("=" * 30)
