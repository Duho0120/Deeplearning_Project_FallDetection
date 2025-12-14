"""
유틸리티 함수 모음
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes=['정상', '낙상'], save_path=None):
    """
    혼동 행렬 시각화
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        classes: 클래스 이름 리스트
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.ylabel('실제 레이블')
    plt.xlabel('예측 레이블')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """
    특징 중요도 시각화 (Random Forest 전용)
    
    Args:
        model: 학습된 Random Forest 모델
        feature_names: 특징 이름 리스트
        top_n: 표시할 상위 특징 개수
        save_path: 저장 경로
    """
    if not hasattr(model.model, 'feature_importances_'):
        print("이 모델은 특징 중요도를 지원하지 않습니다.")
        return
    
    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    if feature_names is None:
        feature_names = [f'특징_{i}' for i in range(len(importances))]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'상위 {top_n}개 특징 중요도')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('중요도')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"특징 중요도 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true, y_pred, target_names=['정상', '낙상']):
    """
    분류 리포트 출력
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        target_names: 클래스 이름 리스트
    """
    print("\n===== 상세 분류 리포트 =====")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("=" * 30)


def create_sample_dataset_structure(base_path='data/raw'):
    """
    샘플 데이터셋 구조 생성
    
    Args:
        base_path: 데이터 디렉토리 경로
    """
    os.makedirs(os.path.join(base_path, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'fall'), exist_ok=True)
    
    print(f"데이터셋 구조가 생성되었습니다: {base_path}")
    print("  - normal/ : 정상 자세 이미지를 이 폴더에 넣으세요")
    print("  - fall/   : 낙상 자세 이미지를 이 폴더에 넣으세요")
