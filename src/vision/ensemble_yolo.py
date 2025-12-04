"""
YOLO 앙상블 모델 모듈
Segmentation과 Keypoint 검출을 통합하여 수행
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("[YOLO] ultralytics 패키지 없음")

class EnsembleYOLO:
    """
    앙상블 YOLO 모델
    Segmentation과 Keypoint 모델을 결합하여 사용
    """
    
    def __init__(self, segmentation_model_path: str, keypoint_model_path: str):
        """
        Args:
            segmentation_model_path: Segmentation 모델 경로
            keypoint_model_path: Keypoint 검출 모델 경로
        """
        if not HAS_YOLO:
            raise ImportError("ultralytics 패키지가 필요합니다")
        
        # 모델 경로 확인
        if not os.path.exists(segmentation_model_path):
            raise FileNotFoundError(f"Segmentation 모델을 찾을 수 없습니다: {segmentation_model_path}")
        if not os.path.exists(keypoint_model_path):
            raise FileNotFoundError(f"Keypoint 모델을 찾을 수 없습니다: {keypoint_model_path}")
        
        # 모델 로드
        logger.info(f"[YOLO] Segmentation 모델 로드: {segmentation_model_path}")
        self.seg_model = YOLO(segmentation_model_path)
        
        logger.info(f"[YOLO] Keypoint 모델 로드: {keypoint_model_path}")
        self.pose_model = YOLO(keypoint_model_path)
        
        # 클래스 이름 저장
        self.class_names = self.seg_model.names
        logger.info(f"[YOLO] 클래스: {self.class_names}")
    
    def predict_ensemble(self, image: np.ndarray, conf_threshold: float = 0.5) -> Dict[str, Any]:
        """
        앙상블 예측 수행
        기존 코드의 predict_ensemble 로직 완전 유지
        
        Args:
            image: 입력 이미지 (BGR 또는 RGB)
            conf_threshold: 신뢰도 임계값
            
        Returns:
            예측 결과 딕셔너리:
                - boxes: 바운딩 박스 배열
                - masks: 세그멘테이션 마스크 배열
                - keypoints: 키포인트 좌표 배열
                - classes: 클래스 ID 배열
                - confidences: 신뢰도 배열
        """
        results = {
            'boxes': [],
            'masks': [],
            'keypoints': [],
            'classes': [],
            'confidences': []
        }
        
        # Segmentation 모델 추론
        logger.debug(f"[YOLO] Segmentation 추론 시작 (conf={conf_threshold})")
        seg_results = self.seg_model.predict(image, conf=conf_threshold, verbose=False)
        
        if seg_results and len(seg_results) > 0:
            res = seg_results[0]
            
            # 박스 정보 추출
            if res.boxes is not None and len(res.boxes) > 0:
                results['boxes'] = res.boxes.xyxy.cpu().numpy()
                results['classes'] = res.boxes.cls.cpu().numpy()
                results['confidences'] = res.boxes.conf.cpu().numpy()
                logger.debug(f"[YOLO] {len(results['boxes'])}개 객체 검출")
            
            # 마스크 정보 추출
            if res.masks is not None:
                results['masks'] = res.masks.data.cpu().numpy()
                logger.debug(f"[YOLO] {len(results['masks'])}개 마스크 생성")
        
        # Keypoint 모델 추론
        logger.debug(f"[YOLO] Keypoint 추론 시작 (conf={conf_threshold})")
        pose_results = self.pose_model.predict(image, conf=conf_threshold, verbose=False)
        
        if pose_results and len(pose_results) > 0:
            res = pose_results[0]
            
            # 키포인트 정보 추출
            if res.keypoints is not None:
                results['keypoints'] = res.keypoints.xy.cpu().numpy()
                logger.debug(f"[YOLO] {len(results['keypoints'])}개 키포인트 검출")
            
            # Segmentation에서 박스를 못 찾았으면 Keypoint 모델의 박스 사용
            if len(results['boxes']) == 0 and res.boxes is not None and len(res.boxes) > 0:
                results['boxes'] = res.boxes.xyxy.cpu().numpy()
                results['classes'] = res.boxes.cls.cpu().numpy()
                results['confidences'] = res.boxes.conf.cpu().numpy()
                logger.debug(f"[YOLO] Keypoint 모델에서 {len(results['boxes'])}개 박스 사용")
        
        # 결과 요약 로그
        logger.info(f"[YOLO] 앙상블 결과: 박스={len(results['boxes'])}, "
                   f"마스크={len(results['masks'])}, 키포인트={len(results['keypoints'])}")
        
        return results
    
    def predict_quick(self, image: np.ndarray, conf_threshold: float = 0.3) -> bool:
        """
        빠른 객체 존재 확인
        기존 코드의 quick_object_detection 로직 구현
        
        Args:
            image: 입력 이미지
            conf_threshold: 신뢰도 임계값 (낮게 설정)
            
        Returns:
            객체 검출 여부
        """
        try:
            # Segmentation만 사용하여 빠른 검출
            seg_results = self.seg_model.predict(image, conf=conf_threshold, verbose=False)
            
            if seg_results and len(seg_results) > 0:
                res = seg_results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    logger.debug(f"[YOLO] 빠른 검출: {len(res.boxes)}개 객체 발견")
                    return True
            
            # Segmentation에서 못 찾으면 Keypoint로 시도
            pose_results = self.pose_model.predict(image, conf=conf_threshold, verbose=False)
            
            if pose_results and len(pose_results) > 0:
                res = pose_results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    logger.debug(f"[YOLO] 빠른 검출(Keypoint): {len(res.boxes)}개 객체 발견")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"[YOLO] 빠른 검출 오류: {e}")
            return False
    
    def get_class_name(self, class_id: int) -> str:
        """
        클래스 ID를 이름으로 변환
        
        Args:
            class_id: 클래스 ID
            
        Returns:
            클래스 이름
        """
        if class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"Unknown_{class_id}"