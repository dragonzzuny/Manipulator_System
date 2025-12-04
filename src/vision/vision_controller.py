"""
비전 시스템 컨트롤러
카메라(RealSense) + YOLO 앙상블 + 파지 플래너를 통합 관리
"""

import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass

from src.config_loader import Config, VisionConfig, RobotConfig, SystemConfig
from src.vision.ensemble_yolo import EnsembleYOLO
from src.vision.grasp_planner import GraspPlanner, GraspCandidate

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    logger.warning("[VISION] pyrealsense2 없음 - 카메라 기능 제한")

class VisionController:
    """
    통합 비전 시스템 컨트롤러
    기존 코드의 모든 비전 관련 기능을 통합
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.vision_config = config.vision
        self.robot_config = config.robot
        self.system_config = config.system
        
        # 카메라 관련
        self.pipeline = None
        self.align = None
        self.color_intrinsics = None
        self.depth_scale = None
        
        # 모델 및 플래너
        self.ensemble_model = self._init_ensemble_model()
        self.grasp_planner = GraspPlanner(self.vision_config)
        
        # 캘리브레이션
        self.calibration_matrix = config.calibration_matrix
        self.y_offset = config.robot.y_offset
        
        # 캡처 관련
        self.capture_count = 0
        self.save_dir = Path(config.system.save_dir)
        if config.system.enable_debug_capture:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[VISION] 비전 컨트롤러 초기화 완료")
    
    def _init_ensemble_model(self) -> Optional[EnsembleYOLO]:
        """YOLO 앙상블 모델 초기화"""
        try:
            if not (os.path.exists(self.vision_config.segmentation_model_path) and 
                   os.path.exists(self.vision_config.keypoint_model_path)):
                logger.error("[VISION] 모델 파일을 찾을 수 없습니다")
                return None
            
            return EnsembleYOLO(
                self.vision_config.segmentation_model_path,
                self.vision_config.keypoint_model_path
            )
        except Exception as e:
            logger.error(f"[VISION] 모델 로드 실패: {e}")
            return None
    
    def start_camera(self) -> bool:
        """
        RealSense 카메라 시작
        기존 코드의 초기화 로직 완전 유지
        """
        if not HAS_REALSENSE:
            logger.error("[VISION] RealSense 라이브러리 없음")
            return False
        
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 스트림 설정
            config.enable_stream(rs.stream.depth, 
                               self.vision_config.camera_width, 
                               self.vision_config.camera_height, 
                               rs.format.z16, 
                               self.vision_config.camera_fps)
            config.enable_stream(rs.stream.color, 
                               self.vision_config.camera_width, 
                               self.vision_config.camera_height, 
                               rs.format.bgr8, 
                               self.vision_config.camera_fps)
            
            # 파이프라인 시작
            profile = self.pipeline.start(config)
            
            # Align 객체 생성
            self.align = rs.align(rs.stream.color)
            
            # 카메라 내부 파라미터 획득
            self.color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            
            # Warmup frames (기존 코드의 30프레임 유지)
            logger.info(f"[VISION] 카메라 워밍업 ({self.vision_config.warmup_frames} 프레임)")
            for _ in range(self.vision_config.warmup_frames):
                self.pipeline.wait_for_frames()
            
            logger.info("[VISION] 카메라 시작 성공")
            return True
            
        except Exception as e:
            logger.error(f"[VISION] 카메라 시작 실패: {e}")
            return False
    
    def stop_camera(self):
        """카메라 정지"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("[VISION] 카메라 정지")
            except:
                pass
    
    def capture_frames(self) -> Optional[Dict[str, Any]]:
        """
        멀티프레임 캡처
        기존 _capture_frames 로직 유지
        
        Returns:
            캡처된 프레임 데이터 딕셔너리
        """
        if not self.pipeline:
            logger.error("[VISION] 카메라가 시작되지 않음")
            return None
        
        depth_accumulator = None
        last_frames = None
        
        # 멀티프레임 캡처로 노이즈 감소
        for i in range(self.vision_config.multi_frame_count):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 깊이 이미지 누적
            depth_img = np.asanyarray(depth_frame.get_data())
            if depth_accumulator is None:
                depth_accumulator = depth_img.astype(np.float32)
            else:
                depth_accumulator += depth_img
            
            last_frames = (color_frame, depth_frame)
            time.sleep(self.vision_config.frame_delay)
        
        if last_frames is None:
            logger.error("[VISION] 프레임 캡처 실패")
            return None
        
        self.capture_count += 1
        
        captured_data = {
            'color': np.asanyarray(last_frames[0].get_data()),
            'depth': (depth_accumulator / self.vision_config.multi_frame_count).astype(np.uint16),
            'depth_frame': last_frames[1],
            'capture_id': self.capture_count
        }
        
        logger.debug(f"[VISION] 캡처 #{self.capture_count} 완료")
        return captured_data
    
    def detect_objects(self, image: np.ndarray, 
                       conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        객체 탐지 수행
        
        Args:
            image: 입력 이미지
            conf_threshold: 신뢰도 임계값
            
        Returns:
            탐지 결과 딕셔너리
        """
        if not self.ensemble_model:
            logger.error("[VISION] 모델이 로드되지 않음")
            return {}
        
        if conf_threshold is None:
            conf_threshold = self.vision_config.confidence_threshold
        
        return self.ensemble_model.predict_ensemble(image, conf_threshold)
    
    def quick_object_detection(self, image: np.ndarray) -> bool:
        """
        빠른 객체 존재 확인
        기존 quick_object_detection 로직 유지
        """
        if not self.ensemble_model:
            return False
        
        return self.ensemble_model.predict_quick(image, conf_threshold=0.3)
    
    def find_object_and_get_coords(self) -> Optional[GraspCandidate]:
        """
        객체 탐지 및 파지점 계산
        카메라 시작/정지 포함
        
        Returns:
            파지 후보 또는 None
        """
        if not self.start_camera():
            logger.error("[VISION] 카메라 시작 실패")
            return None
        
        try:
            # 대기 시간 (안정화)
            time.sleep(self.system_config.capture_wait_time)
            
            # 프레임 캡처
            captured_data = self.capture_frames()
            if captured_data is None:
                return None
            
            color_image = captured_data['color']
            depth_frame = captured_data['depth_frame']
            
            # 객체 탐지
            results = self.detect_objects(color_image)
            
            if 'masks' not in results or len(results['masks']) == 0:
                logger.error("[VISION] 마스크 검출 실패")
                return None
            
            # 첫 번째 객체 사용
            mask = cv2.resize(results['masks'][0], 
                            (self.vision_config.camera_width, self.vision_config.camera_height))
            bbox = results['boxes'][0] if len(results['boxes']) > 0 else None
            
            if bbox is None:
                logger.error("[VISION] 바운딩 박스 없음")
                return None
            
            # 키포인트 확인
            keypoint_2d = None
            if 'keypoints' in results and len(results['keypoints']) > 0:
                kpts = results['keypoints'][0]
                if len(kpts) > 0 and kpts[0][0] > 0 and kpts[0][1] > 0:
                    keypoint_2d = (int(kpts[0][0]), int(kpts[0][1]))
            
            if keypoint_2d is None:
                logger.warning("[VISION] 키포인트 없음, 마스크 중심 사용")
                y_coords, x_coords = np.where(mask > 0)
                if len(x_coords) > 0:
                    keypoint_2d = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                else:
                    return None
            
            # 그리퍼 파라미터
            gripper_params = {
                'open_width_mm': self.robot_config.gripper_open_width_mm,
                'jaw_length_mm': self.robot_config.gripper_jaw_length_mm,
                'jaw_thickness_mm': self.robot_config.gripper_jaw_thickness_mm
            }
            
            # 파지점 계산
            best_grasp = self.grasp_planner.find_best_grasp(
                mask, 
                depth_frame, 
                keypoint_2d, 
                bbox,
                gripper_params,
                self.color_intrinsics,
                self.y_offset,
                self.calibration_matrix
            )
            
            if best_grasp:
                best_grasp.object_id = 0
                best_grasp.object_conf = results['confidences'][0] if len(results['confidences']) > 0 else 0.0
                
                # 디버그 캡처가 활성화된 경우
                if self.system_config.enable_debug_capture:
                    self._save_debug_image(color_image, mask, best_grasp)
            
            return best_grasp
            
        finally:
            self.stop_camera()
    
    def find_object_and_get_coords_with_retry(self, max_retries: int = 5) -> Optional[GraspCandidate]:
        """
        재시도를 포함한 객체 탐지 및 파지점 계산
        기존 find_object_and_get_coords_with_retry 로직 완전 유지
        
        Args:
            max_retries: 최대 재시도 횟수
            
        Returns:
            파지 후보 또는 None
        """
        for attempt in range(1, max_retries + 1):
            logger.info(f"[VISION][ATTEMPT {attempt}/{max_retries}] 객체 탐지 시도")
            
            try:
                grasp = self.find_object_and_get_coords()
                
                if grasp:
                    logger.info(f"[VISION][ATTEMPT {attempt}] 객체 탐지 성공!")
                    self._log_grasp_result(grasp)
                    return grasp
                
                logger.warning(f"[VISION][ATTEMPT {attempt}] 객체 탐지 실패")
                
                # 재시도 전 대기
                if attempt < max_retries:
                    time.sleep(2.0)
                    
            except Exception as e:
                logger.error(f"[VISION][ATTEMPT {attempt}] 오류 발생: {e}")
                if attempt < max_retries:
                    time.sleep(2.0)
        
        logger.error(f"[VISION] {max_retries}회 시도 모두 실패")
        return None
    
    def quick_redetect_object(self, max_attempts: int = 3) -> bool:
        """
        빠른 객체 재탐지
        기존 quick_redetect_object 로직 유지
        
        Args:
            max_attempts: 최대 시도 횟수
            
        Returns:
            객체 탐지 성공 여부
        """
        if not self.start_camera():
            return False
        
        try:
            for attempt in range(1, max_attempts + 1):
                logger.info(f"[VISION][QUICK {attempt}/{max_attempts}] 빠른 재탐지")
                
                captured = self.capture_frames()
                if captured is None:
                    continue
                
                if self.quick_object_detection(captured['color']):
                    logger.info(f"[VISION][QUICK {attempt}] 객체 재발견!")
                    return True
                
                time.sleep(0.5)
            
            return False
            
        finally:
            self.stop_camera()
    
    def _save_debug_image(self, image: np.ndarray, mask: np.ndarray, grasp: GraspCandidate):
        """디버그용 이미지 저장"""
        try:
            # 시각화 이미지 생성
            vis_img = image.copy()
            
            # 마스크 오버레이
            mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_colored, 0.3, 0)
            
            # 파지점 표시
            cv2.circle(vis_img, tuple(grasp.center_2d.astype(int)), 5, (0, 255, 0), -1)
            
            # 키포인트 표시
            cv2.circle(vis_img, grasp.keypoint_pos, 5, (255, 0, 0), -1)
            
            # 벡터 표시
            cv2.line(vis_img, 
                    (int(grasp.bbox_center[0]), int(grasp.bbox_center[1])),
                    grasp.keypoint_pos,
                    (0, 255, 255), 2)
            
            # 파일 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = self.save_dir / f"grasp_{timestamp}_{self.capture_count}.png"
            cv2.imwrite(str(filepath), vis_img)
            
            logger.debug(f"[VISION] 디버그 이미지 저장: {filepath}")
            
        except Exception as e:
            logger.error(f"[VISION] 디버그 이미지 저장 실패: {e}")
    
    def _log_grasp_result(self, grasp: GraspCandidate):
        """파지 결과 로그 출력"""
        logger.info("="*50)
        logger.info("[VISION] 파지점 검출 결과:")
        logger.info(f"  방식: {grasp.grasp_method}")
        logger.info(f"  벡터 각도: {grasp.angle_deg:.1f}°")
        logger.info(f"  벡터 거리: {grasp.vector_distance:.1f}px")
        logger.info(f"  품질 점수: {grasp.quality_score:.0f}")
        logger.info(f"  충실도: {grasp.fill_factor:.1%}")
        logger.info(f"  충돌: {grasp.collision_score:.0f}px")
        logger.info(f"  2D 위치: {grasp.center_2d.astype(int)}")
        logger.info(f"  3D 위치: {grasp.center_3d}")
        logger.info(f"  로봇 좌표: {grasp.robot_coords}")
        logger.info("="*50)