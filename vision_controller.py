#!/usr/bin/env python3
"""
Vision System Controller
MQTT 기반 객체 탐지 및 좌표 계산 모듈
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import paho.mqtt.client as mqtt


class VisionState(Enum):
    """비전 시스템 상태"""
    IDLE = "idle"
    SCANNING = "scanning"
    DETECTED = "detected"
    ERROR = "error"


@dataclass
class DetectedObject:
    """탐지된 객체 정보"""
    x: float  # mm
    y: float  # mm
    z: float  # mm
    angle: float  # degrees
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'angle': self.angle,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


class VisionController:
    """비전 시스템 제어 클래스"""
    
    def __init__(self, config: Dict[str, Any], mqtt_client: mqtt.Client):
        self.config = config
        self.mqtt_client = mqtt_client
        self.logger = logging.getLogger(__name__)
        
        # 비전 설정
        self.camera_config = config['Vision']['camera']
        self.model_config = config['Vision']['model']
        self.calib_config = config['Vision']['calibration']
        self.detect_config = config['Vision']['detection']
        
        # 캘리브레이션 행렬
        self.calibration_matrix = np.array(self.calib_config['matrix'])
        self.y_offset = self.calib_config['y_offset']
        
        # YOLO 모델
        self.model: Optional[YOLO] = None
        
        # RealSense 파이프라인
        self.pipeline: Optional[rs.pipeline] = None
        self.rs_config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.pipeline_started = False
        
        # 상태 변수
        self.current_state = VisionState.IDLE
        self.last_detection: Optional[DetectedObject] = None
        
        # 스레드 관리
        self.scan_thread: Optional[threading.Thread] = None
        self.stop_scanning = threading.Event()
        
    def initialize(self) -> bool:
        """비전 시스템 초기화"""
        try:
            # YOLO 모델 로드
            self.logger.info("YOLO 모델 로딩 중...")
            self.model = YOLO(self.model_config['weights'])
            self.logger.info(f"✅ YOLO 모델 로드 완료: {self.model_config['weights']}")
            
            # RealSense 설정
            self.logger.info("RealSense 카메라 설정 중...")
            self._setup_camera()
            
            self._publish_status("initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"비전 시스템 초기화 실패: {e}")
            self._publish_error(str(e))
            return False
    
    def _setup_camera(self):
        """카메라 설정"""
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        # 스트림 설정
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.camera_config['width'],
            self.camera_config['height'],
            rs.format.z16,
            self.camera_config['fps']
        )
        
        self.rs_config.enable_stream(
            rs.stream.color,
            self.camera_config['width'],
            self.camera_config['height'],
            rs.format.bgr8,
            self.camera_config['fps']
        )
        
        # 깊이와 컬러 정렬
        self.align = rs.align(rs.stream.color)
        self.logger.info("✅ RealSense 카메라 설정 완료")
    
    def start_scanning(self, scan_type: str = "pick") -> bool:
        """스캔 시작"""
        if self.current_state == VisionState.SCANNING:
            self.logger.warning("이미 스캔 중입니다")
            return False
        
        try:
            # 카메라 시작
            if not self.pipeline_started:
                profile = self.pipeline.start(self.rs_config)
                self.pipeline_started = True
                
                # 안정화 대기
                self.logger.info("카메라 안정화 중...")
                for _ in range(self.detect_config['stabilization_frames']):
                    self.pipeline.wait_for_frames()
                    time.sleep(0.1)
            
            # 스캔 스레드 시작
            self.stop_scanning.clear()
            self.scan_thread = threading.Thread(
                target=self._scan_loop,
                args=(scan_type,)
            )
            self.scan_thread.daemon = True
            self.scan_thread.start()
            
            self.current_state = VisionState.SCANNING
            self._publish_status(f"scanning_{scan_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"스캔 시작 실패: {e}")
            self._publish_error(str(e))
            return False
    
    def stop_scanning(self):
        """스캔 중지"""
        self.stop_scanning.set()
        
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=2.0)
        
        if self.pipeline_started:
            self.pipeline.stop()
            self.pipeline_started = False
        
        self.current_state = VisionState.IDLE
        self._publish_status("idle")
        self.logger.info("스캔 중지됨")
    
    def detect_object(self, timeout: Optional[float] = None) -> Optional[DetectedObject]:
        """객체 탐지 (동기 방식)"""
        if timeout is None:
            timeout = self.detect_config['timeout']
        
        self.logger.info(f"객체 탐지 시작 (타임아웃: {timeout}초)")
        
        # 카메라 시작
        if not self.pipeline_started:
            profile = self.pipeline.start(self.rs_config)
            self.pipeline_started = True
            
            # 안정화
            for _ in range(self.detect_config['stabilization_frames']):
                self.pipeline.wait_for_frames()
        
        # 카메라 내부 파라미터
        color_stream = profile.get_stream(rs.stream.color)
        color_intrin = color_stream.as_video_stream_profile().get_intrinsics()
        
        start_time = time.time()
        best_detection = None
        best_confidence = 0.0
        
        while time.time() - start_time < timeout:
            # 프레임 획득
            frames = self.pipeline.wait_for_frames(1000)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 이미지 변환
            color_img = np.asanyarray(color_frame.get_data())
            
            # YOLO 추론
            results = self.model.predict(
                color_img,
                conf=self.model_config['confidence'],
                device=self.model_config['device'],
                verbose=False
            )
            
            for res in results:
                if len(res.boxes) > 0:
                    # 가장 신뢰도 높은 객체
                    for idx in range(len(res.boxes)):
                        confidence = res.boxes.conf[idx].item()
                        
                        if confidence > best_confidence:
                            box = res.boxes.xyxy[idx].cpu().numpy().astype(int)
                            
                            # 중심점
                            cx = int((box[0] + box[2]) / 2)
                            cy = int((box[1] + box[3]) / 2)
                            
                            # 깊이 획득
                            dist = depth_frame.get_distance(cx, cy)
                            
                            if self.detect_config['min_depth'] < dist < self.detect_config['max_depth']:
                                # 3D 좌표 변환
                                depth_point = rs.rs2_deproject_pixel_to_point(
                                    color_intrin, [cx, cy], dist
                                )
                                
                                # 카메라 좌표 (mm)
                                x_cam = depth_point[0] * 1000
                                y_cam = depth_point[1] * 1000 + self.y_offset
                                z_cam = depth_point[2] * 1000
                                
                                # 각도 계산 (마스크가 있는 경우)
                                angle = 0.0
                                if res.masks is not None and idx < len(res.masks.data):
                                    mask = (res.masks.data[idx].cpu().numpy() > 0.5).astype(np.uint8)
                                    angle = self._compute_orientation(mask)
                                
                                # 로봇 좌표 변환
                                robot_coords = self._transform_to_robot(x_cam, y_cam, z_cam, angle)
                                
                                best_detection = DetectedObject(
                                    x=robot_coords[0],
                                    y=robot_coords[1],
                                    z=robot_coords[2],
                                    angle=robot_coords[3],
                                    confidence=confidence,
                                    timestamp=time.time()
                                )
                                best_confidence = confidence
                                
                                self.logger.info(f"객체 탐지: {best_detection}")
        
        if best_detection:
            self.last_detection = best_detection
            self.current_state = VisionState.DETECTED
            self._publish_detection(best_detection, "pick")
        
        return best_detection
    
    def _scan_loop(self, scan_type: str):
        """연속 스캔 루프"""
        self.logger.info(f"연속 스캔 시작: {scan_type}")
        
        while not self.stop_scanning.is_set():
            try:
                detection = self.detect_object(timeout=1.0)
                
                if detection:
                    self._publish_detection(detection, scan_type)
                    
                    # 한 번만 탐지하고 중지할지 설정에 따라 결정
                    if scan_type == "pick":
                        break
                
            except Exception as e:
                self.logger.error(f"스캔 중 오류: {e}")
                time.sleep(0.5)
        
        self.logger.info("스캔 루프 종료")
    
    def _compute_orientation(self, mask: np.ndarray) -> float:
        """마스크에서 객체 방향 계산"""
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return 0.0
        
        # PCA를 통한 주축 계산
        coords = np.vstack((xs, ys)).astype(np.float64)
        mean = coords.mean(axis=1, keepdims=True)
        centered = coords - mean
        cov = np.cov(centered)
        
        eigvals, eigvecs = np.linalg.eig(cov)
        vx, vy = eigvecs[:, np.argmax(eigvals)]
        
        angle = np.degrees(np.arctan2(vy, vx))
        
        # 각도 정규화 (-180 ~ 180)
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        
        return angle
    
    def _transform_to_robot(self, x_cam: float, y_cam: float, z_cam: float, angle: float) -> Tuple[float, float, float, float]:
        """카메라 좌표를 로봇 좌표로 변환"""
        # 동차 좌표
        cam_vec = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # 변환 행렬 적용
        robot_vec = cam_vec @ self.calibration_matrix
        
        return (robot_vec[0], robot_vec[1], robot_vec[2], angle)
    
    def _publish_detection(self, detection: DetectedObject, detection_type: str):
        """탐지 결과 발행"""
        # 탐지 알림
        topic = "vision/detected"
        self.mqtt_client.publish(topic, json.dumps({
            "detected": True,
            "type": detection_type,
            "confidence": detection.confidence,
            "timestamp": detection.timestamp
        }))
        
        # 좌표 발행
        topic = f"vision/coordinates/{detection_type}"
        self.mqtt_client.publish(topic, json.dumps(detection.to_dict()))
        
        self.logger.info(f"탐지 결과 발행: {topic}")
    
    def _publish_status(self, status: str):
        """상태 발행"""
        topic = "vision/status"
        payload = {
            "status": status,
            "state": self.current_state.value,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
    
    def _publish_error(self, error: str):
        """에러 발행"""
        topic = "vision/error"
        payload = {
            "error": error,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
    
    def cleanup(self):
        """정리"""
        self.stop_scanning()
        if self.pipeline:
            self.pipeline = None
        self.logger.info("비전 시스템 정리 완료")


def main():
    """테스트용 메인 함수"""
    # 설정 로드
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, config['Operation']['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # MQTT 클라이언트
    mqtt_client = mqtt.Client(client_id="vision_controller")
    mqtt_client.connect(config['MQTT']['broker'], config['MQTT']['port'])
    mqtt_client.loop_start()
    
    # 비전 컨트롤러
    vision = VisionController(config, mqtt_client)
    
    if vision.initialize():
        # 테스트: 객체 탐지
        detection = vision.detect_object()
        if detection:
            print(f"객체 탐지 성공: {detection}")
    
    vision.cleanup()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


if __name__ == "__main__":
    main()