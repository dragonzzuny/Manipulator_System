"""
설정 파일 로더 모듈
YAML 설정을 읽어 dataclass로 변환하여 타입 안정성과 자동완성 지원
"""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

@dataclass
class RobotConfig:
    """로봇 팔 관련 설정"""
    ip: str
    gripper_open_width_mm: float
    gripper_jaw_length_mm: float
    gripper_jaw_thickness_mm: float
    gripper_depth: float
    approach_clearance: float
    
    # 조인트 위치
    j_home: List[float]
    j_vision_scan: List[float]
    j_down: List[float]
    
    # 오프셋
    camera_z_offset: float
    grasp_depth_adjust: float
    approach_height: float
    camera_base_offset_deg: float
    y_offset: float
    
    # 작업 공간
    x_limit: float
    y_limit: float
    z_min: float
    z_max: float

@dataclass
class AMRConfig:
    """AMR 관련 설정"""
    host: str
    move_threshold: float
    check_interval: float
    max_wait_time: float
    stop_count_threshold: int
    
    # 지점 이름
    grab_point: str
    release_point: str
    start_point: str

@dataclass
class VisionConfig:
    """비전 시스템 관련 설정"""
    # 모델 경로
    segmentation_model_path: str
    keypoint_model_path: str
    
    # 카메라
    camera_width: int
    camera_height: int
    camera_fps: int
    warmup_frames: int
    multi_frame_count: int
    frame_delay: float
    
    # 탐지 파라미터
    confidence_threshold: float
    min_vector_distance: float
    min_inside_score: int
    roi_radius_factor: float
    search_step: int
    
    # 안전 마스크
    primary_erosion_kernel: int
    secondary_erosion_kernel: int
    min_safe_area_primary: int
    min_safe_area_secondary: int
    
    # 점수 계산
    distance_penalty_weight: float
    collision_penalty_weight: float
    edge_bonus_weight: float
    angle_confidence_threshold: float

@dataclass
class RetryConfig:
    """재시도 정책 설정"""
    vision_max_retries: int
    quick_redetect_attempts: int
    pick_max_attempts: int
    gripper_verification_attempts: int

@dataclass
class SystemConfig:
    """시스템 전반 설정"""
    enable_debug_capture: bool
    save_dir: str
    log_file: str
    log_level: str
    
    # 타이밍
    capture_wait_time: float
    analysis_display_time: float
    gripper_close_wait: float
    post_pick_wait: float

@dataclass
class Config:
    """전체 설정 컨테이너"""
    robot: RobotConfig
    amr: AMRConfig
    vision: VisionConfig
    retry_policy: RetryConfig
    system: SystemConfig
    calibration_matrix: np.ndarray

class ConfigLoader:
    """YAML 설정 파일 로더"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 기본 경로: 현재 파일 기준으로 상대 경로 찾기
            current_file = Path(__file__).absolute()
            project_root = current_file.parent.parent  # src의 상위 디렉토리
            config_path = project_root / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        
        # 경로 존재 확인
        if not self.config_path.exists():
            # 대체 경로 시도
            alternative_paths = [
                Path.cwd() / "config" / "settings.yaml",
                Path(__file__).parent / ".." / "config" / "settings.yaml",
                Path("config/settings.yaml"),
            ]
            
            for alt_path in alternative_paths:
                alt_path = alt_path.resolve()
                if alt_path.exists():
                    self.config_path = alt_path
                    print(f"설정 파일 발견: {self.config_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"설정 파일을 찾을 수 없습니다.\n"
                    f"시도한 경로:\n"
                    f"  - {config_path}\n" +
                    "\n".join(f"  - {p.resolve()}" for p in alternative_paths)
                )
    
    def load(self) -> Config:
        """설정 파일을 로드하여 Config 객체 반환"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 로봇 설정
        robot_data = data['robot']
        robot_config = RobotConfig(
            ip=robot_data['ip'],
            gripper_open_width_mm=robot_data['gripper']['open_width_mm'],
            gripper_jaw_length_mm=robot_data['gripper']['jaw_length_mm'],
            gripper_jaw_thickness_mm=robot_data['gripper']['jaw_thickness_mm'],
            gripper_depth=robot_data['gripper']['depth'],
            approach_clearance=robot_data['gripper']['approach_clearance'],
            j_home=robot_data['joints']['home'],
            j_vision_scan=robot_data['joints']['vision_scan'],
            j_down=robot_data['joints']['down'],
            camera_z_offset=robot_data['offsets']['camera_z_offset'],
            grasp_depth_adjust=robot_data['offsets']['grasp_depth_adjust'],
            approach_height=robot_data['offsets']['approach_height'],
            camera_base_offset_deg=robot_data['offsets']['camera_base_offset_deg'],
            y_offset=robot_data['offsets']['y_offset'],
            x_limit=robot_data['workspace']['x_limit'],
            y_limit=robot_data['workspace']['y_limit'],
            z_min=robot_data['workspace']['z_min'],
            z_max=robot_data['workspace']['z_max']
        )
        
        # AMR 설정
        amr_data = data['amr']
        amr_config = AMRConfig(
            host=amr_data['host'],
            move_threshold=amr_data['move_threshold'],
            check_interval=amr_data['check_interval'],
            max_wait_time=amr_data['max_wait_time'],
            stop_count_threshold=amr_data['stop_count_threshold'],
            grab_point=amr_data['points']['grab'],
            release_point=amr_data['points']['release'],
            start_point=amr_data['points']['start']
        )
        
        # 비전 설정
        vision_data = data['vision']
        vision_config = VisionConfig(
            segmentation_model_path=vision_data['models']['segmentation'],
            keypoint_model_path=vision_data['models']['keypoint'],
            camera_width=vision_data['camera']['width'],
            camera_height=vision_data['camera']['height'],
            camera_fps=vision_data['camera']['fps'],
            warmup_frames=vision_data['camera']['warmup_frames'],
            multi_frame_count=vision_data['camera']['multi_frame_count'],
            frame_delay=vision_data['camera']['frame_delay'],
            confidence_threshold=vision_data['detection']['confidence_threshold'],
            min_vector_distance=vision_data['detection']['min_vector_distance'],
            min_inside_score=vision_data['detection']['min_inside_score'],
            roi_radius_factor=vision_data['detection']['roi_radius_factor'],
            search_step=vision_data['detection']['search_step'],
            primary_erosion_kernel=vision_data['safety_mask']['primary_erosion_kernel'],
            secondary_erosion_kernel=vision_data['safety_mask']['secondary_erosion_kernel'],
            min_safe_area_primary=vision_data['safety_mask']['min_safe_area_primary'],
            min_safe_area_secondary=vision_data['safety_mask']['min_safe_area_secondary'],
            distance_penalty_weight=vision_data['scoring']['distance_penalty_weight'],
            collision_penalty_weight=vision_data['scoring']['collision_penalty_weight'],
            edge_bonus_weight=vision_data['scoring']['edge_bonus_weight'],
            angle_confidence_threshold=vision_data['scoring']['angle_confidence_threshold']
        )
        
        # 재시도 정책
        retry_data = data['retry_policy']
        retry_config = RetryConfig(
            vision_max_retries=retry_data['vision_max_retries'],
            quick_redetect_attempts=retry_data['quick_redetect_attempts'],
            pick_max_attempts=retry_data['pick_max_attempts'],
            gripper_verification_attempts=retry_data['gripper_verification_attempts']
        )
        
        # 시스템 설정
        system_data = data['system']
        system_config = SystemConfig(
            enable_debug_capture=system_data['enable_debug_capture'],
            save_dir=system_data['save_dir'],
            log_file=system_data['log_file'],
            log_level=system_data['log_level'],
            capture_wait_time=system_data['timing']['capture_wait'],
            analysis_display_time=system_data['timing']['analysis_display'],
            gripper_close_wait=system_data['timing']['gripper_close_wait'],
            post_pick_wait=system_data['timing']['post_pick_wait']
        )
        
        # 캘리브레이션 행렬
        calibration_matrix = np.array(data['calibration']['matrix'])
        
        return Config(
            robot=robot_config,
            amr=amr_config,
            vision=vision_config,
            retry_policy=retry_config,
            system=system_config,
            calibration_matrix=calibration_matrix
        )
    
    def setup_logging(self, config: Config):
        """로깅 설정"""
        log_level = getattr(logging, config.system.log_level)
        
        # 로그 디렉토리 생성
        log_path = Path(config.system.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - [%(levelname)s][%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(config.system.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )