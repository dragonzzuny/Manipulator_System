"""
로봇 팔(Arm) 컨트롤러
frrpc를 통한 로봇 제어
"""

import time
import numpy as np
import logging
from typing import List, Optional, Tuple
from src.config_loader import RobotConfig

logger = logging.getLogger(__name__)

try:
    import frrpc
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False
    logger.warning("[ARM] frrpc 모듈 없음 - 시뮬레이션 모드")

class ArmController:
    """
    로봇 팔 제어 클래스
    기존 RobotController의 모든 메서드와 호출 방식을 그대로 유지
    """
    
    def __init__(self, config: RobotConfig):
        """
        Args:
            config: 로봇 설정 객체
        """
        self.config = config
        self.robot = None
        self.connected = False
        
        # 조인트 위치 설정
        self.J_HOME = config.j_home
        self.J_VISION_SCAN = config.j_vision_scan
        self.J_DOWN = config.j_down
        
        # 오프셋 설정
        self.camera_z_offset = config.camera_z_offset
        self.grasp_depth_adjust = config.grasp_depth_adjust
        self.approach_height = config.approach_height
        self.camera_base_offset_deg = config.camera_base_offset_deg
        self.y_offset = config.y_offset
        
        # 작업 공간 제한
        self.x_limit = config.x_limit
        self.y_limit = config.y_limit
        self.z_min = config.z_min
        self.z_max = config.z_max
        
        if HAS_ROBOT:
            self.connect()
    
    def connect(self) -> bool:
        """
        로봇 연결
        기존 코드와 완전히 동일한 방식 유지
        """
        if not HAS_ROBOT:
            logger.warning("[ARM] frrpc 없음 - 연결 건너뜀")
            return False
        
        try:
            self.robot = frrpc.RPC(self.config.ip)
            self.connected = True
            logger.info(f"[ARM] 로봇 연결 성공 - IP: {self.config.ip}")
            return True
        except Exception as e:
            logger.error(f"[ARM] 로봇 연결 실패: {e}")
            self.connected = False
            return False
    
    def move_joint(self, joint_angles: List[float], speed: int = 50) -> bool:
        """
        조인트 각도로 이동
        기존 코드의 파라미터와 호출 방식을 100% 유지
        
        Args:
            joint_angles: 6개 조인트 각도 리스트
            speed: 이동 속도 (0-100)
        """
        if not self.connected:
            logger.debug("[ARM] 시뮬레이션 모드 - move_joint 스킵")
            return True
        
        try:
            # 기존 코드와 완전히 동일한 순서와 타입 변환 유지
            joint_angles = [float(j) for j in joint_angles]
            
            # Forward Kinematics로 pose 계산
            fk_result = self.robot.GetForwardKin(joint_angles)
            if fk_result[0] != 0:
                logger.error(f"[ARM] FK 계산 실패: {fk_result[0]}")
                return False
            
            pose = [float(fk_result[i]) for i in range(1, 7)]
            
            # 기존 코드와 동일한 파라미터 유지
            EP = [0.0, 0.0, 0.0, 0.0]
            DP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            # MoveJ 호출 - 파라미터 순서와 타입 완전 유지
            self.robot.MoveJ(joint_angles, pose, 0, 0, float(speed), 60.0, 100.0, EP, -1.0, 0, DP)
            
            time.sleep(0.5)  # 기존 코드의 대기 시간 유지
            return self.wait_motion_done()
            
        except Exception as e:
            logger.error(f"[ARM] 조인트 이동 실패: {e}")
            return False
    
    def move_pose(self, target_pose: List[float], speed: int = 30) -> bool:
        """
        카테시안 포즈로 이동
        기존 코드의 IK 계산과 MoveJ 호출 방식 완전 유지
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] 6DOF 포즈
            speed: 이동 속도
        """
        if not self.connected:
            logger.debug("[ARM] 시뮬레이션 모드 - move_pose 스킵")
            return True
        
        try:
            target_pose = [float(p) for p in target_pose]
            
            # Inverse Kinematics 계산
            ik_result = self.robot.GetInverseKin(0, target_pose, -1)
            if ik_result[0] != 0:
                logger.error(f"[ARM] IK 계산 실패: {ik_result[0]}")
                return False
            
            joints = [float(ik_result[i]) for i in range(1, 7)]
            
            # 기존 코드와 동일한 파라미터
            EP = [0.0, 0.0, 0.0, 0.0]
            DP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            self.robot.MoveJ(joints, target_pose, 0, 0, float(speed), 60.0, 100.0, EP, -1.0, 0, DP)
            
            return self.wait_motion_done()
            
        except Exception as e:
            logger.error(f"[ARM] 포즈 이동 실패: {e}")
            return False
    
    def wait_motion_done(self, timeout: float = 20) -> bool:
        """
        모션 완료 대기
        기존 코드의 대기 로직 완전 유지
        
        Args:
            timeout: 최대 대기 시간 (초)
        """
        if not self.connected:
            return True
        
        start_time = time.time()
        last_joints = np.array(self.get_joint_pose())
        
        while time.time() - start_time < timeout:
            time.sleep(0.2)
            current_joints = np.array(self.get_joint_pose())
            
            # 기존 코드의 임계값 0.1 유지
            if np.sum(np.abs(current_joints - last_joints)) < 0.1:
                logger.debug("[ARM] 모션 완료")
                return True
            
            last_joints = current_joints
        
        logger.error(f"[ARM] 모션 대기 시간 초과 ({timeout}초)")
        return False
    
    def move_gripper(self, position: float, speed: int = 50) -> bool:
        """
        그리퍼 제어
        기존 MoveGripper 호출 방식 완전 유지
        
        Args:
            position: 그리퍼 위치 (0: 닫힘, 100: 열림)
            speed: 동작 속도
        """
        if not self.connected:
            logger.debug(f"[ARM] 시뮬레이션 모드 - 그리퍼 {position}")
            return True
        
        try:
            # 기존 코드의 파라미터 완전 유지
            self.robot.MoveGripper(1, int(position), int(speed), 50, 5000, 0, 0, 0.0, 0, 0)
            time.sleep(1)  # 기존 대기 시간 유지
            return True
        except Exception as e:
            logger.error(f"[ARM] 그리퍼 제어 실패: {e}")
            return False
    
    def get_joint_pose(self) -> List[float]:
        """
        현재 조인트 각도 조회
        """
        if not self.connected:
            return self.J_HOME.copy()
        
        try:
            joints = self.robot.GetActualJointPosDegree(0)
            return [float(joints[i]) for i in range(1, 7)]
        except Exception as e:
            logger.error(f"[ARM] 조인트 각도 조회 실패: {e}")
            return self.J_HOME.copy()
    
    def get_tool_pose(self) -> List[float]:
        """
        현재 툴 포즈 조회
        """
        if not self.connected:
            return [0, 0, 200, 0, 0, 0]
        
        try:
            pose = self.robot.GetActualToolFlangePose(0)
            return [float(pose[i]) for i in range(1, 7)]
        except Exception as e:
            logger.error(f"[ARM] 툴 포즈 조회 실패: {e}")
            return [0, 0, 200, 0, 0, 0]
    
    def get_gripper_position(self) -> Optional[float]:
        """
        그리퍼 현재 위치 조회
        
        Returns:
            그리퍼 위치 (0-100) 또는 None
        """
        if not self.connected:
            return 100.0
        
        try:
            # 실제 API 확인 필요 - 현재는 더미 구현
            # TODO: GetGripperPosition API 확인
            return 100.0
        except Exception as e:
            logger.error(f"[ARM] 그리퍼 위치 조회 실패: {e}")
            return None
    
    def check_workspace(self, x: float, y: float, z: float) -> bool:
        """
        작업 공간 범위 확인
        기존 코드의 범위 체크 로직 유지
        
        Args:
            x, y, z: 확인할 좌표
            
        Returns:
            작업 공간 내부 여부
        """
        # X, Y 범위 체크
        if abs(x) > self.x_limit or abs(y) > self.y_limit:
            logger.warning(f"[ARM] 작업공간 벗어남: x={x:.1f}, y={y:.1f} (제한: ±{self.x_limit})")
            return False
        
        # Z 범위 체크
        if z < self.z_min or z > self.z_max:
            logger.warning(f"[ARM] Z축 범위 벗어남: z={z:.1f} (범위: {self.z_min}~{self.z_max})")
            return False
        
        return True
    
    def normalize_angle(self, angle: float) -> float:
        """
        각도를 -180~180 범위로 정규화
        기존 코드의 normalize_angle 함수 동일 구현
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def calculate_robot_angle(self, vision_angle_deg: float, current_rz: float) -> float:
        """
        비전 각도를 로봇 회전각으로 변환
        기존 코드의 변환 로직 완전 유지
        
        Args:
            vision_angle_deg: 비전에서 계산한 각도
            current_rz: 현재 로봇 Z축 회전각
            
        Returns:
            로봇 Z축 회전각
        """
        # 기존 코드의 변환 공식 그대로 유지
        target_rz = self.camera_base_offset_deg - vision_angle_deg
        
        # 각도 정규화
        target_rz = self.normalize_angle(target_rz)
        target_rz_alt = self.normalize_angle(target_rz + 180)
        
        # 현재 각도에서 가까운 각도 선택
        if abs(self.normalize_angle(target_rz_alt - current_rz)) < abs(self.normalize_angle(target_rz - current_rz)):
            target_rz = target_rz_alt
        
        logger.info(f"[ARM] 각도 변환: 비전={vision_angle_deg:.1f}° → 로봇={target_rz:.1f}°")
        
        return target_rz
    
    def move_to_home(self, speed: int = 50) -> bool:
        """홈 위치로 이동"""
        logger.info("[ARM] 홈 위치로 이동")
        return self.move_joint(self.J_HOME, speed)
    
    def move_to_vision_scan(self, speed: int = 50) -> bool:
        """비전 스캔 위치로 이동"""
        logger.info("[ARM] 비전 스캔 위치로 이동")
        return self.move_joint(self.J_VISION_SCAN, speed)
    
    def move_to_down(self, speed: int = 50) -> bool:
        """내려놓기 위치로 이동"""
        logger.info("[ARM] 내려놓기 위치로 이동")
        return self.move_joint(self.J_DOWN, speed)
    
    def open_gripper(self, speed: int = 80) -> bool:
        """그리퍼 열기"""
        logger.info("[ARM] 그리퍼 열기")
        return self.move_gripper(100, speed)
    
    def close_gripper(self, speed: int = 50) -> bool:
        """그리퍼 닫기"""
        logger.info("[ARM] 그리퍼 닫기")
        return self.move_gripper(0, speed)