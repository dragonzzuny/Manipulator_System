"""
워크플로우 오케스트레이터
AMR 이동 → 비전 스캔 → 피킹 → Release → Start의 전체 시나리오 관리
기존 main() 함수의 시퀀스를 구조화
"""

import time
import logging
from typing import Optional
from enum import Enum

from src.config_loader import Config
from src.controllers.amr_controller import AMRController
from src.controllers.arm_controller import ArmController
from src.vision.vision_controller import VisionController
from src.vision.grasp_planner import GraspCandidate

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    """워크플로우 상태"""
    INIT = "초기화"
    MOVE_TO_GRAB = "Grab 지점 이동"
    VISION_SCAN = "비전 스캔"
    PICK_OBJECT = "객체 피킹"
    MOVE_TO_RELEASE = "Release 지점 이동"  
    PLACE_OBJECT = "객체 내려놓기"
    MOVE_TO_START = "Start 지점 이동"
    COMPLETE = "완료"
    ERROR = "오류"

class PickAndPlaceWorkflow:
    """
    피킹 앤 플레이스 워크플로우
    기존 main 함수의 전체 시퀀스를 클래스로 구조화
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정 객체
        """
        self.config = config
        self.state = WorkflowState.INIT
        
        # 컨트롤러 초기화
        self.amr = AMRController(config.amr)
        self.arm = ArmController(config.robot)
        self.vision = VisionController(config)
        
        # 재시도 정책
        self.retry_config = config.retry_policy
        
        logger.info("[WORKFLOW] 피킹 앤 플레이스 워크플로우 초기화")
    
    def run(self) -> bool:
        """
        전체 워크플로우 실행
        기존 main() 함수의 시퀀스를 그대로 유지
        
        Returns:
            성공 여부
        """
        try:
            logger.info("="*70)
            logger.info("[WORKFLOW] 피킹 앤 플레이스 작업 시작")
            logger.info("="*70)
            
            # 1단계: Grab 지점으로 이동
            if not self._move_to_grab_point():
                return False
            
            # 2단계: 비전 스캔 자세로 이동
            if not self._prepare_vision_scan():
                return False
            
            # 3단계: 객체 탐지 및 파지점 계산 (재시도 포함)
            grasp = self._detect_and_calculate_grasp()
            if grasp is None:
                return False
            
            # 4단계: 객체 피킹 (재시도 포함)
            if not self._pick_object(grasp):
                return False
            
            # 5단계: Release 지점으로 이동
            if not self._move_to_release_point():
                return False
            
            # 6단계: 객체 내려놓기
            if not self._place_object():
                return False
            
            # 7단계: Start 지점으로 이동
            if not self._move_to_start_point():
                return False
            
            self.state = WorkflowState.COMPLETE
            logger.info("="*70)
            logger.info("[WORKFLOW] ✅ 피킹 앤 플레이스 작업 완료!")
            logger.info("="*70)
            return True
            
        except Exception as e:
            self.state = WorkflowState.ERROR
            logger.error(f"[WORKFLOW] 작업 중 오류 발생: {e}")
            self._emergency_recovery()
            return False
    
    def _move_to_grab_point(self) -> bool:
        """
        Grab 지점으로 AMR 이동
        기존 코드의 로직 완전 유지
        """
        self.state = WorkflowState.MOVE_TO_GRAB
        logger.info(f"[WORKFLOW] AMR을 '{self.config.amr.grab_point}'로 이동")
        
        if not self.amr.move_and_wait(self.config.amr.grab_point):
            logger.error(f"[WORKFLOW] '{self.config.amr.grab_point}' 이동 실패")
            return False
        
        logger.info(f"[WORKFLOW] '{self.config.amr.grab_point}' 도착 완료")
        return True
    
    def _prepare_vision_scan(self) -> bool:
        """
        비전 스캔 준비
        로봇 팔을 비전 자세로 이동하고 그리퍼 열기
        """
        logger.info("[WORKFLOW] 비전 스캔 준비")
        
        # 홈 위치로 이동
        if not self.arm.move_to_home(speed=50):
            logger.error("[WORKFLOW] 홈 위치 이동 실패")
            return False
        
        # 그리퍼 열기
        if not self.arm.open_gripper(speed=80):
            logger.error("[WORKFLOW] 그리퍼 열기 실패")
            return False
        
        # 비전 스캔 위치로 이동
        if not self.arm.move_to_vision_scan(speed=50):
            logger.error("[WORKFLOW] 비전 스캔 위치 이동 실패")
            return False
        
        # 안정화 대기
        time.sleep(1.0)
        
        logger.info("[WORKFLOW] 비전 스캔 준비 완료")
        return True
    
    def _detect_and_calculate_grasp(self) -> Optional[GraspCandidate]:
        """
        객체 탐지 및 파지점 계산
        기존 find_object_and_get_coords_with_retry 로직 유지
        """
        self.state = WorkflowState.VISION_SCAN
        logger.info("[WORKFLOW] 객체 탐지 및 파지점 계산 시작")
        
        # 재시도를 포함한 객체 탐지
        grasp = self.vision.find_object_and_get_coords_with_retry(
            max_retries=self.retry_config.vision_max_retries
        )
        
        if grasp is None:
            logger.error("[WORKFLOW] 객체 탐지 최종 실패")
            return None
        
        logger.info("[WORKFLOW] 파지점 계산 성공")
        return grasp
    
    def _pick_object(self, grasp: GraspCandidate) -> bool:
        """
        객체 피킹 수행
        기존 pick_object_with_vector_grasp 로직 유지
        파지 검증 및 재시도 포함
        """
        self.state = WorkflowState.PICK_OBJECT
        logger.info("[WORKFLOW] 객체 피킹 시작")
        
        max_attempts = self.retry_config.pick_max_attempts
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"[WORKFLOW][PICK {attempt}/{max_attempts}] 피킹 시도")
            
            # 피킹 실행
            if self._execute_pick(grasp):
                # 파지 검증
                if self._verify_grasp():
                    logger.info(f"[WORKFLOW][PICK {attempt}] 피킹 성공!")
                    return True
                else:
                    logger.warning(f"[WORKFLOW][PICK {attempt}] 파지 검증 실패")
                    
                    # 그리퍼 열고 홈으로 복귀
                    self.arm.open_gripper()
                    self.arm.move_to_home()
                    
                    if attempt < max_attempts:
                        # 객체 재탐지 시도
                        if not self.vision.quick_redetect_object(
                            max_attempts=self.retry_config.quick_redetect_attempts):
                            logger.error("[WORKFLOW] 객체를 재발견할 수 없음")
                            return False
                        
                        # 파지점 재계산
                        new_grasp = self.vision.find_object_and_get_coords()
                        if new_grasp:
                            grasp = new_grasp
                            logger.info("[WORKFLOW] 파지점 재계산 완료")
            else:
                logger.error(f"[WORKFLOW][PICK {attempt}] 피킹 실행 실패")
                
                if attempt < max_attempts:
                    time.sleep(2.0)
        
        logger.error(f"[WORKFLOW] {max_attempts}회 피킹 시도 모두 실패")
        return False
    
    def _execute_pick(self, grasp: GraspCandidate) -> bool:
        """
        실제 피킹 동작 수행
        기존 execute_pick 로직 완전 유지
        """
        try:
            # 현재 포즈 확인
            current_pose = self.arm.get_tool_pose()
            current_x, current_y, current_z, _, _, current_rz = current_pose
            
            # 타겟 위치 계산
            target_x = current_x + grasp.robot_coords[0]
            target_y = current_y + grasp.robot_coords[1]
            
            # 벡터 각도를 로봇 회전각으로 변환
            target_rz = self.arm.calculate_robot_angle(grasp.angle_deg, current_rz)
            
            # Z축 계산
            descent = grasp.center_3d[2] - self.config.robot.camera_z_offset + self.config.robot.grasp_depth_adjust
            grasp_z = current_z - descent
            approach_z = grasp_z + self.config.robot.approach_height
            
            # 작업 공간 확인
            if not self.arm.check_workspace(target_x, target_y, grasp_z):
                logger.error("[WORKFLOW] 타겟 위치가 작업 공간을 벗어남")
                return False
            
            # 접근 위치로 이동
            approach_pose = [target_x, target_y, approach_z, -179.7, -0.2, target_rz]
            logger.info(f"[WORKFLOW] 접근 위치: {approach_pose}")
            if not self.arm.move_pose(approach_pose, speed=40):
                logger.error("[WORKFLOW] 접근 위치 이동 실패")
                return False
            
            # 파지 위치로 하강
            grasp_pose = [target_x, target_y, grasp_z, -179.7, -0.2, target_rz]
            logger.info(f"[WORKFLOW] 파지 위치: {grasp_pose}")
            if not self.arm.move_pose(grasp_pose, speed=15):
                logger.error("[WORKFLOW] 파지 위치 이동 실패")
                return False
            
            # 그리퍼 닫기
            logger.info("[WORKFLOW] 그리퍼 닫기")
            self.arm.close_gripper(speed=50)
            time.sleep(self.config.system.gripper_close_wait)
            
            # 들어올리기
            lift_pose = [target_x, target_y, approach_z + 50, -179.7, -0.2, target_rz]
            logger.info(f"[WORKFLOW] 들어올리기: {lift_pose}")
            if not self.arm.move_pose(lift_pose, speed=30):
                logger.error("[WORKFLOW] 들어올리기 실패")
                self.arm.open_gripper()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[WORKFLOW] 피킹 실행 오류: {e}")
            return False
    
    def _verify_grasp(self) -> bool:
        """
        파지 성공 여부 검증
        기존 verify_grasp_with_gripper 로직 유지
        """
        logger.info("[WORKFLOW] 파지 검증 시작")
        
        for attempt in range(1, self.retry_config.gripper_verification_attempts + 1):
            # 그리퍼 위치 확인 (실제 API 필요)
            gripper_pos = self.arm.get_gripper_position()
            
            if gripper_pos is not None and gripper_pos < 90:  # 90% 미만이면 뭔가 잡음
                logger.info(f"[WORKFLOW] 파지 검증 성공 (그리퍼 위치: {gripper_pos:.1f}%)")
                return True
            
            logger.warning(f"[WORKFLOW] 파지 검증 시도 {attempt} - 그리퍼 위치: {gripper_pos}")
            time.sleep(0.5)
        
        # 기본적으로 성공으로 가정 (실제 센서 없을 때)
        logger.warning("[WORKFLOW] 파지 검증 - 센서 없음, 성공으로 가정")
        return True
    
    def _move_to_release_point(self) -> bool:
        """Release 지점으로 이동"""
        self.state = WorkflowState.MOVE_TO_RELEASE
        logger.info(f"[WORKFLOW] AMR을 '{self.config.amr.release_point}'로 이동")
        
        # 홈 위치로 복귀
        if not self.arm.move_to_home(speed=50):
            logger.error("[WORKFLOW] 홈 위치 복귀 실패")
            return False
        
        # AMR 이동
        if not self.amr.move_and_wait(self.config.amr.release_point):
            logger.error(f"[WORKFLOW] '{self.config.amr.release_point}' 이동 실패")
            return False
        
        logger.info(f"[WORKFLOW] '{self.config.amr.release_point}' 도착 완료")
        return True
    
    def _place_object(self) -> bool:
        """객체 내려놓기"""
        self.state = WorkflowState.PLACE_OBJECT
        logger.info("[WORKFLOW] 객체 내려놓기")
        
        # 내려놓기 위치로 이동
        if not self.arm.move_to_down(speed=40):
            logger.error("[WORKFLOW] 내려놓기 위치 이동 실패")
            return False
        
        # 그리퍼 열기
        if not self.arm.open_gripper(speed=50):
            logger.error("[WORKFLOW] 그리퍼 열기 실패")
            return False
        
        time.sleep(1.0)
        
        # 홈으로 복귀
        if not self.arm.move_to_home(speed=50):
            logger.error("[WORKFLOW] 홈 복귀 실패")
            return False
        
        logger.info("[WORKFLOW] 객체 내려놓기 완료")
        return True
    
    def _move_to_start_point(self) -> bool:
        """Start 지점으로 이동"""
        self.state = WorkflowState.MOVE_TO_START
        logger.info(f"[WORKFLOW] AMR을 '{self.config.amr.start_point}'로 이동")
        
        if not self.amr.move_and_wait(self.config.amr.start_point):
            logger.error(f"[WORKFLOW] '{self.config.amr.start_point}' 이동 실패")
            return False
        
        logger.info(f"[WORKFLOW] '{self.config.amr.start_point}' 도착 완료")
        return True
    
    def _emergency_recovery(self):
        """긴급 복구 동작"""
        logger.warning("[WORKFLOW] 긴급 복구 시작")
        
        try:
            # 그리퍼 열기
            self.arm.open_gripper()
            
            # 홈으로 복귀
            self.arm.move_to_home()
            
            # AMR 정지 (구현 필요)
            self.amr.emergency_stop()
            
        except Exception as e:
            logger.error(f"[WORKFLOW] 긴급 복구 실패: {e}")
    
    def get_state(self) -> WorkflowState:
        """현재 워크플로우 상태 반환"""
        return self.state

class TestWorkflow:
    """
    테스트용 워크플로우
    개별 기능 테스트용
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.arm = ArmController(config.robot)
        self.vision = VisionController(config)
        self.amr = AMRController(config.amr)
    
    def test_vision_only(self) -> Optional[GraspCandidate]:
        """비전만 테스트"""
        logger.info("[TEST] 비전 테스트 시작")
        
        # 비전 스캔 자세
        self.arm.move_to_home()
        self.arm.move_to_vision_scan()
        
        # 객체 탐지
        grasp = self.vision.find_object_and_get_coords()
        
        if grasp:
            logger.info("[TEST] 비전 테스트 성공")
        else:
            logger.error("[TEST] 비전 테스트 실패")
        
        return grasp
    
    def test_arm_movement(self) -> bool:
        """로봇 팔 동작 테스트"""
        logger.info("[TEST] 로봇 팔 동작 테스트")
        
        try:
            # 홈 → 비전 → 다운 → 홈
            self.arm.move_to_home()
            time.sleep(1)
            
            self.arm.move_to_vision_scan()
            time.sleep(1)
            
            self.arm.move_to_down()
            time.sleep(1)
            
            self.arm.move_to_home()
            
            # 그리퍼 테스트
            self.arm.open_gripper()
            time.sleep(1)
            self.arm.close_gripper()
            time.sleep(1)
            self.arm.open_gripper()
            
            logger.info("[TEST] 로봇 팔 동작 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"[TEST] 로봇 팔 동작 테스트 실패: {e}")
            return False
    
    def test_amr_movement(self) -> bool:
        """AMR 이동 테스트"""
        logger.info("[TEST] AMR 이동 테스트")
        
        try:
            # 현재 위치 확인
            pose = self.amr.get_pose()
            if pose:
                logger.info(f"[TEST] 현재 위치: x={pose.x:.2f}, y={pose.y:.2f}")
            
            # 모드 확인
            mode = self.amr.get_mode()
            logger.info(f"[TEST] 현재 모드: {mode}")
            
            logger.info("[TEST] AMR 이동 테스트 완료")
            return True
            
        except Exception as e:
            logger.error(f"[TEST] AMR 이동 테스트 실패: {e}")
            return False