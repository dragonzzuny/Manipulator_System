#!/usr/bin/env python3
"""
Robot Arm Controller
MQTT 기반 Fair 로봇팔 제어 모듈
"""

import os
import json
import time
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import yaml
import numpy as np
import frrpc
import paho.mqtt.client as mqtt


class RobotState(Enum):
    """로봇 상태"""
    IDLE = "idle"
    MOVING = "moving"
    PICKING = "picking"
    PLACING = "placing"
    ERROR = "error"


class GripperState(Enum):
    """그리퍼 상태"""
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"
    HOLDING = "holding"


@dataclass
class RobotPose:
    """로봇 포즈 (카테시안)"""
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


class RobotController:
    """Fair 로봇팔 제어 클래스"""
    
    def __init__(self, config: Dict[str, Any], mqtt_client: mqtt.Client):
        self.config = config
        self.mqtt_client = mqtt_client
        self.logger = logging.getLogger(__name__)
        
        # 로봇 설정
        self.robot_ip = config['Robot']['ip']
        self.positions = config['Robot']['positions']
        self.gripper_config = config['Robot']['gripper']
        self.motion_config = config['Robot']['motion']
        self.tool_config = config['Robot']['tool']
        self.workspace_config = config['Robot']['workspace']
        
        # Fair 로봇 연결
        self.robot: Optional[frrpc.RPC] = None
        
        # 동작 파라미터
        self.EP = [0.0, 0.0, 0.0, 0.0]  # 외부 축
        self.DP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 다이나믹 파라미터
        
        # 상태 변수
        self.current_state = RobotState.IDLE
        self.gripper_state = GripperState.OPEN
        self.current_joint: Optional[List[float]] = None
        self.current_pose: Optional[RobotPose] = None
        
        # 안전 플래그
        self.emergency_stop = False
        
    def initialize(self) -> bool:
        """로봇 초기화"""
        try:
            # Fair 로봇 연결
            self.logger.info(f"Fair 로봇 연결 중: {self.robot_ip}")
            self.robot = frrpc.RPC(self.robot_ip)
            
            # 연결 확인
            mode = self.robot.GetControllerMode()
            self.logger.info(f"✅ 로봇 연결 성공 - 모드: {mode}")
            
            # 서보 활성화 확인
            servo_state = self.robot.GetServoState()
            if not servo_state or servo_state[0] != 1:
                self.logger.warning("서보 비활성화 상태, 활성화 중...")
                # self.robot.ServoOn()  # 필요시 활성화
            
            # 홈 포지션으로 이동
            self.go_home()
            
            # 그리퍼 초기화
            self.open_gripper()
            
            self._publish_status("initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"로봇 초기화 실패: {e}")
            self._publish_error(str(e))
            return False
    
    def go_home(self) -> bool:
        """홈 포지션으로 이동"""
        return self.move_to_position("home")
    
    def move_to_position(self, position_name: str) -> bool:
        """사전 정의된 위치로 이동"""
        if position_name not in self.positions:
            self.logger.error(f"알 수 없는 위치: {position_name}")
            return False
        
        try:
            joint_angles = self.positions[position_name]
            pose = self._forward_kinematics(joint_angles)
            
            self.logger.info(f"{position_name} 위치로 이동 중...")
            self.current_state = RobotState.MOVING
            
            # 이동 속도 결정
            if position_name == "home":
                velocity = self.motion_config['velocity_fast']
            else:
                velocity = self.motion_config['velocity_slow']
            
            # MoveJ 명령
            self.robot.MoveJ(
                joint_angles,
                pose,
                0,  # user
                0,  # tool
                velocity,
                self.motion_config['acceleration'],
                self.motion_config['jerk'],
                self.EP,
                -1.0,  # offset_flag
                0,  # offset_pos
                self.DP
            )
            
            # 동작 완료 대기
            if self._wait_for_motion_done():
                self.current_state = RobotState.IDLE
                self._publish_status(f"at_{position_name}")
                self.logger.info(f"✅ {position_name} 위치 도착")
                return True
            
        except Exception as e:
            self.logger.error(f"이동 실패: {e}")
            self.current_state = RobotState.ERROR
            self._publish_error(str(e))
        
        return False
    
    def pick_object(self, coordinates: Dict[str, float]) -> bool:
        """객체 피킹"""
        try:
            self.current_state = RobotState.PICKING
            self.logger.info("=" * 60)
            self.logger.info("객체 피킹 시작")
            
            # 좌표 추출
            tx = float(coordinates.get('x', 0))
            ty = float(coordinates.get('y', 0))
            tz = float(coordinates.get('z', 0))
            angle = float(coordinates.get('angle', 0))
            
            self.logger.info(f"목표 좌표: X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f}, Angle={angle:.1f}")
            
            # 작업 공간 체크
            distance = np.sqrt(tx**2 + ty**2)
            if distance > self.workspace_config['max_reach']:
                self.logger.error(f"도달 불가: {distance:.1f}mm > {self.workspace_config['max_reach']}mm")
                return False
            
            # 현재 포즈 획득
            curr_pose_list = self.robot.GetActualToolFlangePose(0)
            curr_pose = RobotPose(*curr_pose_list[1:7])
            
            # Z축 오프셋 적용
            tz_adjusted = tz - self.tool_config['z_offset']
            
            # 각도 보정 (조건부 90도 회전)
            final_angle = self._calculate_rotation(curr_pose.rz, angle)
            
            # 1단계: 접근 위치
            approach_pose = RobotPose(
                curr_pose.x + tx,
                curr_pose.y + ty,
                curr_pose.z - tz_adjusted + self.tool_config['approach_height'],
                curr_pose.rx,
                curr_pose.ry,
                final_angle
            )
            
            if not self._move_to_pose(approach_pose, self.motion_config['velocity_slow']):
                return False
            
            # 2단계: 파지 위치로 하강
            grasp_pose = RobotPose(
                approach_pose.x,
                approach_pose.y,
                approach_pose.z - self.tool_config['grasp_depth'],
                approach_pose.rx,
                approach_pose.ry,
                approach_pose.rz
            )
            
            if not self._move_to_pose(grasp_pose, self.motion_config['velocity_pick']):
                return False
            
            # 3단계: 그리퍼 닫기
            if not self.close_gripper():
                return False
            
            self.gripper_state = GripperState.HOLDING
            
            # 4단계: 들어올리기
            lift_pose = RobotPose(
                grasp_pose.x,
                grasp_pose.y,
                grasp_pose.z + self.tool_config['lift_height'],
                grasp_pose.rx,
                grasp_pose.ry,
                grasp_pose.rz
            )
            
            if not self._move_to_pose(lift_pose, self.motion_config['velocity_slow']):
                return False
            
            # 5단계: 홈 복귀
            self.go_home()
            
            self.current_state = RobotState.IDLE
            self._publish_status("object_picked")
            self.logger.info("✅ 객체 피킹 완료")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"피킹 실패: {e}")
            self.current_state = RobotState.ERROR
            self._publish_error(str(e))
            return False
    
    def place_object(self, coordinates: Optional[Dict[str, float]] = None) -> bool:
        """객체 배치"""
        try:
            self.current_state = RobotState.PLACING
            self.logger.info("객체 배치 시작")
            
            if coordinates:
                # 특정 위치에 배치
                tx = float(coordinates.get('x', 0))
                ty = float(coordinates.get('y', 0))
                tz = float(coordinates.get('z', 0))
                
                curr_pose_list = self.robot.GetActualToolFlangePose(0)
                curr_pose = RobotPose(*curr_pose_list[1:7])
                
                place_pose = RobotPose(
                    curr_pose.x + tx,
                    curr_pose.y + ty,
                    curr_pose.z - tz,
                    curr_pose.rx,
                    curr_pose.ry,
                    curr_pose.rz
                )
                
                if not self._move_to_pose(place_pose, self.motion_config['velocity_slow']):
                    return False
            
            # 그리퍼 열기
            if not self.open_gripper():
                return False
            
            self.gripper_state = GripperState.OPEN
            time.sleep(0.5)
            
            # 홈 복귀
            self.go_home()
            
            self.current_state = RobotState.IDLE
            self._publish_status("object_placed")
            self.logger.info("✅ 객체 배치 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"배치 실패: {e}")
            self.current_state = RobotState.ERROR
            self._publish_error(str(e))
            return False
    
    def open_gripper(self) -> bool:
        """그리퍼 열기"""
        return self._move_gripper(self.gripper_config['open_position'])
    
    def close_gripper(self) -> bool:
        """그리퍼 닫기"""
        return self._move_gripper(self.gripper_config['close_position'])
    
    def _move_gripper(self, position: int) -> bool:
        """그리퍼 이동"""
        try:
            self.logger.info(f"그리퍼 이동: {position}%")
            self.gripper_state = GripperState.MOVING
            
            self.robot.MoveGripper(
                self.gripper_config['index'],
                position,
                self.gripper_config['velocity'],
                self.gripper_config['force'],
                self.gripper_config['max_time'],
                0, 0, 0.0, 0, 0
            )
            
            # 동작 완료 대기
            start_time = time.time()
            timeout = self.gripper_config['max_time'] / 1000.0 + 1.0
            
            while time.time() - start_time < timeout:
                try:
                    state = self.robot.GetGripperMotionDone()
                    if len(state) >= 3 and state[2] == 1:
                        self.gripper_state = GripperState.OPEN if position >= 90 else GripperState.CLOSED
                        self.logger.info(f"✅ 그리퍼 이동 완료: {self.gripper_state.value}")
                        return True
                except:
                    pass
                time.sleep(0.1)
            
            self.logger.warning("그리퍼 동작 타임아웃")
            return False
            
        except Exception as e:
            self.logger.error(f"그리퍼 제어 실패: {e}")
            return False
    
    def _move_to_pose(self, pose: RobotPose, velocity: float) -> bool:
        """포즈로 이동"""
        try:
            # IK 계산
            joint_angles = self._inverse_kinematics(pose.to_list())
            
            if not self._validate_joint_angles(joint_angles):
                self.logger.error("조인트 각도 범위 초과")
                return False
            
            # MoveJ 명령
            self.robot.MoveJ(
                joint_angles,
                pose.to_list(),
                0, 0,
                velocity,
                self.motion_config['acceleration'],
                self.motion_config['jerk'],
                self.EP,
                -1.0, 0,
                self.DP
            )
            
            return self._wait_for_motion_done()
            
        except Exception as e:
            self.logger.error(f"포즈 이동 실패: {e}")
            return False
    
    def _calculate_rotation(self, current_rz: float, object_angle: float) -> float:
        """조건부 90도 회전 계산"""
        # 기본적으로 90도 추가
        predicted_rz = current_rz + object_angle + 90.0
        
        # 각도 범위 체크
        if predicted_rz > 180:
            perpendicular_offset = -90.0
        elif predicted_rz < -180:
            perpendicular_offset = 45.0
        else:
            perpendicular_offset = 45.0
        
        final_angle = object_angle + perpendicular_offset
        final_rz = current_rz + final_angle
        
        # 정규화
        while final_rz > 180:
            final_rz -= 360
        while final_rz < -180:
            final_rz += 360
        
        return final_rz
    
    def _forward_kinematics(self, joint_angles: List[float]) -> List[float]:
        """순기구학"""
        result = self.robot.GetForwardKin(joint_angles)
        return result[1:7] if len(result) > 6 else [0] * 6
    
    def _inverse_kinematics(self, pose: List[float]) -> List[float]:
        """역기구학"""
        result = self.robot.GetInverseKin(0, pose, -1)
        return result[1:7] if len(result) > 6 else [0] * 6
    
    def _validate_joint_angles(self, joint_angles: List[float]) -> bool:
        """조인트 각도 검증"""
        for i, angle in enumerate(joint_angles):
            if abs(angle) > 360:
                self.logger.warning(f"조인트 {i+1} 범위 초과: {angle}°")
                return False
        return True
    
    def _wait_for_motion_done(self, timeout: float = 20.0) -> bool:
        """동작 완료 대기"""
        start_time = time.time()
        
        try:
            last_joints = np.array(self.robot.GetActualJointPosDegree(0)[1:7])
        except:
            return False
        
        while time.time() - start_time < timeout:
            if self.emergency_stop:
                self.logger.warning("비상 정지 요청")
                self.robot.StopMotion()
                return False
            
            time.sleep(0.2)
            
            try:
                current_joints = np.array(self.robot.GetActualJointPosDegree(0)[1:7])
                joint_diff = np.sum(np.abs(current_joints - last_joints))
                
                if joint_diff < 0.1:  # 0.1도 미만 변화
                    return True
                    
                last_joints = current_joints
            except:
                time.sleep(1)
        
        self.logger.warning("동작 완료 타임아웃")
        return False
    
    def emergency_stop(self):
        """비상 정지"""
        self.emergency_stop = True
        try:
            self.robot.StopMotion()
            self.logger.warning("⚠️ 비상 정지 실행")
            self._publish_status("emergency_stop")
        except Exception as e:
            self.logger.error(f"비상 정지 실패: {e}")
    
    def reset_emergency_stop(self):
        """비상 정지 해제"""
        self.emergency_stop = False
        self.logger.info("비상 정지 해제")
        self._publish_status("emergency_stop_reset")
    
    def _publish_status(self, status: str):
        """상태 발행"""
        topic = "robot/status"
        payload = {
            "status": status,
            "state": self.current_state.value,
            "gripper": self.gripper_state.value,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
    
    def _publish_error(self, error: str):
        """에러 발행"""
        topic = "robot/error"
        payload = {
            "error": error,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(topic, json.dumps(payload))


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
    mqtt_client = mqtt.Client(client_id="robot_controller")
    mqtt_client.connect(config['MQTT']['broker'], config['MQTT']['port'])
    mqtt_client.loop_start()
    
    # 로봇 컨트롤러
    robot = RobotController(config, mqtt_client)
    
    if robot.initialize():
        # 테스트: 비전 스캔 위치로 이동
        robot.move_to_position("vision_scan")
        
        # 테스트: 홈으로 복귀
        robot.go_home()
    
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


if __name__ == "__main__":
    main()