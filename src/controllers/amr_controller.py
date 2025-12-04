"""
AMR(Autonomous Mobile Robot) 컨트롤러
"""

import time
import json
import requests
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from src.config_loader import AMRConfig

logger = logging.getLogger(__name__)

@dataclass
class AMRPose:
    """AMR 위치 정보"""
    x: float
    y: float
    theta: float
    
    def distance_to(self, other: 'AMRPose') -> float:
        """다른 포즈까지의 거리 계산"""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class AMRController:
    """
    AMR 제어 클래스
    실제 AMR API에 맞춰 수정
    """
    
    def __init__(self, config: AMRConfig):
        """
        Args:
            config: AMR 설정 객체
        """
        self.config = config
        
        # URL 스킴 확인 및 자동 추가
        if not config.host.startswith(('http://', 'https://')):
            self.base_url = f"http://{config.host}"
            logger.warning(f"[AMR] URL 스킴 누락, 자동 추가: {self.base_url}")
        else:
            self.base_url = config.host
        
        # 도착 판정용 변수
        self.move_threshold = config.move_threshold
        self.check_interval = config.check_interval
        self.max_wait_time = config.max_wait_time
        self.stop_count_threshold = config.stop_count_threshold
        
        logger.info(f"[AMR] 컨트롤러 초기화 - Host: {self.base_url}")
        
        # 연결 테스트
        self._test_connection()
    
    def _test_connection(self):
        """초기 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/reeman/current_version", timeout=2)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"[AMR] 연결 성공 - 버전: {version_info}")
            else:
                logger.warning(f"[AMR] 버전 조회 실패 - Status: {response.status_code}")
        except Exception as e:
            logger.warning(f"[AMR] 초기 연결 테스트 실패: {e}")
    
    def get_pose(self) -> Optional[AMRPose]:
        """
        AMR의 현재 위치 조회
        기존 코드와 동일한 로직 유지
        """
        try:
            response = requests.get(f"{self.base_url}/reeman/pose")
            
            if response.status_code == 200:
                data = response.json()
                pose = AMRPose(
                    x=data['x'],
                    y=data['y'],
                    theta=data['theta']
                )
                logger.debug(f"[AMR][POSE] 현재 위치: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
                return pose
            else:
                logger.error(f"[AMR][POSE] 조회 실패 - Status: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"[AMR][POSE] 통신 오류: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"[AMR][POSE] 응답 파싱 오류: {e}")
            return None
    
    def get_mode(self) -> Optional[int]:
        """
        AMR 현재 모드 조회
        0: 정지, 1: 수동, 2: 자율주행 등
        """
        try:
            response = requests.get(f"{self.base_url}/reeman/get_mode")
            
            if response.status_code == 200:
                data = response.json()
                mode = data.get('mode', -1)
                logger.debug(f"[AMR][MODE] 현재 모드: {mode}")
                return mode
            else:
                logger.error(f"[AMR][MODE] 조회 실패 - Status: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"[AMR][MODE] 통신 오류: {e}")
            return None
    
    def navigate_to_point(self, point_name: str) -> bool:
        """
        지정된 포인트로 이동 명령 전송
        
        Args:
            point_name: 이동할 포인트 이름 (Grab_point, Release_point, Charging_point 등)
            
        Returns:
            명령 전송 성공 여부
        """
        try:
            url = f"{self.base_url}/cmd/nav_name"
            # 실제 API는 "point" 파라미터 사용
            payload = {"point": point_name}
            
            logger.info(f"[AMR][NAV] '{point_name}' 이동 명령 전송")
            logger.debug(f"[AMR][NAV] URL: {url}, Payload: {payload}")
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[AMR][NAV] '{point_name}' 이동 명령 성공 - 응답: {result}")
                return True
            else:
                logger.error(f"[AMR][NAV] '{point_name}' 이동 명령 실패 - Status: {response.status_code}")
                logger.error(f"[AMR][NAV] 응답: {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"[AMR][NAV] 통신 오류: {e}")
            return False
    
    def wait_for_arrival(self, target_point: str, timeout: Optional[float] = None) -> bool:
        """
        AMR이 목표 지점에 도착할 때까지 대기
        기존 코드의 도착 판정 로직을 100% 유지
        
        Args:
            target_point: 목표 지점 이름
            timeout: 최대 대기 시간 (None이면 config의 max_wait_time 사용)
            
        Returns:
            도착 성공 여부
        """
        if timeout is None:
            timeout = self.max_wait_time
        
        logger.info(f"[AMR][WAIT] '{target_point}' 도착 대기 시작 (최대 {timeout}초)")
        
        start_time = time.time()
        last_pose = None
        stop_count = 0
        
        while time.time() - start_time < timeout:
            # 현재 위치 조회
            current_pose = self.get_pose()
            if current_pose is None:
                logger.warning("[AMR][WAIT] 위치 조회 실패, 재시도...")
                time.sleep(self.check_interval)
                continue
            
            # 이전 위치와 비교하여 정지 판정
            if last_pose is not None:
                distance = current_pose.distance_to(last_pose)
                
                if distance < self.move_threshold:
                    stop_count += 1
                    logger.debug(f"[AMR][WAIT] 정지 감지 (카운트: {stop_count}/{self.stop_count_threshold})")
                    
                    # 충분한 시간동안 정지했으면 도착으로 판정
                    if stop_count >= self.stop_count_threshold:
                        logger.info(f"[AMR][WAIT] '{target_point}' 도착 완료!")
                        return True
                else:
                    # 움직임이 감지되면 카운트 리셋
                    if stop_count > 0:
                        logger.debug(f"[AMR][WAIT] 움직임 감지, 정지 카운트 리셋")
                    stop_count = 0
            
            last_pose = current_pose
            time.sleep(self.check_interval)
        
        logger.error(f"[AMR][WAIT] '{target_point}' 도착 시간 초과 ({timeout}초)")
        return False
    
    def move_and_wait(self, point_name: str, timeout: Optional[float] = None) -> bool:
        """
        이동 명령 전송 후 도착까지 대기하는 통합 함수
        
        Args:
            point_name: 이동할 포인트 이름
            timeout: 최대 대기 시간
            
        Returns:
            이동 및 도착 성공 여부
        """
        # 이동 명령 전송
        if not self.navigate_to_point(point_name):
            logger.error(f"[AMR] '{point_name}' 이동 명령 실패")
            return False
        
        # 도착 대기
        if not self.wait_for_arrival(point_name, timeout):
            logger.error(f"[AMR] '{point_name}' 도착 실패")
            return False
        
        return True
    
    def emergency_stop(self) -> bool:
        """
        AMR 긴급 정지
        필요 시 구현 (API 엔드포인트 확인 필요)
        """
        logger.warning("[AMR] 긴급 정지 명령 (구현 필요)")
        # TODO: 실제 API 엔드포인트 확인 후 구현
        # 예시: requests.post(f"{self.base_url}/cmd/stop")
        return True
    
    def get_battery_level(self) -> Optional[float]:
        """
        AMR 배터리 레벨 조회
        필요 시 구현 (API 엔드포인트 확인 필요)
        """
        # TODO: 실제 API 엔드포인트 확인 후 구현
        # 예시: response = requests.get(f"{self.base_url}/reeman/battery")
        return None
    
    def get_version(self) -> Optional[Dict[str, Any]]:
        """
        AMR 버전 정보 조회
        """
        try:
            response = requests.get(f"{self.base_url}/reeman/current_version")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"[AMR] 버전 조회 실패: {e}")
            return None