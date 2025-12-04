#!/usr/bin/env python3
"""
AMR (Autonomous Mobile Robot) Controller
MQTT 기반 AMR 제어 및 상태 관리 모듈
"""

import os
import json
import time
import math
import logging
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

import yaml
import requests
import paho.mqtt.client as mqtt


# AMR 상태 정의
class AMRState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    ARRIVED = "arrived"
    ERROR = "error"
    CHARGING = "charging"


@dataclass
class Position:
    """위치 데이터 클래스"""
    x: float
    y: float
    theta: float
    
    def distance_to(self, other: 'Position') -> float:
        """다른 위치까지의 거리 계산"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class AMRController:
    """AMR 제어 클래스"""
    
    def __init__(self, config: Dict[str, Any], mqtt_client: mqtt.Client):
        self.config = config
        self.mqtt_client = mqtt_client
        self.logger = logging.getLogger(__name__)
        
        # AMR 설정
        self.host = config['AMR']['host']
        self.port = config['AMR']['port']
        self.timeout = config['AMR']['timeout']
        
        # 위치 정의
        self.positions = {
            name: Position(*coords) 
            for name, coords in config['AMR']['positions'].items()
        }
        
        # 이동 파라미터
        self.nav_params = config['AMR']['navigation']
        self.arrival_params = config['AMR']['arrival']
        
        # 상태 변수
        self.current_state = AMRState.IDLE
        self.current_position: Optional[Position] = None
        self.target_position: Optional[str] = None
        self.is_moving = False
        
        # 모니터링 스레드
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
    def initialize(self) -> bool:
        """AMR 초기화"""
        try:
            # AMR 상태 확인
            response = self._api_request("GET", "/reeman/get_mode")
            if response:
                self.logger.info(f"AMR 초기화 성공 - 모드: {response.get('mode')}")
                self._publish_status("initialized")
                return True
            return False
        except Exception as e:
            self.logger.error(f"AMR 초기화 실패: {e}")
            return False
    
    def get_current_position(self) -> Optional[Position]:
        """현재 위치 조회"""
        try:
            response = self._api_request("GET", "/reeman/pose")
            if response:
                return Position(
                    x=response.get('x', 0),
                    y=response.get('y', 0),
                    theta=response.get('theta', 0)
                )
        except Exception as e:
            self.logger.error(f"위치 조회 실패: {e}")
        return None
    
    def navigate_to(self, location: str) -> bool:
        """지정된 위치로 이동"""
        if location not in self.positions:
            self.logger.error(f"알 수 없는 위치: {location}")
            return False
        
        target = self.positions[location]
        self.target_position = location
        
        try:
            # 이동 명령 전송
            data = {
                "point": location,  # 또는 좌표 직접 전송
                "x": target.x,
                "y": target.y,
                "theta": target.theta
            }
            
            response = self._api_request("POST", "/cmd/nav_name", data)
            
            if response and response.get("status") == "success":
                self.is_moving = True
                self.current_state = AMRState.MOVING
                
                # 상태 발행
                self._publish_status(f"moving_to_{location}")
                
                # 도착 모니터링 시작
                self._start_arrival_monitor(location)
                
                return True
                
        except Exception as e:
            self.logger.error(f"이동 명령 실패: {e}")
            self._publish_error(str(e))
            
        return False
    
    def wait_for_arrival(self, location: str, timeout: int = 120) -> bool:
        """도착 대기 - 엄격한 정지 확인"""
        start_time = time.time()
        params = self.arrival_params
        
        self.logger.info(f"{location} 도착 대기 시작 (타임아웃: {timeout}초)")
        
        # 초기 위치 기록
        initial_pos = self.get_current_position()
        if not initial_pos:
            return False
        
        # 1단계: 이동 시작 확인
        nav_started = self._check_navigation_started(initial_pos)
        if not nav_started:
            self.logger.warning("네비게이션 시작 실패")
            return self._check_already_arrived()
        
        # 2단계: 도착 대기
        consecutive_stops = 0
        required_stops = params['consecutive_stops']
        
        while time.time() - start_time < timeout:
            # 위치 측정
            pos1 = self.get_current_position()
            if not pos1:
                time.sleep(0.5)
                continue
            
            time.sleep(params['check_interval'])
            
            pos2 = self.get_current_position()
            if not pos2:
                continue
            
            distance_moved = pos1.distance_to(pos2)
            
            # 정지 판단
            if distance_moved < params['move_threshold']:
                consecutive_stops += 1
                self.logger.debug(f"정지 감지 {consecutive_stops}/{required_stops}")
                
                if consecutive_stops >= required_stops:
                    # 최종 검증
                    if self._final_verification():
                        self.logger.info(f"✅ {location} 도착 확인")
                        self._handle_arrival(location)
                        return True
                    else:
                        consecutive_stops = 0
            else:
                consecutive_stops = 0
        
        self.logger.warning(f"도착 대기 타임아웃: {location}")
        return False
    
    def stop(self) -> bool:
        """AMR 정지"""
        try:
            response = self._api_request("POST", "/cmd/stop")
            if response:
                self.is_moving = False
                self.current_state = AMRState.IDLE
                self._publish_status("stopped")
                return True
        except Exception as e:
            self.logger.error(f"정지 명령 실패: {e}")
        return False
    
    def _check_navigation_started(self, initial_pos: Position) -> bool:
        """네비게이션 시작 확인"""
        for _ in range(10):
            current_pos = self.get_current_position()
            if current_pos:
                # 이동 거리 확인
                if initial_pos.distance_to(current_pos) > 0.05:
                    return True
                
                # 모드 확인
                response = self._api_request("GET", "/reeman/get_mode")
                if response and response.get('mode') == 3:  # 네비게이션 모드
                    return True
            
            time.sleep(1)
        
        return False
    
    def _check_already_arrived(self) -> bool:
        """이미 도착했는지 확인"""
        stop_count = 0
        
        for _ in range(5):
            pos1 = self.get_current_position()
            time.sleep(0.5)
            pos2 = self.get_current_position()
            
            if pos1 and pos2:
                if pos1.distance_to(pos2) < 0.02:
                    stop_count += 1
        
        return stop_count >= 4
    
    def _final_verification(self) -> bool:
        """최종 정지 검증"""
        verification_time = self.arrival_params['final_verification_time']
        checks = int(verification_time / 0.5)
        
        for _ in range(checks):
            pos1 = self.get_current_position()
            time.sleep(0.5)
            pos2 = self.get_current_position()
            
            if pos1 and pos2:
                if pos1.distance_to(pos2) > 0.02:
                    return False
        
        return True
    
    def _handle_arrival(self, location: str):
        """도착 처리"""
        self.is_moving = False
        self.current_state = AMRState.ARRIVED
        self.current_position = self.positions[location]
        
        # MQTT 발행
        topic = f"amr/arrived/{location}"
        payload = {
            "location": location,
            "timestamp": time.time(),
            "position": {
                "x": self.current_position.x,
                "y": self.current_position.y,
                "theta": self.current_position.theta
            }
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
        self.logger.info(f"도착 알림 발행: {topic}")
    
    def _start_arrival_monitor(self, location: str):
        """도착 모니터링 스레드 시작"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitoring.set()
            self.monitor_thread.join()
        
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_arrival,
            args=(location,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_arrival(self, location: str):
        """백그라운드에서 도착 모니터링"""
        if self.wait_for_arrival(location):
            self.logger.info(f"모니터링: {location} 도착 완료")
        else:
            self.logger.error(f"모니터링: {location} 도착 실패")
            self._publish_error(f"Failed to reach {location}")
    
    def _api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """AMR API 요청"""
        url = f"http://{self.host}:{self.port}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"지원하지 않는 메소드: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API 요청 실패: {url} - {e}")
            return None
    
    def _publish_status(self, status: str):
        """상태 발행"""
        topic = "amr/status"
        payload = {
            "status": status,
            "state": self.current_state.value,
            "timestamp": time.time()
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
    
    def _publish_error(self, error: str):
        """에러 발행"""
        topic = "amr/error"
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
    
    # MQTT 클라이언트 생성
    mqtt_client = mqtt.Client(client_id="amr_controller")
    mqtt_client.connect(config['MQTT']['broker'], config['MQTT']['port'])
    mqtt_client.loop_start()
    
    # AMR 컨트롤러 생성
    amr = AMRController(config, mqtt_client)
    
    # 초기화
    if amr.initialize():
        # 테스트: grab_point로 이동
        if amr.navigate_to("grab_point"):
            amr.wait_for_arrival("grab_point")
        
        # 테스트: release_point로 이동
        if amr.navigate_to("release_point"):
            amr.wait_for_arrival("release_point")
    
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


if __name__ == "__main__":
    main()