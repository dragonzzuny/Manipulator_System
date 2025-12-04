#!/usr/bin/env python3
"""
Main System Orchestrator
MQTT 기반 전체 시스템 조정 및 작업 흐름 관리
"""

import os
import sys
import json
import time
import yaml
import signal
import logging
import threading
from enum import Enum
from typing import Optional, Dict, Any

import paho.mqtt.client as mqtt

# 컴포넌트 임포트
from amr_controller import AMRController, AMRState
from vision_controller import VisionController, VisionState
from robot_controller import RobotController, RobotState


class SystemState(Enum):
    """시스템 상태"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class WorkflowState(Enum):
    """작업 흐름 상태"""
    WAITING = "waiting"
    MOVE_TO_PICK = "move_to_pick"
    SCANNING = "scanning"
    PICKING = "picking"
    MOVE_TO_PLACE = "move_to_place"
    PLACING = "placing"
    MOVE_TO_HOME = "move_to_home"
    COMPLETE = "complete"


class MainOrchestrator:
    """메인 시스템 오케스트레이터"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 로깅 설정
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # MQTT 클라이언트
        self