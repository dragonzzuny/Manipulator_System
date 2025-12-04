"""
파지 계획(Grasp Planning) 모듈
벡터 각도 기반 파지점 계산 알고리즘
안전 마스크 ROI 내에서 최적 파지점 탐색
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from src.config_loader import VisionConfig

logger = logging.getLogger(__name__)

@dataclass
class GraspCandidate:
    """
    파지 후보 데이터 클래스
    기존 코드의 모든 필드 유지
    """
    object_id: int = -1
    object_conf: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    grasp_method: str = "VECTOR_ANGLE"
    center_2d: np.ndarray = field(default_factory=lambda: np.zeros(2))
    center_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    robot_coords: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inside_score: float = 0.0
    collision_score: float = 0.0
    quality_score: float = 0.0
    fill_factor: float = 0.0
    angle_deg: float = 0.0
    vector_distance: float = 0.0  # bbox중심-keypoint 거리
    keypoint_pos: Tuple[int, int] = (0, 0)
    bbox_center: Tuple[float, float] = (0.0, 0.0)

class GraspPlanner:
    """
    파지점 계산 클래스
    벡터 각도 기반 알고리즘 구현
    """
    
    def __init__(self, config: VisionConfig):
        """
        Args:
            config: 비전 시스템 설정
        """
        self.config = config
        
        # 탐지 파라미터
        self.min_vector_distance = config.min_vector_distance
        self.min_inside_score = config.min_inside_score
        self.roi_radius_factor = config.roi_radius_factor
        self.search_step = config.search_step
        
        # 안전 마스크 설정
        self.primary_erosion_kernel = config.primary_erosion_kernel
        self.secondary_erosion_kernel = config.secondary_erosion_kernel
        self.min_safe_area_primary = config.min_safe_area_primary
        self.min_safe_area_secondary = config.min_safe_area_secondary
        
        # 점수 계산 가중치
        self.distance_penalty_weight = config.distance_penalty_weight
        self.collision_penalty_weight = config.collision_penalty_weight
        self.edge_bonus_weight = config.edge_bonus_weight
        self.angle_confidence_threshold = config.angle_confidence_threshold
        
        logger.info("[GRASP] 파지 플래너 초기화 완료")
    
    def calculate_vector_angle(self, 
                              bbox: np.ndarray, 
                              keypoint_2d: Tuple[int, int]) -> Tuple[float, float, Tuple[float, float]]:
        """
        bbox 중심에서 keypoint로의 벡터 각도 계산
        기존 코드의 계산 로직 100% 유지
        
        Args:
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            keypoint_2d: 키포인트 좌표 (x, y)
            
        Returns:
            angle_deg: 계산된 각도 (0~180도)
            distance: 중심-키포인트 거리 (픽셀)
            center: bbox 중심 좌표
        """
        # bbox가 (x1, y1, x2, y2) 형식
        x1, y1, x2, y2 = bbox
        
        # bbox 중심점 계산
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # 키포인트 좌표
        kp_x, kp_y = keypoint_2d
        
        # 중심 → 키포인트 벡터
        vx = kp_x - cx
        vy = kp_y - cy
        
        # 벡터 길이 (노이즈 필터링용)
        distance = np.hypot(vx, vy)
        
        if distance < self.min_vector_distance:
            logger.warning(f"[GRASP] 키포인트가 중심에 너무 가까움 (거리: {distance:.1f}px), 기본각 0° 사용")
            return 0.0, distance, (cx, cy)
        
        # atan2로 각도 계산 (라디안 → 도)
        # +x축(오른쪽) 기준, 반시계 방향이 양수
        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)
        
        # 0~180도 범위로 정규화 (그리퍼는 180도 대칭)
        if angle_deg < 0:
            angle_deg += 180
        elif angle_deg >= 180:
            angle_deg -= 180
        
        logger.info(f"[GRASP] 벡터 각도: bbox중심({cx:.1f}, {cy:.1f}) → "
                   f"키포인트({kp_x}, {kp_y}) = {angle_deg:.1f}°, 거리={distance:.1f}px")
        
        return angle_deg, distance, (cx, cy)
    
    def create_safe_grasp_mask(self, full_mask: np.ndarray) -> np.ndarray:
        """
        안전한 파지 영역 마스크 생성
        가장자리를 제외한 안전 영역만 남김
        
        Args:
            full_mask: 원본 마스크
            
        Returns:
            안전 영역 마스크
        """
        # 안전한 파지 영역 생성: 가장자리 5px 침식
        erosion_kernel = np.ones((self.primary_erosion_kernel, self.primary_erosion_kernel), np.uint8)
        safe_grasp_mask = cv2.erode(full_mask.astype(np.uint8), erosion_kernel, iterations=1)
        
        # 안전 영역이 너무 작아지면 3px로 재시도
        if np.sum(safe_grasp_mask) < self.min_safe_area_primary:
            erosion_kernel = np.ones((self.secondary_erosion_kernel, self.secondary_erosion_kernel), np.uint8)
            safe_grasp_mask = cv2.erode(full_mask.astype(np.uint8), erosion_kernel, iterations=1)
            logger.info("[GRASP] 안전 영역이 작아서 3px 침식으로 조정")
        
        # 그래도 너무 작으면 원본 마스크 사용
        if np.sum(safe_grasp_mask) < self.min_safe_area_secondary:
            safe_grasp_mask = full_mask.astype(np.uint8)
            logger.warning("[GRASP] 물체가 너무 작아서 침식 없이 진행")
        
        return safe_grasp_mask
    
    def find_best_grasp(self, 
                       full_mask: np.ndarray, 
                       depth_frame,
                       keypoint_2d: Tuple[int, int], 
                       bbox: np.ndarray,
                       gripper_params: Dict[str, float],
                       color_intrinsics,
                       y_offset: float = -68.0,
                       calibration_matrix: Optional[np.ndarray] = None) -> Optional[GraspCandidate]:
        """
        최적 파지점 찾기
        2단계 접근법:
        1) bbox 중심-keypoint 벡터로 각도 결정
        2) 결정된 각도로 키포인트 주변 ROI에서 최적 파지점 탐색
        
        기존 find_best_grasp 로직 100% 유지
        
        Args:
            full_mask: 객체 마스크
            depth_frame: 깊이 프레임 (RealSense depth_frame 객체)
            keypoint_2d: 키포인트 좌표
            bbox: 바운딩 박스
            gripper_params: 그리퍼 파라미터 (width, length, thickness)
            color_intrinsics: 카메라 내부 파라미터
            y_offset: Y축 오프셋
            calibration_matrix: 캘리브레이션 행렬
            
        Returns:
            최적 파지 후보 또는 None
        """
        h, w = full_mask.shape
        
        # 마스크 안정화 (팽창)
        kernel = np.ones((3, 3), np.uint8)
        stable_mask = cv2.dilate(full_mask.astype(np.uint8), kernel, iterations=1)
        
        # 안전한 파지 영역 생성
        safe_grasp_mask = self.create_safe_grasp_mask(full_mask)
        
        # 마스크 영역 확인
        y_coords, x_coords = np.where(full_mask > 0)
        if len(x_coords) == 0:
            logger.error("[GRASP] 마스크가 비어있음")
            return None
        
        # ============ 1단계: 벡터 각도 계산 ============
        grasp_angle, vec_distance, bbox_center = self.calculate_vector_angle(bbox, keypoint_2d)
        logger.info(f"[GRASP] 1단계: 벡터 각도 결정 = {grasp_angle:.1f}°")
        
        # 기준 깊이값 계산 (키포인트 주변)
        kp_x, kp_y = keypoint_2d
        depth_values = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                px, py = kp_x + dx, kp_y + dy
                if 0 <= px < w and 0 <= py < h:
                    d = depth_frame.get_distance(px, py)
                    if d > 0:
                        depth_values.append(d)
        
        if not depth_values:
            logger.error("[GRASP] 깊이 정보를 얻을 수 없음")
            return None
        
        dist_m = np.mean(depth_values)
        
        # 그리퍼 크기를 픽셀 단위로 변환
        px_per_mm = color_intrinsics.fx / (dist_m * 1000.0) if dist_m > 0 else 1
        w_px = int(gripper_params['open_width_mm'] * px_per_mm)
        l_px = int(gripper_params['jaw_length_mm'] * px_per_mm)
        t_px = int(gripper_params['jaw_thickness_mm'] * px_per_mm)
        
        # ============ 2단계: 안전 영역 내에서 최적 파지점 탐색 ============
        # ROI 설정: 키포인트 중심으로 반경 설정
        roi_radius = max(10, int(w_px * self.roi_radius_factor))
        search_step = self.search_step
        
        best_position = None
        best_score = float('-inf')
        best_collision = float('inf')
        
        # 각도는 고정하고 위치만 변경하며 탐색
        angle_rad = np.radians(grasp_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        half_w, half_l = w_px / 2, l_px / 2
        
        # jaw 형상 정의 (각도는 고정)
        jaw1_base = np.array([
            [-half_l, -half_w - t_px],
            [half_l, -half_w - t_px],
            [half_l, -half_w],
            [-half_l, -half_w]
        ])
        
        jaw2_base = np.array([
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, half_w + t_px],
            [-half_l, half_w + t_px]
        ])
        
        inner_base = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        
        # ROI 내 후보 위치 탐색
        candidates = []
        
        for dy in range(-roi_radius, roi_radius + 1, search_step):
            for dx in range(-roi_radius, roi_radius + 1, search_step):
                # 후보 위치
                test_x = kp_x + dx
                test_y = kp_y + dy
                
                # 이미지 경계 체크
                if not (half_l < test_x < w - half_l and half_w + t_px < test_y < h - half_w - t_px):
                    continue
                
                # 안전 마스크 내부 체크 (가장자리 제외된 영역)
                if safe_grasp_mask[test_y, test_x] == 0:
                    continue
                
                # 그리퍼 위치에서 충돌 검사
                jaw1_rot = (jaw1_base @ R.T + np.array([test_x, test_y])).astype(np.int32)
                jaw2_rot = (jaw2_base @ R.T + np.array([test_x, test_y])).astype(np.int32)
                
                # jaw 충돌 마스크
                test_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(test_mask, [jaw1_rot], 1)
                cv2.fillPoly(test_mask, [jaw2_rot], 1)
                
                # 충돌 픽셀 수 (원본 마스크 기준)
                jaw_collision = np.logical_and(test_mask > 0, stable_mask > 0)
                collision_pixels = np.sum(jaw_collision)
                
                # 내부 영역 점수
                inner_rot = (inner_base @ R.T + np.array([test_x, test_y])).astype(np.int32)
                inner_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(inner_mask, [inner_rot], 1)
                
                inside_score = np.sum(np.logical_and(inner_mask > 0, stable_mask > 0))
                
                # 충실도
                inner_area = w_px * l_px
                fill_factor = inside_score / inner_area if inner_area > 0 else 0.0
                
                # 키포인트로부터의 거리 (가까울수록 선호)
                dist_from_kp = np.hypot(test_x - kp_x, test_y - kp_y)
                distance_penalty = dist_from_kp / roi_radius  # 0~1 정규화
                
                # 가장자리로부터의 거리 보너스 (안쪽일수록 높음)
                edge_bonus = 0
                if safe_grasp_mask[test_y, test_x] > 0:
                    # 주변 8방향 확인하여 모두 안전 영역이면 보너스
                    safe_neighbors = 0
                    for ndy in [-1, 0, 1]:
                        for ndx in [-1, 0, 1]:
                            if ndy == 0 and ndx == 0:
                                continue
                            nx, ny = test_x + ndx, test_y + ndy
                            if 0 <= nx < w and 0 <= ny < h and safe_grasp_mask[ny, nx] > 0:
                                safe_neighbors += 1
                    edge_bonus = safe_neighbors / 8.0 * self.edge_bonus_weight
                
                # 종합 점수 계산 (기존 수식 완전 유지)
                score = (inside_score * (1 - distance_penalty * self.distance_penalty_weight) + 
                        edge_bonus - collision_pixels * self.collision_penalty_weight)
                
                candidates.append({
                    'x': test_x,
                    'y': test_y,
                    'score': score,
                    'collision': collision_pixels,
                    'inside_score': inside_score,
                    'fill_factor': fill_factor,
                    'dist_from_kp': dist_from_kp,
                    'edge_bonus': edge_bonus
                })
        
        if not candidates:
            logger.warning("[GRASP] 안전 ROI 내 유효한 파지점 없음, 키포인트가 안전 영역인지 확인")
            # 키포인트가 안전 영역에 있는지 확인
            if safe_grasp_mask[kp_y, kp_x] > 0:
                best_x, best_y = kp_x, kp_y
            else:
                # 안전 영역의 중심점 찾기
                safe_y, safe_x = np.where(safe_grasp_mask > 0)
                if len(safe_x) > 0:
                    best_x = int(np.mean(safe_x))
                    best_y = int(np.mean(safe_y))
                    logger.info(f"[GRASP] 안전 영역 중심 사용: ({best_x}, {best_y})")
                else:
                    best_x, best_y = kp_x, kp_y  # 최후의 수단
            
            # 선택된 점에서 기본 점수 계산
            jaw1_rot = (jaw1_base @ R.T + np.array([best_x, best_y])).astype(np.int32)
            jaw2_rot = (jaw2_base @ R.T + np.array([best_x, best_y])).astype(np.int32)
            test_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(test_mask, [jaw1_rot], 1)
            cv2.fillPoly(test_mask, [jaw2_rot], 1)
            collision_pixels = np.sum(np.logical_and(test_mask > 0, stable_mask > 0))
            
            inner_rot = (inner_base @ R.T + np.array([best_x, best_y])).astype(np.int32)
            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(inner_mask, [inner_rot], 1)
            inside_score = np.sum(np.logical_and(inner_mask > 0, stable_mask > 0))
            fill_factor = inside_score / (w_px * l_px) if (w_px * l_px) > 0 else 0.0
            quality_score = inside_score - collision_pixels * 2
        else:
            # 최적 후보 선택
            # 1차: 충돌이 적은 후보들 필터링
            min_collision = min(c['collision'] for c in candidates)
            low_collision_candidates = [c for c in candidates if c['collision'] <= min_collision + 5]
            
            # 2차: 그 중에서 점수가 가장 높은 후보 선택
            best_candidate = max(low_collision_candidates, key=lambda x: x['score'])
            
            best_x = best_candidate['x']
            best_y = best_candidate['y']
            collision_pixels = best_candidate['collision']
            inside_score = best_candidate['inside_score']
            fill_factor = best_candidate['fill_factor']
            quality_score = best_candidate['score']
            
            logger.info(f"[GRASP] 2단계: 최적 파지점 - 키포인트에서 {best_candidate['dist_from_kp']:.1f}px 이동, "
                       f"가장자리 보너스 {best_candidate['edge_bonus']:.1f}")
        
        # 최종 3D 좌표 계산
        z_mm = dist_m * 1000.0
        x_mm = (best_x - color_intrinsics.ppx) * z_mm / color_intrinsics.fx
        y_mm = (best_y - color_intrinsics.ppy) * z_mm / color_intrinsics.fy
        
        # 로봇 좌표계 변환
        robot_coords = np.array([x_mm, y_mm + y_offset, z_mm])
        if calibration_matrix is not None:
            robot_coords = (calibration_matrix @ np.array([x_mm, y_mm + y_offset, z_mm, 1]))[:3]
        
        # 벡터 거리 기반 각도 신뢰도
        angle_confidence = min(vec_distance / self.angle_confidence_threshold, 1.0)
        final_quality = quality_score * angle_confidence
        
        best_grasp = GraspCandidate(
            center_2d=np.array([best_x, best_y]),
            center_3d=np.array([x_mm, y_mm, z_mm]),
            robot_coords=robot_coords,
            angle_deg=float(grasp_angle),
            quality_score=float(final_quality),
            inside_score=float(inside_score),
            collision_score=float(collision_pixels),
            fill_factor=fill_factor,
            vector_distance=float(vec_distance),
            keypoint_pos=keypoint_2d,
            bbox=tuple(bbox),
            bbox_center=bbox_center,
            grasp_method="VECTOR_ANGLE_SAFE_ROI"
        )
        
        logger.info(f"[GRASP] 최종 파지: 위치=({best_x}, {best_y}), 각도={grasp_angle:.1f}°, "
                   f"품질={final_quality:.0f}, 충돌={collision_pixels}px, 안전영역 내부")
        
        return best_grasp