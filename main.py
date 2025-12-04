#!/usr/bin/env python3
"""
AMR + 로봇 팔 + 비전 통합 시스템
벡터 각도 기반 정밀 피킹 시스템
"""

import os
import sys
import time
import logging
from pathlib import Path

# 환경 변수 설정 (기존 코드 유지)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config_loader import ConfigLoader
from src.orchestrator.workflow import PickAndPlaceWorkflow, TestWorkflow

def print_banner():
    """시스템 배너 출력"""
    print("\n" + "="*70)
    print("  AMR + 로봇 팔 + 비전 통합 시스템")
    print("  벡터 각도 기반 정밀 피킹 시스템")
    print("="*70)

def print_menu():
    """메뉴 출력"""
    print("\n" + "="*50)
    print("시스템 메뉴")
    print("="*50)
    print("1. 전체 피킹 앤 플레이스 실행")
    print("2. 비전 테스트")
    print("3. 로봇 팔 테스트")
    print("4. AMR 테스트")
    print("5. 설정 다시 로드")
    print("0. 종료")
    print("-"*50)

def main():
    """
    메인 함수
    기존 코드의 메뉴 시스템을 유지하면서 워크플로우 활용
    """
    # 설정 로드
    try:
        config_loader = ConfigLoader()
        config = config_loader.load()
        config_loader.setup_logging(config)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        print("config/settings.yaml 파일을 확인하세요.")
        return
    
    logger = logging.getLogger(__name__)
    
    # 배너 출력
    print_banner()
    logger.info("[MAIN] 시스템 시작")
    
    # 워크플로우 초기화
    try:
        main_workflow = PickAndPlaceWorkflow(config)
        test_workflow = TestWorkflow(config)
        logger.info("[MAIN] 워크플로우 초기화 완료")
    except Exception as e:
        logger.error(f"[MAIN] 워크플로우 초기화 실패: {e}")
        return
    
    # 메인 루프
    while True:
        print_menu()
        
        try:
            choice = input("선택: ").strip()
            
            if choice == '0':
                logger.info("[MAIN] 사용자 종료 요청")
                break
            
            elif choice == '1':
                # 전체 피킹 앤 플레이스 실행
                logger.info("[MAIN] 전체 피킹 앤 플레이스 시작")
                start_time = time.time()
                
                success = main_workflow.run()
                
                elapsed_time = time.time() - start_time
                
                if success:
                    print(f"\n✅ 작업 성공! (소요시간: {elapsed_time:.1f}초)")
                    logger.info(f"[MAIN] 작업 성공 - 소요시간: {elapsed_time:.1f}초")
                else:
                    print(f"\n❌ 작업 실패 (소요시간: {elapsed_time:.1f}초)")
                    logger.error(f"[MAIN] 작업 실패 - 소요시간: {elapsed_time:.1f}초")
                
                # 워크플로우 재초기화 (다음 실행을 위해)
                main_workflow = PickAndPlaceWorkflow(config)
            
            elif choice == '2':
                # 비전 테스트
                logger.info("[MAIN] 비전 테스트 시작")
                grasp = test_workflow.test_vision_only()
                
                if grasp:
                    print(f"\n✅ 비전 테스트 성공!")
                    print(f"  벡터 각도: {grasp.angle_deg:.1f}°")
                    print(f"  벡터 거리: {grasp.vector_distance:.1f}px")
                    print(f"  품질 점수: {grasp.quality_score:.0f}")
                    print(f"  충실도: {grasp.fill_factor:.1%}")
                else:
                    print("\n❌ 비전 테스트 실패")
            
            elif choice == '3':
                # 로봇 팔 테스트
                logger.info("[MAIN] 로봇 팔 테스트 시작")
                if test_workflow.test_arm_movement():
                    print("\n✅ 로봇 팔 테스트 성공!")
                else:
                    print("\n❌ 로봇 팔 테스트 실패")
            
            elif choice == '4':
                # AMR 테스트
                logger.info("[MAIN] AMR 테스트 시작")
                if test_workflow.test_amr_movement():
                    print("\n✅ AMR 테스트 성공!")
                else:
                    print("\n❌ AMR 테스트 실패")
            
            elif choice == '5':
                # 설정 다시 로드
                logger.info("[MAIN] 설정 다시 로드")
                try:
                    config = config_loader.load()
                    main_workflow = PickAndPlaceWorkflow(config)
                    test_workflow = TestWorkflow(config)
                    print("✅ 설정 다시 로드 완료")
                    logger.info("[MAIN] 설정 다시 로드 성공")
                except Exception as e:
                    print(f"❌ 설정 로드 실패: {e}")
                    logger.error(f"[MAIN] 설정 로드 실패: {e}")
            
            else:
                print("잘못된 선택입니다.")
        
        except KeyboardInterrupt:
            logger.info("[MAIN] 사용자 인터럽트")
            print("\n\n프로그램을 종료합니다.")
            break
        
        except Exception as e:
            logger.error(f"[MAIN] 예상치 못한 오류: {e}")
            print(f"\n❌ 오류 발생: {e}")
            print("계속하려면 Enter를 누르세요...")
            input()
    
    # 종료
    print("\n" + "="*50)
    print("시스템 종료")
    print("="*50)
    logger.info("[MAIN] 시스템 정상 종료")

if __name__ == "__main__":
    main()