#!/usr/bin/env python3
"""
24시간 학습 시스템 설정
실제 24시간 학습을 위한 시간 설정
"""

class LearningConfig:
    """학습 설정"""
    
    # 학습 모드
    DEMO_MODE = False  # True: 빠른 데모, False: 실제 24시간 학습
    
    # 시간 설정 (분 단위)
    if DEMO_MODE:
        # 데모 모드: 전체 1시간
        SESSION_DURATION_MIN = 1  # 각 세션 1분
        SESSION_DURATION_MAX = 2  # 최대 2분
        EXERCISE_DURATION = 0.5   # 실습 30초
        BREAK_BETWEEN_BLOCKS = 1  # 블록 간 휴식 1분
    else:
        # 실제 모드: 24시간
        SESSION_DURATION_MIN = 20  # 각 세션 최소 20분
        SESSION_DURATION_MAX = 40  # 최대 40분
        EXERCISE_DURATION = 15     # 실습 15분
        BREAK_BETWEEN_BLOCKS = 30  # 블록 간 휴식 30분
    
    # 진행 표시 설정
    PROGRESS_UPDATE_INTERVAL = 30  # 30초마다 진행률 업데이트
    SAVE_INTERVAL = 300           # 5분마다 진행상황 저장
    
    # 학습 설정
    ENABLE_BREAKS = True          # 휴식 시간 활성화
    AUTO_RESUME = True           # 중단 후 자동 재개
    DETAILED_LOGGING = True      # 상세 로그
    
    @classmethod
    def get_actual_duration(cls, base_minutes: float) -> float:
        """실제 대기 시간 계산 (초 단위)"""
        if cls.DEMO_MODE:
            # 데모 모드: 분을 초로 (1분 -> 1초)
            return base_minutes
        else:
            # 실제 모드: 분을 초로 변환
            return base_minutes * 60
    
    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """시간을 읽기 좋은 형식으로 변환"""
        if seconds < 60:
            return f"{seconds:.0f}초"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}분"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}시간"