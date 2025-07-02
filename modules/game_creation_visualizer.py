#!/usr/bin/env python3
"""
AutoCI 게임 제작 과정 실시간 시각화 시스템
사용자에게 게임이 만들어지는 과정을 단계별로 보여줍니다
"""

import os
import sys
import time
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class GameCreationPhase(Enum):
    """게임 제작 단계"""
    PLANNING = "🎯 기획 단계"
    DESIGN = "🎨 디자인 단계"
    PROTOTYPING = "🔨 프로토타입 제작"
    CORE_MECHANICS = "⚙️ 핵심 메커니즘 구현"
    LEVEL_DESIGN = "🗺️ 레벨 디자인"
    ENEMY_AI = "🤖 적 AI 구현"
    PLAYER_CONTROLS = "🎮 플레이어 조작 구현"
    UI_CREATION = "📱 UI/UX 제작"
    SOUND_INTEGRATION = "🔊 사운드 통합"
    VISUAL_EFFECTS = "✨ 시각 효과 추가"
    BALANCING = "⚖️ 게임 밸런싱"
    OPTIMIZATION = "🚀 최적화"
    TESTING = "🧪 테스트"
    POLISHING = "💎 마무리 작업"
    RELEASE = "🎉 출시 준비"

@dataclass
class CreationStep:
    """제작 단계 상세 정보"""
    phase: GameCreationPhase
    description: str
    duration: float  # 예상 소요 시간 (초)
    details: List[str]
    progress: float = 0.0
    completed: bool = False

class GameCreationVisualizer:
    """게임 제작 과정 시각화"""
    
    def __init__(self):
        self.current_phase = None
        self.current_step = None
        self.total_progress = 0.0
        self.creation_log = []
        self.is_active = False
        
        # 게임 타입별 제작 단계
        self.game_type_phases = {
            "platformer": self._get_platformer_phases(),
            "racing": self._get_racing_phases(),
            "puzzle": self._get_puzzle_phases(),
            "rpg": self._get_rpg_phases(),
            "fps": self._get_fps_phases(),
            "strategy": self._get_strategy_phases()
        }
        
    def _get_platformer_phases(self) -> List[CreationStep]:
        """플랫포머 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "플랫포머 게임 기획 중...",
                5.0,
                [
                    "📝 게임 컨셉 정의: 고전적인 2D 플랫포머",
                    "🎯 목표 설정: 스테이지 클리어 방식",
                    "👥 타겟 유저: 캐주얼 게이머",
                    "📊 난이도 곡선 설계"
                ]
            ),
            CreationStep(
                GameCreationPhase.DESIGN,
                "캐릭터 및 월드 디자인...",
                8.0,
                [
                    "🎨 주인공 캐릭터 스프라이트 제작",
                    "🏞️ 배경 타일셋 디자인",
                    "🎭 적 캐릭터 디자인 (3종)",
                    "💫 아이템 및 파워업 디자인"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "핵심 플레이 메커니즘 구현...",
                15.0,
                [
                    "🏃 캐릭터 이동 시스템 구현",
                    "🦘 점프 메커니즘 (더블점프 포함)",
                    "⚔️ 공격 시스템 구현",
                    "💥 충돌 감지 시스템",
                    "🎯 중력 및 물리 엔진 설정"
                ]
            ),
            CreationStep(
                GameCreationPhase.LEVEL_DESIGN,
                "스테이지 디자인 및 제작...",
                20.0,
                [
                    "🗺️ 튜토리얼 스테이지 제작",
                    "🏔️ 메인 스테이지 1-1 ~ 1-4 제작",
                    "🌋 보스 스테이지 디자인",
                    "🎨 각 스테이지별 고유 기믹 추가",
                    "🔀 비밀 경로 및 숨겨진 아이템 배치"
                ]
            ),
            CreationStep(
                GameCreationPhase.ENEMY_AI,
                "적 AI 시스템 구현...",
                12.0,
                [
                    "🤖 기본 적 AI: 패트롤 패턴",
                    "👾 추적형 적 AI 구현",
                    "🎯 원거리 공격 적 AI",
                    "👹 보스 AI 패턴 (3단계)"
                ]
            )
        ]
    
    def _get_racing_phases(self) -> List[CreationStep]:
        """레이싱 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "레이싱 게임 기획 중...",
                5.0,
                [
                    "🏁 게임 컨셉: 아케이드 레이싱",
                    "🚗 차량 종류 및 특성 기획",
                    "🛣️ 트랙 테마 설정",
                    "🏆 진행 시스템 설계"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "차량 물리 시스템 구현...",
                18.0,
                [
                    "🚗 차량 이동 및 조향 시스템",
                    "💨 가속/감속 메커니즘",
                    "🌀 드리프트 시스템 구현",
                    "💥 충돌 물리 구현",
                    "⚡ 부스터 시스템"
                ]
            ),
            CreationStep(
                GameCreationPhase.LEVEL_DESIGN,
                "레이싱 트랙 제작...",
                25.0,
                [
                    "🏙️ 도시 트랙 제작",
                    "🏔️ 산악 트랙 제작",
                    "🏖️ 해변 트랙 제작",
                    "🌉 각 트랙별 고유 장애물 배치",
                    "✨ 숏컷 경로 디자인"
                ]
            )
        ]
    
    def _get_puzzle_phases(self) -> List[CreationStep]:
        """퍼즐 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "퍼즐 게임 메커니즘 설계...",
                6.0,
                [
                    "🧩 핵심 퍼즐 메커니즘 정의",
                    "📈 난이도 진행 곡선 설계",
                    "💡 힌트 시스템 기획",
                    "🎯 스테이지 구성 계획"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "퍼즐 로직 구현...",
                20.0,
                [
                    "🔲 그리드 시스템 구현",
                    "🔄 블록 이동/회전 메커니즘",
                    "✨ 매칭 및 제거 로직",
                    "🎯 목표 달성 조건 시스템",
                    "⏱️ 시간/이동 제한 시스템"
                ]
            )
        ]
    
    def _get_rpg_phases(self) -> List[CreationStep]:
        """RPG 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "RPG 세계관 및 스토리 설계...",
                10.0,
                [
                    "🌍 게임 세계관 설정",
                    "📖 메인 스토리라인 작성",
                    "🎭 주요 캐릭터 설정",
                    "⚔️ 전투 시스템 기획",
                    "📊 캐릭터 성장 시스템 설계"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "RPG 핵심 시스템 구현...",
                30.0,
                [
                    "⚔️ 턴제 전투 시스템 구현",
                    "📊 스탯 및 레벨업 시스템",
                    "🎒 인벤토리 시스템",
                    "💬 대화 시스템 구현",
                    "🗺️ 월드맵 탐험 시스템"
                ]
            )
        ]
    
    def _get_fps_phases(self) -> List[CreationStep]:
        """FPS 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "FPS 게임 컨셉 기획...",
                5.0,
                [
                    "🎯 게임 컨셉: 택티컬 슈터",
                    "🔫 무기 시스템 기획",
                    "🗺️ 맵 디자인 컨셉",
                    "🎮 게임 모드 설계"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "FPS 핵심 메커니즘 구현...",
                25.0,
                [
                    "🎯 1인칭 카메라 시스템",
                    "🔫 무기 발사 메커니즘",
                    "💥 탄도학 및 데미지 시스템",
                    "🏃 캐릭터 이동 (달리기, 웅크리기, 점프)",
                    "🎯 조준 시스템 (ADS)"
                ]
            )
        ]
    
    def _get_strategy_phases(self) -> List[CreationStep]:
        """전략 게임 제작 단계"""
        return [
            CreationStep(
                GameCreationPhase.PLANNING,
                "전략 게임 시스템 설계...",
                8.0,
                [
                    "🏰 게임 컨셉: 실시간 전략",
                    "⚔️ 유닛 및 건물 시스템 기획",
                    "💰 자원 시스템 설계",
                    "🗺️ 맵 및 지형 시스템"
                ]
            ),
            CreationStep(
                GameCreationPhase.CORE_MECHANICS,
                "전략 게임 핵심 구현...",
                35.0,
                [
                    "🏗️ 건물 건설 시스템",
                    "👥 유닛 생산 및 제어",
                    "💰 자원 채집 및 관리",
                    "⚔️ 전투 시스템",
                    "🤖 적 AI 전략 시스템"
                ]
            )
        ]
    
    async def start_visualization(self, game_type: str, game_name: str):
        """게임 제작 시각화 시작"""
        self.is_active = True
        self.total_progress = 0.0
        self.creation_log.clear()
        
        print(f"\n{'='*60}")
        print(f"🎮 {game_name} ({game_type}) 제작을 시작합니다!")
        print(f"{'='*60}\n")
        
        # 게임 타입에 맞는 제작 단계 가져오기
        phases = self.game_type_phases.get(game_type, self._get_platformer_phases())
        
        # 각 단계별로 진행
        for phase_index, step in enumerate(phases):
            if not self.is_active:
                break
                
            self.current_step = step
            self.current_phase = step.phase
            
            # 단계 시작 표시
            print(f"\n{step.phase.value}")
            print(f"📋 {step.description}")
            print("-" * 50)
            
            # 세부 작업 표시
            for i, detail in enumerate(step.details):
                if not self.is_active:
                    break
                    
                # 작업 진행 애니메이션
                await self._show_progress_animation(detail, step.duration / len(step.details))
                
                # 진행률 업데이트
                step.progress = (i + 1) / len(step.details) * 100
                self.total_progress = (phase_index + step.progress / 100) / len(phases) * 100
                
                # 로그에 추가
                self.creation_log.append({
                    "time": datetime.now(),
                    "phase": step.phase.value,
                    "detail": detail,
                    "progress": self.total_progress
                })
            
            step.completed = True
            print(f"✅ {step.phase.value} 완료!\n")
            
            # 사용자 피드백을 받을 수 있는 타이밍
            if phase_index < len(phases) - 1:
                print("💬 이 단계에서 수정하고 싶은 부분이 있나요? (명령어를 입력하거나 Enter로 계속)")
                await asyncio.sleep(2)  # 사용자 입력 대기 시간
    
    async def _show_progress_animation(self, task: str, duration: float):
        """작업 진행 애니메이션 표시"""
        print(f"  {task}", end="", flush=True)
        
        # 진행 바 애니메이션
        steps = 20
        for i in range(steps + 1):
            if not self.is_active:
                break
                
            progress = i / steps
            bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))
            print(f"\r  {task} [{bar}] {int(progress * 100)}%", end="", flush=True)
            await asyncio.sleep(duration / steps)
        
        print(f"\r  {task} [██████████] 100% ✓")
    
    def add_custom_step(self, description: str, details: List[str]):
        """사용자 요청에 의한 커스텀 단계 추가"""
        custom_step = CreationStep(
            GameCreationPhase.CORE_MECHANICS,
            f"사용자 요청: {description}",
            10.0,
            details
        )
        
        print(f"\n🔧 사용자 요청 작업 추가됨: {description}")
        return custom_step
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 제작 상태 반환"""
        return {
            "current_phase": self.current_phase.value if self.current_phase else None,
            "current_step": self.current_step.description if self.current_step else None,
            "total_progress": round(self.total_progress, 2),
            "is_active": self.is_active
        }
    
    def show_creation_summary(self):
        """제작 과정 요약 표시"""
        print(f"\n{'='*60}")
        print("📊 게임 제작 과정 요약")
        print(f"{'='*60}")
        
        if not self.creation_log:
            print("아직 제작이 시작되지 않았습니다.")
            return
        
        # 단계별 요약
        phase_summary = {}
        for log_entry in self.creation_log:
            phase = log_entry["phase"]
            if phase not in phase_summary:
                phase_summary[phase] = []
            phase_summary[phase].append(log_entry["detail"])
        
        for phase, details in phase_summary.items():
            print(f"\n{phase}")
            for detail in details[:3]:  # 각 단계별 주요 작업 3개만 표시
                print(f"  • {detail}")
            if len(details) > 3:
                print(f"  • ... 외 {len(details) - 3}개 작업")
        
        print(f"\n전체 진행률: {round(self.total_progress, 2)}%")
        print(f"총 작업 수: {len(self.creation_log)}개")
    
    def stop(self):
        """시각화 중지"""
        self.is_active = False


# 전역 인스턴스
_visualizer = None

def get_game_creation_visualizer() -> GameCreationVisualizer:
    """게임 제작 시각화 싱글톤 인스턴스 반환"""
    global _visualizer
    if _visualizer is None:
        _visualizer = GameCreationVisualizer()
    return _visualizer