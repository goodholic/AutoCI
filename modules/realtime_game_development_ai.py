#!/usr/bin/env python3
"""
실시간 24시간 게임 개발 AI 시스템

이 시스템은 24시간 동안 실시간으로 게임을 개발하며,
사용자가 명령을 내릴 수 있고, 개발 과정에서 학습한 데이터로
자가 학습을 수행하는 통합 AI 모델입니다.
"""

import os
import sys
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from queue import Queue, Empty
from pathlib import Path

# 기존 모듈들 임포트
from game_factory_24h import GameFactory24Hour
from realtime_24h_monitor import Realtime24HMonitor
from self_evolution_system import SelfEvolutionSystem
from realtime_learning_integrator import RealtimeLearningIntegrator
from development_experience_collector import DevelopmentExperienceCollector
from autoci_monitor_client import AutoCIMonitorClient
from ai_model_controller import AIModelController
from progressive_learning_manager import ProgressiveLearningManager


class RealtimeGameDevelopmentAI:
    """24시간 실시간 게임 개발 AI 시스템"""
    
    def __init__(self):
        self.base_path = Path("/mnt/d/AutoCI/AutoCI")
        self.is_running = False
        self.command_queue = Queue()
        self.development_thread = None
        self.monitor_thread = None
        self.learning_thread = None
        
        # 핵심 컴포넌트 초기화
        self.game_factory = None
        self.monitor = None
        self.evolution_system = None
        self.learning_integrator = None
        self.experience_collector = None
        self.ai_controller = None
        self.progress_manager = None
        
        # 상태 관리
        self.current_state = {
            "start_time": None,
            "elapsed_time": 0,
            "current_phase": "대기중",
            "game_type": None,
            "game_name": None,
            "progress": 0,
            "quality_score": 0,
            "iterations": 0,
            "errors_fixed": 0,
            "features_added": 0,
            "learning_cycles": 0,
            "user_commands": [],
            "development_log": []
        }
        
        # 학습 데이터
        self.learning_data = {
            "successful_solutions": [],
            "error_patterns": [],
            "user_feedback": [],
            "performance_metrics": [],
            "creative_approaches": []
        }
        
    def initialize_components(self):
        """모든 컴포넌트 초기화"""
        try:
            # 게임 팩토리 초기화
            self.game_factory = GameFactory24Hour()
            
            # 모니터 초기화
            self.monitor = Realtime24HMonitor()
            
            # 자가 진화 시스템 초기화
            self.evolution_system = SelfEvolutionSystem()
            
            # 실시간 학습 통합기 초기화
            self.learning_integrator = RealtimeLearningIntegrator()
            
            # 개발 경험 수집기 초기화
            self.experience_collector = DevelopmentExperienceCollector()
            
            # AI 모델 컨트롤러 초기화
            self.ai_controller = AIModelController()
            
            # 진행 관리자 초기화
            self.progress_manager = ProgressiveLearningManager()
            
            return True
            
        except Exception as e:
            print(f"컴포넌트 초기화 실패: {e}")
            return False
    
    async def start_development(self, game_type: str, game_name: str):
        """24시간 게임 개발 시작"""
        self.is_running = True
        self.current_state["start_time"] = datetime.now()
        self.current_state["game_type"] = game_type
        self.current_state["game_name"] = game_name
        
        # 개발 스레드 시작
        self.development_thread = threading.Thread(
            target=self._development_loop,
            args=(game_type, game_name)
        )
        self.development_thread.start()
        
        # 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
        # 학습 스레드 시작
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.start()
        
        # 명령 처리 루프
        await self._command_loop()
    
    def _development_loop(self, game_type: str, game_name: str):
        """게임 개발 메인 루프"""
        try:
            # 게임 팩토리로 24시간 개발 시작
            self.game_factory.create_game_24h(
                game_type=game_type,
                game_name=game_name,
                callback=self._development_callback
            )
        except Exception as e:
            self._log_error(f"개발 루프 오류: {e}")
    
    def _monitor_loop(self):
        """실시간 모니터링 루프"""
        while self.is_running:
            try:
                # 현재 상태 업데이트
                self._update_state()
                
                # 모니터에 상태 전송
                if self.monitor:
                    self.monitor.update_status(self.current_state)
                
                # 1초마다 업데이트
                time.sleep(1)
                
            except Exception as e:
                self._log_error(f"모니터 루프 오류: {e}")
    
    def _learning_loop(self):
        """자가 학습 루프"""
        while self.is_running:
            try:
                # 30분마다 학습 수행
                time.sleep(1800)  # 30분
                
                if self.learning_data["successful_solutions"]:
                    # 성공적인 솔루션으로 학습
                    self._perform_learning()
                    
                # 진화 시스템 업데이트
                if self.evolution_system:
                    self.evolution_system.check_evolution()
                    
            except Exception as e:
                self._log_error(f"학습 루프 오류: {e}")
    
    async def _command_loop(self):
        """사용자 명령 처리 루프"""
        while self.is_running:
            try:
                # 비동기적으로 명령 확인
                await asyncio.sleep(0.1)
                
                # 큐에서 명령 가져오기
                try:
                    command = self.command_queue.get_nowait()
                    await self._process_command(command)
                except Empty:
                    pass
                    
            except Exception as e:
                self._log_error(f"명령 루프 오류: {e}")
    
    async def process_user_command(self, command: str) -> Dict[str, Any]:
        """사용자 명령 처리"""
        # 명령을 큐에 추가
        self.command_queue.put(command)
        
        # 명령 기록
        self.current_state["user_commands"].append({
            "time": datetime.now().isoformat(),
            "command": command
        })
        
        # 즉시 응답
        return {
            "status": "accepted",
            "message": f"명령 '{command}'가 처리 대기열에 추가되었습니다."
        }
    
    async def _process_command(self, command: str):
        """실제 명령 처리"""
        try:
            # 명령 파싱
            parts = command.lower().split()
            cmd = parts[0] if parts else ""
            
            if cmd == "add" and len(parts) > 1:
                # 기능 추가
                feature = " ".join(parts[1:])
                await self._add_feature(feature)
                
            elif cmd == "modify" and len(parts) > 1:
                # 게임 수정
                aspect = " ".join(parts[1:])
                await self._modify_game(aspect)
                
            elif cmd == "status":
                # 상태 보고
                await self._report_status()
                
            elif cmd == "pause":
                # 일시 정지
                await self._pause_development()
                
            elif cmd == "resume":
                # 재개
                await self._resume_development()
                
            elif cmd == "learn":
                # 즉시 학습
                await self._immediate_learning()
                
            elif cmd == "save":
                # 상태 저장
                await self._save_state()
                
            elif cmd == "report":
                # 보고서 생성
                await self._generate_report()
                
            else:
                # AI에게 자연어 명령 해석 요청
                await self._interpret_natural_command(command)
                
        except Exception as e:
            self._log_error(f"명령 처리 오류: {e}")
    
    async def _add_feature(self, feature: str):
        """게임에 기능 추가"""
        # 게임 팩토리에 기능 추가 요청
        if self.game_factory:
            result = await self.game_factory.add_feature_async(feature)
            
            # 성공한 경우 학습 데이터에 추가
            if result["success"]:
                self.learning_data["successful_solutions"].append({
                    "type": "feature_addition",
                    "feature": feature,
                    "implementation": result.get("implementation", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
            self._log_event(f"기능 추가: {feature} - {result['message']}")
    
    async def _modify_game(self, aspect: str):
        """게임 수정"""
        if self.game_factory:
            result = await self.game_factory.modify_game_async(aspect)
            self._log_event(f"게임 수정: {aspect} - {result['message']}")
    
    def _update_state(self):
        """현재 상태 업데이트"""
        if self.current_state["start_time"]:
            # 경과 시간 계산
            elapsed = datetime.now() - self.current_state["start_time"]
            self.current_state["elapsed_time"] = elapsed.total_seconds()
            
            # 진행률 계산 (24시간 기준)
            self.current_state["progress"] = min(
                (elapsed.total_seconds() / (24 * 3600)) * 100, 100
            )
            
        # 게임 팩토리에서 상태 가져오기
        if self.game_factory:
            factory_state = self.game_factory.get_current_state()
            self.current_state.update(factory_state)
    
    def _perform_learning(self):
        """학습 수행"""
        try:
            # 학습 데이터 준비
            learning_batch = {
                "solutions": self.learning_data["successful_solutions"][-100:],
                "errors": self.learning_data["error_patterns"][-50:],
                "feedback": self.learning_data["user_feedback"][-50:]
            }
            
            # 실시간 학습 통합기로 학습
            if self.learning_integrator:
                result = self.learning_integrator.integrate_learning_data(
                    learning_batch,
                    self.current_state["game_type"]
                )
                
                self.current_state["learning_cycles"] += 1
                self._log_event(f"학습 사이클 {self.current_state['learning_cycles']} 완료")
                
        except Exception as e:
            self._log_error(f"학습 수행 오류: {e}")
    
    def _development_callback(self, event: Dict[str, Any]):
        """개발 이벤트 콜백"""
        # 이벤트 기록
        self.current_state["development_log"].append({
            "time": datetime.now().isoformat(),
            "event": event
        })
        
        # 이벤트 타입별 처리
        if event.get("type") == "error":
            self.learning_data["error_patterns"].append(event)
            self.current_state["errors_fixed"] += 1
            
        elif event.get("type") == "feature_complete":
            self.current_state["features_added"] += 1
            
        elif event.get("type") == "phase_change":
            self.current_state["current_phase"] = event.get("phase", "")
            
        elif event.get("type") == "quality_update":
            self.current_state["quality_score"] = event.get("score", 0)
    
    async def _save_state(self):
        """현재 상태 저장"""
        state_file = self.base_path / f"realtime_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        save_data = {
            "current_state": self.current_state,
            "learning_data": self.learning_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        self._log_event(f"상태 저장됨: {state_file}")
    
    async def _generate_report(self):
        """개발 보고서 생성"""
        report = {
            "title": f"24시간 게임 개발 보고서 - {self.current_state['game_name']}",
            "duration": self.current_state["elapsed_time"],
            "progress": self.current_state["progress"],
            "quality_score": self.current_state["quality_score"],
            "statistics": {
                "총 반복 횟수": self.current_state["iterations"],
                "수정된 오류": self.current_state["errors_fixed"],
                "추가된 기능": self.current_state["features_added"],
                "학습 사이클": self.current_state["learning_cycles"],
                "사용자 명령": len(self.current_state["user_commands"])
            },
            "phases_completed": self._get_completed_phases(),
            "learning_insights": self._get_learning_insights(),
            "user_interactions": self.current_state["user_commands"][-10:],
            "timestamp": datetime.now().isoformat()
        }
        
        report_file = self.base_path / f"development_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        self._log_event(f"보고서 생성됨: {report_file}")
        return report
    
    def _get_completed_phases(self) -> List[str]:
        """완료된 단계 목록"""
        # 게임 팩토리에서 완료된 단계 가져오기
        if self.game_factory:
            return self.game_factory.get_completed_phases()
        return []
    
    def _get_learning_insights(self) -> List[str]:
        """학습 인사이트"""
        insights = []
        
        # 가장 많이 수정된 오류 패턴
        if self.learning_data["error_patterns"]:
            error_types = {}
            for error in self.learning_data["error_patterns"]:
                error_type = error.get("type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
            most_common = max(error_types.items(), key=lambda x: x[1])
            insights.append(f"가장 빈번한 오류: {most_common[0]} ({most_common[1]}회)")
        
        # 성공적인 솔루션 수
        insights.append(f"성공적인 솔루션: {len(self.learning_data['successful_solutions'])}개")
        
        # 사용자 피드백 요약
        if self.learning_data["user_feedback"]:
            insights.append(f"사용자 피드백 수: {len(self.learning_data['user_feedback'])}개")
        
        return insights
    
    async def _interpret_natural_command(self, command: str):
        """자연어 명령 해석"""
        # AI 모델로 명령 해석
        if self.ai_controller:
            interpretation = await self.ai_controller.interpret_command(command)
            
            # 해석된 명령 실행
            if interpretation.get("action"):
                await self._process_command(interpretation["action"])
            else:
                self._log_event(f"명령 해석 실패: {command}")
    
    def _log_event(self, message: str):
        """이벤트 로깅"""
        log_entry = {
            "time": datetime.now().isoformat(),
            "message": message
        }
        self.current_state["development_log"].append(log_entry)
        
        # 콘솔에도 출력
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def _log_error(self, message: str):
        """오류 로깅"""
        error_entry = {
            "time": datetime.now().isoformat(),
            "error": message
        }
        self.learning_data["error_patterns"].append(error_entry)
        
        # 콘솔에 오류 출력
        print(f"[ERROR] {message}")
    
    def stop_development(self):
        """개발 중지"""
        self.is_running = False
        
        # 모든 스레드 종료 대기
        if self.development_thread:
            self.development_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        # 최종 상태 저장
        asyncio.run(self._save_state())
        
        # 최종 보고서 생성
        asyncio.run(self._generate_report())


async def main():
    """메인 실행 함수"""
    # 실시간 개발 AI 생성
    ai = RealtimeGameDevelopmentAI()
    
    # 컴포넌트 초기화
    if not ai.initialize_components():
        print("초기화 실패")
        return
    
    # 게임 타입과 이름 설정
    game_type = "platformer"
    game_name = "SuperPlatformer24H"
    
    print(f"24시간 실시간 게임 개발 시작: {game_name}")
    print("명령어: add feature [기능], modify [항목], status, pause, resume, learn, save, report")
    print("-" * 80)
    
    try:
        # 개발 시작
        await ai.start_development(game_type, game_name)
        
    except KeyboardInterrupt:
        print("\n개발 중단됨")
        ai.stop_development()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        ai.stop_development()


if __name__ == "__main__":
    asyncio.run(main())