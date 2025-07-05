#!/usr/bin/env python3
"""
AutoCI 24시간 실시간 게임 개발 시스템

24시간 동안 실시간으로 게임을 개발하며, 사용자가 명령을 내릴 수 있고,
개발 과정에서 학습한 데이터로 자가 학습하는 통합 시스템
"""

import os
import sys
import asyncio
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 시스템 경로 추가
sys.path.append(str(Path(__file__).parent))

# 모듈 임포트
from modules.realtime_game_development_ai import RealtimeGameDevelopmentAI
from modules.realtime_visual_monitor import RealtimeVisualMonitor, MonitorController
from modules.realtime_command_interface import RealtimeCommandInterface
from modules.persistent_self_learning_system import PersistentSelfLearningSystem
from modules.game_factory_24h import GameFactory24Hour
from modules.ai_model_controller import AIModelController


class AutoCIRealtime24H:
    """AutoCI 24시간 실시간 통합 시스템"""
    
    def __init__(self):
        self.base_path = Path("/mnt/d/AutoCI/AutoCI")
        self.is_running = False
        
        # 핵심 컴포넌트
        self.development_ai = None
        self.visual_monitor = None
        self.monitor_controller = None
        self.command_interface = None
        self.learning_system = None
        self.ai_controller = None
        
        # 상태
        self.current_game = {
            "type": None,
            "name": None,
            "start_time": None,
            "status": "준비중"
        }
        
        # 설정
        self.config = {
            "auto_save_interval": 300,  # 5분마다 자동 저장
            "learning_interval": 600,   # 10분마다 학습
            "report_interval": 3600,    # 1시간마다 보고서
            "use_rich_display": True,   # Rich 디스플레이 사용
            "enable_ai_suggestions": True,
            "max_retries": 1000,        # 최대 재시도 횟수
            "persistence_level": "extreme"  # 끈질김 수준
        }
    
    def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            print("🚀 AutoCI 24시간 실시간 게임 개발 시스템 초기화 중...")
            
            # 개발 AI 초기화
            print("  - 개발 AI 초기화...")
            self.development_ai = RealtimeGameDevelopmentAI()
            if not self.development_ai.initialize_components():
                raise Exception("개발 AI 초기화 실패")
            
            # 시각적 모니터 초기화
            print("  - 시각적 모니터 초기화...")
            self.visual_monitor = RealtimeVisualMonitor(use_rich=self.config["use_rich_display"])
            self.monitor_controller = MonitorController(self.visual_monitor)
            
            # 명령 인터페이스 초기화
            print("  - 명령 인터페이스 초기화...")
            self.command_interface = RealtimeCommandInterface(
                command_handler=self._handle_user_command
            )
            
            # 자가 학습 시스템 초기화
            print("  - 자가 학습 시스템 초기화...")
            self.learning_system = PersistentSelfLearningSystem(
                base_path=self.base_path / "learning_data"
            )
            
            # AI 컨트롤러 초기화
            print("  - AI 모델 컨트롤러 초기화...")
            self.ai_controller = AIModelController()
            
            print("✅ 시스템 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            return False
    
    async def start(self, game_type: str = None, game_name: str = None):
        """24시간 개발 시작"""
        if not game_type:
            game_type, game_name = await self._select_game_type()
        
        self.current_game = {
            "type": game_type,
            "name": game_name or f"{game_type}_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now(),
            "status": "개발중"
        }
        
        self.is_running = True
        
        # 모니터 시작
        self.visual_monitor.start()
        self.monitor_controller.update_phase("시작", f"{game_type} 게임 개발 준비 중...")
        
        # 학습 시스템에 시작 기록
        self.learning_system.add_learning_entry(
            category="game_start",
            context={"game_type": game_type, "game_name": self.current_game["name"]},
            solution={"action": "initialize"},
            outcome={"status": "started"},
            quality_score=1.0,
            tags=["start", game_type]
        )
        
        # 비동기 태스크 시작
        tasks = [
            asyncio.create_task(self._development_loop()),
            asyncio.create_task(self._monitor_update_loop()),
            asyncio.create_task(self._learning_loop()),
            asyncio.create_task(self._auto_save_loop()),
            asyncio.create_task(self._command_processing_loop())
        ]
        
        # 명령 인터페이스는 별도 스레드에서 실행
        command_thread = threading.Thread(target=self._run_command_interface)
        command_thread.start()
        
        try:
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n\n🛑 개발 중단 요청됨...")
        finally:
            # 정리
            await self._shutdown()
            command_thread.join(timeout=5)
    
    async def _development_loop(self):
        """개발 메인 루프"""
        try:
            # 개발 AI 시작
            await self.development_ai.start_development(
                self.current_game["type"],
                self.current_game["name"]
            )
        except Exception as e:
            self._log_error(f"개발 루프 오류: {e}")
            
            # 학습 시스템에 오류 기록
            self.learning_system.add_learning_entry(
                category="development_error",
                context={"error": str(e), "game": self.current_game},
                solution={"attempted": "continue_development"},
                outcome={"success": False, "error": str(e)},
                quality_score=0.3,
                tags=["error", "development"]
            )
    
    async def _monitor_update_loop(self):
        """모니터 업데이트 루프"""
        while self.is_running:
            try:
                # 개발 AI 상태 가져오기
                if self.development_ai:
                    state = self.development_ai.current_state
                    
                    # 모니터 업데이트
                    self.visual_monitor.update_state(state)
                    
                    # 성능 데이터 업데이트 (시뮬레이션)
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent
                    gpu = 50.0  # GPU 모니터링은 실제 구현 필요
                    
                    self.visual_monitor.update_performance(cpu, memory, gpu)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self._log_error(f"모니터 업데이트 오류: {e}")
                await asyncio.sleep(1)
    
    async def _learning_loop(self):
        """학습 루프"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["learning_interval"])
                
                # 현재 개발 상태로 학습
                if self.development_ai:
                    # 성공적인 액션들 학습
                    for solution in self.development_ai.learning_data["successful_solutions"]:
                        self.learning_system.add_learning_entry(
                            category="successful_solution",
                            context=solution.get("context", {}),
                            solution=solution.get("solution", {}),
                            outcome=solution.get("outcome", {"success": True}),
                            quality_score=0.8,
                            confidence=0.9,
                            tags=["success", self.current_game["type"]]
                        )
                    
                    # 오류 패턴 학습
                    for error in self.development_ai.learning_data["error_patterns"]:
                        self.learning_system.add_learning_entry(
                            category="error_pattern",
                            context=error.get("context", {}),
                            solution=error.get("attempted_solution", {}),
                            outcome=error.get("outcome", {"success": False}),
                            quality_score=0.4,
                            confidence=0.7,
                            tags=["error", self.current_game["type"]]
                        )
                
                # 학습 보고서 생성
                report = self.learning_system.get_learning_report()
                self.monitor_controller.set_learning_status(
                    f"학습 사이클 {report['statistics']['learning_cycles']} 완료"
                )
                
            except Exception as e:
                self._log_error(f"학습 루프 오류: {e}")
    
    async def _auto_save_loop(self):
        """자동 저장 루프"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["auto_save_interval"])
                
                # 상태 저장
                await self._save_state()
                
                self.monitor_controller.log_action("save", "자동 저장 완료")
                
            except Exception as e:
                self._log_error(f"자동 저장 오류: {e}")
    
    async def _command_processing_loop(self):
        """명령 처리 루프"""
        while self.is_running:
            try:
                # 개발 AI의 명령 큐 확인
                if self.development_ai and not self.development_ai.command_queue.empty():
                    command = self.development_ai.command_queue.get()
                    
                    # 명령 처리
                    response = await self._process_ai_command(command)
                    
                    # 모니터에 표시
                    self.visual_monitor.add_user_command(command)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self._log_error(f"명령 처리 오류: {e}")
    
    def _run_command_interface(self):
        """명령 인터페이스 실행 (별도 스레드)"""
        try:
            self.command_interface.start()
        except Exception as e:
            self._log_error(f"명령 인터페이스 오류: {e}")
    
    def _handle_user_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 명령 처리"""
        cmd_type = command["type"]
        
        try:
            # 명령 타입별 처리
            if cmd_type == "create":
                return self._handle_create_command(command)
            
            elif cmd_type == "add_feature":
                return self._handle_add_feature(command)
            
            elif cmd_type == "modify":
                return self._handle_modify(command)
            
            elif cmd_type == "pause":
                return self._handle_pause()
            
            elif cmd_type == "resume":
                return self._handle_resume()
            
            elif cmd_type == "save":
                return self._handle_save(command)
            
            elif cmd_type == "status":
                return self._handle_status()
            
            elif cmd_type == "report":
                return self._handle_report(command)
            
            elif cmd_type == "learn":
                return self._handle_learn(command)
            
            elif cmd_type == "ask":
                return self._handle_ask(command)
            
            else:
                # 개발 AI로 전달
                if self.development_ai:
                    asyncio.run(self.development_ai.process_user_command(
                        json.dumps(command)
                    ))
                    return {"success": True, "message": "명령이 개발 AI로 전달되었습니다."}
                
                return {"success": False, "message": "알 수 없는 명령입니다."}
                
        except Exception as e:
            return {"success": False, "message": f"명령 처리 오류: {e}"}
    
    def _handle_create_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """게임 생성 명령 처리"""
        if self.current_game["status"] == "개발중":
            return {
                "success": False,
                "message": "이미 게임이 개발 중입니다. 먼저 현재 개발을 중지하세요."
            }
        
        # 새 게임 시작
        game_type = command["game_type"]
        game_name = command["game_name"]
        
        self.current_game = {
            "type": game_type,
            "name": game_name,
            "start_time": datetime.now(),
            "status": "개발중"
        }
        
        # 개발 시작
        asyncio.create_task(self._restart_development(game_type, game_name))
        
        return {
            "success": True,
            "message": f"{game_type} 게임 '{game_name}' 개발을 시작합니다."
        }
    
    def _handle_add_feature(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """기능 추가 명령 처리"""
        feature = command["feature"]
        
        # 개발 AI에 전달
        if self.development_ai:
            asyncio.run(self.development_ai.process_user_command(f"add feature {feature}"))
            
            # 모니터 업데이트
            self.monitor_controller.log_action("add", f"기능 추가: {feature}")
            
            # 학습 시스템에 기록
            self.learning_system.add_learning_entry(
                category="user_command",
                context={"command": "add_feature", "feature": feature},
                solution={"action": "implement_feature"},
                outcome={"status": "processing"},
                quality_score=0.7,
                tags=["user_command", "feature"]
            )
            
            return {
                "success": True,
                "message": f"'{feature}' 기능 추가를 시작합니다."
            }
        
        return {"success": False, "message": "개발 AI가 준비되지 않았습니다."}
    
    def _handle_pause(self) -> Dict[str, Any]:
        """일시정지 명령 처리"""
        if self.current_game["status"] != "개발중":
            return {"success": False, "message": "개발 중이 아닙니다."}
        
        self.current_game["status"] = "일시정지"
        self.monitor_controller.update_phase("일시정지", "개발이 일시정지되었습니다")
        
        return {"success": True, "message": "개발이 일시정지되었습니다."}
    
    def _handle_resume(self) -> Dict[str, Any]:
        """재개 명령 처리"""
        if self.current_game["status"] != "일시정지":
            return {"success": False, "message": "일시정지 상태가 아닙니다."}
        
        self.current_game["status"] = "개발중"
        self.monitor_controller.update_phase("재개", "개발을 재개합니다")
        
        return {"success": True, "message": "개발을 재개합니다."}
    
    async def _handle_save(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """저장 명령 처리"""
        save_name = command.get("name", f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        await self._save_state(save_name)
        
        return {"success": True, "message": f"상태가 '{save_name}'으로 저장되었습니다."}
    
    def _handle_status(self) -> Dict[str, Any]:
        """상태 명령 처리"""
        if not self.development_ai:
            return {"success": False, "message": "시스템이 준비되지 않았습니다."}
        
        state = self.development_ai.current_state
        elapsed = datetime.now() - self.current_game["start_time"] if self.current_game["start_time"] else timedelta(0)
        
        status_text = f"""
📊 현재 상태:
  게임: {self.current_game['name']} ({self.current_game['type']})
  상태: {self.current_game['status']}
  경과 시간: {self._format_timedelta(elapsed)}
  진행률: {state['progress']:.1f}%
  현재 단계: {state['current_phase']}
  현재 작업: {state['current_task']}
  품질 점수: {state['quality_score']}/100
  
📈 통계:
  반복 횟수: {state['iterations']}
  수정된 오류: {state['errors_fixed']}
  추가된 기능: {state['features_added']}
  학습 사이클: {state['learning_cycles']}
"""
        
        return {
            "success": True,
            "message": status_text,
            "data": {
                "game": self.current_game,
                "state": state
            }
        }
    
    async def _handle_report(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """보고서 명령 처리"""
        report_type = command.get("report_type", "summary")
        
        # 개발 보고서 생성
        if self.development_ai:
            dev_report = await self.development_ai._generate_report()
        else:
            dev_report = {}
        
        # 학습 보고서
        learning_report = self.learning_system.get_learning_report()
        
        # 통합 보고서
        report = {
            "title": f"24시간 AI 게임 개발 보고서 - {self.current_game['name']}",
            "timestamp": datetime.now().isoformat(),
            "game_info": self.current_game,
            "development_report": dev_report,
            "learning_report": learning_report,
            "summary": self._generate_summary(dev_report, learning_report)
        }
        
        # 파일로 저장
        report_path = self.base_path / f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": f"보고서가 생성되었습니다: {report_path}",
            "data": report["summary"]
        }
    
    def _handle_learn(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """학습 명령 처리"""
        topic = command.get("topic")
        
        if topic:
            # 특정 주제 학습
            self.learning_system.add_learning_entry(
                category="focused_learning",
                context={"topic": topic},
                solution={"action": "study_topic"},
                outcome={"status": "learning"},
                quality_score=0.8,
                tags=["learning", topic]
            )
            
            return {
                "success": True,
                "message": f"'{topic}' 주제에 대한 집중 학습을 시작합니다."
            }
        else:
            # 일반 학습
            report = self.learning_system.get_learning_report()
            return {
                "success": True,
                "message": f"학습 진행 중... (사이클: {report['statistics']['learning_cycles']})"
            }
    
    def _handle_ask(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """질문 명령 처리"""
        question = command["question"]
        
        # AI에게 질문
        if self.ai_controller:
            try:
                response = self.ai_controller.ask_question(question)
                
                # 학습 시스템에 기록
                self.learning_system.add_learning_entry(
                    category="user_question",
                    context={"question": question},
                    solution={"answer": response},
                    outcome={"status": "answered"},
                    quality_score=0.8,
                    tags=["question", "user_interaction"]
                )
                
                return {
                    "success": True,
                    "message": response,
                    "ai_response": True
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"AI 응답 오류: {e}"
                }
        
        return {
            "success": False,
            "message": "AI 컨트롤러가 준비되지 않았습니다."
        }
    
    async def _save_state(self, name: Optional[str] = None):
        """상태 저장"""
        if not name:
            name = f"autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        state_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "game": self.current_game,
            "development_state": self.development_ai.current_state if self.development_ai else {},
            "learning_stats": self.learning_system.get_learning_report()["statistics"],
            "config": self.config
        }
        
        # 상태 파일 저장
        state_path = self.base_path / f"states/{name}.json"
        state_path.parent.mkdir(exist_ok=True)
        
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        # 개발 AI 상태도 저장
        if self.development_ai:
            await self.development_ai._save_state()
    
    async def _restart_development(self, game_type: str, game_name: str):
        """개발 재시작"""
        # 기존 개발 중지
        if self.development_ai:
            self.development_ai.stop_development()
        
        # 새로운 개발 시작
        self.development_ai = RealtimeGameDevelopmentAI()
        self.development_ai.initialize_components()
        
        await self.development_ai.start_development(game_type, game_name)
    
    def _generate_summary(self, dev_report: Dict, learning_report: Dict) -> Dict[str, Any]:
        """요약 생성"""
        return {
            "total_time": self._format_timedelta(
                datetime.now() - self.current_game["start_time"]
            ) if self.current_game["start_time"] else "N/A",
            "progress": dev_report.get("progress", 0),
            "quality_score": dev_report.get("quality_score", 0),
            "features_added": dev_report.get("statistics", {}).get("추가된 기능", 0),
            "errors_fixed": dev_report.get("statistics", {}).get("수정된 오류", 0),
            "learning_entries": learning_report["statistics"]["total_entries"],
            "patterns_discovered": learning_report["statistics"]["patterns_discovered"],
            "insights_generated": learning_report["statistics"]["insights_generated"]
        }
    
    def _format_timedelta(self, td: timedelta) -> str:
        """시간 포맷팅"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    async def _select_game_type(self) -> Tuple[str, str]:
        """게임 타입 선택"""
        print("\n게임 타입을 선택하세요:")
        print("1. platformer - 플랫폼 게임")
        print("2. rpg - RPG 게임")
        print("3. puzzle - 퍼즐 게임")
        print("4. racing - 레이싱 게임")
        print("5. custom - 직접 입력")
        
        choice = input("\n선택 (1-5): ").strip()
        
        game_types = {
            "1": "platformer",
            "2": "rpg",
            "3": "puzzle",
            "4": "racing"
        }
        
        if choice in game_types:
            game_type = game_types[choice]
            game_name = input(f"\n게임 이름 (Enter로 자동 생성): ").strip()
            return game_type, game_name or None
        elif choice == "5":
            game_type = input("\n게임 타입: ").strip()
            game_name = input("게임 이름: ").strip()
            return game_type, game_name
        else:
            print("잘못된 선택입니다. 기본값(platformer)을 사용합니다.")
            return "platformer", None
    
    def _log_error(self, message: str):
        """오류 로깅"""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": message
        }
        
        # 콘솔 출력
        print(f"[ERROR] {message}")
        
        # 파일 로깅
        log_path = self.base_path / "logs/errors.log"
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_log, ensure_ascii=False) + "\n")
    
    async def _shutdown(self):
        """시스템 종료"""
        print("\n📦 시스템 종료 중...")
        
        # 최종 상태 저장
        await self._save_state("final_state")
        
        # 최종 보고서 생성
        await self._handle_report({"report_type": "final"})
        
        # 컴포넌트 종료
        self.is_running = False
        
        if self.development_ai:
            self.development_ai.stop_development()
        
        if self.visual_monitor:
            self.visual_monitor.stop()
        
        if self.learning_system:
            self.learning_system.shutdown()
        
        print("✅ 시스템이 안전하게 종료되었습니다.")


async def main():
    """메인 함수"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AutoCI 24시간 실시간 AI 게임 개발 시스템 v5.0          ║
║                                                              ║
║  AI가 24시간 동안 끈질기게 게임을 개발하며 학습합니다      ║
║  사용자는 실시간으로 명령을 내리고 개발 과정을 관찰합니다   ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 시스템 생성
    system = AutoCIRealtime24H()
    
    # 초기화
    if not system.initialize():
        print("시스템 초기화에 실패했습니다.")
        return
    
    # 개발 시작
    try:
        await system.start()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())