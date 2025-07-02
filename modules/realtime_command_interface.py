#!/usr/bin/env python3
"""
실시간 명령 인터페이스

사용자가 24시간 게임 개발 중에 실시간으로 명령을 내리고
AI와 상호작용할 수 있는 인터페이스
"""

import os
import sys
import asyncio
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from queue import Queue, Empty
import readline  # 명령어 히스토리 지원

# AI 모델 컨트롤러 임포트
try:
    from ai_model_controller import AIModelController
except ImportError:
    AIModelController = None


class CommandParser:
    """명령어 파서"""
    
    def __init__(self):
        self.commands = {
            # 게임 개발 명령
            "create": self._parse_create,
            "add": self._parse_add,
            "modify": self._parse_modify,
            "remove": self._parse_remove,
            "test": self._parse_test,
            "build": self._parse_build,
            
            # 제어 명령
            "pause": self._parse_pause,
            "resume": self._parse_resume,
            "stop": self._parse_stop,
            "save": self._parse_save,
            "load": self._parse_load,
            
            # 정보 명령
            "status": self._parse_status,
            "report": self._parse_report,
            "logs": self._parse_logs,
            "stats": self._parse_stats,
            
            # 학습 명령
            "learn": self._parse_learn,
            "train": self._parse_train,
            "evaluate": self._parse_evaluate,
            
            # AI 상호작용
            "ask": self._parse_ask,
            "explain": self._parse_explain,
            "suggest": self._parse_suggest,
            
            # 시스템 명령
            "help": self._parse_help,
            "quit": self._parse_quit,
            "exit": self._parse_quit
        }
        
        # 명령어 별칭
        self.aliases = {
            "?": "help",
            "q": "quit",
            "s": "status",
            "p": "pause",
            "r": "resume"
        }
    
    def parse(self, command_line: str) -> Dict[str, Any]:
        """명령어 파싱"""
        if not command_line.strip():
            return {"type": "empty"}
        
        parts = command_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # 별칭 확인
        cmd = self.aliases.get(cmd, cmd)
        
        # 명령어 파서 호출
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            # 자연어 명령으로 처리
            return {
                "type": "natural",
                "command": command_line,
                "original": command_line
            }
    
    def _parse_create(self, args: List[str]) -> Dict[str, Any]:
        """create 명령 파싱"""
        if not args:
            return {"type": "error", "message": "게임 타입을 지정하세요"}
        
        game_type = args[0]
        game_name = args[1] if len(args) > 1 else f"{game_type}_game"
        
        return {
            "type": "create",
            "game_type": game_type,
            "game_name": game_name
        }
    
    def _parse_add(self, args: List[str]) -> Dict[str, Any]:
        """add 명령 파싱"""
        if not args:
            return {"type": "error", "message": "추가할 기능을 지정하세요"}
        
        if args[0] == "feature":
            feature = " ".join(args[1:])
            return {"type": "add_feature", "feature": feature}
        elif args[0] == "level":
            level_name = " ".join(args[1:])
            return {"type": "add_level", "level": level_name}
        elif args[0] == "enemy":
            enemy_type = " ".join(args[1:])
            return {"type": "add_enemy", "enemy": enemy_type}
        else:
            return {"type": "add_generic", "item": " ".join(args)}
    
    def _parse_modify(self, args: List[str]) -> Dict[str, Any]:
        """modify 명령 파싱"""
        if not args:
            return {"type": "error", "message": "수정할 항목을 지정하세요"}
        
        return {
            "type": "modify",
            "aspect": " ".join(args)
        }
    
    def _parse_remove(self, args: List[str]) -> Dict[str, Any]:
        """remove 명령 파싱"""
        if not args:
            return {"type": "error", "message": "제거할 항목을 지정하세요"}
        
        return {
            "type": "remove",
            "item": " ".join(args)
        }
    
    def _parse_test(self, args: List[str]) -> Dict[str, Any]:
        """test 명령 파싱"""
        test_type = args[0] if args else "all"
        return {"type": "test", "test_type": test_type}
    
    def _parse_build(self, args: List[str]) -> Dict[str, Any]:
        """build 명령 파싱"""
        platform = args[0] if args else "windows"
        return {"type": "build", "platform": platform}
    
    def _parse_pause(self, args: List[str]) -> Dict[str, Any]:
        """pause 명령 파싱"""
        return {"type": "pause"}
    
    def _parse_resume(self, args: List[str]) -> Dict[str, Any]:
        """resume 명령 파싱"""
        return {"type": "resume"}
    
    def _parse_stop(self, args: List[str]) -> Dict[str, Any]:
        """stop 명령 파싱"""
        return {"type": "stop"}
    
    def _parse_save(self, args: List[str]) -> Dict[str, Any]:
        """save 명령 파싱"""
        save_name = args[0] if args else None
        return {"type": "save", "name": save_name}
    
    def _parse_load(self, args: List[str]) -> Dict[str, Any]:
        """load 명령 파싱"""
        if not args:
            return {"type": "error", "message": "로드할 파일을 지정하세요"}
        
        return {"type": "load", "name": args[0]}
    
    def _parse_status(self, args: List[str]) -> Dict[str, Any]:
        """status 명령 파싱"""
        return {"type": "status"}
    
    def _parse_report(self, args: List[str]) -> Dict[str, Any]:
        """report 명령 파싱"""
        report_type = args[0] if args else "summary"
        return {"type": "report", "report_type": report_type}
    
    def _parse_logs(self, args: List[str]) -> Dict[str, Any]:
        """logs 명령 파싱"""
        count = int(args[0]) if args and args[0].isdigit() else 10
        return {"type": "logs", "count": count}
    
    def _parse_stats(self, args: List[str]) -> Dict[str, Any]:
        """stats 명령 파싱"""
        return {"type": "stats"}
    
    def _parse_learn(self, args: List[str]) -> Dict[str, Any]:
        """learn 명령 파싱"""
        topic = " ".join(args) if args else None
        return {"type": "learn", "topic": topic}
    
    def _parse_train(self, args: List[str]) -> Dict[str, Any]:
        """train 명령 파싱"""
        model = args[0] if args else "current"
        return {"type": "train", "model": model}
    
    def _parse_evaluate(self, args: List[str]) -> Dict[str, Any]:
        """evaluate 명령 파싱"""
        return {"type": "evaluate"}
    
    def _parse_ask(self, args: List[str]) -> Dict[str, Any]:
        """ask 명령 파싱"""
        if not args:
            return {"type": "error", "message": "질문을 입력하세요"}
        
        return {"type": "ask", "question": " ".join(args)}
    
    def _parse_explain(self, args: List[str]) -> Dict[str, Any]:
        """explain 명령 파싱"""
        if not args:
            return {"type": "error", "message": "설명할 항목을 지정하세요"}
        
        return {"type": "explain", "topic": " ".join(args)}
    
    def _parse_suggest(self, args: List[str]) -> Dict[str, Any]:
        """suggest 명령 파싱"""
        context = " ".join(args) if args else "improvements"
        return {"type": "suggest", "context": context}
    
    def _parse_help(self, args: List[str]) -> Dict[str, Any]:
        """help 명령 파싱"""
        topic = args[0] if args else None
        return {"type": "help", "topic": topic}
    
    def _parse_quit(self, args: List[str]) -> Dict[str, Any]:
        """quit 명령 파싱"""
        return {"type": "quit"}


class RealtimeCommandInterface:
    """실시간 명령 인터페이스"""
    
    def __init__(self, command_handler: Optional[Callable] = None):
        self.parser = CommandParser()
        self.command_handler = command_handler
        self.is_running = False
        self.command_queue = Queue()
        self.response_queue = Queue()
        self.command_history = []
        self.ai_controller = AIModelController() if AIModelController else None
        
        # 명령어 자동완성 설정
        self._setup_autocomplete()
    
    def _setup_autocomplete(self):
        """자동완성 설정"""
        # 명령어 목록
        commands = list(self.parser.commands.keys())
        
        # readline 자동완성 함수
        def completer(text, state):
            options = [cmd for cmd in commands if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            return None
        
        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    
    def start(self):
        """인터페이스 시작"""
        self.is_running = True
        
        # 명령 처리 스레드 시작
        process_thread = threading.Thread(target=self._process_commands)
        process_thread.start()
        
        # 메인 입력 루프
        self._input_loop()
        
        # 종료 대기
        self.is_running = False
        process_thread.join()
    
    def _input_loop(self):
        """사용자 입력 루프"""
        print("🎮 24시간 AI 게임 개발 - 실시간 명령 인터페이스")
        print("도움말을 보려면 'help'를 입력하세요.")
        print("-" * 60)
        
        while self.is_running:
            try:
                # 프롬프트 표시
                command_line = input("\n> ").strip()
                
                if not command_line:
                    continue
                
                # 명령 파싱
                parsed = self.parser.parse(command_line)
                
                # 히스토리에 추가
                self.command_history.append({
                    "time": datetime.now().isoformat(),
                    "command": command_line,
                    "parsed": parsed
                })
                
                # 종료 명령 확인
                if parsed["type"] == "quit":
                    if self._confirm_quit():
                        self.is_running = False
                        break
                    else:
                        continue
                
                # 명령 큐에 추가
                self.command_queue.put(parsed)
                
                # 응답 대기 및 출력
                self._wait_for_response()
                
            except KeyboardInterrupt:
                print("\n\n중단하려면 'quit'를 입력하세요.")
                continue
            except EOFError:
                # Ctrl+D
                if self._confirm_quit():
                    self.is_running = False
                    break
            except Exception as e:
                print(f"\n오류: {e}")
    
    def _process_commands(self):
        """명령 처리 스레드"""
        while self.is_running:
            try:
                # 명령 가져오기
                command = self.command_queue.get(timeout=0.1)
                
                # 명령 처리
                response = self._handle_command(command)
                
                # 응답 큐에 추가
                self.response_queue.put(response)
                
            except Empty:
                continue
            except Exception as e:
                error_response = {
                    "success": False,
                    "message": f"명령 처리 오류: {e}"
                }
                self.response_queue.put(error_response)
    
    def _handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """명령 처리"""
        cmd_type = command["type"]
        
        # 에러 명령
        if cmd_type == "error":
            return {
                "success": False,
                "message": command["message"]
            }
        
        # 도움말
        if cmd_type == "help":
            return self._show_help(command.get("topic"))
        
        # 상태 정보
        if cmd_type == "status":
            return self._show_status()
        
        # 통계
        if cmd_type == "stats":
            return self._show_stats()
        
        # 자연어 명령
        if cmd_type == "natural":
            return self._handle_natural_command(command["command"])
        
        # 외부 핸들러로 전달
        if self.command_handler:
            return self.command_handler(command)
        
        return {
            "success": False,
            "message": "명령 처리기가 설정되지 않았습니다"
        }
    
    def _handle_natural_command(self, command: str) -> Dict[str, Any]:
        """자연어 명령 처리"""
        if self.ai_controller:
            # AI에게 명령 해석 요청
            try:
                interpretation = self.ai_controller.interpret_command(command)
                
                if interpretation.get("action"):
                    # 해석된 명령 실행
                    parsed = self.parser.parse(interpretation["action"])
                    return self._handle_command(parsed)
                else:
                    return {
                        "success": True,
                        "message": interpretation.get("response", "명령을 이해했습니다"),
                        "ai_response": True
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "message": f"AI 해석 오류: {e}"
                }
        
        return {
            "success": False,
            "message": "자연어 처리를 위한 AI가 준비되지 않았습니다"
        }
    
    def _show_help(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """도움말 표시"""
        if topic:
            # 특정 명령어 도움말
            help_text = self._get_command_help(topic)
        else:
            # 전체 도움말
            help_text = self._get_general_help()
        
        return {
            "success": True,
            "message": help_text,
            "type": "help"
        }
    
    def _get_general_help(self) -> str:
        """일반 도움말"""
        return """
📋 사용 가능한 명령어:

게임 개발:
  create [type] [name]  - 새 게임 생성 (platformer, rpg, puzzle, racing)
  add feature [name]    - 게임에 기능 추가
  add level [name]      - 레벨 추가
  add enemy [type]      - 적 추가
  modify [aspect]       - 게임 요소 수정
  remove [item]         - 항목 제거
  test [type]          - 게임 테스트
  build [platform]     - 게임 빌드

제어:
  pause                - 개발 일시정지
  resume               - 개발 재개
  stop                 - 개발 중지
  save [name]          - 상태 저장
  load [name]          - 상태 불러오기

정보:
  status (s)           - 현재 상태 확인
  report [type]        - 보고서 생성
  logs [count]         - 최근 로그 확인
  stats                - 통계 확인

학습:
  learn [topic]        - AI 학습 시작
  train [model]        - 모델 훈련
  evaluate             - 성능 평가

AI 상호작용:
  ask [question]       - AI에게 질문
  explain [topic]      - 설명 요청
  suggest [context]    - 제안 요청

시스템:
  help (?)             - 도움말 표시
  quit (q)             - 종료

💡 팁: 자연어로도 명령할 수 있습니다!
예) "플레이어 점프력을 높여줘", "현재 어떤 작업 중이야?"
"""
    
    def _get_command_help(self, command: str) -> str:
        """특정 명령어 도움말"""
        help_texts = {
            "create": """
create 명령어 - 새 게임 생성

사용법: create [게임타입] [게임이름]

게임 타입:
  - platformer : 플랫폼 게임
  - rpg : 롤플레잉 게임
  - puzzle : 퍼즐 게임
  - racing : 레이싱 게임

예시:
  create platformer MyPlatformer
  create rpg FantasyQuest
""",
            "add": """
add 명령어 - 게임에 요소 추가

사용법: 
  add feature [기능명]  - 새 기능 추가
  add level [레벨명]    - 새 레벨 추가
  add enemy [적 타입]   - 새 적 추가

예시:
  add feature double jump
  add level underground cave
  add enemy flying bat
""",
            "modify": """
modify 명령어 - 게임 요소 수정

사용법: modify [수정할 항목]

예시:
  modify player speed
  modify jump height
  modify enemy damage
  modify level difficulty
"""
        }
        
        return help_texts.get(command, f"'{command}' 명령어에 대한 도움말이 없습니다.")
    
    def _show_status(self) -> Dict[str, Any]:
        """상태 표시"""
        # 실제 구현에서는 게임 개발 상태를 가져옴
        status_info = {
            "current_phase": "개발 중",
            "progress": 45.2,
            "elapsed_time": "5:23:15",
            "quality_score": 72,
            "current_task": "플레이어 애니메이션 구현"
        }
        
        status_text = f"""
📊 현재 상태:
  단계: {status_info['current_phase']}
  진행률: {status_info['progress']:.1f}%
  경과 시간: {status_info['elapsed_time']}
  품질 점수: {status_info['quality_score']}/100
  현재 작업: {status_info['current_task']}
"""
        
        return {
            "success": True,
            "message": status_text,
            "data": status_info
        }
    
    def _show_stats(self) -> Dict[str, Any]:
        """통계 표시"""
        stats = {
            "iterations": 234,
            "errors_fixed": 45,
            "features_added": 12,
            "learning_cycles": 8,
            "user_commands": len(self.command_history)
        }
        
        stats_text = f"""
📈 개발 통계:
  반복 횟수: {stats['iterations']}
  수정된 오류: {stats['errors_fixed']}
  추가된 기능: {stats['features_added']}
  학습 사이클: {stats['learning_cycles']}
  사용자 명령: {stats['user_commands']}
"""
        
        return {
            "success": True,
            "message": stats_text,
            "data": stats
        }
    
    def _wait_for_response(self):
        """응답 대기 및 출력"""
        try:
            # 타임아웃으로 응답 대기
            response = self.response_queue.get(timeout=10)
            
            # 응답 출력
            if response.get("success"):
                if response.get("ai_response"):
                    print(f"\n🤖 AI: {response['message']}")
                else:
                    print(f"\n✅ {response['message']}")
            else:
                print(f"\n❌ {response['message']}")
            
            # 추가 데이터가 있으면 출력
            if response.get("data"):
                print(f"\n데이터: {json.dumps(response['data'], ensure_ascii=False, indent=2)}")
                
        except Empty:
            print("\n⏱️ 응답 시간 초과")
    
    def _confirm_quit(self) -> bool:
        """종료 확인"""
        response = input("\n정말 종료하시겠습니까? (y/n): ").lower()
        return response == 'y' or response == 'yes'
    
    def add_external_handler(self, handler: Callable):
        """외부 명령 핸들러 추가"""
        self.command_handler = handler
    
    def get_command_history(self) -> List[Dict[str, Any]]:
        """명령 히스토리 반환"""
        return self.command_history.copy()


def demo():
    """데모 실행"""
    def demo_handler(command: Dict[str, Any]) -> Dict[str, Any]:
        """데모 명령 핸들러"""
        cmd_type = command["type"]
        
        if cmd_type == "create":
            return {
                "success": True,
                "message": f"{command['game_type']} 게임 '{command['game_name']}' 생성을 시작합니다..."
            }
        elif cmd_type == "add_feature":
            return {
                "success": True,
                "message": f"'{command['feature']}' 기능을 게임에 추가합니다..."
            }
        elif cmd_type == "pause":
            return {
                "success": True,
                "message": "개발이 일시정지되었습니다."
            }
        elif cmd_type == "resume":
            return {
                "success": True,
                "message": "개발을 재개합니다."
            }
        else:
            return {
                "success": True,
                "message": f"명령 '{cmd_type}'를 처리했습니다."
            }
    
    # 인터페이스 생성 및 시작
    interface = RealtimeCommandInterface(command_handler=demo_handler)
    
    print("\n🎮 실시간 명령 인터페이스 데모")
    print("24시간 AI 게임 개발 시스템에 오신 것을 환영합니다!\n")
    
    interface.start()


if __name__ == "__main__":
    demo()