#!/usr/bin/env python3
"""
AutoCI 대화형 인터페이스
자연어로 Godot 엔진을 제어하는 대화형 시스템
"""

import asyncio
import sys
import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import readline  # 명령어 히스토리
from colorama import init, Fore, Back, Style
import re

from modules.autoci_orchestrator import AutoCIOrchestrator
from autoci_24h_learning_system import ContinuousLearningSystem

# Colorama 초기화
init(autoreset=True)

class ConversationalInterface:
    """대화형 인터페이스 클래스"""
    
    def __init__(self):
        self.orchestrator = None
        self.learning_system = None
        self.conversation_context = {
            "current_project": None,
            "current_scene": None,
            "selected_nodes": [],
            "history": []
        }
        self.commands = self._init_commands()
        self.logger = logging.getLogger(__name__)
        
    def _init_commands(self) -> Dict[str, str]:
        """명령어 초기화"""
        return {
            "/help": "도움말 표시",
            "/status": "현재 상태 확인",
            "/projects": "프로젝트 목록",
            "/scenes": "씬 목록",
            "/tasks": "작업 목록",
            "/learn": "학습 통계",
            "/exit": "종료",
            "/clear": "화면 지우기"
        }
        
    async def start(self):
        """인터페이스 시작"""
        # 시작 메시지
        self._print_welcome()
        
        # 시스템 초기화
        await self._initialize_systems()
        
        # 메인 루프
        await self._main_loop()
        
    async def _initialize_systems(self):
        """시스템 초기화"""
        print(f"{Fore.YELLOW}시스템을 초기화하는 중...")
        
        # Orchestrator 초기화
        self.orchestrator = AutoCIOrchestrator()
        await self.orchestrator.start()
        
        # 학습 시스템 초기화
        self.learning_system = ContinuousLearningSystem(self.orchestrator)
        await self.learning_system.start()
        
        print(f"{Fore.GREEN}✓ 시스템 초기화 완료!")
        print(f"{Fore.CYAN}Godot 엔진과 연결되었습니다.")
        print()
        
    def _print_welcome(self):
        """환영 메시지 출력"""
        print(f"""
{Fore.MAGENTA}{'='*60}
{Fore.CYAN}🤖 AutoCI - 24시간 게임 제작 AI Agent
{Fore.MAGENTA}{'='*60}

{Fore.GREEN}안녕하세요! 저는 당신의 게임 제작을 도와드릴 AI 어시스턴트입니다.
자연스러운 한국어나 영어로 말씀해 주시면, Godot 엔진을 제어해 드립니다.

{Fore.YELLOW}예시:
  • "간단한 2D 플랫포머 게임을 만들어줘"
  • "플레이어 캐릭터를 추가해줘"
  • "점프 높이를 2배로 늘려줘"
  • "적 캐릭터 3개를 만들어줘"

{Fore.BLUE}명령어:
  • /help - 도움말
  • /status - 현재 상태
  • /exit - 종료

{Fore.MAGENTA}{'='*60}
        """)
        
    async def _main_loop(self):
        """메인 대화 루프"""
        while True:
            try:
                # 프롬프트 표시
                prompt = self._get_prompt()
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                    
                # 히스토리에 추가
                self.conversation_context["history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "user",
                    "content": user_input
                })
                
                # 명령어 처리
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    # 자연어 처리
                    await self._handle_natural_language(user_input)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}종료하시려면 /exit를 입력하세요.")
            except Exception as e:
                print(f"{Fore.RED}오류 발생: {e}")
                self.logger.error(f"대화 루프 오류: {e}")
                
    def _get_prompt(self) -> str:
        """프롬프트 생성"""
        project = self.conversation_context.get("current_project", "없음")
        scene = self.conversation_context.get("current_scene", "없음")
        
        return f"{Fore.BLUE}[프로젝트: {project} | 씬: {scene}] {Fore.GREEN}➤ {Style.RESET_ALL}"
        
    async def _handle_command(self, command: str):
        """명령어 처리"""
        cmd = command.lower().split()[0]
        
        if cmd == "/help":
            self._show_help()
        elif cmd == "/status":
            await self._show_status()
        elif cmd == "/projects":
            await self._show_projects()
        elif cmd == "/scenes":
            await self._show_scenes()
        elif cmd == "/tasks":
            await self._show_tasks()
        elif cmd == "/learn":
            await self._show_learning_stats()
        elif cmd == "/clear":
            os.system('clear' if os.name == 'posix' else 'cls')
        elif cmd == "/exit":
            await self._shutdown()
        else:
            print(f"{Fore.RED}알 수 없는 명령어: {cmd}")
            
    def _show_help(self):
        """도움말 표시"""
        print(f"\n{Fore.CYAN}=== 도움말 ===")
        print(f"{Fore.YELLOW}사용 가능한 명령어:")
        for cmd, desc in self.commands.items():
            print(f"  {Fore.GREEN}{cmd:<15} {Fore.WHITE}{desc}")
            
        print(f"\n{Fore.YELLOW}자연어 예시:")
        examples = [
            "새로운 게임 프로젝트를 만들어줘",
            "플레이어 캐릭터를 생성해줘",
            "배경 음악을 추가해줘",
            "게임을 실행해줘",
            "코드를 최적화해줘"
        ]
        for example in examples:
            print(f"  {Fore.WHITE}• {example}")
        print()
        
    async def _show_status(self):
        """현재 상태 표시"""
        print(f"\n{Fore.CYAN}=== 시스템 상태 ===")
        
        # Godot 연결 상태
        godot_connected = self.orchestrator.godot.ping()
        print(f"Godot 엔진: {Fore.GREEN if godot_connected else Fore.RED}{'연결됨' if godot_connected else '연결 안됨'}")
        
        # 현재 프로젝트/씬
        print(f"현재 프로젝트: {Fore.YELLOW}{self.conversation_context.get('current_project', '없음')}")
        print(f"현재 씬: {Fore.YELLOW}{self.conversation_context.get('current_scene', '없음')}")
        
        # 활성 작업
        active_tasks = len(self.orchestrator.active_tasks)
        queued_tasks = self.orchestrator.task_queue.qsize()
        print(f"활성 작업: {Fore.YELLOW}{active_tasks}")
        print(f"대기 중인 작업: {Fore.YELLOW}{queued_tasks}")
        
        # 학습 통계
        stats = self.learning_system.get_statistics()
        print(f"학습된 패턴: {Fore.YELLOW}{stats['total_patterns']}")
        print(f"처리된 작업: {Fore.YELLOW}{stats['total_entries']}")
        print()
        
    async def _show_projects(self):
        """프로젝트 목록 표시"""
        print(f"\n{Fore.CYAN}=== 프로젝트 목록 ===")
        # TODO: 실제 프로젝트 목록 가져오기
        print(f"{Fore.YELLOW}현재 구현 중...")
        print()
        
    async def _show_scenes(self):
        """씬 목록 표시"""
        print(f"\n{Fore.CYAN}=== 씬 목록 ===")
        if self.conversation_context.get("current_project"):
            # 현재 프로젝트의 씬 목록
            scene_tree = self.orchestrator.godot.get_scene_tree()
            self._print_scene_tree(scene_tree)
        else:
            print(f"{Fore.YELLOW}프로젝트가 선택되지 않았습니다.")
        print()
        
    def _print_scene_tree(self, tree: Dict[str, Any], indent: int = 0):
        """씬 트리 출력"""
        # TODO: 실제 씬 트리 파싱 및 출력
        print(f"{Fore.YELLOW}씬 트리 표시...")
        
    async def _show_tasks(self):
        """작업 목록 표시"""
        print(f"\n{Fore.CYAN}=== 작업 목록 ===")
        
        # 활성 작업
        if self.orchestrator.active_tasks:
            print(f"{Fore.YELLOW}진행 중:")
            for task_id, task in self.orchestrator.active_tasks.items():
                print(f"  {Fore.GREEN}• {task.description} ({task.status})")
                
        # 최근 완료 작업
        recent_completed = self.orchestrator.completed_tasks[-5:]
        if recent_completed:
            print(f"\n{Fore.YELLOW}최근 완료:")
            for task in recent_completed:
                status_color = Fore.GREEN if task.status == "completed" else Fore.RED
                print(f"  {status_color}• {task.description} ({task.status})")
        print()
        
    async def _show_learning_stats(self):
        """학습 통계 표시"""
        stats = self.learning_system.get_statistics()
        
        print(f"\n{Fore.CYAN}=== 학습 통계 ===")
        print(f"총 학습 항목: {Fore.YELLOW}{stats['total_entries']}")
        print(f"발견된 패턴: {Fore.YELLOW}{stats['total_patterns']}")
        
        if stats['metrics']:
            print(f"\n{Fore.CYAN}성능 메트릭:")
            for metric, data in stats['metrics'].items():
                trend_color = Fore.GREEN if data['trend'] == 'improving' else Fore.YELLOW
                print(f"  {metric}: {Fore.WHITE}{data['current']:.2f} {trend_color}({data['trend']})")
        print()
        
    async def _handle_natural_language(self, user_input: str):
        """자연어 입력 처리"""
        print(f"{Fore.CYAN}처리 중...")
        
        # 컨텍스트 추가
        context = {
            "current_project": self.conversation_context.get("current_project"),
            "current_scene": self.conversation_context.get("current_scene"),
            "conversation_history": self.conversation_context["history"][-5:]  # 최근 5개
        }
        
        # Orchestrator로 처리
        response = await self.orchestrator.process_user_input(user_input, context)
        
        # 응답 출력
        print(f"{Fore.GREEN}AutoCI: {response}")
        
        # 컨텍스트 업데이트
        self._update_context_from_response(response)
        
        # 히스토리에 추가
        self.conversation_context["history"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "assistant",
            "content": response
        })
        
    def _update_context_from_response(self, response: str):
        """응답에서 컨텍스트 업데이트"""
        # 프로젝트 생성 감지
        if "프로젝트 생성" in response or "project created" in response.lower():
            # 프로젝트 이름 추출 시도
            match = re.search(r'프로젝트[:\s]+(\S+)', response)
            if match:
                self.conversation_context["current_project"] = match.group(1)
                
        # 씬 변경 감지
        if "씬" in response or "scene" in response.lower():
            match = re.search(r'씬[:\s]+(\S+)', response)
            if match:
                self.conversation_context["current_scene"] = match.group(1)
                
    async def _shutdown(self):
        """시스템 종료"""
        print(f"\n{Fore.YELLOW}시스템을 종료하는 중...")
        
        # 학습 시스템 종료
        if self.learning_system:
            await self.learning_system.stop()
            
        # Orchestrator 종료
        if self.orchestrator:
            await self.orchestrator.stop()
            
        print(f"{Fore.GREEN}안녕히 가세요! 👋")
        sys.exit(0)

class InteractiveAutoCI:
    """대화형 AutoCI 메인 클래스"""
    
    def __init__(self):
        self.interface = ConversationalInterface()
        
    async def run(self):
        """실행"""
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('autoci_conversation.log'),
                logging.StreamHandler()
            ]
        )
        
        # 인터페이스 시작
        await self.interface.start()

def main():
    """메인 함수"""
    app = InteractiveAutoCI()
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n{Fore.RED}오류 발생: {e}")
        logging.error(f"치명적 오류: {e}", exc_info=True)

if __name__ == "__main__":
    main()