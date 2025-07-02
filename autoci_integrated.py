#!/usr/bin/env python3
"""
AutoCI 통합 시스템 - 모든 기능을 하나로 통합
Godot 제어 시각화와 하단 Input UI 포함
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path
from datetime import datetime
import threading
from typing import Optional, Dict, List
import queue
import curses
import json

# 프로젝트 루트 설정
AUTOCI_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(AUTOCI_ROOT))

class AutoCIIntegrated:
    """통합된 AutoCI 시스템"""
    
    def __init__(self):
        self.root = AUTOCI_ROOT
        self.running = True
        self.current_project = None
        self.godot_process = None
        self.message_queue = queue.Queue()
        self.command_history = []
        
        # 실시간 모니터링용
        self.monitor_data = {
            "status": {},
            "progress": {},
            "logs": []
        }
        self.max_logs = 10
        self.log_dir = Path("logs/24h_improvement")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 모듈 초기화
        self.init_modules()
        
    def init_modules(self):
        """필요한 모듈 초기화"""
        # AI 컨트롤러
        self.ai_controller = None
        try:
            from modules.godot_ai_controller import get_ai_controller
            self.ai_controller = get_ai_controller()
        except:
            pass
            
        # 터미널 UI
        self.ui = None
        try:
            from modules.terminal_ui import get_terminal_ui
            self.ui = get_terminal_ui()
        except:
            pass
            
        # 게임 빌더
        self.game_builder = None
        try:
            from modules.progressive_game_builder import get_progressive_builder
            self.game_builder = get_progressive_builder()
        except:
            pass
    
    def clear_screen(self):
        """화면 지우기"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        """상단 헤더 표시"""
        print("╔" + "═"*78 + "╗")
        print("║" + " "*20 + "🤖 AutoCI - 통합 AI 게임 개발 시스템" + " "*22 + "║")
        print("║" + " "*15 + "Godot을 AI가 제어하며 24시간 자동 개발합니다" + " "*18 + "║")
        print("╚" + "═"*78 + "╝")
    
    def show_status_bar(self):
        """상태 바 표시"""
        print("\n" + "─"*80)
        status = f"프로젝트: {self.current_project or '없음'}"
        godot_status = "Godot: 실행중" if self.godot_process else "Godot: 미실행"
        time_str = datetime.now().strftime("%H:%M:%S")
        
        print(f"│ {status:<30} │ {godot_status:<25} │ {time_str:<20} │")
        print("─"*80)
    
    def show_command_help(self):
        """명령어 도움말 표시"""
        print("\n📋 명령어 목록:")
        print("─"*80)
        commands = [
            ("create [type] game", "게임 제작 시작 (platformer, racing, puzzle, rpg)"),
            ("resume", "중단된 게임 개발 재개"),
            ("monitor", "실시간 모니터링 상태 보기"),
            ("open_godot", "Godot 에디터 열기"),
            ("learn", "AI 모델 기반 연속 학습 시작"),
            ("learn low", "메모리 최적화 연속 학습"),
            ("fix", "학습 기반 AI 게임 제작 능력 업데이트"),
            ("build-godot", "Windows 버전 Godot 빌드"),
            ("build-godot-linux", "Linux 버전 Godot 빌드"),
            ("ai demo", "AI가 Godot을 제어하는 데모"),
            ("status", "시스템 상태 확인"),
            ("help", "도움말 표시"),
            ("exit", "종료"),
        ]
        
        for cmd, desc in commands:
            print(f"  {cmd:<25} - {desc}")
        print("─"*80)
    
    def show_input_ui(self):
        """하단 Input UI 표시"""
        print("\n" + "═"*80)
        print("💬 명령어를 입력하세요 (help: 도움말)")
        print("─"*80)
    
    async def handle_create_game(self, game_type: str):
        """게임 생성 처리"""
        # 기존 상태 확인
        improvement_status = Path("improvement_status.json")
        if improvement_status.exists():
            with open(improvement_status, 'r', encoding='utf-8') as f:
                status = json.load(f)
                if status.get("status") == "running":
                    self.monitor_data["logs"].append("⚠️ 진행 중인 게임 개발이 있습니다.")
                    self.monitor_data["logs"].append(f"   프로젝트: {status.get('game_name')}")
                    self.monitor_data["logs"].append("   'resume' 명령어로 계속할 수 있습니다.")
        
        # 게임 이름 설정
        game_name = f"{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_project = game_name
        
        # 모니터링 로그 추가
        self.monitor_data["logs"].append(f"🎮 {game_type} 게임 제작을 시작합니다...")
        
        # 24시간 게임 제작 공장 시작
        try:
            from modules.game_factory_24h import get_game_factory
            factory = get_game_factory()
            
            self.monitor_data["logs"].append("🏭 24시간 게임 제작 공장을 시작합니다...")
            result = await factory.start_factory(game_name, game_type)
            
            # 24시간 개선 태스크가 실행 중인지 확인
            if hasattr(factory, 'improvement_task') and factory.improvement_task:
                self.monitor_data["logs"].append("✅ 24시간 백그라운드 개선이 시작되었습니다.")
                self.monitor_data["logs"].append("📊 실시간 진행 상황이 표시됩니다.")
            
        except Exception as e:
            self.monitor_data["logs"].append(f"❌ 게임 제작 중 오류: {e}")
    
    async def handle_open_godot(self):
        """Godot 열기"""
        print("\n🚀 Godot 에디터를 엽니다...")
        
        godot_paths = [
            self.root / "godot_ai_build" / "output" / "godot.ai.editor.windows.x86_64.exe",
            self.root / "godot_ai_build" / "godot-source" / "bin" / "godot.windows.editor.x86_64.exe",
            self.root / "godot_ai_build" / "Godot_v4.3-stable_win64.exe",
        ]
        
        godot_exe = None
        for path in godot_paths:
            if path.exists():
                godot_exe = str(path)
                break
        
        if godot_exe:
            # WSL에서 Windows 프로그램 실행
            windows_path = godot_exe.replace("/mnt/c", "C:").replace("/mnt/d", "D:").replace("/", "\\")
            cmd = ["cmd.exe", "/c", "start", "", windows_path]
            
            try:
                subprocess.run(cmd, check=True)
                print("✅ Godot 에디터가 열렸습니다!")
                
                # AI 제어 데모 옵션
                if self.ai_controller:
                    show_demo = input("\nAI 제어 데모를 보시겠습니까? (y/N): ")
                    if show_demo.lower() == 'y':
                        await self.ai_controller.interactive_ai_control("create node")
            except Exception as e:
                print(f"❌ Godot 실행 실패: {e}")
        else:
            print("❌ Godot 실행 파일을 찾을 수 없습니다.")
    
    async def handle_learn(self, low_memory: bool = False):
        """AI 학습 시작"""
        if low_memory:
            print("\n📚 메모리 최적화 연속 학습을 시작합니다...")
            # autoci learn low 실행
            subprocess.run([sys.executable, str(self.root / "autoci.py"), "learn", "low"])
        else:
            print("\n📚 AI 모델 기반 연속 학습을 시작합니다...")
            # autoci learn 실행
            subprocess.run([sys.executable, str(self.root / "autoci.py"), "learn"])
    
    async def handle_fix(self):
        """AI 게임 제작 능력 업데이트"""
        print("\n🔧 학습을 토대로 AI의 게임 제작 능력을 업데이트합니다...")
        subprocess.run([sys.executable, str(self.root / "autoci.py"), "fix"])
    
    async def handle_build_godot(self, linux: bool = False):
        """Godot 빌드"""
        if linux:
            print("\n🔨 Linux 버전 Godot을 빌드합니다...")
            subprocess.run(["bash", str(self.root / "build-godot-linux")])
        else:
            print("\n🔨 Windows 버전 Godot을 빌드합니다...")
            subprocess.run(["bash", str(self.root / "build-godot")])
            print("\n✅ 빌드 완료!")
            print(f"📁 빌드된 파일: {self.root}/godot_ai_build/output/godot.windows.editor.x86_64.exe")
    
    async def handle_ai_demo(self):
        """AI 데모"""
        if self.ai_controller:
            print("\n🤖 AI가 Godot을 제어하는 데모를 시작합니다...")
            await self.ai_controller.start_ai_control_demo()
        else:
            print("❌ AI 컨트롤러를 사용할 수 없습니다.")
    
    def show_status(self):
        """시스템 상태 표시"""
        print("\n" + "="*80)
        print("📊 AutoCI 시스템 상태")
        print("="*80)
        print(f"현재 프로젝트: {self.current_project or '없음'}")
        print(f"Godot 상태: {'실행중' if self.godot_process else '미실행'}")
        print(f"AI 컨트롤러: {'활성' if self.ai_controller else '비활성'}")
        print(f"게임 빌더: {'활성' if self.game_builder else '비활성'}")
        print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    async def handle_resume_game(self):
        """중단된 게임 개발 재개"""
        try:
            from modules.game_factory_24h import get_game_factory
            factory = get_game_factory()
            await factory.resume_factory()
            
            # 재개된 프로젝트 이름 가져오기
            improvement_status = Path("improvement_status.json")
            if improvement_status.exists():
                with open(improvement_status, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    self.current_project = status.get('game_name')
                    
            self.monitor_data["logs"].append("✅ 게임 개발을 재개했습니다.")
        except Exception as e:
            self.monitor_data["logs"].append(f"❌ 재개 중 오류: {e}")
    
    def show_monitor_info(self):
        """실시간 모니터링 정보 표시"""
        print("\n" + "="*80)
        print("📊 24시간 게임 개발 실시간 모니터링")
        print("="*80)
        
        if self.monitor_data["status"]:
            status = self.monitor_data["status"]
            print(f"🎮 프로젝트: {status.get('project_name', '없음')}")
            print(f"⏰ 경과: {status.get('elapsed_time', '00:00:00')} | 남은 시간: {status.get('remaining_time', '24:00:00')}")
            print(f"📈 진행률: {status.get('progress_percent', 0):.1f}%")
            print(f"🔄 반복: {status.get('iteration_count', 0)} | 🔨 수정: {status.get('fixes_count', 0)} | ✨ 개선: {status.get('improvements_count', 0)}")
            print(f"📊 품질 점수: {status.get('quality_score', 0)}/100")
        
        if self.monitor_data["progress"]:
            progress = self.monitor_data["progress"]
            print(f"\n🔧 현재 작업: {progress.get('current_task', '대기 중')}")
            print(f"💪 끈질김: {progress.get('persistence_level', 'NORMAL')}")
            print(f"🎨 창의성: {progress.get('creativity_level', 0)}/10")
        
        print("\n📋 최근 로그:")
        for log in self.monitor_data["logs"][-5:]:
            print(f"  {log}")
        
        print("="*80)
    
    async def process_command(self, command: str):
        """명령어 처리"""
        parts = command.strip().lower().split()
        if not parts:
            return
        
        cmd = parts[0]
        
        if cmd == "create" and len(parts) >= 3 and parts[2] == "game":
            game_type = parts[1]
            if game_type in ["platformer", "racing", "puzzle", "rpg"]:
                await self.handle_create_game(game_type)
            else:
                print("❌ 지원하는 게임 타입: platformer, racing, puzzle, rpg")
        
        elif cmd == "resume":
            # 중단된 게임 개발 재개
            await self.handle_resume_game()
        
        elif cmd == "monitor":
            # 실시간 모니터링 표시
            self.show_monitor_info()
        
        elif cmd == "open_godot":
            await self.handle_open_godot()
        
        elif cmd == "learn":
            if len(parts) > 1 and parts[1] == "low":
                await self.handle_learn(low_memory=True)
            else:
                await self.handle_learn()
        
        elif cmd == "fix":
            await self.handle_fix()
        
        elif cmd == "build-godot":
            await self.handle_build_godot()
        
        elif cmd == "build-godot-linux":
            await self.handle_build_godot(linux=True)
        
        elif cmd == "ai" and len(parts) > 1 and parts[1] == "demo":
            await self.handle_ai_demo()
        
        elif cmd == "status":
            self.show_status()
        
        elif cmd == "help":
            self.show_command_help()
        
        elif cmd == "exit" or cmd == "quit":
            self.running = False
            print("\n👋 AutoCI를 종료합니다.")
        
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print("💡 'help'를 입력하여 사용 가능한 명령어를 확인하세요.")
    
    async def run_with_curses(self, stdscr):
        """커서스 모드로 실행"""
        # 커서 숨기기
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.timeout(100)  # 100ms
        
        # 색상 설정
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        
        # 입력 버퍼
        input_buffer = ""
        
        # 모니터링 업데이트 스레드
        monitor_thread = threading.Thread(target=self._monitor_update_thread, daemon=True)
        monitor_thread.start()
        
        while self.running:
            try:
                # 화면 크기
                height, width = stdscr.getmaxyx()
                
                # 화면 지우기
                stdscr.clear()
                
                # 상단 헤더 (3줄)
                self._draw_header(stdscr, width)
                
                # 모니터링 영역 (height - 6줄)
                monitor_height = height - 6
                self._draw_monitor(stdscr, width, monitor_height, 3)
                
                # 하단 입력 영역 (3줄)
                self._draw_input(stdscr, width, height - 3, input_buffer)
                
                # 화면 새로고침
                stdscr.refresh()
                
                # 키 입력 처리
                key = stdscr.getch()
                if key != -1:
                    if key == ord('\n'):  # Enter
                        if input_buffer.strip():
                            # 비동기로 명령어 처리
                            asyncio.create_task(self.process_command(input_buffer))
                            self.command_history.append(input_buffer)
                            input_buffer = ""
                    elif key == curses.KEY_BACKSPACE or key == 127:
                        input_buffer = input_buffer[:-1]
                    elif key == 27:  # ESC
                        self.running = False
                    elif 32 <= key <= 126:  # 일반 문자
                        input_buffer += chr(key)
                    elif key >= 0x80:  # 한글 및 유니코드 문자 처리
                        try:
                            # UTF-8로 디코드 시도
                            input_buffer += chr(key)
                        except:
                            pass
                
                # 비동기 작업 처리
                await asyncio.sleep(0.01)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self.monitor_data["logs"].append(f"❌ 오류: {str(e)}")
    
    def _draw_header(self, stdscr, width):
        """헤더 그리기"""
        header_lines = [
            "╔" + "═"*78 + "╗",
            "║" + " "*20 + "🤖 AutoCI - 통합 AI 게임 개발 시스템" + " "*22 + "║",
            "╚" + "═"*78 + "╝"
        ]
        
        for i, line in enumerate(header_lines):
            if i < stdscr.getmaxyx()[0]:
                stdscr.addstr(i, 0, line[:width], curses.color_pair(1) | curses.A_BOLD)
    
    def _draw_monitor(self, stdscr, width, height, start_y):
        """모니터링 영역 그리기"""
        # 제목
        if start_y < stdscr.getmaxyx()[0]:
            stdscr.addstr(start_y, 0, "┌" + "─"*30 + " 실시간 모니터링 " + "─"*30 + "┐", curses.color_pair(3))
        
        # 상태 정보
        y = start_y + 1
        if self.monitor_data["status"]:
            status = self.monitor_data["status"]
            info_lines = [
                f"🎮 프로젝트: {status.get('project_name', '없음')}",
                f"⏰ 경과: {status.get('elapsed_time', '00:00:00')} | 남은 시간: {status.get('remaining_time', '24:00:00')}",
                f"📈 진행률: {status.get('progress_percent', 0):.1f}%",
                f"🔄 반복: {status.get('iteration_count', 0)} | 🔨 수정: {status.get('fixes_count', 0)} | ✨ 개선: {status.get('improvements_count', 0)}",
                f"📊 품질 점수: {status.get('quality_score', 0)}/100"
            ]
            
            for line in info_lines:
                if y < start_y + height - 2 and y < stdscr.getmaxyx()[0]:
                    stdscr.addstr(y, 2, line[:width-4], curses.color_pair(2))
                    y += 1
        
        # 진행 상황
        if self.monitor_data["progress"]:
            progress = self.monitor_data["progress"]
            y += 1
            if y < start_y + height - 2 and y < stdscr.getmaxyx()[0]:
                stdscr.addstr(y, 2, f"🔧 현재 작업: {progress.get('current_task', '대기 중')[:width-20]}", curses.color_pair(5))
                y += 1
            
            # 끈질김 레벨
            persistence = progress.get('persistence_level', 'NORMAL')
            color = curses.color_pair(2)
            if persistence in ['STUBBORN', 'OBSESSIVE']:
                color = curses.color_pair(3)
            elif persistence == 'INFINITE':
                color = curses.color_pair(4)
            
            if y < start_y + height - 2 and y < stdscr.getmaxyx()[0]:
                stdscr.addstr(y, 2, f"💪 끈질김: {persistence}", color)
                y += 1
            
            # 창의성 레벨
            creativity = progress.get('creativity_level', 0)
            creativity_bar = "█" * creativity + "░" * (10 - creativity)
            if y < start_y + height - 2 and y < stdscr.getmaxyx()[0]:
                stdscr.addstr(y, 2, f"🎨 창의성: [{creativity_bar}] {creativity}/10", curses.color_pair(5))
                y += 1
        
        # 실시간 로그
        y += 1
        if y < start_y + height - 2 and y < stdscr.getmaxyx()[0]:
            stdscr.addstr(y, 2, "📋 실시간 로그:", curses.color_pair(3))
            y += 1
        
        # 로그 메시지
        log_start = max(0, len(self.monitor_data["logs"]) - (start_y + height - y - 2))
        for log in self.monitor_data["logs"][log_start:]:
            if y < start_y + height - 1 and y < stdscr.getmaxyx()[0]:
                # 로그 줄 자르기
                if len(log) > width - 4:
                    log = log[:width-7] + "..."
                stdscr.addstr(y, 4, log, curses.color_pair(5))
                y += 1
        
        # 하단 테두리
        if start_y + height - 1 < stdscr.getmaxyx()[0]:
            stdscr.addstr(start_y + height - 1, 0, "└" + "─"*(width-2) + "┘", curses.color_pair(3))
    
    def _draw_input(self, stdscr, width, y, input_buffer):
        """입력 영역 그리기"""
        # 상단 경계선
        if y < stdscr.getmaxyx()[0]:
            stdscr.addstr(y, 0, "═"*width, curses.color_pair(1))
        
        # 입력 프롬프트
        if y + 1 < stdscr.getmaxyx()[0]:
            prompt = "💬 명령어: "
            stdscr.addstr(y + 1, 0, prompt, curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(y + 1, len(prompt), input_buffer + "_", curses.color_pair(5))
        
        # 도움말
        if y + 2 < stdscr.getmaxyx()[0]:
            help_text = "(help: 도움말, ESC: 종료)"
            stdscr.addstr(y + 2, 0, help_text, curses.color_pair(5) | curses.A_DIM)
    
    def _monitor_update_thread(self):
        """모니터링 데이터 업데이트 스레드"""
        while self.running:
            try:
                # 현재 프로젝트 찾기
                if self.current_project:
                    # 상태 파일 읽기
                    status_file = self.log_dir / f"{self.current_project}_status.json"
                    if status_file.exists():
                        with open(status_file, 'r', encoding='utf-8') as f:
                            self.monitor_data["status"] = json.load(f)
                    
                    # 진행 파일 읽기
                    progress_file = self.log_dir / f"{self.current_project}_progress.json"
                    if progress_file.exists():
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            self.monitor_data["progress"] = json.load(f)
                    
                    # 로그 파일 읽기
                    log_file = self.log_dir / "latest_improvement.log"
                    if log_file.exists():
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-20:]  # 마지막 20줄
                            for line in lines:
                                line = line.strip()
                                if line and line not in self.monitor_data["logs"]:
                                    self.monitor_data["logs"].append(line)
                                    if len(self.monitor_data["logs"]) > self.max_logs:
                                        self.monitor_data["logs"] = self.monitor_data["logs"][-self.max_logs:]
                
                time.sleep(0.5)  # 0.5초마다 업데이트
                
            except Exception:
                pass
    
    async def run(self):
        """메인 실행 루프"""
        # curses 모드로 실행
        try:
            await self._run_curses_wrapper()
        except Exception as e:
            print(f"❌ Curses 모드 실퇨, 일반 모드로 전환: {e}")
            await self.run_simple_mode()
    
    async def _run_curses_wrapper(self):
        """커서스 래퍼"""
        loop = asyncio.get_event_loop()
        
        def curses_main(stdscr):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_with_curses(stdscr))
        
        await loop.run_in_executor(None, curses.wrapper, curses_main)
    
    async def run_simple_mode(self):
        """일반 모드 실행 (커서스 없이)"""
        self.clear_screen()
        self.show_header()
        
        self.show_command_help()
        
        # 명령어 입력 루프
        while self.running:
            self.show_input_ui()
            try:
                command = input("> ").strip()
                if command:
                    self.command_history.append(command)
                    await self.process_command(command)
            except KeyboardInterrupt:
                print("\n\n👋 종료하려면 'exit'를 입력하세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
        if self.godot_process:
            self.godot_process.terminate()


def main():
    """메인 함수"""
    autoci = AutoCIIntegrated()
    
    try:
        asyncio.run(autoci.run())
    except KeyboardInterrupt:
        print("\n\nAutoCI가 종료되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()