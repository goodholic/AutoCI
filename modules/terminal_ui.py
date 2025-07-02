#!/usr/bin/env python3
"""
AutoCI 터미널 UI - 하단에 명령어 인터페이스를 제공
"""

import os
import sys
from typing import List, Optional
from datetime import datetime

class TerminalUI:
    """터미널 하단에 표시되는 UI 시스템"""
    
    def __init__(self):
        self.commands = [
            ("1", "🎮 새 게임 만들기", "create [type] game"),
            ("2", "🤖 AI 제어 데모", "ai demo"),
            ("3", "💬 한글 대화 모드", "chat"),
            ("4", "📊 시스템 상태", "status"),
            ("5", "🔧 게임 수정", "modify"),
            ("6", "📚 AI 학습", "learn"),
            ("7", "🌐 멀티플레이어", "create multiplayer"),
            ("8", "❓ 도움말", "help"),
            ("9", "🚪 종료", "exit"),
        ]
        self.quick_commands = {
            "p": "create platformer game",
            "r": "create racing game",
            "z": "create puzzle game",
            "m": "modify",
            "s": "status",
            "h": "help",
        }
    
    def clear_screen(self):
        """화면 지우기"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_header(self):
        """상단 헤더 표시"""
        print("╔" + "═"*78 + "╗")
        print("║" + " "*25 + "🤖 AutoCI - AI 게임 개발 시스템" + " "*22 + "║")
        print("║" + " "*20 + "AI가 실시간으로 Godot을 제어하여 게임을 만듭니다" + " "*11 + "║")
        print("╚" + "═"*78 + "╝")
    
    def show_main_menu(self):
        """메인 메뉴 표시"""
        print("\n┌─ 주요 명령어 " + "─"*64 + "┐")
        
        # 두 열로 명령어 표시
        for i in range(0, len(self.commands), 2):
            left = self.commands[i]
            right = self.commands[i+1] if i+1 < len(self.commands) else None
            
            left_text = f"│ [{left[0]}] {left[1]:<20}"
            if right:
                right_text = f"[{right[0]}] {right[1]:<20}"
                print(f"{left_text} {right_text:>35} │")
            else:
                print(f"{left_text}" + " "*38 + "│")
        
        print("└" + "─"*78 + "┘")
    
    def show_quick_commands(self):
        """빠른 명령어 표시"""
        print("\n┌─ 빠른 명령어 " + "─"*64 + "┐")
        print("│ ", end="")
        for key, desc in self.quick_commands.items():
            print(f"[{key}] {desc.split()[1][:8]:<8} ", end="")
        print(" │")
        print("└" + "─"*78 + "┘")
    
    def show_current_status(self, project_name: Optional[str] = None, ai_status: str = "대기중"):
        """현재 상태 표시"""
        print("\n┌─ 현재 상태 " + "─"*66 + "┐")
        if project_name:
            print(f"│ 🎮 현재 프로젝트: {project_name:<58} │")
        else:
            print(f"│ 🎮 현재 프로젝트: {'없음':<58} │")
        print(f"│ 🤖 AI 상태: {ai_status:<62} │")
        print(f"│ ⏰ 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<64} │")
        print("└" + "─"*78 + "┘")
    
    def show_input_prompt(self) -> str:
        """입력 프롬프트 표시"""
        print("\n" + "─"*80)
        return "AutoCI > "
    
    def show_ai_action(self, action: str):
        """AI 액션 표시"""
        print(f"\n🤖 AI: {action}")
    
    def show_progress(self, task: str, progress: int, max_progress: int = 100):
        """진행률 표시"""
        bar_length = 40
        filled = int(bar_length * progress / max_progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r{task}: [{bar}] {progress}/{max_progress}", end="", flush=True)
    
    def show_error(self, message: str):
        """에러 메시지 표시"""
        print(f"\n❌ 오류: {message}")
    
    def show_success(self, message: str):
        """성공 메시지 표시"""
        print(f"\n✅ {message}")
    
    def show_info(self, message: str):
        """정보 메시지 표시"""
        print(f"\nℹ️ {message}")
    
    def show_game_creation_ui(self):
        """게임 생성 UI 표시"""
        print("\n┌─ 게임 타입 선택 " + "─"*62 + "┐")
        print("│ [1] 🏃 Platformer - 점프와 달리기가 있는 2D 플랫폼 게임" + " "*22 + "│")
        print("│ [2] 🏎️  Racing     - 스피드를 즐기는 레이싱 게임" + " "*28 + "│")
        print("│ [3] 🧩 Puzzle     - 머리를 쓰는 퍼즐 게임" + " "*35 + "│")
        print("│ [4] ⚔️  RPG        - 모험과 성장이 있는 롤플레잉 게임" + " "*23 + "│")
        print("│ [5] 🔫 FPS        - 1인칭 슈팅 게임" + " "*42 + "│")
        print("│ [6] 🏰 Strategy   - 전략적 사고가 필요한 전략 게임" + " "*26 + "│")
        print("└" + "─"*78 + "┘")
    
    def format_command_help(self, command: str, description: str) -> str:
        """명령어 도움말 포맷"""
        return f"  {command:<25} - {description}"
    
    def show_welcome_animation(self):
        """환영 애니메이션"""
        frames = [
            "🤖", "🤖💭", "🤖💭🎮", "🤖💭🎮✨"
        ]
        import time
        for frame in frames:
            print(f"\r{frame} AutoCI 시작 중...", end="", flush=True)
            time.sleep(0.3)
        print("\r" + " "*30 + "\r", end="")


# 전역 인스턴스
_ui = None

def get_terminal_ui() -> TerminalUI:
    """터미널 UI 싱글톤 인스턴스 반환"""
    global _ui
    if _ui is None:
        _ui = TerminalUI()
    return _ui