#!/usr/bin/env python3
"""
실시간 시각적 모니터링 시스템

24시간 게임 개발 과정을 실시간으로 시각화하고
사용자가 직관적으로 AI의 작업을 관찰할 수 있게 하는 시스템
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import deque
import curses

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RealtimeVisualMonitor:
    """실시간 시각적 모니터링 시스템"""
    
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.is_running = False
        self.update_thread = None
        
        # 상태 데이터
        self.state = {
            "start_time": None,
            "elapsed_time": timedelta(0),
            "remaining_time": timedelta(hours=24),
            "current_phase": "초기화",
            "current_task": "시스템 준비 중...",
            "progress": 0,
            "quality_score": 0,
            "iterations": 0,
            "errors_fixed": 0,
            "features_added": 0,
            "ai_actions": deque(maxlen=10),  # 최근 10개 액션
            "godot_status": "대기중",
            "learning_status": "대기중",
            "user_commands": deque(maxlen=5),  # 최근 5개 명령
            "performance": {
                "cpu": 0,
                "memory": 0,
                "gpu": 0
            }
        }
        
        # Rich 컴포넌트
        if self.use_rich:
            self.console = Console()
            self.layout = self._create_layout()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            )
    
    def _create_layout(self) -> Layout:
        """Rich 레이아웃 생성"""
        layout = Layout()
        
        # 메인 레이아웃 구성
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # 바디 레이아웃 분할
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # 왼쪽 패널 분할
        layout["left"].split_column(
            Layout(name="status", size=10),
            Layout(name="actions", size=12),
            Layout(name="progress_bars", size=6)
        )
        
        # 오른쪽 패널 분할
        layout["right"].split_column(
            Layout(name="stats", size=12),
            Layout(name="commands", size=8),
            Layout(name="performance", size=8)
        )
        
        return layout
    
    def start(self):
        """모니터링 시작"""
        self.is_running = True
        self.state["start_time"] = datetime.now()
        
        if self.use_rich:
            self._start_rich_monitor()
        else:
            self._start_simple_monitor()
    
    def _start_rich_monitor(self):
        """Rich 기반 모니터 시작"""
        def update_display():
            with Live(self.layout, refresh_per_second=2, console=self.console) as live:
                while self.is_running:
                    self._update_rich_display()
                    time.sleep(0.5)
        
        self.update_thread = threading.Thread(target=update_display)
        self.update_thread.start()
    
    def _start_simple_monitor(self):
        """단순 터미널 모니터 시작"""
        def update_display():
            while self.is_running:
                self._update_simple_display()
                time.sleep(1)
        
        self.update_thread = threading.Thread(target=update_display)
        self.update_thread.start()
    
    def _update_rich_display(self):
        """Rich 디스플레이 업데이트"""
        # 헤더 업데이트
        self.layout["header"].update(self._create_header())
        
        # 상태 패널 업데이트
        self.layout["status"].update(self._create_status_panel())
        
        # 액션 로그 업데이트
        self.layout["actions"].update(self._create_actions_panel())
        
        # 진행 바 업데이트
        self.layout["progress_bars"].update(self._create_progress_panel())
        
        # 통계 업데이트
        self.layout["stats"].update(self._create_stats_panel())
        
        # 명령 히스토리 업데이트
        self.layout["commands"].update(self._create_commands_panel())
        
        # 성능 모니터 업데이트
        self.layout["performance"].update(self._create_performance_panel())
        
        # 푸터 업데이트
        self.layout["footer"].update(self._create_footer())
    
    def _create_header(self) -> Panel:
        """헤더 생성"""
        elapsed = self._format_time(self.state["elapsed_time"])
        remaining = self._format_time(self.state["remaining_time"])
        
        header_text = Text(
            f"🎮 24시간 AI 게임 개발 | ⏱️ 경과: {elapsed} | ⏳ 남은 시간: {remaining}",
            style="bold white on blue"
        )
        
        return Panel(header_text, box=box.DOUBLE_EDGE)
    
    def _create_status_panel(self) -> Panel:
        """상태 패널 생성"""
        table = Table(show_header=False, box=None)
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("🎯 현재 단계:", self.state["current_phase"])
        table.add_row("🔧 현재 작업:", self.state["current_task"])
        table.add_row("📊 진행률:", f"{self.state['progress']:.1f}%")
        table.add_row("⭐ 품질 점수:", f"{self.state['quality_score']}/100")
        table.add_row("🎮 Godot 상태:", self.state["godot_status"])
        table.add_row("🧠 학습 상태:", self.state["learning_status"])
        
        return Panel(table, title="📊 현재 상태", border_style="green")
    
    def _create_actions_panel(self) -> Panel:
        """AI 액션 로그 패널"""
        actions_text = ""
        
        for action in self.state["ai_actions"]:
            timestamp = action.get("time", "").split('T')[1].split('.')[0] if "time" in action else ""
            action_type = action.get("type", "")
            description = action.get("description", "")
            
            # 액션 타입별 아이콘
            icon = {
                "click": "🖱️",
                "type": "⌨️",
                "menu": "📋",
                "create": "✨",
                "modify": "🔧",
                "test": "🧪",
                "fix": "🔨",
                "optimize": "⚡"
            }.get(action_type, "📌")
            
            actions_text += f"[{timestamp}] {icon} {description}\n"
        
        return Panel(
            actions_text.strip() or "대기 중...",
            title="🤖 AI 실시간 액션",
            border_style="blue"
        )
    
    def _create_progress_panel(self) -> Panel:
        """진행 상황 바 패널"""
        # 전체 진행률
        overall_bar = self._create_progress_bar("전체 진행", self.state["progress"], 100)
        
        # 현재 단계 진행률
        phase_progress = self.state.get("phase_progress", 0)
        phase_bar = self._create_progress_bar("단계 진행", phase_progress, 100)
        
        # 품질 점수
        quality_bar = self._create_progress_bar("품질 점수", self.state["quality_score"], 100)
        
        content = f"{overall_bar}\n{phase_bar}\n{quality_bar}"
        
        return Panel(content, title="📈 진행 상황", border_style="yellow")
    
    def _create_stats_panel(self) -> Panel:
        """통계 패널"""
        table = Table(show_header=False, box=None)
        table.add_column("Stat", style="magenta")
        table.add_column("Value", style="white")
        
        table.add_row("🔄 반복 횟수:", str(self.state["iterations"]))
        table.add_row("🔧 수정된 오류:", str(self.state["errors_fixed"]))
        table.add_row("✨ 추가된 기능:", str(self.state["features_added"]))
        table.add_row("🧠 학습 사이클:", str(self.state.get("learning_cycles", 0)))
        table.add_row("💬 사용자 명령:", str(len(self.state["user_commands"])))
        
        return Panel(table, title="📊 통계", border_style="cyan")
    
    def _create_commands_panel(self) -> Panel:
        """사용자 명령 히스토리"""
        commands_text = ""
        
        for cmd in self.state["user_commands"]:
            timestamp = cmd.get("time", "").split('T')[1].split('.')[0] if "time" in cmd else ""
            command = cmd.get("command", "")
            commands_text += f"[{timestamp}] > {command}\n"
        
        return Panel(
            commands_text.strip() or "명령 대기 중...",
            title="💬 사용자 명령",
            border_style="green"
        )
    
    def _create_performance_panel(self) -> Panel:
        """성능 모니터 패널"""
        perf = self.state["performance"]
        
        cpu_bar = self._create_progress_bar("CPU", perf["cpu"], 100)
        mem_bar = self._create_progress_bar("RAM", perf["memory"], 100)
        gpu_bar = self._create_progress_bar("GPU", perf["gpu"], 100)
        
        content = f"{cpu_bar}\n{mem_bar}\n{gpu_bar}"
        
        return Panel(content, title="⚡ 시스템 성능", border_style="red")
    
    def _create_footer(self) -> Panel:
        """푸터 생성"""
        commands = "add feature [기능] | modify [항목] | status | pause | resume | save | report | quit"
        footer_text = Text(f"명령어: {commands}", style="dim white")
        
        return Panel(footer_text, box=box.DOUBLE_EDGE)
    
    def _create_progress_bar(self, label: str, value: float, max_value: float) -> str:
        """진행 바 생성"""
        percentage = (value / max_value) * 100 if max_value > 0 else 0
        bar_length = 20
        filled_length = int(bar_length * percentage / 100)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        return f"{label:<12} [{bar}] {percentage:>5.1f}%"
    
    def _update_simple_display(self):
        """단순 터미널 디스플레이 업데이트"""
        # 화면 클리어
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # 상태 출력
        print("=" * 80)
        print(f"🎮 24시간 AI 게임 개발 시스템")
        print("=" * 80)
        
        elapsed = self._format_time(self.state["elapsed_time"])
        remaining = self._format_time(self.state["remaining_time"])
        
        print(f"⏱️  경과 시간: {elapsed} | 남은 시간: {remaining}")
        print(f"📊 진행률: {self.state['progress']:.1f}% | 품질: {self.state['quality_score']}/100")
        print(f"🎯 현재 단계: {self.state['current_phase']}")
        print(f"🔧 현재 작업: {self.state['current_task']}")
        print()
        
        print("📊 통계:")
        print(f"  반복: {self.state['iterations']} | 오류 수정: {self.state['errors_fixed']} | 기능 추가: {self.state['features_added']}")
        print()
        
        print("🤖 최근 AI 액션:")
        for action in list(self.state["ai_actions"])[-5:]:
            print(f"  - {action.get('description', '')}")
        print()
        
        print("💬 최근 사용자 명령:")
        for cmd in list(self.state["user_commands"])[-3:]:
            print(f"  > {cmd.get('command', '')}")
        print()
        
        print("-" * 80)
        print("명령어: add feature [기능] | modify [항목] | status | pause | resume | quit")
    
    def update_state(self, new_state: Dict[str, Any]):
        """상태 업데이트"""
        # 시간 업데이트
        if new_state.get("start_time"):
            elapsed = datetime.now() - new_state["start_time"]
            self.state["elapsed_time"] = elapsed
            self.state["remaining_time"] = timedelta(hours=24) - elapsed
        
        # 다른 상태 업데이트
        for key, value in new_state.items():
            if key in self.state and key not in ["ai_actions", "user_commands"]:
                self.state[key] = value
    
    def add_ai_action(self, action: Dict[str, Any]):
        """AI 액션 추가"""
        action["time"] = datetime.now().isoformat()
        self.state["ai_actions"].append(action)
    
    def add_user_command(self, command: str):
        """사용자 명령 추가"""
        self.state["user_commands"].append({
            "time": datetime.now().isoformat(),
            "command": command
        })
    
    def update_performance(self, cpu: float, memory: float, gpu: float):
        """성능 데이터 업데이트"""
        self.state["performance"] = {
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu
        }
    
    def _format_time(self, td: timedelta) -> str:
        """시간 포맷팅"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def stop(self):
        """모니터링 중지"""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=2)
        
        if self.use_rich and hasattr(self, 'console'):
            self.console.clear()


class MonitorController:
    """모니터 컨트롤러 - 외부에서 모니터 상태 업데이트"""
    
    def __init__(self, monitor: RealtimeVisualMonitor):
        self.monitor = monitor
    
    def log_action(self, action_type: str, description: str):
        """AI 액션 로깅"""
        self.monitor.add_ai_action({
            "type": action_type,
            "description": description
        })
    
    def update_phase(self, phase: str, task: str = ""):
        """단계 및 작업 업데이트"""
        self.monitor.update_state({
            "current_phase": phase,
            "current_task": task
        })
    
    def update_progress(self, progress: float, quality: float = None):
        """진행률 업데이트"""
        update = {"progress": progress}
        if quality is not None:
            update["quality_score"] = quality
        self.monitor.update_state(update)
    
    def increment_stats(self, stat_name: str, amount: int = 1):
        """통계 증가"""
        current_value = self.monitor.state.get(stat_name, 0)
        self.monitor.update_state({stat_name: current_value + amount})
    
    def set_godot_status(self, status: str):
        """Godot 상태 설정"""
        self.monitor.update_state({"godot_status": status})
    
    def set_learning_status(self, status: str):
        """학습 상태 설정"""
        self.monitor.update_state({"learning_status": status})


def demo():
    """데모 실행"""
    import random
    
    # 모니터 생성 및 시작
    monitor = RealtimeVisualMonitor(use_rich=True)
    controller = MonitorController(monitor)
    
    monitor.start()
    
    # 시뮬레이션
    try:
        phases = [
            ("초기화", "프로젝트 설정 중..."),
            ("기획", "게임 컨셉 정의 중..."),
            ("프로토타입", "기본 메커니즘 구현 중..."),
            ("개발", "핵심 기능 구현 중..."),
            ("폴리싱", "그래픽 및 사운드 추가 중..."),
            ("테스트", "버그 수정 중..."),
            ("최적화", "성능 개선 중...")
        ]
        
        action_types = [
            ("click", "Godot 메뉴 클릭"),
            ("type", "스크립트 작성"),
            ("create", "새 씬 생성"),
            ("modify", "플레이어 속도 조정"),
            ("test", "게임 플레이 테스트"),
            ("fix", "충돌 버그 수정"),
            ("optimize", "렌더링 최적화")
        ]
        
        start_time = datetime.now()
        
        for i in range(100):
            # 상태 업데이트
            progress = (i / 100) * 100
            quality = min(100, 30 + i * 0.7)
            
            controller.update_progress(progress, quality)
            controller.update_phase(
                phases[i % len(phases)][0],
                phases[i % len(phases)][1]
            )
            
            # 랜덤 액션
            if i % 3 == 0:
                action = random.choice(action_types)
                controller.log_action(action[0], action[1])
            
            # 통계 업데이트
            if i % 5 == 0:
                controller.increment_stats("iterations")
            if i % 10 == 0:
                controller.increment_stats("errors_fixed")
            if i % 15 == 0:
                controller.increment_stats("features_added")
            
            # 성능 업데이트
            monitor.update_performance(
                cpu=random.uniform(20, 80),
                memory=random.uniform(30, 70),
                gpu=random.uniform(40, 90)
            )
            
            # Godot 상태
            if i % 20 == 0:
                controller.set_godot_status(random.choice(["실행 중", "편집 중", "빌드 중"]))
            
            # 학습 상태
            if i % 25 == 0:
                controller.set_learning_status(random.choice(["학습 중", "분석 중", "대기 중"]))
            
            # 사용자 명령 시뮬레이션
            if i % 30 == 0:
                monitor.add_user_command(random.choice([
                    "add feature double jump",
                    "modify player speed",
                    "status",
                    "optimize rendering"
                ]))
            
            time.sleep(0.5)
        
    except KeyboardInterrupt:
        print("\n데모 중단됨")
    
    finally:
        monitor.stop()


if __name__ == "__main__":
    demo()