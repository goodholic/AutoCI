#!/usr/bin/env python3
"""
24시간 게임 개선 실시간 모니터링 시스템
"""

import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import threading
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

class RealtimeMonitor:
    """24시간 게임 개선 실시간 모니터링"""
    
    def __init__(self):
        self.console = Console()
        self.monitoring = False
        self.monitor_task = None
        
        # 로그 디렉토리
        self.log_dir = Path("logs/24h_improvement")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 상태
        self.current_status = {
            "project_name": "없음",
            "start_time": None,
            "elapsed_time": "00:00:00",
            "remaining_time": "24:00:00",
            "progress_percent": 0,
            "iteration_count": 0,
            "fixes_count": 0,
            "improvements_count": 0,
            "quality_score": 0,
            "current_task": "대기 중",
            "last_activity": "시스템 시작",
            "persistence_level": "NORMAL",
            "creativity_level": 0,
            "is_desperate": False
        }
        
        # 최근 활동 로그
        self.recent_logs = []
        self.max_logs = 10
        
        # 레이아웃 설정
        self.layout = Layout()
        self.setup_layout()
    
    def setup_layout(self):
        """레이아웃 설정"""
        self.layout.split_column(
            Layout(name="header", size=12),
            Layout(name="status", size=8),
            Layout(name="logs", size=10),
            Layout(name="input", size=3)
        )
    
    def start_monitoring(self, project_name: str):
        """실시간 모니터링 시작"""
        self.current_status["project_name"] = project_name
        self.current_status["start_time"] = datetime.now()
        self.monitoring = True
        
        # 비동기 모니터링 태스크 시작
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        print(f"🎯 24시간 개선 실시간 모니터링을 시작했습니다: {project_name}")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
        print("🛑 24시간 개선 모니터링을 중지했습니다.")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        try:
            while self.monitoring:
                # 로그 파일들 확인
                await self._check_log_files()
                
                # 상태 업데이트
                self._update_status()
                
                # 1초 대기
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ 모니터링 오류: {e}")
    
    async def _check_log_files(self):
        """로그 파일들 확인"""
        project_name = self.current_status["project_name"]
        if project_name == "없음":
            return
        
        # 상태 파일 확인
        status_file = self.log_dir / f"{project_name}_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                    self.current_status.update(status_data)
            except (json.JSONDecodeError, Exception):
                pass
        
        # 진행 상황 파일 확인
        progress_file = self.log_dir / f"{project_name}_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.current_status.update(progress_data)
            except (json.JSONDecodeError, Exception):
                pass
        
        # 최신 로그 파일 확인
        latest_log = self.log_dir / "latest_improvement.log"
        if latest_log.exists():
            try:
                # 마지막 몇 줄만 읽기
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    new_lines = lines[-5:]  # 마지막 5줄
                    
                    for line in new_lines:
                        line = line.strip()
                        if line and line not in [log["message"] for log in self.recent_logs]:
                            self.recent_logs.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "message": line[:80] + "..." if len(line) > 80 else line
                            })
                    
                    # 최대 로그 수 제한
                    if len(self.recent_logs) > self.max_logs:
                        self.recent_logs = self.recent_logs[-self.max_logs:]
                        
            except Exception:
                pass
    
    def _update_status(self):
        """상태 업데이트"""
        if self.current_status["start_time"]:
            # 경과 시간 계산
            elapsed = datetime.now() - self.current_status["start_time"]
            self.current_status["elapsed_time"] = str(elapsed).split('.')[0]
            
            # 남은 시간 계산 (24시간 - 경과 시간)
            total_seconds = elapsed.total_seconds()
            remaining_seconds = max(0, 24 * 3600 - total_seconds)
            remaining = timedelta(seconds=remaining_seconds)
            self.current_status["remaining_time"] = str(remaining).split('.')[0]
            
            # 진행률 계산
            self.current_status["progress_percent"] = min(100, total_seconds / (24 * 3600) * 100)
    
    def create_header_panel(self) -> Panel:
        """헤더 패널 생성"""
        content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🏭 24시간 게임 개선 공장 - 실시간 모니터링                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎮 프로젝트: {self.current_status['project_name']}
⏰ 경과: {self.current_status['elapsed_time']} | 남은 시간: {self.current_status['remaining_time']} | 진행률: {self.current_status['progress_percent']:.1f}%
🔄 반복: {self.current_status['iteration_count']} | 수정: {self.current_status['fixes_count']} | 개선: {self.current_status['improvements_count']}
📊 게임 품질 점수: {self.current_status['quality_score']}/100
        """
        return Panel(content.strip(), style="bold blue")
    
    def create_status_panel(self) -> Panel:
        """상태 패널 생성"""
        # 끈질김 레벨 표시
        persistence_color = "green"
        if self.current_status['persistence_level'] == "DETERMINED":
            persistence_color = "yellow"
        elif self.current_status['persistence_level'] in ["STUBBORN", "OBSESSIVE"]:
            persistence_color = "orange"
        elif self.current_status['persistence_level'] == "INFINITE":
            persistence_color = "red"
        
        # 창의성 레벨 표시
        creativity_bar = "█" * self.current_status['creativity_level'] + "░" * (10 - self.current_status['creativity_level'])
        
        content = f"""
🔧 현재 작업: {self.current_status['current_task']}
💪 끈질김 레벨: [{persistence_color}]{self.current_status['persistence_level']}[/{persistence_color}]
🎨 창의성 레벨: [{creativity_bar}] {self.current_status['creativity_level']}/10
🚨 절망 모드: {"🔥 활성화" if self.current_status['is_desperate'] else "⭕ 비활성"}
📍 최근 활동: {self.current_status['last_activity']}
        """
        return Panel(content.strip(), style="bold green")
    
    def create_logs_panel(self) -> Panel:
        """로그 패널 생성"""
        if not self.recent_logs:
            content = "📋 실시간 로그를 기다리는 중..."
        else:
            content = "\n".join([
                f"[{log['time']}] {log['message']}" 
                for log in self.recent_logs[-8:]  # 최근 8개만 표시
            ])
        
        return Panel(content, title="📋 실시간 로그", style="dim")
    
    def create_input_panel(self) -> Panel:
        """입력 패널 생성"""
        content = "💬 명령어를 입력하세요 (help: 도움말, stop: 모니터링 중지)"
        return Panel(content, style="bold white")
    
    def get_display_content(self) -> Layout:
        """표시할 전체 콘텐츠 생성"""
        self.layout["header"].update(self.create_header_panel())
        self.layout["status"].update(self.create_status_panel())
        self.layout["logs"].update(self.create_logs_panel())
        self.layout["input"].update(self.create_input_panel())
        return self.layout
    
    def add_log(self, message: str):
        """수동으로 로그 추가"""
        self.recent_logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message
        })
        
        if len(self.recent_logs) > self.max_logs:
            self.recent_logs = self.recent_logs[-self.max_logs:]
    
    def update_current_task(self, task: str):
        """현재 작업 업데이트"""
        self.current_status["current_task"] = task
        self.current_status["last_activity"] = f"{datetime.now().strftime('%H:%M:%S')} - {task}"


# 전역 인스턴스
_monitor = None

def get_realtime_monitor() -> RealtimeMonitor:
    """실시간 모니터 싱글톤 인스턴스 반환"""
    global _monitor
    if _monitor is None:
        _monitor = RealtimeMonitor()
    return _monitor 