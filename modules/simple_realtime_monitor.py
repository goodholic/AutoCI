#!/usr/bin/env python3
"""
간단한 24시간 게임 개선 실시간 모니터링 시스템 (Rich 라이브러리 없이)
"""

import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

class SimpleRealtimeMonitor:
    """24시간 게임 개선 간단한 실시간 모니터링"""
    
    def __init__(self):
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
        self.max_logs = 5
        
        # 마지막 표시 시간
        self.last_display_time = 0
        self.display_interval = 3  # 3초마다 업데이트
    
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
                    new_lines = lines[-3:]  # 마지막 3줄
                    
                    for line in new_lines:
                        line = line.strip()
                        if line and line not in [log["message"] for log in self.recent_logs]:
                            self.recent_logs.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "message": line[:60] + "..." if len(line) > 60 else line
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
    
    def show_status_header(self):
        """상태 헤더를 터미널 상단에 표시 (주기적으로)"""
        current_time = time.time()
        
        # 3초마다만 업데이트
        if current_time - self.last_display_time < self.display_interval:
            return
        
        self.last_display_time = current_time
        
        if not self.monitoring:
            return
        
        # 터미널 상단에 상태 표시
        print("\033[H\033[2J", end="")  # 화면 지우기
        
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "🏭 24시간 게임 개선 공장 - 실시간 모니터링" + " " * 18 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        
        # 기본 정보
        print(f"🎮 프로젝트: {self.current_status['project_name']}")
        print(f"⏰ 경과: {self.current_status['elapsed_time']} | 남은 시간: {self.current_status['remaining_time']} | 진행률: {self.current_status['progress_percent']:.1f}%")
        print(f"🔄 반복: {self.current_status['iteration_count']} | 수정: {self.current_status['fixes_count']} | 개선: {self.current_status['improvements_count']}")
        print(f"📊 게임 품질 점수: {self.current_status['quality_score']}/100")
        print()
        
        # 현재 작업 상태
        print(f"🔧 현재 작업: {self.current_status['current_task']}")
        print(f"💪 끈질김 레벨: {self.current_status['persistence_level']}")
        
        # 창의성 레벨 바
        creativity_level = self.current_status['creativity_level']
        creativity_bar = "█" * creativity_level + "░" * (10 - creativity_level)
        print(f"🎨 창의성 레벨: [{creativity_bar}] {creativity_level}/10")
        
        # 절망 모드
        desperate_status = "🔥 활성화" if self.current_status['is_desperate'] else "⭕ 비활성"
        print(f"🚨 절망 모드: {desperate_status}")
        print()
        
        # 최근 로그
        print("📋 최근 활동:")
        if self.recent_logs:
            for log in self.recent_logs[-3:]:  # 최근 3개만
                print(f"  [{log['time']}] {log['message']}")
        else:
            print("  실시간 로그를 기다리는 중...")
        
        print()
        print("─" * 80)
        print("💬 명령어를 입력하세요 (help: 도움말, stop: 모니터링 중지)")
        print("─" * 80)
    
    def show_simple_status(self):
        """간단한 상태 표시 (명령어 입력 전)"""
        if not self.monitoring:
            return
            
        print()
        print("┌─ 24시간 개선 상태 " + "─" * 58 + "┐")
        print(f"│ 🎮 {self.current_status['project_name']:<25} │ ⏰ {self.current_status['elapsed_time']:<12} │ 📊 {self.current_status['progress_percent']:.1f}% │")
        print(f"│ 🔧 {self.current_status['current_task'][:40]:<40} │")
        print("└" + "─" * 78 + "┘")
    
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
_simple_monitor = None

def get_simple_realtime_monitor() -> SimpleRealtimeMonitor:
    """간단한 실시간 모니터 싱글톤 인스턴스 반환"""
    global _simple_monitor
    if _simple_monitor is None:
        _simple_monitor = SimpleRealtimeMonitor()
    return _simple_monitor 