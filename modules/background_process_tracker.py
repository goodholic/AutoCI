#!/usr/bin/env python3
"""
AutoCI 백그라운드 프로세스 추적 시스템
24시간 개선 프로세스의 상태를 실시간으로 추적하고 로그 파일에 저장
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import psutil

class BackgroundProcessTracker:
    """백그라운드 프로세스 추적기"""
    
    def __init__(self, project_name: str, log_dir: str = "logs/24h_improvement"):
        self.project_name = project_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일들
        self.status_file = self.log_dir / f"{project_name}_status.json"
        self.progress_file = self.log_dir / f"{project_name}_progress.json"
        self.log_file = self.log_dir / "latest_improvement.log"
        
        # 추적 상태
        self.start_time = datetime.now()
        self.status = {
            "project_name": project_name,
            "start_time": self.start_time.isoformat(),
            "elapsed_time": "00:00:00",
            "remaining_time": "24:00:00",
            "progress_percent": 0,
            "iteration_count": 0,
            "fixes_count": 0,
            "improvements_count": 0,
            "quality_score": 0,
            "last_update": datetime.now().isoformat()
        }
        
        self.progress = {
            "current_task": "시스템 초기화",
            "last_activity": "프로세스 시작",
            "persistence_level": "NORMAL",
            "creativity_level": 5,
            "is_desperate": False,
            "current_module": None,
            "current_phase": "startup"
        }
        
        # 백그라운드 업데이트 스레드
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # 초기 상태 저장
        self.save_status()
        self.log("🚀 24시간 개선 프로세스를 시작했습니다")
    
    def _update_loop(self):
        """백그라운드 업데이트 루프"""
        while self.running:
            try:
                # 경과 시간 업데이트
                elapsed = datetime.now() - self.start_time
                self.status["elapsed_time"] = str(elapsed).split('.')[0]
                
                # 남은 시간 계산
                total_seconds = elapsed.total_seconds()
                remaining_seconds = max(0, 24 * 3600 - total_seconds)
                hours = int(remaining_seconds // 3600)
                minutes = int((remaining_seconds % 3600) // 60)
                seconds = int(remaining_seconds % 60)
                self.status["remaining_time"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # 진행률 계산
                self.status["progress_percent"] = min(100, total_seconds / (24 * 3600) * 100)
                
                # 마지막 업데이트 시간
                self.status["last_update"] = datetime.now().isoformat()
                
                # 상태 저장
                self.save_status()
                
                # 1초 대기
                time.sleep(1)
                
            except Exception as e:
                self.log(f"❌ 업데이트 루프 오류: {e}")
                time.sleep(5)
    
    def save_status(self):
        """상태를 파일에 저장"""
        try:
            # status 파일 저장
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, ensure_ascii=False, indent=2)
            
            # progress 파일 저장
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ 상태 저장 오류: {e}")
    
    def log(self, message: str):
        """로그 메시지 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"❌ 로그 기록 오류: {e}")
    
    def update_task(self, task: str, phase: Optional[str] = None):
        """현재 작업 업데이트"""
        self.progress["current_task"] = task
        self.progress["last_activity"] = f"{datetime.now().strftime('%H:%M:%S')} - {task}"
        if phase:
            self.progress["current_phase"] = phase
        
        self.log(f"🔧 작업: {task}")
        self.save_status()
    
    def increment_iteration(self):
        """반복 횟수 증가"""
        self.status["iteration_count"] += 1
        self.log(f"🔄 반복 #{self.status['iteration_count']}")
        self.save_status()
    
    def increment_fixes(self):
        """수정 횟수 증가"""
        self.status["fixes_count"] += 1
        self.log(f"🔨 수정 #{self.status['fixes_count']}")
        self.save_status()
    
    def increment_improvements(self):
        """개선 횟수 증가"""
        self.status["improvements_count"] += 1
        self.log(f"✨ 개선 #{self.status['improvements_count']}")
        self.save_status()
    
    def update_quality_score(self, score: int):
        """품질 점수 업데이트"""
        self.status["quality_score"] = max(0, min(100, score))
        self.log(f"📊 품질 점수: {self.status['quality_score']}/100")
        self.save_status()
    
    def update_persistence_level(self, level: str):
        """끈질김 레벨 업데이트"""
        self.progress["persistence_level"] = level
        self.log(f"💪 끈질김 레벨: {level}")
        self.save_status()
    
    def update_creativity_level(self, level: int):
        """창의성 레벨 업데이트"""
        self.progress["creativity_level"] = max(0, min(10, level))
        self.log(f"🎨 창의성 레벨: {self.progress['creativity_level']}/10")
        self.save_status()
    
    def set_desperate_mode(self, is_desperate: bool):
        """절망 모드 설정"""
        self.progress["is_desperate"] = is_desperate
        if is_desperate:
            self.log("🔥 절망 모드 활성화!")
        else:
            self.log("😌 절망 모드 해제")
        self.save_status()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계 가져오기"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": memory_info.rss / 1024 / 1024,
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        except:
            return {}
    
    def stop(self):
        """추적 중지"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        self.log("🛑 24시간 개선 프로세스를 중지했습니다")
        
        # 최종 상태 저장
        self.save_status()


# 전역 추적기 인스턴스
_tracker: Optional[BackgroundProcessTracker] = None

def get_process_tracker(project_name: Optional[str] = None) -> Optional[BackgroundProcessTracker]:
    """프로세스 추적기 싱글톤 인스턴스 반환"""
    global _tracker
    
    if project_name and (not _tracker or _tracker.project_name != project_name):
        # 기존 추적기 중지
        if _tracker:
            _tracker.stop()
        
        # 새 추적기 생성
        _tracker = BackgroundProcessTracker(project_name)
    
    return _tracker

def stop_process_tracker():
    """프로세스 추적기 중지"""
    global _tracker
    if _tracker:
        _tracker.stop()
        _tracker = None