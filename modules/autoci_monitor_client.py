#!/usr/bin/env python3
"""
AutoCI 백그라운드 프로세스 모니터링 클라이언트
WSL 환경에서 실시간으로 24시간 개선 과정을 볼 수 있는 모니터
"""

import os
import sys
import json
import time
import signal
import curses
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

class AutoCIMonitorClient:
    """AutoCI 프로세스 모니터링 클라이언트"""
    
    def __init__(self, log_dir: str = "logs/24h_improvement", mode: str = None):
        self.log_dir = Path(log_dir)
        self.running = True
        self.current_project = None
        self.mode = mode  # simple, curses, 또는 None
        
        # 모니터링할 로그 파일들
        self.status_file = None
        self.progress_file = None
        self.log_file = None
        
        # 화면 표시용 데이터
        self.status_data = {}
        self.progress_data = {}
        self.recent_logs = []
        self.max_logs = 20
        
    def find_latest_project(self) -> Optional[str]:
        """최신 프로젝트 찾기"""
        if not self.log_dir.exists():
            return None
            
        # status 파일들 찾기
        status_files = list(self.log_dir.glob("*_status.json"))
        if not status_files:
            return None
            
        # 가장 최근 수정된 파일 찾기
        latest = max(status_files, key=lambda f: f.stat().st_mtime)
        project_name = latest.stem.replace("_status", "")
        
        return project_name
    
    def setup_files(self, project_name: str):
        """모니터링할 파일 설정"""
        self.current_project = project_name
        self.status_file = self.log_dir / f"{project_name}_status.json"
        self.progress_file = self.log_dir / f"{project_name}_progress.json"
        self.log_file = self.log_dir / "latest_improvement.log"
    
    def read_status(self):
        """상태 파일 읽기"""
        if self.status_file and self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    self.status_data = json.load(f)
            except:
                pass
                
        if self.progress_file and self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress_data.update(json.load(f))
            except:
                pass
    
    def read_logs(self):
        """로그 파일 읽기"""
        if self.log_file and self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    # 파일 끝에서부터 읽기
                    f.seek(0, 2)  # 파일 끝으로
                    file_size = f.tell()
                    
                    # 마지막 5KB만 읽기
                    read_size = min(file_size, 5120)
                    f.seek(max(0, file_size - read_size))
                    
                    lines = f.read().splitlines()
                    # 마지막 줄부터 저장
                    self.recent_logs = lines[-self.max_logs:] if lines else []
            except:
                pass
    
    def display_with_curses(self, stdscr):
        """curses를 사용한 화면 표시"""
        # 초기 설정
        curses.curs_set(0)  # 커서 숨기기
        stdscr.nodelay(1)   # 논블로킹 입력
        stdscr.timeout(1000)  # 1초 타임아웃
        
        # 색상 설정
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        
        while self.running:
            try:
                # 데이터 읽기
                self.read_status()
                self.read_logs()
                
                # 화면 지우기
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                
                # 헤더
                header = "╔══ AutoCI 24시간 개선 모니터 ══╗"
                stdscr.addstr(0, (width - len(header)) // 2, header, curses.color_pair(1) | curses.A_BOLD)
                
                # 프로젝트 정보
                y = 2
                if self.current_project:
                    stdscr.addstr(y, 2, f"📁 프로젝트: {self.current_project}", curses.color_pair(2))
                    y += 1
                
                # 상태 정보
                if self.status_data:
                    y += 1
                    stdscr.addstr(y, 2, "📊 상태 정보:", curses.color_pair(3) | curses.A_BOLD)
                    y += 1
                    
                    info_items = [
                        ("경과 시간", self.status_data.get("elapsed_time", "00:00:00")),
                        ("남은 시간", self.status_data.get("remaining_time", "24:00:00")),
                        ("진행률", f"{self.status_data.get('progress_percent', 0):.1f}%"),
                        ("반복 횟수", self.status_data.get("iteration_count", 0)),
                        ("수정 횟수", self.status_data.get("fixes_count", 0)),
                        ("개선 횟수", self.status_data.get("improvements_count", 0)),
                        ("품질 점수", f"{self.status_data.get('quality_score', 0)}/100"),
                    ]
                    
                    for label, value in info_items:
                        if y < height - 10:
                            stdscr.addstr(y, 4, f"• {label}: {value}")
                            y += 1
                
                # 진행 상황
                if self.progress_data:
                    y += 1
                    if y < height - 8:
                        stdscr.addstr(y, 2, "🔧 진행 상황:", curses.color_pair(3) | curses.A_BOLD)
                        y += 1
                        
                        current_task = self.progress_data.get("current_task", "대기 중")
                        persistence = self.progress_data.get("persistence_level", "NORMAL")
                        creativity = self.progress_data.get("creativity_level", 0)
                        desperate = self.progress_data.get("is_desperate", False)
                        
                        if y < height - 6:
                            stdscr.addstr(y, 4, f"• 현재 작업: {current_task[:width-10]}")
                            y += 1
                        if y < height - 5:
                            color = curses.color_pair(2)
                            if persistence in ["STUBBORN", "OBSESSIVE"]:
                                color = curses.color_pair(3)
                            elif persistence == "INFINITE":
                                color = curses.color_pair(4)
                            stdscr.addstr(y, 4, f"• 끈질김: {persistence}", color)
                            y += 1
                        if y < height - 4:
                            creativity_bar = "█" * creativity + "░" * (10 - creativity)
                            stdscr.addstr(y, 4, f"• 창의성: [{creativity_bar}] {creativity}/10")
                            y += 1
                        if y < height - 3 and desperate:
                            stdscr.addstr(y, 4, "• 🔥 절망 모드 활성화!", curses.color_pair(4) | curses.A_BLINK)
                            y += 1
                
                # 최근 로그
                y = max(y + 1, height - 15)
                if y < height - 2:
                    stdscr.addstr(y, 2, "📋 실시간 로그:", curses.color_pair(3) | curses.A_BOLD)
                    y += 1
                    
                    log_start = max(0, len(self.recent_logs) - (height - y - 2))
                    for i, log in enumerate(self.recent_logs[log_start:]):
                        if y < height - 2:
                            # 로그 줄 자르기
                            if len(log) > width - 4:
                                log = log[:width-7] + "..."
                            stdscr.addstr(y, 4, log)
                            y += 1
                
                # 하단 정보
                footer = "Press 'q' to quit, 'r' to refresh"
                stdscr.addstr(height - 1, 2, footer, curses.A_DIM)
                
                # 화면 새로고침
                stdscr.refresh()
                
                # 키 입력 처리
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                elif key == ord('r') or key == ord('R'):
                    continue
                    
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                # 오류 발생 시 하단에 표시
                try:
                    stdscr.addstr(height - 2, 2, f"Error: {str(e)}", curses.color_pair(4))
                    stdscr.refresh()
                except:
                    pass
    
    def run_simple_display(self):
        """curses 없이 간단한 표시"""
        print("🖥️  AutoCI 24시간 개선 모니터 (Simple Mode)")
        print("=" * 60)
        print("Press Ctrl+C to quit\n")
        
        while self.running:
            try:
                # 데이터 읽기
                self.read_status()
                self.read_logs()
                
                # 화면 지우기
                os.system('clear' if os.name != 'nt' else 'cls')
                
                # 헤더
                print("╔══ AutoCI 24시간 개선 모니터 ══╗")
                print(f"📁 프로젝트: {self.current_project or '없음'}")
                print(f"🕐 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                # 상태 정보
                if self.status_data:
                    print("\n📊 상태 정보:")
                    print(f"  • 경과 시간: {self.status_data.get('elapsed_time', '00:00:00')}")
                    print(f"  • 남은 시간: {self.status_data.get('remaining_time', '24:00:00')}")
                    print(f"  • 진행률: {self.status_data.get('progress_percent', 0):.1f}%")
                    print(f"  • 반복: {self.status_data.get('iteration_count', 0)} | "
                          f"수정: {self.status_data.get('fixes_count', 0)} | "
                          f"개선: {self.status_data.get('improvements_count', 0)}")
                    print(f"  • 품질 점수: {self.status_data.get('quality_score', 0)}/100")
                
                # 진행 상황
                if self.progress_data:
                    print("\n🔧 진행 상황:")
                    print(f"  • 현재 작업: {self.progress_data.get('current_task', '대기 중')}")
                    print(f"  • 끈질김 레벨: {self.progress_data.get('persistence_level', 'NORMAL')}")
                    print(f"  • 창의성 레벨: {self.progress_data.get('creativity_level', 0)}/10")
                    if self.progress_data.get('is_desperate', False):
                        print("  • 🔥 절망 모드 활성화!")
                
                # 최근 로그
                print("\n📋 실시간 로그:")
                if self.recent_logs:
                    for log in self.recent_logs[-10:]:  # 마지막 10줄만
                        print(f"  {log[:80]}{'...' if len(log) > 80 else ''}")
                else:
                    print("  로그를 기다리는 중...")
                
                print("\n" + "=" * 60)
                print("Press Ctrl+C to quit")
                
                # 1초 대기
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n\n👋 모니터링을 종료합니다.")
                self.running = False
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                time.sleep(2)
    
    def run(self, simple_mode: bool = False):
        """모니터 실행"""
        # 최신 프로젝트 찾기
        project = self.find_latest_project()
        if not project:
            print("❌ 진행 중인 프로젝트를 찾을 수 없습니다.")
            print(f"   로그 디렉토리를 확인하세요: {self.log_dir}")
            return
        
        self.setup_files(project)
        print(f"✅ 프로젝트 '{project}' 모니터링을 시작합니다...")
        time.sleep(1)
        
        # mode가 지정되어 있으면 우선 사용
        if self.mode == "simple" or simple_mode or os.name == 'nt':
            # Windows나 simple 모드는 curses 없이
            self.run_simple_display()
        else:
            # Linux/WSL은 curses 사용
            try:
                curses.wrapper(self.display_with_curses)
            except:
                # curses 실패 시 simple 모드로
                print("⚠️  Curses 모드 실패, Simple 모드로 전환합니다...")
                self.run_simple_display()
    
    async def run_async(self):
        """비동기 실행 래퍼"""
        # 동기 함수를 비동기로 래핑
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.run, self.mode == "simple")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AutoCI 백그라운드 프로세스 모니터")
    parser.add_argument(
        "--log-dir",
        default="logs/24h_improvement",
        help="로그 디렉토리 경로"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple 모드 사용 (curses 없이)"
    )
    parser.add_argument(
        "--project",
        help="특정 프로젝트 모니터링"
    )
    
    args = parser.parse_args()
    
    # 모니터 생성 및 실행
    monitor = AutoCIMonitorClient(log_dir=args.log_dir)
    
    if args.project:
        monitor.setup_files(args.project)
    
    try:
        monitor.run(simple_mode=args.simple)
    except Exception as e:
        print(f"❌ 모니터 실행 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()