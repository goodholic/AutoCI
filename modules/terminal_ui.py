import curses
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any, List

class TerminalUI:
    def __init__(self):
        self.stdscr = None
        self.log_window = None
        self.status_window = None
        self.input_window = None
        self.header_window = None
        self.cot_window = None # Chain of Thought (사고의 사슬) 윈도우

        self.log_messages = []
        self.status_data: Dict[str, Any] = {
            "project_name": "없음",
            "ai_status": "대기중",
            "current_task": "초기화 중",
            "progress_percent": 0,
            "quality_score": 0,
            "elapsed_time": "00:00:00"
        }
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = True
        self.input_thread = None
        self.cot_messages = []

    def _input_loop(self):
        while self.running:
            try:
                user_input = self.input_window.getstr().decode('utf-8')
                self.input_queue.put(user_input)
                self._clear_input_line()
            except Exception:
                # 윈도우가 닫히거나 다른 오류 발생 시 스레드 종료
                break

    def _clear_input_line(self):
        h, w = self.input_window.getmaxyx()
        self.input_window.move(0, 0)
        self.input_window.clrtoeol()
        self.input_window.addstr(0, 0, "AutoCI > ")
        self.input_window.refresh()

    def _draw_header(self):
        self.header_window.erase()
        h, w = self.header_window.getmaxyx()
        title = "🤖 AutoCI - AI 게임 개발 시스템 🎮"
        subtitle = "AI가 실시간으로 Godot을 제어하여 게임을 만듭니다"
        self.header_window.addstr(0, (w - len(title)) // 2, title, curses.A_BOLD)
        self.header_window.addstr(1, (w - len(subtitle)) // 2, subtitle)
        self.header_window.box()
        self.header_window.refresh()

    def _draw_status(self):
        self.status_window.erase()
        self.status_window.box()
        h, w = self.status_window.getmaxyx()

        self.status_window.addstr(1, 2, f"프로젝트: {self.status_data['project_name']}")
        self.status_window.addstr(2, 2, f"AI 상태: {self.status_data['ai_status']}")
        self.status_window.addstr(3, 2, f"현재 작업: {self.status_data['current_task']}")

        # 진행률 바
        progress_bar_len = w - 20
        progress_filled = int(progress_bar_len * self.status_data['progress_percent'] / 100)
        progress_bar = "█" * progress_filled + "░" * (progress_bar_len - progress_filled)
        self.status_window.addstr(4, 2, f"진행률: [{progress_bar}] {self.status_data['progress_percent']:.1f}%")

        self.status_window.addstr(5, 2, f"품질 점수: {self.status_data['quality_score']}/100")
        self.status_window.addstr(6, 2, f"경과 시간: {self.status_data['elapsed_time']}")
        self.status_window.refresh()

    def _draw_log(self):
        self.log_window.erase()
        self.log_window.box()
        h, w = self.log_window.getmaxyx()
        self.log_window.addstr(0, 2, "로그")

        display_logs = self.log_messages[-h + 2:] # 화면 크기에 맞춰 최신 로그 표시
        for i, msg in enumerate(display_logs):
            self.log_window.addstr(i + 1, 1, msg[:w-2]) # 윈도우 너비에 맞춰 자르기
        self.log_window.refresh()

    def _draw_cot(self):
        self.cot_window.erase()
        self.cot_window.box()
        h, w = self.cot_window.getmaxyx()
        self.cot_window.addstr(0, 2, "AI 사고 과정 (Chain of Thought)")

        display_cots = self.cot_messages[-h + 2:]
        for i, msg in enumerate(display_cots):
            self.cot_window.addstr(i + 1, 1, msg[:w-2])
        self.cot_window.refresh()

    def _update_all_windows(self):
        self._draw_header()
        self._draw_status()
        self._draw_log()
        self._draw_cot()
        self._clear_input_line()

    def _main_loop(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(1) # 커서 보이게
        self.stdscr.nodelay(True) # Non-blocking input

        # 윈도우 분할
        sh, sw = self.stdscr.getmaxyx()
        
        # 헤더 (3줄)
        self.header_window = curses.newwin(3, sw, 0, 0)
        
        # 상태 (8줄)
        self.status_window = curses.newwin(8, sw, 3, 0)
        
        # 사고 과정 (Chain of Thought) 윈도우 (로그 윈도우의 절반)
        cot_height = (sh - 3 - 8 - 2) // 2 # 전체 높이 - 헤더 - 상태 - 입력 - 로그 제목
        self.cot_window = curses.newwin(cot_height, sw, 3 + 8, 0)

        # 로그 윈도우 (나머지 공간)
        log_height = sh - 3 - 8 - cot_height - 2 # 전체 높이 - 헤더 - 상태 - cot - 입력
        self.log_window = curses.newwin(log_height, sw, 3 + 8 + cot_height, 0)
        
        # 입력 (2줄)
        self.input_window = curses.newwin(2, sw, sh - 2, 0)

        self.input_thread = threading.Thread(target=self._input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

        self._update_all_windows()

        while self.running:
            self._draw_log()
            self._draw_status()
            self._draw_cot()
            self.stdscr.refresh()
            time.sleep(0.1) # 100ms마다 업데이트

    def start(self):
        curses.wrapper(self._main_loop)

    def stop(self):
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            # 스레드 종료를 위해 입력 윈도우를 닫음
            self.input_window.nodelay(False) # Blocking mode for getstr to allow interrupt
            self.input_window.keypad(True)
            curses.endwin()
            self.input_thread.join(timeout=1)
        
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        self._draw_log()

    def update_status(self, data: Dict[str, Any]):
        self.status_data.update(data)
        self._draw_status()

    def get_user_input(self) -> Optional[str]:
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def add_cot_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.cot_messages.append(f"[{timestamp}] {message}")
        self._draw_cot()

# Singleton instance
_ui_instance = None

def get_terminal_ui() -> TerminalUI:
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = TerminalUI()
    return _ui_instance

if __name__ == "__main__":
    ui = get_terminal_ui()
    ui.start()
    # 테스트를 위해 잠시 대기
    ui.add_log("시스템 초기화 중...")
    ui.update_status({"project_name": "TestGame", "ai_status": "작업 중", "current_task": "게임 생성", "progress_percent": 25, "quality_score": 50, "elapsed_time": "00:01:30"})
    ui.add_cot_message("문제 분석: 플레이어 점프가 너무 낮음.")
    ui.add_cot_message("해결 계획: Player.gd의 JUMP_VELOCITY를 증가.")
    time.sleep(5)
    ui.add_log("게임 생성 완료.")
    ui.update_status({"progress_percent": 100, "current_task": "완료", "quality_score": 80})
    ui.add_cot_message("최종 해결책: JUMP_VELOCITY = -500.0으로 변경.")
    time.sleep(5)
    ui.stop()
