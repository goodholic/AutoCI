#!/usr/bin/env python3
"""
신경망 연속 학습 시스템 컨트롤러
24시간 백그라운드 학습 관리
"""

import os
import sys
import json
import psutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# 색상 코드
class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class NeuralLearningController:
    """신경망 학습 컨트롤러"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.pid_file = self.base_path / "neural_learning.pid"
        self.log_file = self.base_path / "neural_continuous_learning.log"
        self.status_file = self.base_path / "neural_learning_status.json"
        
    def print_banner(self):
        """배너 출력"""
        print(f"{Colors.PURPLE}")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║                                                               ║")
        print("║      🧠 24시간 C# 신경망 연속 학습 시스템 🧠                 ║")
        print("║                                                               ║")
        print("║         실제로 학습하는 ChatGPT 수준 AI                       ║")
        print("║                                                               ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        
    def start(self):
        """학습 시작"""
        self.print_banner()
        
        if self.is_running():
            print(f"{Colors.YELLOW}⚠️  신경망 학습이 이미 실행 중입니다.{Colors.ENDC}")
            self.status()
            return
            
        print(f"{Colors.GREEN}🚀 신경망 학습을 시작합니다...{Colors.ENDC}")
        
        # 백그라운드 프로세스 시작
        cmd = [
            sys.executable,
            str(self.base_path / "neural_continuous_learning.py")
        ]
        
        # 로그 파일 열기
        with open(self.log_file, 'a') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                start_new_session=True
            )
            
        # PID 저장
        with open(self.pid_file, 'w') as f:
            f.write(str(process.pid))
            
        # 초기 상태 저장
        self._save_status({
            'status': 'running',
            'pid': process.pid,
            'started_at': datetime.now().isoformat(),
            'mode': 'neural_network'
        })
        
        print(f"{Colors.GREEN}✅ 신경망 학습이 시작되었습니다! (PID: {process.pid}){Colors.ENDC}")
        print(f"{Colors.CYAN}📊 로그 파일: {self.log_file}{Colors.ENDC}")
        print(f"{Colors.CYAN}📈 상태 확인: autoci neural status{Colors.ENDC}")
        
        # 초기 정보 표시
        time.sleep(2)
        self._show_learning_info()
        
    def stop(self):
        """학습 중지"""
        if not self.is_running():
            print(f"{Colors.YELLOW}⚠️  실행 중인 학습 프로세스가 없습니다.{Colors.ENDC}")
            return
            
        print(f"{Colors.YELLOW}🛑 신경망 학습을 중지합니다...{Colors.ENDC}")
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # 프로세스 종료
            os.kill(pid, signal.SIGTERM)
            
            # PID 파일 삭제
            self.pid_file.unlink()
            
            # 상태 업데이트
            self._save_status({
                'status': 'stopped',
                'stopped_at': datetime.now().isoformat()
            })
            
            print(f"{Colors.GREEN}✅ 신경망 학습이 중지되었습니다.{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ 중지 중 오류 발생: {e}{Colors.ENDC}")
            
    def restart(self):
        """학습 재시작"""
        print(f"{Colors.YELLOW}🔄 신경망 학습을 재시작합니다...{Colors.ENDC}")
        self.stop()
        time.sleep(2)
        self.start()
        
    def status(self):
        """상태 확인"""
        if not self.is_running():
            print(f"{Colors.RED}❌ 신경망 학습이 실행되고 있지 않습니다.{Colors.ENDC}")
            return
            
        print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}🧠 신경망 학습 상태{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        # 프로세스 정보
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            process = psutil.Process(pid)
            
            print(f"📍 PID: {pid}")
            print(f"⏱️  실행 시간: {self._format_runtime(process.create_time())}")
            print(f"💾 메모리 사용: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"🔥 CPU 사용률: {process.cpu_percent(interval=1)}%")
            
        except Exception as e:
            print(f"{Colors.RED}프로세스 정보 읽기 실패: {e}{Colors.ENDC}")
            
        # 학습 상태
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    
                if 'learning_stats' in status:
                    stats = status['learning_stats']
                    print(f"\n{Colors.GREEN}📊 학습 통계:{Colors.ENDC}")
                    print(f"  • 총 학습 단계: {stats.get('total_steps', 0):,}")
                    print(f"  • 총 학습 샘플: {stats.get('total_samples', 0):,}")
                    print(f"  • 현재 Loss: {stats.get('current_loss', 0):.4f}")
                    print(f"  • 최고 Loss: {stats.get('best_loss', float('inf')):.4f}")
                    print(f"  • 모델 파라미터: {stats.get('model_parameters', 0):,}")
                    
        except Exception as e:
            print(f"{Colors.YELLOW}학습 통계를 읽을 수 없습니다: {e}{Colors.ENDC}")
            
        # 최근 로그
        print(f"\n{Colors.CYAN}📄 최근 로그:{Colors.ENDC}")
        self.logs(10)
        
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        
    def logs(self, lines: int = 50):
        """로그 확인"""
        if not self.log_file.exists():
            print(f"{Colors.YELLOW}로그 파일이 없습니다.{Colors.ENDC}")
            return
            
        try:
            # tail 명령어 사용
            result = subprocess.run(
                ['tail', '-n', str(lines), str(self.log_file)],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                for line in result.stdout.splitlines():
                    # 로그 레벨에 따라 색상 적용
                    if '[ERROR]' in line:
                        print(f"{Colors.RED}{line}{Colors.ENDC}")
                    elif '[WARNING]' in line:
                        print(f"{Colors.YELLOW}{line}{Colors.ENDC}")
                    elif '[INFO]' in line and '✅' in line:
                        print(f"{Colors.GREEN}{line}{Colors.ENDC}")
                    else:
                        print(line)
                        
        except Exception as e:
            print(f"{Colors.RED}로그 읽기 실패: {e}{Colors.ENDC}")
            
    def monitor(self):
        """실시간 모니터링"""
        print(f"{Colors.CYAN}📊 실시간 신경망 학습 모니터링 (Ctrl+C로 종료){Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        try:
            # tail -f 명령어로 실시간 모니터링
            subprocess.run(['tail', '-f', str(self.log_file)])
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}모니터링을 종료합니다.{Colors.ENDC}")
            
    def is_running(self) -> bool:
        """실행 중인지 확인"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # 프로세스 존재 확인
            return psutil.pid_exists(pid)
            
        except:
            return False
            
    def _save_status(self, data: Dict[str, Any]):
        """상태 저장"""
        try:
            existing = {}
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    existing = json.load(f)
                    
            existing.update(data)
            
            with open(self.status_file, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            print(f"{Colors.RED}상태 저장 실패: {e}{Colors.ENDC}")
            
    def _format_runtime(self, start_time: float) -> str:
        """실행 시간 포맷"""
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        return f"{hours}시간 {minutes}분 {seconds}초"
        
    def _show_learning_info(self):
        """학습 정보 표시"""
        print(f"\n{Colors.CYAN}🎓 신경망 학습 정보:{Colors.ENDC}")
        print(f"• 학습 소스: GitHub, StackOverflow, Unity Docs, MS Docs")
        print(f"• 학습 주기: 10분마다 배치 학습")
        print(f"• 데이터 수집: 30분마다")
        print(f"• 모델 평가: 2시간마다")
        print(f"• 일일 백업: 매일 자정")
        print(f"• 최적화: 매일 새벽 3시")
        
        print(f"\n{Colors.GREEN}💡 유용한 명령어:{Colors.ENDC}")
        print(f"• 상태 확인: autoci neural status")
        print(f"• 로그 보기: autoci neural logs")
        print(f"• 실시간 모니터링: autoci neural monitor")
        print(f"• 학습 중지: autoci neural stop")
        

def main():
    """메인 함수"""
    controller = NeuralLearningController()
    
    # 명령어 파싱
    if len(sys.argv) < 2:
        controller.print_banner()
        print(f"{Colors.YELLOW}사용법: {sys.argv[0]} [start|stop|restart|status|logs|monitor]{Colors.ENDC}")
        return
        
    command = sys.argv[1].lower()
    
    if command == 'start':
        controller.start()
    elif command == 'stop':
        controller.stop()
    elif command == 'restart':
        controller.restart()
    elif command == 'status':
        controller.status()
    elif command == 'logs':
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        controller.logs(lines)
    elif command == 'monitor':
        controller.monitor()
    else:
        print(f"{Colors.RED}❌ 알 수 없는 명령어: {command}{Colors.ENDC}")
        print(f"{Colors.YELLOW}사용 가능한 명령어: start, stop, restart, status, logs, monitor{Colors.ENDC}")


if __name__ == "__main__":
    main()