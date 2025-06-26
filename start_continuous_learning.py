#!/usr/bin/env python3
"""
AutoCI 백그라운드 연속 학습 시작 스크립트
24시간 C# 지식 크롤링 및 학습
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
import json
from datetime import datetime

class AutoCIBackgroundLearning:
    """AutoCI 백그라운드 학습 관리자"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.pid_file = self.script_dir / "autoci_learning.pid"
        self.log_file = self.script_dir / "logs" / "continuous_learning.log"
        self.status_file = self.script_dir / "learning_status.json"
        
        # 로그 디렉토리 생성
        self.log_file.parent.mkdir(exist_ok=True)
        
    def start_learning(self):
        """연속 학습 시작"""
        if self.is_running():
            print("❌ AutoCI 연속 학습이 이미 실행 중입니다.")
            return False
        
        print("🚀 AutoCI 24시간 연속 학습 시작...")
        
        try:
            # Python 실행 파일 경로 찾기
            python_cmd = self._find_python()
            
            # 연속 학습 스크립트 실행
            learning_script = self.script_dir / "autoci_continuous_learning.py"
            
            # 백그라운드에서 실행
            process = subprocess.Popen(
                [python_cmd, str(learning_script)],
                stdout=open(self.log_file, 'w', encoding='utf-8'),
                stderr=subprocess.STDOUT,
                cwd=str(self.script_dir)
            )
            
            # PID 저장
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # 상태 저장
            self._save_status("running", process.pid)
            
            print(f"✅ 연속 학습 시작됨 (PID: {process.pid})")
            print(f"📄 로그 파일: {self.log_file}")
            print(f"💡 중지하려면: python start_continuous_learning.py stop")
            
            return True
            
        except Exception as e:
            print(f"❌ 연속 학습 시작 실패: {e}")
            return False
    
    def stop_learning(self):
        """연속 학습 중지"""
        if not self.is_running():
            print("❌ 실행 중인 AutoCI 연속 학습이 없습니다.")
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            print(f"🛑 AutoCI 연속 학습 중지 중... (PID: {pid})")
            
            # Windows와 Unix 모두 지원
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                             capture_output=True)
            else:  # Unix/Linux
                os.kill(pid, signal.SIGTERM)
            
            # PID 파일 삭제
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            # 상태 업데이트
            self._save_status("stopped", None)
            
            print("✅ AutoCI 연속 학습이 중지되었습니다.")
            return True
            
        except Exception as e:
            print(f"❌ 연속 학습 중지 실패: {e}")
            # PID 파일이 남아있으면 삭제
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
    
    def restart_learning(self):
        """연속 학습 재시작"""
        print("🔄 AutoCI 연속 학습 재시작...")
        self.stop_learning()
        time.sleep(2)
        return self.start_learning()
    
    def is_running(self):
        """실행 상태 확인"""
        if not self.pid_file.exists():
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 프로세스 실행 확인
            if os.name == 'nt':  # Windows
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                      capture_output=True, text=True)
                return str(pid) in result.stdout
            else:  # Unix/Linux
                try:
                    os.kill(pid, 0)  # 신호 0은 프로세스 존재 확인
                    return True
                except OSError:
                    return False
                    
        except Exception:
            return False
    
    def get_status(self):
        """학습 상태 확인"""
        print("📊 AutoCI 연속 학습 상태")
        print("=" * 40)
        
        # 실행 상태
        is_running = self.is_running()
        print(f"🔄 실행 상태: {'🟢 실행중' if is_running else '🔴 중지됨'}")
        
        # PID 정보
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    pid = f.read().strip()
                print(f"🆔 프로세스 ID: {pid}")
            except:
                pass
        
        # 로그 파일 정보
        if self.log_file.exists():
            log_size = self.log_file.stat().st_size
            print(f"📄 로그 파일: {self.log_file} ({log_size} bytes)")
            
            # 최근 로그 출력
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("\n📝 최근 로그 (마지막 5줄):")
                        for line in lines[-5:]:
                            print(f"   {line.strip()}")
            except:
                pass
        
        # 상태 파일 정보
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                print(f"\n📈 마지막 업데이트: {status_data.get('last_update', 'Unknown')}")
                print(f"🧠 학습 세션: {status_data.get('sessions', 0)}")
                print(f"📚 수집된 문서: {status_data.get('documents', 0)}")
            except:
                pass
        
        # 학습 데이터 디렉토리 확인
        learning_data_dir = self.script_dir / "learning_data"
        if learning_data_dir.exists():
            db_file = learning_data_dir / "csharp_knowledge.db"
            if db_file.exists():
                db_size = db_file.stat().st_size
                print(f"🗄️ 지식 데이터베이스: {db_size} bytes")
    
    def view_logs(self, lines=50):
        """로그 보기"""
        if not self.log_file.exists():
            print("❌ 로그 파일이 없습니다.")
            return
        
        print(f"📄 AutoCI 연속 학습 로그 (최근 {lines}줄)")
        print("=" * 50)
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
                
            # 최근 N줄 출력
            for line in log_lines[-lines:]:
                print(line.rstrip())
                
        except Exception as e:
            print(f"❌ 로그 읽기 실패: {e}")
    
    def _find_python(self):
        """Python 실행 파일 찾기"""
        # 현재 Python 실행 파일 사용
        return sys.executable
    
    def _save_status(self, status: str, pid: int = None):
        """상태 저장"""
        try:
            status_data = {
                "status": status,
                "pid": pid,
                "last_update": datetime.now().isoformat(),
                "log_file": str(self.log_file),
                "sessions": 0,
                "documents": 0
            }
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ 상태 저장 실패: {e}")

def print_usage():
    """사용법 출력"""
    print("📚 AutoCI 24시간 연속 학습 시스템")
    print("=" * 40)
    print("사용법:")
    print("  python start_continuous_learning.py start    # 학습 시작")
    print("  python start_continuous_learning.py stop     # 학습 중지")
    print("  python start_continuous_learning.py restart  # 학습 재시작")
    print("  python start_continuous_learning.py status   # 상태 확인")
    print("  python start_continuous_learning.py logs     # 로그 보기")
    print("  python start_continuous_learning.py help     # 도움말")

def main():
    """메인 함수"""
    manager = AutoCIBackgroundLearning()
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        manager.start_learning()
    elif command == "stop":
        manager.stop_learning()
    elif command == "restart":
        manager.restart_learning()
    elif command == "status":
        manager.get_status()
    elif command == "logs":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        manager.view_logs(lines)
    elif command == "help":
        print_usage()
    else:
        print(f"❌ 알 수 없는 명령어: {command}")
        print_usage()

if __name__ == "__main__":
    main() 