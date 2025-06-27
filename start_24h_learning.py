#!/usr/bin/env python3
"""
24시간 지속 학습 안정적 시작 스크립트
경로 문제나 가상환경 문제 없이 확실하게 실행
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def start_24h_learning():
    """24시간 지속 학습 시작"""
    print("🧠 24시간 지속 신경망 학습 시작...")
    
    # 현재 디렉토리 확인
    current_dir = Path(__file__).parent
    learning_script = current_dir / "continuous_neural_learning.py"
    
    if not learning_script.exists():
        print(f"❌ 학습 스크립트를 찾을 수 없습니다: {learning_script}")
        return False
    
    try:
        # neural_venv의 python 사용
        neural_python = current_dir / "neural_venv" / "bin" / "python3"
        if neural_python.exists():
            python_cmd = str(neural_python)
        else:
            python_cmd = sys.executable
            
        print(f"🐍 Python 경로: {python_cmd}")
        
        # 백그라운드에서 직접 실행
        process = subprocess.Popen(
            [python_cmd, str(learning_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(current_dir)
        )
        
        print(f"✅ 백그라운드에서 실행 중 (PID: {process.pid})")
        
        # PID 저장
        pid_file = current_dir / "neural_learning.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        # 잠시 기다린 후 상태 확인
        time.sleep(3)
        
        if process.poll() is None:  # 프로세스가 아직 실행 중
            print("🎉 24시간 지속 학습이 성공적으로 시작되었습니다!")
            print("📊 상태 확인: python3 continuous_neural_learning.py status")
            print("🛑 중지: python3 start_24h_learning.py stop")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 학습 시작 실패:")
            print(f"출력: {stdout.decode()}")
            print(f"오류: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        return False

def stop_24h_learning():
    """24시간 지속 학습 중지"""
    print("🛑 24시간 지속 학습 중지...")
    
    current_dir = Path(__file__).parent
    pid_file = current_dir / "neural_learning.pid"
    
    if not pid_file.exists():
        print("⚠️ 실행 중인 프로세스를 찾을 수 없습니다.")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # 프로세스 종료
        os.kill(pid, 15)  # SIGTERM
        time.sleep(2)
        
        # 프로세스가 여전히 실행 중이면 강제 종료
        try:
            os.kill(pid, 0)  # 프로세스 존재 확인
            os.kill(pid, 9)  # SIGKILL
            print("🔥 강제 종료됨")
        except OSError:
            print("✅ 정상 종료됨")
        
        pid_file.unlink()
        
    except Exception as e:
        print(f"❌ 종료 오류: {e}")

def check_status():
    """상태 확인"""
    current_dir = Path(__file__).parent
    
    # 통계 파일 확인
    stats_file = current_dir / "neural_learning_stats.json"
    if stats_file.exists():
        subprocess.run([sys.executable, "continuous_neural_learning.py", "status"])
    else:
        print("⚠️ 학습 통계를 찾을 수 없습니다. 아직 시작되지 않았을 수 있습니다.")
    
    # 프로세스 상태 확인
    pid_file = current_dir / "neural_learning.pid"
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # 프로세스 존재 확인
            print(f"✅ 백그라운드 프로세스 실행 중 (PID: {pid})")
        except (OSError, ValueError):
            print("❌ 백그라운드 프로세스가 실행되지 않음")
            pid_file.unlink()
    else:
        print("❌ 백그라운드 프로세스가 실행되지 않음")

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop":
            stop_24h_learning()
        elif sys.argv[1] == "status":
            check_status()
        else:
            print("사용법: python3 start_24h_learning.py [start|stop|status]")
    else:
        # 기본값: 시작
        start_24h_learning()

if __name__ == "__main__":
    main() 