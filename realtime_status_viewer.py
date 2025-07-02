#!/usr/bin/env python3
"""
실시간 AutoCI 상태 모니터링 도구
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_process_info():
    """현재 실행 중인 AutoCI 프로세스 확인"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        processes = []
        for line in lines:
            if 'autoci' in line.lower() and 'grep' not in line:
                processes.append(line.strip())
        
        return processes
    except:
        return []

def get_latest_status():
    """최신 상태 정보 가져오기"""
    status_dir = Path("logs/24h_improvement")
    if not status_dir.exists():
        return None, None
    
    # 가장 최신 status.json 파일 찾기
    status_files = list(status_dir.glob("*_status.json"))
    if not status_files:
        return None, None
    
    latest_status_file = max(status_files, key=lambda x: x.stat().st_mtime)
    latest_progress_file = latest_status_file.with_name(latest_status_file.name.replace('_status.json', '_progress.json'))
    
    try:
        with open(latest_status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
    except:
        status = {}
    
    try:
        with open(latest_progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except:
        progress = {}
    
    return status, progress

def get_recent_logs():
    """최근 로그 가져오기"""
    log_file = Path("logs/24h_improvement/latest_improvement.log")
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return [line.strip() for line in lines[-10:] if line.strip()]
    except:
        return []

def show_status():
    """상태 표시"""
    clear_screen()
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "🎮 AutoCI 실시간 상태 모니터" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # 현재 시간
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 현재 시간: {current_time}")
    print()
    
    # 프로세스 상태
    print("🤖 실행 중인 프로세스:")
    processes = get_process_info()
    if processes:
        for proc in processes:
            print(f"  {proc}")
    else:
        print("  ❌ AutoCI 프로세스를 찾을 수 없습니다")
    print()
    
    # 최신 상태
    status, progress = get_latest_status()
    if status:
        print("📊 24시간 개발 상태:")
        print(f"  🎮 프로젝트: {status.get('project_name', '없음')}")
        print(f"  ⏰ 시작: {status.get('start_time', '알 수 없음')[:19]}")
        print(f"  📈 경과: {status.get('elapsed_time', '0:00:00')} | 남은 시간: {status.get('remaining_time', '24:00:00')}")
        print(f"  📊 진행률: {status.get('progress_percent', 0):.1f}%")
        print(f"  🔄 반복: {status.get('iteration_count', 0)} | 🔨 수정: {status.get('fixes_count', 0)} | ✨ 개선: {status.get('improvements_count', 0)}")
        print(f"  🎯 품질 점수: {status.get('quality_score', 0)}/100")
        print()
        
        if progress:
            print("🔧 진행 상황:")
            print(f"  현재 작업: {progress.get('current_task', '대기 중')}")
            print(f"  현재 단계: {progress.get('current_phase', '알 수 없음')}")
            print(f"  끈질김 레벨: {progress.get('persistence_level', 'NORMAL')}")
            print(f"  창의성: {progress.get('creativity_level', 0)}/10")
            print(f"  마지막 활동: {progress.get('last_activity', '없음')}")
            print()
    else:
        print("❌ 24시간 개발 상태를 찾을 수 없습니다")
        print()
    
    # 최근 로그
    print("📋 최근 로그 (최근 10줄):")
    logs = get_recent_logs()
    if logs:
        for log in logs:
            print(f"  {log}")
    else:
        print("  📝 로그가 없습니다")
    print()
    
    print("─" * 80)
    print("💡 Ctrl+C로 종료 | 1초마다 자동 업데이트")
    print("─" * 80)

def main():
    """메인 실행"""
    try:
        print("🚀 AutoCI 실시간 모니터링을 시작합니다...")
        time.sleep(1)
        
        while True:
            show_status()
            time.sleep(1)  # 1초마다 업데이트
            
    except KeyboardInterrupt:
        print("\n\n👋 모니터링을 종료합니다.")

if __name__ == "__main__":
    main() 