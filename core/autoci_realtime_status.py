#!/usr/bin/env python3
"""
AutoCI 실시간 상태 진단 및 문제 해결 도구
"""

import os
import sys
import json
import time
import psutil
import threading
import traceback
from pathlib import Path
from datetime import datetime

def diagnose_autoci_processes():
    """AutoCI 프로세스 진단"""
    print("🔍 AutoCI 프로세스 진단 중...")
    
    # 1. 실행 중인 프로세스 확인
    autoci_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'autoci' in cmdline.lower() or 'autoci' in proc.info['name'].lower():
                autoci_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline,
                    'status': proc.status(),
                    'cpu_percent': proc.cpu_percent(),
                    'memory_info': proc.memory_info()
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(f"📊 발견된 AutoCI 프로세스: {len(autoci_processes)}개")
    for proc in autoci_processes:
        print(f"  PID {proc['pid']}: {proc['name']} - {proc['status']}")
        print(f"    명령: {proc['cmdline'][:100]}...")
        print(f"    CPU: {proc['cpu_percent']:.1f}% | Memory: {proc['memory_info'].rss / 1024 / 1024:.1f}MB")
        print()
    
    return autoci_processes

def check_24h_improvement_status():
    """24시간 개선 상태 확인"""
    print("🔧 24시간 개선 시스템 상태 확인...")
    
    status_dir = Path("logs/24h_improvement")
    if not status_dir.exists():
        print("❌ 24시간 개선 로그 디렉토리가 없습니다")
        return None
    
    # 최신 상태 파일 찾기
    status_files = list(status_dir.glob("*_status.json"))
    if not status_files:
        print("❌ 상태 파일이 없습니다")
        return None
    
    latest_status_file = max(status_files, key=lambda x: x.stat().st_mtime)
    latest_progress_file = latest_status_file.with_name(latest_status_file.name.replace('_status.json', '_progress.json'))
    
    try:
        with open(latest_status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        print(f"📊 상태 파일: {latest_status_file.name}")
        print(f"  프로젝트: {status.get('project_name')}")
        print(f"  시작 시간: {status.get('start_time')}")
        print(f"  경과 시간: {status.get('elapsed_time')}")
        print(f"  반복 횟수: {status.get('iteration_count', 0)}")
        print(f"  수정 횟수: {status.get('fixes_count', 0)}")
        print(f"  개선 횟수: {status.get('improvements_count', 0)}")
        
        if latest_progress_file.exists():
            with open(latest_progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"  현재 작업: {progress.get('current_task')}")
            print(f"  현재 단계: {progress.get('current_phase')}")
            print(f"  마지막 활동: {progress.get('last_activity')}")
        
        return status, progress if latest_progress_file.exists() else {}
        
    except Exception as e:
        print(f"❌ 상태 파일 읽기 실패: {e}")
        return None

def check_improvement_logs():
    """개선 로그 분석"""
    print("📋 개선 로그 분석...")
    
    log_file = Path("logs/24h_improvement/latest_improvement.log")
    if not log_file.exists():
        print("❌ 개선 로그 파일이 없습니다")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📝 로그 라인 수: {len(lines)}")
        print("📋 최근 로그 (마지막 20줄):")
        for line in lines[-20:]:
            print(f"  {line.strip()}")
            
        # 특정 패턴 검색
        iteration_logs = [line for line in lines if "개선 반복" in line]
        error_logs = [line for line in lines if "오류 검사" in line]
        
        print(f"\n📊 분석 결과:")
        print(f"  개선 반복 로그: {len(iteration_logs)}개")
        print(f"  오류 검사 로그: {len(error_logs)}개")
        
        if not iteration_logs:
            print("⚠️ 개선 반복이 시작되지 않았습니다!")
        if not error_logs:
            print("⚠️ 오류 검사가 실행되지 않았습니다!")
            
    except Exception as e:
        print(f"❌ 로그 분석 실패: {e}")

def restart_improvement_system():
    """개선 시스템 재시작"""
    print("🔄 24시간 개선 시스템 재시작 시도...")
    
    try:
        # 현재 상태 백업
        status_dir = Path("logs/24h_improvement")
        backup_dir = status_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상태 파일 백업
        for status_file in status_dir.glob("*_status.json"):
            backup_file = backup_dir / f"{status_file.stem}_{timestamp}.json"
            with open(status_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
        
        print("✅ 상태 백업 완료")
        
        # 새로운 개선 프로세스 시작 스크립트 생성
        restart_script = """#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# AutoCI 경로 추가
sys.path.append(str(Path(__file__).parent))

async def restart_improvement():
    try:
        from modules.persistent_game_improver import PersistentGameImprover
        
        # 최신 프로젝트 찾기
        mvp_dir = Path("mvp_games")
        if mvp_dir.exists():
            projects = sorted(mvp_dir.glob("rpg_*"), key=lambda x: x.stat().st_mtime)
            if projects:
                latest_project = projects[-1]
                print(f"🎮 재시작할 프로젝트: {latest_project}")
                
                # 개선 시스템 시작
                improver = PersistentGameImprover()
                await improver.start_24h_improvement(latest_project)
            else:
                print("❌ 재시작할 프로젝트를 찾을 수 없습니다")
        else:
            print("❌ mvp_games 디렉토리가 없습니다")
            
    except Exception as e:
        print(f"❌ 재시작 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(restart_improvement())
"""
        
        restart_file = Path("restart_improvement.py")
        with open(restart_file, 'w', encoding='utf-8') as f:
            f.write(restart_script)
        
        print(f"✅ 재시작 스크립트 생성: {restart_file}")
        print("💡 실행 명령: python restart_improvement.py")
        
    except Exception as e:
        print(f"❌ 재시작 준비 실패: {e}")

def main():
    """메인 진단 실행"""
    print("╔" + "═" * 60 + "╗")
    print("║" + " " * 18 + "🩺 AutoCI 진단 도구" + " " * 18 + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    # 1. 프로세스 진단
    processes = diagnose_autoci_processes()
    print()
    
    # 2. 24시간 개선 상태 확인
    status_result = check_24h_improvement_status()
    print()
    
    # 3. 로그 분석
    check_improvement_logs()
    print()
    
    # 4. 문제 진단
    print("🎯 진단 결과:")
    
    if not processes:
        print("❌ AutoCI 프로세스가 실행되지 않고 있습니다")
    elif not status_result:
        print("❌ 24시간 개선 시스템이 시작되지 않았습니다")
    else:
        status, progress = status_result
        if status.get('iteration_count', 0) == 0:
            print("⚠️ 24시간 개선이 시작되었지만 실제 작업이 진행되지 않고 있습니다")
            print("💡 가능한 원인:")
            print("  - 메인 루프에서 예외 발생")
            print("  - 의존성 모듈 import 실패")
            print("  - 블로킹 상태")
            
            user_input = input("\n🔄 개선 시스템을 재시작하시겠습니까? (y/n): ")
            if user_input.lower() == 'y':
                restart_improvement_system()
        else:
            print("✅ 24시간 개선 시스템이 정상적으로 작동 중입니다")
    
    print("\n🏁 진단 완료")

if __name__ == "__main__":
    main() 