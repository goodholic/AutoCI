#!/usr/bin/env python3
"""
AI 학습 진행 상태 실시간 모니터링
"""

import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

def get_file_size(file_path):
    """파일 크기를 KB 단위로 반환"""
    try:
        size = os.path.getsize(file_path)
        return size / 1024  # KB 단위
    except:
        return 0

def get_process_count():
    """실행 중인 AI 프로세스 개수 확인"""
    try:
        result = subprocess.run(
            ['bash', '-c', 'ps aux | grep -E "(csharp_expert_crawler|enhanced_server)" | grep -v grep | wc -l'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except:
        return 0

def get_training_data_count():
    """학습 데이터 항목 개수 확인"""
    try:
        with open('expert_training_data/training_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except:
        return 0

def get_log_tail():
    """최근 로그 확인"""
    try:
        result = subprocess.run(
            ['tail', '-3', 'csharp_expert_learning.log'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return lines[-1] if lines else "로그 없음"
        return "로그 읽기 실패"
    except:
        return "로그 파일 없음"

def monitor_progress():
    """진행 상태 모니터링"""
    print("🤖 AI 학습 진행 상태 실시간 모니터링 시작")
    print("=" * 60)
    print("종료하려면 Ctrl+C를 누르세요\n")
    
    previous_data = {
        'size': 0,
        'count': 0,
        'processes': 0
    }
    
    try:
        while True:
            # 현재 시간
            now = datetime.now().strftime("%H:%M:%S")
            
            # 데이터 수집
            current_data = {
                'size': get_file_size('expert_training_data/training_dataset.json'),
                'count': get_training_data_count(),
                'processes': get_process_count()
            }
            
            # 변화량 계산
            size_diff = current_data['size'] - previous_data['size']
            count_diff = current_data['count'] - previous_data['count']
            
            # 화면 클리어 (간단한 방법)
            print("\033[2J\033[H")  # 화면 클리어 및 커서를 맨 위로
            
            # 상태 출력
            print("🤖 AI 학습 진행 상태 실시간 모니터링")
            print("=" * 60)
            print(f"📅 현재 시간: {now}")
            print()
            
            # 프로세스 상태
            print(f"🔄 실행 중인 AI 프로세스: {current_data['processes']}개")
            if current_data['processes'] > 0:
                print("   ✅ AI가 활발히 학습 중입니다!")
            else:
                print("   ❌ AI 프로세스가 실행되지 않음")
            print()
            
            # 학습 데이터 상태
            print(f"📊 학습 데이터:")
            print(f"   파일 크기: {current_data['size']:.1f} KB")
            print(f"   데이터 항목: {current_data['count']}개")
            
            if size_diff > 0:
                print(f"   📈 최근 증가: +{size_diff:.1f} KB, +{count_diff}개 항목")
            elif previous_data['size'] > 0:
                print(f"   ⏸️  변화 없음 (안정 상태)")
            print()
            
            # 최근 활동
            recent_log = get_log_tail()
            print(f"🔍 최근 활동:")
            print(f"   {recent_log}")
            print()
            
            # GPU 메모리 사용량 (선택사항)
            try:
                gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True)
                if gpu_result.returncode == 0:
                    gpu_info = gpu_result.stdout.strip().split(',')
                    if len(gpu_info) >= 2:
                        used = int(gpu_info[0].strip())
                        total = int(gpu_info[1].strip())
                        gpu_percent = (used / total) * 100
                        print(f"🎮 GPU 메모리: {used}MB / {total}MB ({gpu_percent:.1f}%)")
                        print()
            except:
                pass
            
            # 성과 요약
            print("🎯 현재까지의 성과:")
            if current_data['count'] > 100:
                print(f"   🏆 우수! {current_data['count']}개의 고품질 학습 데이터 수집")
            elif current_data['count'] > 50:
                print(f"   👍 양호! {current_data['count']}개의 학습 데이터 수집")
            elif current_data['count'] > 0:
                print(f"   🌱 시작 단계: {current_data['count']}개의 학습 데이터 수집")
            else:
                print("   🔍 데이터 수집 시작 중...")
            
            print()
            print("=" * 60)
            print("📢 1분마다 자동 업데이트됩니다... (Ctrl+C로 종료)")
            
            # 이전 데이터 저장
            previous_data = current_data.copy()
            
            # 1분 대기
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n✅ 모니터링을 종료합니다.")
        print("AI는 백그라운드에서 계속 학습하고 있습니다! 🤖")

if __name__ == "__main__":
    monitor_progress() 