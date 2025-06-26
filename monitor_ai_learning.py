#!/usr/bin/env python3
"""
AutoCI AI 학습 모니터링 전용 도구
실시간 학습 진행상황 및 성능 지표 모니터링
"""

import time
import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import sqlite3

def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_learning_stats():
    """학습 통계 수집"""
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'learning_files': 0,
        'training_files': 0,
        'total_size_mb': 0,
        'recent_updates': 0,
        'active_processes': 0,
        'database_records': 0
    }
    
    try:
        # 학습 데이터 파일 수
        learning_path = Path('expert_learning_data')
        if learning_path.exists():
            stats['learning_files'] = len(list(learning_path.rglob('*.json')))
            stats['total_size_mb'] = sum(f.stat().st_size for f in learning_path.rglob('*') if f.is_file()) / (1024*1024)
        
        # 훈련 데이터 파일 수
        training_path = Path('expert_training_data')
        if training_path.exists():
            stats['training_files'] = len(list(training_path.rglob('*.json*')))
        
        # 최근 업데이트 (5분 내)
        current_time = time.time()
        for file_path in Path('.').rglob('*.json'):
            if current_time - file_path.stat().st_mtime < 300:  # 5분
                stats['recent_updates'] += 1
        
        # 활성 프로세스
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        autoci_lines = [line for line in result.stdout.split('\n') if 'autoci' in line.lower() or 'learning' in line.lower()]
        stats['active_processes'] = len(autoci_lines)
        
        # 데이터베이스 레코드
        db_path = Path('autoci_cache/autoci.db')
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tasks")
            stats['database_records'] = cursor.fetchone()[0]
            conn.close()
            
    except Exception as e:
        print(f"통계 수집 오류: {e}")
    
    return stats

def get_model_performance():
    """모델 성능 지표 수집"""
    performance = {
        'improvement_ratio': 0,
        'files_analyzed': 0,
        'quality_score': 0,
        'last_update': '알 수 없음'
    }
    
    try:
        # 모델 개선 보고서
        report_path = Path('model_improvement_report.json')
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                performance['improvement_ratio'] = data.get('improvement_ratio', 0)
                performance['files_analyzed'] = data.get('total_files', 0)
                performance['last_update'] = data.get('timestamp', '알 수 없음')
        
        # 학습 통계
        stats_path = Path('expert_learning_data/deep_collection_stats.json')
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                performance['quality_score'] = len(data.get('categories', {}))
                
    except Exception as e:
        print(f"성능 지표 수집 오류: {e}")
    
    return performance

def display_dashboard():
    """모니터링 대시보드 표시"""
    clear_screen()
    
    print("🤖 AutoCI AI 학습 모니터링 대시보드")
    print("=" * 60)
    
    stats = get_learning_stats()
    performance = get_model_performance()
    
    print(f"⏰ 업데이트 시간: {stats['timestamp']}")
    print("")
    
    print("📊 학습 데이터 현황:")
    print(f"   📚 학습 파일: {stats['learning_files']:,}개")
    print(f"   📁 훈련 파일: {stats['training_files']:,}개")
    print(f"   💾 총 크기: {stats['total_size_mb']:.2f}MB")
    print(f"   🔄 최근 업데이트: {stats['recent_updates']}개 (5분 내)")
    print("")
    
    print("🎯 모델 성능 지표:")
    print(f"   📈 개선율: {performance['improvement_ratio']:.1f}%")
    print(f"   📋 분석된 파일: {performance['files_analyzed']:,}개")
    print(f"   ⭐ 품질 점수: {performance['quality_score']}")
    print(f"   🕒 마지막 업데이트: {performance['last_update']}")
    print("")
    
    print("⚙️ 시스템 상태:")
    print(f"   🔄 활성 프로세스: {stats['active_processes']}개")
    print(f"   🗄️ DB 레코드: {stats['database_records']:,}개")
    
    # 메모리 사용량
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            memory_line = lines[1].split()
            used = memory_line[2]
            total = memory_line[1]
            print(f"   💾 메모리: {used}/{total}")
    except:
        pass
    
    print("")
    print("🔍 활성 프로세스:")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        autoci_lines = [line for line in result.stdout.split('\n') 
                       if ('autoci' in line.lower() or 'learning' in line.lower()) and 'grep' not in line]
        
        if autoci_lines:
            for line in autoci_lines[:5]:  # 최대 5개만 표시
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    command = ' '.join(parts[10:])[:50]
                    print(f"   🟢 PID:{pid} CPU:{cpu}% MEM:{mem}% {command}")
        else:
            print("   📴 관련 프로세스 없음")
    except:
        print("   ❌ 프로세스 정보 수집 실패")
    
    print("")
    print("=" * 60)
    print("📘 조작법: Ctrl+C로 종료, Enter로 새로고침")

def main():
    """메인 모니터링 루프"""
    print("🚀 AutoCI AI 학습 모니터링 시작...")
    print("💡 3초마다 자동 갱신됩니다.")
    
    try:
        while True:
            display_dashboard()
            time.sleep(3)
    except KeyboardInterrupt:
        clear_screen()
        print("👋 AutoCI 학습 모니터링을 종료합니다.")
        print("감사합니다! 🤖")

if __name__ == "__main__":
    main() 