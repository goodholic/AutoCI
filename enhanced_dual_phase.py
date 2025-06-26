#!/usr/bin/env python3
"""
향상된 Dual Phase System
1단계: RAG 시스템으로 즉시 활용
2단계: 백그라운드에서 파인튜닝
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import psutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_dual_phase.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedDualPhaseSystem:
    """향상된 이중 단계 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_path = self.base_path / "dual_phase_config.json"
        self.status_path = self.base_path / "dual_phase_status.json"
        
        # 설정 로드
        self.config = self.load_config()
        
        # 시스템 상태
        self.rag_process = None
        self.tuning_process = None
        self.is_running = False
        self.start_time = None
        
    def load_config(self) -> Dict:
        """설정 로드"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            default_config = {
                "rag_port": 8000,
                "enable_gpu": False,
                "max_memory_gb": 8,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "checkpoint_interval": 1000,
                "auto_restart": True
            }
            self.save_config(default_config)
            return default_config
            
    def save_config(self, config: Dict):
        """설정 저장"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def start_system(self):
        """이중 단계 시스템 시작"""
        logger.info("🚀 Enhanced Dual Phase System 시작")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # 시스템 상태 초기화
        self.update_status({
            'status': 'running',
            'start_time': self.start_time.isoformat(),
            'rag_status': 'starting',
            'tuning_status': 'starting'
        })
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 단계 1: RAG 시스템 (즉시)
            rag_future = executor.submit(self.start_rag_phase)
            
            # 단계 2: 파인튜닝 (백그라운드)
            tuning_future = executor.submit(self.start_tuning_phase)
            
            # 모니터링 루프
            try:
                while self.is_running:
                    time.sleep(30)  # 30초마다 체크
                    self.monitor_system()
                    
            except KeyboardInterrupt:
                logger.info("🛑 시스템 종료 신호")
                self.stop_system()
                
    def start_rag_phase(self):
        """RAG 단계 시작 (즉시 활용)"""
        logger.info("📚 단계 1: RAG 시스템 시작")
        
        try:
            # 데이터가 있는지 확인
            expert_data_path = self.base_path / "expert_learning_data"
            if not expert_data_path.exists() or not any(expert_data_path.iterdir()):
                logger.info("📥 전문가 데이터 수집 중...")
                subprocess.run([
                    sys.executable,
                    str(self.base_path / "enhanced_expert_collector.py")
                ], check=True)
            
            # 스마트 인덱싱
            logger.info("🔍 데이터 인덱싱...")
            subprocess.run([
                sys.executable,
                str(self.base_path / "smart_indexer.py")
            ], check=True)
            
            # RAG 서버 시작
            logger.info(f"🌐 RAG 서버 시작 (포트: {self.config['rag_port']})")
            self.rag_process = subprocess.Popen([
                sys.executable,
                str(self.base_path / "enhanced_rag_system_v2.py"),
                "--server",
                "--port", str(self.config['rag_port'])
            ])
            
            self.update_status({'rag_status': 'running'})
            logger.info("✅ RAG 시스템 준비 완료")
            
            # 프로세스 모니터링
            while self.is_running and self.rag_process.poll() is None:
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"RAG 단계 오류: {e}")
            self.update_status({'rag_status': 'error', 'rag_error': str(e)})
            
    def start_tuning_phase(self):
        """파인튜닝 단계 시작 (백그라운드)"""
        logger.info("🎯 단계 2: 백그라운드 파인튜닝 시작")
        
        # RAG가 시작될 때까지 대기
        time.sleep(10)
        
        try:
            # 시스템 리소스 체크
            memory_available = psutil.virtual_memory().available / (1024**3)  # GB
            if memory_available < self.config['max_memory_gb']:
                logger.warning(f"⚠️  메모리 부족: {memory_available:.1f}GB < {self.config['max_memory_gb']}GB")
                
            # 파인튜닝 프로세스 시작
            logger.info("🔧 모델 파인튜닝 시작...")
            self.tuning_process = subprocess.Popen([
                sys.executable,
                str(self.base_path / "hybrid_rag_training_system.py"),
                "--batch-size", str(self.config['batch_size']),
                "--learning-rate", str(self.config['learning_rate']),
                "--num-epochs", str(self.config['num_epochs'])
            ])
            
            self.update_status({'tuning_status': 'running'})
            
            # 프로세스 모니터링
            while self.is_running and self.tuning_process.poll() is None:
                time.sleep(30)
                self.check_tuning_progress()
                
            # 완료 확인
            if self.tuning_process.returncode == 0:
                logger.info("✅ 파인튜닝 완료")
                self.update_status({'tuning_status': 'completed'})
                
                # RAG 시스템 재시작하여 새 모델 적용
                if self.config.get('auto_restart', True):
                    self.restart_rag_with_new_model()
            else:
                logger.error("❌ 파인튜닝 실패")
                self.update_status({'tuning_status': 'failed'})
                
        except Exception as e:
            logger.error(f"파인튜닝 단계 오류: {e}")
            self.update_status({'tuning_status': 'error', 'tuning_error': str(e)})
            
    def monitor_system(self):
        """시스템 모니터링"""
        # CPU 및 메모리 사용량
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # 프로세스 상태 확인
        rag_alive = self.rag_process and self.rag_process.poll() is None
        tuning_alive = self.tuning_process and self.tuning_process.poll() is None
        
        status_update = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'rag_alive': rag_alive,
            'tuning_alive': tuning_alive,
            'last_check': datetime.now().isoformat()
        }
        
        self.update_status(status_update)
        
        # 리소스 경고
        if memory_percent > 90:
            logger.warning(f"⚠️  메모리 사용량 높음: {memory_percent}%")
            
        # 프로세스 재시작
        if not rag_alive and self.config.get('auto_restart', True):
            logger.warning("RAG 프로세스 종료됨. 재시작 중...")
            self.start_rag_phase()
            
    def check_tuning_progress(self):
        """파인튜닝 진행 상황 확인"""
        # 체크포인트 파일 확인
        checkpoint_dir = self.base_path / "models" / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"📊 최신 체크포인트: {latest.name}")
                
    def restart_rag_with_new_model(self):
        """새 모델로 RAG 재시작"""
        logger.info("🔄 새 모델로 RAG 시스템 재시작...")
        
        # 기존 RAG 종료
        if self.rag_process:
            self.rag_process.terminate()
            self.rag_process.wait()
            
        # 새 모델로 재시작
        time.sleep(5)
        self.start_rag_phase()
        
    def stop_system(self):
        """시스템 종료"""
        logger.info("🛑 시스템 종료 중...")
        
        self.is_running = False
        
        # 프로세스 종료
        if self.rag_process:
            self.rag_process.terminate()
            self.rag_process.wait()
            
        if self.tuning_process:
            self.tuning_process.terminate()
            self.tuning_process.wait()
            
        self.update_status({'status': 'stopped'})
        
    def update_status(self, update: Dict):
        """상태 업데이트"""
        if self.status_path.exists():
            with open(self.status_path, 'r', encoding='utf-8') as f:
                status = json.load(f)
        else:
            status = {}
            
        status.update(update)
        
        with open(self.status_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
            
    def get_status(self) -> Dict:
        """현재 상태 가져오기"""
        if self.status_path.exists():
            with open(self.status_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def generate_report(self):
        """상태 리포트 생성"""
        status = self.get_status()
        
        report = f"""# Dual Phase System 상태 리포트

## 🚀 시스템 개요
- **상태**: {status.get('status', 'unknown')}
- **시작 시간**: {status.get('start_time', 'N/A')}
- **마지막 체크**: {status.get('last_check', 'N/A')}

## 📊 단계별 상태

### 1️⃣ RAG 시스템 (즉시 활용)
- **상태**: {status.get('rag_status', 'unknown')}
- **프로세스**: {'실행 중' if status.get('rag_alive', False) else '중지됨'}
- **포트**: {self.config['rag_port']}

### 2️⃣ 파인튜닝 (백그라운드)
- **상태**: {status.get('tuning_status', 'unknown')}
- **프로세스**: {'실행 중' if status.get('tuning_alive', False) else '중지됨'}
- **배치 크기**: {self.config['batch_size']}
- **학습률**: {self.config['learning_rate']}
- **에폭**: {self.config['num_epochs']}

## 💻 시스템 리소스
- **CPU 사용률**: {status.get('cpu_percent', 0):.1f}%
- **메모리 사용률**: {status.get('memory_percent', 0):.1f}%

## 🔧 설정
- **자동 재시작**: {self.config.get('auto_restart', True)}
- **최대 메모리**: {self.config['max_memory_gb']}GB
- **GPU 사용**: {self.config.get('enable_gpu', False)}

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_path = self.base_path / "autoci_reports" / "dual_phase_status.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Dual Phase System")
    parser.add_argument("command", choices=["start", "stop", "status", "report"],
                       help="실행할 명령")
    
    args = parser.parse_args()
    
    system = EnhancedDualPhaseSystem()
    
    if args.command == "start":
        system.start_system()
        
    elif args.command == "stop":
        system.stop_system()
        
    elif args.command == "status":
        status = system.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    elif args.command == "report":
        report = system.generate_report()
        print(report)


if __name__ == "__main__":
    main()