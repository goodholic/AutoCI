#!/usr/bin/env python3
"""
AutoCI Enhanced System - 24시간 자동 코드 수정 시스템
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_enhanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AutoCIEnhancer:
    """향상된 AutoCI 시스템"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.config_path = self.base_path / "autoci_config.json"
        self.expert_data_path = self.base_path / "expert_learning_data"
        self.reports_path = self.base_path / "autoci_reports"
        self.models_path = self.base_path / "models"
        
        # 필요한 디렉토리 생성
        self.expert_data_path.mkdir(exist_ok=True)
        self.reports_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # 설정 로드
        self.config = self.load_config()
        
        # 시스템 상태
        self.is_running = False
        self.start_time = None
        self.processed_files = 0
        self.improvements_made = 0
        
    def load_config(self) -> Dict:
        """설정 파일 로드"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 기본 설정
            default_config = {
                "target_path": str(self.base_path),
                "model_path": "CodeLlama-7b-Instruct-hf",
                "batch_size": 4,
                "max_workers": 4,
                "check_interval": 300,  # 5분마다 체크
                "rag_enabled": True,
                "fine_tuning_enabled": True,
                "auto_save_reports": True,
                "expert_sources": [
                    "https://docs.microsoft.com/en-us/dotnet/csharp/",
                    "https://github.com/dotnet/csharplang",
                    "https://stackoverflow.com/questions/tagged/c%23"
                ]
            }
            self.save_config(default_config)
            return default_config
            
    def save_config(self, config: Dict):
        """설정 저장"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def start_24h_system(self, target_path: Optional[str] = None):
        """24시간 자동 코드 수정 시스템 시작"""
        if target_path:
            self.config["target_path"] = target_path
            self.save_config(self.config)
            
        logger.info(f"🚀 24시간 자동 코드 수정 시스템 시작")
        logger.info(f"📁 대상 경로: {self.config['target_path']}")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # 병렬 처리를 위한 스레드 풀
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 1. RAG 시스템 (즉시 실행)
            rag_future = executor.submit(self.run_rag_system)
            
            # 2. 모델 파인튜닝 (백그라운드)
            tuning_future = executor.submit(self.run_fine_tuning)
            
            # 3. 코드 개선 시스템 (메인 루프)
            improvement_future = executor.submit(self.run_code_improvement)
            
            # 모든 작업 대기
            try:
                while self.is_running:
                    time.sleep(60)  # 1분마다 상태 체크
                    self.generate_status_report()
            except KeyboardInterrupt:
                logger.info("🛑 시스템 종료 신호 받음")
                self.is_running = False
                
    def run_rag_system(self):
        """RAG 시스템 실행 (단계 1)"""
        logger.info("🔍 RAG 시스템 시작...")
        
        try:
            # enhanced_rag_system_v2.py 실행
            subprocess.run([
                sys.executable,
                str(self.base_path / "enhanced_rag_system_v2.py"),
                "--data-path", str(self.expert_data_path),
                "--continuous"
            ], check=True)
        except Exception as e:
            logger.error(f"RAG 시스템 오류: {e}")
            
    def run_fine_tuning(self):
        """모델 파인튜닝 실행 (단계 2)"""
        logger.info("🎯 모델 파인튜닝 시작...")
        
        try:
            # 벡터 인덱싱 사용
            subprocess.run([
                sys.executable,
                str(self.base_path / "vector_indexer.py")
            ], check=True)
            
            # 파인튜닝 실행
            if (self.base_path / "hybrid_rag_training_system.py").exists():
                subprocess.run([
                    sys.executable,
                    str(self.base_path / "hybrid_rag_training_system.py")
                ], check=True)
        except Exception as e:
            logger.error(f"파인튜닝 오류: {e}")
            
    def run_code_improvement(self):
        """코드 개선 시스템 실행"""
        logger.info("🔧 코드 개선 시스템 시작...")
        
        while self.is_running:
            try:
                # auto_code_modifier.py 실행
                result = subprocess.run([
                    sys.executable,
                    str(self.base_path / "auto_code_modifier.py"),
                    self.config["target_path"]
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.improvements_made += 1
                    logger.info(f"✅ 코드 개선 완료 (총 {self.improvements_made}개)")
                
                # 다음 실행까지 대기
                time.sleep(self.config["check_interval"])
                
            except Exception as e:
                logger.error(f"코드 개선 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기
                
    def collect_expert_data(self):
        """C# 전문 데이터 수집"""
        logger.info("📚 C# 전문 데이터 수집 시작...")
        
        # deep_csharp_collector.py 실행
        try:
            subprocess.run([
                sys.executable,
                str(self.base_path / "deep_csharp_collector.py")
            ], check=True)
            
            logger.info("✅ 심층 전문 데이터 수집 완료")
            
            # 수집 후 자동 벡터 인덱싱
            logger.info("🔍 벡터 기반 인덱싱...")
            subprocess.run([
                sys.executable,
                str(self.base_path / "vector_indexer.py")
            ], check=True)
            
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            
    def generate_status_report(self):
        """상태 리포트 생성"""
        if not self.config.get("auto_save_reports", True):
            return
            
        runtime = datetime.now() - self.start_time if self.start_time else "N/A"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "runtime": str(runtime),
            "target_path": self.config["target_path"],
            "improvements_made": self.improvements_made,
            "processed_files": self.processed_files,
            "rag_enabled": self.config.get("rag_enabled", True),
            "fine_tuning_enabled": self.config.get("fine_tuning_enabled", True)
        }
        
        # JSON 리포트 저장
        report_path = self.reports_path / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Markdown 리포트 생성
        self.generate_markdown_report(report)
        
    def generate_markdown_report(self, status: Dict):
        """마크다운 형식의 리포트 생성"""
        md_content = f"""# AutoCI 상태 리포트

## 📊 시스템 상태
- **시작 시간**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}
- **실행 시간**: {status['runtime']}
- **대상 경로**: `{status['target_path']}`

## 📈 진행 상황
- **개선된 파일 수**: {status['improvements_made']}
- **처리된 파일 수**: {status['processed_files']}

## 🔧 시스템 설정
- **RAG 시스템**: {'활성화' if status['rag_enabled'] else '비활성화'}
- **파인튜닝**: {'활성화' if status['fine_tuning_enabled'] else '비활성화'}
- **체크 간격**: {self.config.get('check_interval', 300)}초

## 📝 최근 활동
```
{self.get_recent_logs()}
```

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 마크다운 파일 저장
        md_path = self.reports_path / "latest_status.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
    def get_recent_logs(self) -> str:
        """최근 로그 가져오기"""
        log_file = self.base_path / "autoci_enhanced.log"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return ''.join(lines[-20:])  # 마지막 20줄
        return "로그 없음"


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AutoCI Enhanced System")
    parser.add_argument("command", choices=["start", "stop", "status", "collect", "report"],
                       help="실행할 명령")
    parser.add_argument("--path", type=str, help="대상 경로 (start 명령 시)")
    parser.add_argument("--daemon", action="store_true", help="백그라운드 실행")
    
    args = parser.parse_args()
    
    enhancer = AutoCIEnhancer()
    
    if args.command == "start":
        if args.daemon:
            # 백그라운드 실행
            logger.info("🚀 백그라운드 모드로 시작...")
            # TODO: 데몬 프로세스 구현
        else:
            enhancer.start_24h_system(args.path)
            
    elif args.command == "stop":
        logger.info("🛑 시스템 종료 중...")
        enhancer.is_running = False
        
    elif args.command == "status":
        enhancer.generate_status_report()
        print(f"📊 상태 리포트가 생성되었습니다: {enhancer.reports_path}")
        
    elif args.command == "collect":
        enhancer.collect_expert_data()
        
    elif args.command == "report":
        enhancer.generate_status_report()
        latest_report = enhancer.reports_path / "latest_status.md"
        if latest_report.exists():
            print(latest_report.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()