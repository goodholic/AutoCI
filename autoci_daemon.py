#!/usr/bin/env python3
"""
AutoCI Resume 데몬 - 24시간 안정적 운용을 위한 감시 및 재시작 시스템
"""

import os
import sys
import time
import json
import psutil
import signal
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoCIDaemon:
    """AutoCI Resume 명령어를 24시간 안정적으로 운용하는 데몬"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.status_file = self.project_root / "logs" / "daemon_status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.process = None
        self.project_path = None
        self.restart_count = 0
        self.max_restarts = 100  # 24시간 동안 최대 재시작 횟수
        self.start_time = datetime.now()
        self.target_duration = timedelta(hours=24)
        
        # 메모리 임계값
        self.memory_threshold_percent = 85
        self.memory_check_interval = 60  # 초
        
        # 상태 체크 간격
        self.health_check_interval = 30  # 초
        self.last_health_check = datetime.now()
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"시그널 {signum} 수신 - 안전하게 종료합니다")
        self.running = False
        self._stop_process()
        sys.exit(0)
    
    def _update_status(self, status: Dict[str, Any]):
        """상태 파일 업데이트"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"상태 업데이트 실패: {e}")
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용률 반환"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _check_process_health(self) -> bool:
        """프로세스 상태 확인"""
        if not self.process:
            return False
        
        # 프로세스 실행 여부 확인
        poll = self.process.poll()
        if poll is not None:
            logger.warning(f"프로세스가 종료됨 (코드: {poll})")
            return False
        
        # CPU 사용률 체크 (좀비 프로세스 감지)
        try:
            proc = psutil.Process(self.process.pid)
            cpu_percent = proc.cpu_percent(interval=1)
            
            # 10분 이상 CPU 0%면 문제가 있을 수 있음
            if cpu_percent == 0:
                if not hasattr(self, '_zero_cpu_start'):
                    self._zero_cpu_start = datetime.now()
                elif (datetime.now() - self._zero_cpu_start).seconds > 600:
                    logger.warning("프로세스가 10분 이상 비활성 상태")
                    return False
            else:
                self._zero_cpu_start = None
                
        except psutil.NoSuchProcess:
            return False
        
        return True
    
    def _check_checkpoint_status(self) -> Dict[str, Any]:
        """체크포인트 상태 확인"""
        checkpoint_dir = self.project_root / "logs" / "checkpoints"
        if not checkpoint_dir.exists():
            return {}
        
        # 가장 최근 체크포인트 찾기
        checkpoints = list(checkpoint_dir.glob("*_checkpoint.json"))
        if not checkpoints:
            return {}
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _start_resume_process(self, project_path: str) -> subprocess.Popen:
        """Resume 프로세스 시작"""
        logger.info(f"Resume 프로세스 시작: {project_path}")
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'  # 실시간 출력
        
        # autoci resume 실행
        cmd = [sys.executable, str(self.project_root / "autoci"), "resume"]
        
        # 프로젝트 디렉토리에서 실행
        process = subprocess.Popen(
            cmd,
            cwd=project_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 라인 버퍼링
        )
        
        # 비동기 출력 처리 시작
        asyncio.create_task(self._handle_process_output(process))
        
        return process
    
    async def _handle_process_output(self, process: subprocess.Popen):
        """프로세스 출력 비동기 처리"""
        try:
            while process.poll() is None:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        logger.info(f"[RESUME] {line.strip()}")
                
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"출력 처리 오류: {e}")
    
    def _stop_process(self):
        """프로세스 안전하게 종료"""
        if self.process:
            logger.info("프로세스 종료 중...")
            
            # 먼저 SIGINT 보내기 (Ctrl+C와 동일)
            try:
                self.process.send_signal(signal.SIGINT)
                time.sleep(5)  # 정상 종료 대기
            except:
                pass
            
            # 아직 살아있으면 강제 종료
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                    time.sleep(2)
                    if self.process.poll() is None:
                        self.process.kill()
                except:
                    pass
            
            self.process = None
    
    def _memory_cleanup(self):
        """메모리 정리"""
        import gc
        gc.collect()
        
        # GPU 메모리 정리 (가능한 경우)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    async def run(self, project_path: str):
        """데몬 메인 루프"""
        self.project_path = project_path
        logger.info(f"AutoCI 데몬 시작 - 프로젝트: {project_path}")
        logger.info(f"목표: {self.target_duration.total_seconds() / 3600}시간 연속 운용")
        
        while self.running and datetime.now() - self.start_time < self.target_duration:
            try:
                # 상태 업데이트
                elapsed = datetime.now() - self.start_time
                remaining = self.target_duration - elapsed
                
                status = {
                    "daemon_start": self.start_time,
                    "elapsed": str(elapsed),
                    "remaining": str(remaining),
                    "restart_count": self.restart_count,
                    "memory_usage": self._get_memory_usage(),
                    "process_active": self.process is not None,
                    "last_update": datetime.now()
                }
                
                # 체크포인트 정보 추가
                checkpoint_info = self._check_checkpoint_status()
                if checkpoint_info:
                    status["checkpoint"] = {
                        "iteration": checkpoint_info.get("iteration_count", 0),
                        "progress": checkpoint_info.get("progress_percent", 0)
                    }
                
                self._update_status(status)
                
                # 프로세스가 없거나 죽었으면 시작/재시작
                if not self._check_process_health():
                    if self.restart_count >= self.max_restarts:
                        logger.error("최대 재시작 횟수 초과")
                        break
                    
                    self._stop_process()
                    self._memory_cleanup()
                    
                    logger.info(f"프로세스 재시작 중... (#{self.restart_count + 1})")
                    self.process = self._start_resume_process(project_path)
                    self.restart_count += 1
                    
                    # 재시작 후 안정화 대기
                    await asyncio.sleep(10)
                
                # 메모리 체크
                memory_usage = self._get_memory_usage()
                if memory_usage > self.memory_threshold_percent:
                    logger.warning(f"높은 메모리 사용률: {memory_usage}%")
                    
                    # 프로세스 재시작으로 메모리 정리
                    self._stop_process()
                    self._memory_cleanup()
                    await asyncio.sleep(5)
                
                # 정기적 상태 로그
                if (datetime.now() - self.last_health_check).seconds > 300:  # 5분마다
                    logger.info(f"상태: 경과 {elapsed}, 재시작 {self.restart_count}회, 메모리 {memory_usage:.1f}%")
                    self.last_health_check = datetime.now()
                
                # 대기
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"데몬 루프 오류: {e}")
                await asyncio.sleep(10)
        
        # 정상 종료
        self._stop_process()
        
        final_status = {
            "completed": True,
            "total_runtime": str(datetime.now() - self.start_time),
            "total_restarts": self.restart_count,
            "end_time": datetime.now()
        }
        self._update_status(final_status)
        
        logger.info("AutoCI 데몬 종료")
        logger.info(f"총 실행 시간: {final_status['total_runtime']}")
        logger.info(f"총 재시작 횟수: {self.restart_count}")

async def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python autoci_daemon.py <프로젝트_경로>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    if not Path(project_path).exists():
        print(f"프로젝트 경로를 찾을 수 없습니다: {project_path}")
        sys.exit(1)
    
    daemon = AutoCIDaemon()
    await daemon.run(project_path)

if __name__ == "__main__":
    asyncio.run(main())