#!/usr/bin/env python3
"""
프로세스 관리자 - WSL 환경에서 안정적인 24시간 실행을 위한 모듈
"""

import os
import sys
import signal
import psutil
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProcessManager:
    """프로세스 생명주기 관리"""
    
    def __init__(self):
        self.is_wsl = self._detect_wsl()
        self.keep_alive = True
        self.shutdown_handlers = []
        
    def _detect_wsl(self) -> bool:
        """WSL 환경 감지"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def register_shutdown_handler(self, handler: Callable):
        """종료 핸들러 등록"""
        self.shutdown_handlers.append(handler)
    
    def setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger.info(f"시그널 {signum} 수신됨")
            self.keep_alive = False
            for handler in self.shutdown_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"종료 핸들러 오류: {e}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if self.is_wsl:
            # WSL에서는 추가 시그널 처리
            signal.signal(signal.SIGHUP, signal.SIG_IGN)  # 세션 종료 무시
    
    def check_memory_usage(self) -> float:
        """메모리 사용률 확인"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_percent()
        except:
            return 0.0
    
    def check_system_resources(self) -> dict:
        """시스템 리소스 확인"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except:
            return {}
    
    async def keep_alive_loop(self, check_interval: int = 60):
        """Keep-alive 루프 - WSL 세션 유지"""
        while self.keep_alive:
            try:
                # 시스템 리소스 체크
                resources = self.check_system_resources()
                
                # 메모리 사용률이 높으면 경고
                if resources.get("memory_percent", 0) > 90:
                    logger.warning(f"높은 메모리 사용률: {resources['memory_percent']}%")
                
                # WSL 환경에서는 추가 체크
                if self.is_wsl:
                    # WSL 세션 유지를 위한 더미 작업
                    subprocess.run(["echo", "keep-alive"], capture_output=True)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Keep-alive 오류: {e}")
                await asyncio.sleep(10)
    
    def create_restart_script(self, command: str, project_path: str):
        """재시작 스크립트 생성"""
        script_path = Path.home() / ".autoci_restart.sh"
        
        script_content = f"""#!/bin/bash
# AutoCI 재시작 스크립트
cd "{project_path}"
{command}
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        logger.info(f"재시작 스크립트 생성됨: {script_path}")
        
        return script_path

# 싱글톤 인스턴스
_process_manager = None

def get_process_manager() -> ProcessManager:
    """프로세스 관리자 싱글톤 인스턴스 반환"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager