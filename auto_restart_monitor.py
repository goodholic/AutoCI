#!/usr/bin/env python3
"""
AutoCI 자동 재시작 모니터링 시스템
24시간 무중단 운영을 위한 시스템 감시 및 자동 복구
"""

import os
import sys
import time
import json
import sqlite3
import threading
import subprocess
import psutil
import signal
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import socket
import requests
from dataclasses import dataclass

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """시스템 상태 정보"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: float
    network_latency: float
    process_count: int
    is_healthy: bool
    timestamp: str

@dataclass
class ProcessInfo:
    """프로세스 정보"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: float
    is_running: bool

class SystemMonitor:
    """시스템 모니터링"""
    
    def __init__(self):
        self.health_history = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "gpu_usage": 90.0,
            "network_latency": 5000,  # ms
            "min_free_memory": 1024   # MB
        }
        
        # GPU 모니터링 지원 여부
        self.gpu_available = self._check_gpu_availability()
        
        logger.info("🔍 시스템 모니터 초기화 완료")
    
    def _check_gpu_availability(self) -> bool:
        """GPU 모니터링 가능 여부 확인"""
        try:
            import GPUtil
            return True
        except ImportError:
            logger.warning("GPUtil 없음 - GPU 모니터링 비활성화")
            return False
    
    def get_system_health(self) -> SystemHealth:
        """현재 시스템 상태 조회"""
        
        # CPU 사용률
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # GPU 사용률
        gpu_usage = self._get_gpu_usage()
        
        # 네트워크 지연시간
        network_latency = self._get_network_latency()
        
        # 프로세스 수
        process_count = len(psutil.pids())
        
        # 전체 상태 판단
        is_healthy = (
            cpu_usage < self.alert_thresholds["cpu_usage"] and
            memory_usage < self.alert_thresholds["memory_usage"] and
            disk_usage < self.alert_thresholds["disk_usage"] and
            gpu_usage < self.alert_thresholds["gpu_usage"] and
            network_latency < self.alert_thresholds["network_latency"]
        )
        
        health = SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_usage=gpu_usage,
            network_latency=network_latency,
            process_count=process_count,
            is_healthy=is_healthy,
            timestamp=datetime.now().isoformat()
        )
        
        # 히스토리 저장
        self.health_history.append(health)
        if len(self.health_history) > 1440:  # 24시간 분 단위 (1440분)
            self.health_history.pop(0)
        
        return health
    
    def _get_gpu_usage(self) -> float:
        """GPU 사용률 조회"""
        if not self.gpu_available:
            return 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return max(gpu.load * 100 for gpu in gpus)
            return 0.0
        except Exception as e:
            logger.warning(f"GPU 사용률 조회 실패: {e}")
            return 0.0
    
    def _get_network_latency(self) -> float:
        """네트워크 지연시간 측정"""
        try:
            start_time = time.time()
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            end_time = time.time()
            return (end_time - start_time) * 1000  # ms
        except Exception:
            return 999999  # 네트워크 연결 불가
    
    def get_health_trend(self, hours: int = 1) -> Dict:
        """시스템 상태 트렌드 분석"""
        if not self.health_history:
            return {"trend": "no_data"}
        
        # 최근 N시간 데이터
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_health = [
            h for h in self.health_history 
            if datetime.fromisoformat(h.timestamp) > cutoff_time
        ]
        
        if len(recent_health) < 2:
            return {"trend": "insufficient_data"}
        
        # 트렌드 분석
        cpu_trend = self._calculate_trend([h.cpu_usage for h in recent_health])
        memory_trend = self._calculate_trend([h.memory_usage for h in recent_health])
        
        return {
            "trend": "improving" if cpu_trend < 0 and memory_trend < 0 else "degrading" if cpu_trend > 0 or memory_trend > 0 else "stable",
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "avg_cpu": sum(h.cpu_usage for h in recent_health) / len(recent_health),
            "avg_memory": sum(h.memory_usage for h in recent_health) / len(recent_health),
            "health_rate": sum(1 for h in recent_health if h.is_healthy) / len(recent_health)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """값들의 트렌드 계산 (양수: 증가, 음수: 감소)"""
        if len(values) < 2:
            return 0.0
        
        # 선형 회귀의 기울기 계산
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

class ProcessMonitor:
    """프로세스 모니터링"""
    
    def __init__(self):
        self.monitored_processes = {}  # name -> ProcessInfo
        self.restart_counts = {}  # name -> count
        self.max_restart_attempts = 3
        
        logger.info("🔍 프로세스 모니터 초기화 완료")
    
    def register_process(self, name: str, command: str, working_dir: str = None):
        """모니터링할 프로세스 등록"""
        self.monitored_processes[name] = {
            "command": command,
            "working_dir": working_dir or os.getcwd(),
            "pid": None,
            "last_restart": None,
            "restart_count": 0
        }
        
        logger.info(f"📝 프로세스 등록: {name} -> {command}")
    
    def start_process(self, name: str) -> bool:
        """프로세스 시작"""
        if name not in self.monitored_processes:
            logger.error(f"등록되지 않은 프로세스: {name}")
            return False
        
        process_info = self.monitored_processes[name]
        
        try:
            # 기존 프로세스가 실행 중인지 확인
            if self.is_process_running(name):
                logger.info(f"프로세스 이미 실행 중: {name}")
                return True
            
            # 프로세스 시작
            logger.info(f"🚀 프로세스 시작: {name}")
            
            process = subprocess.Popen(
                process_info["command"],
                shell=True,
                cwd=process_info["working_dir"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            process_info["pid"] = process.pid
            process_info["last_restart"] = datetime.now()
            process_info["restart_count"] += 1
            
            logger.info(f"✅ 프로세스 시작됨: {name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"프로세스 시작 실패 {name}: {e}")
            return False
    
    def stop_process(self, name: str) -> bool:
        """프로세스 중지"""
        if name not in self.monitored_processes:
            return False
        
        process_info = self.monitored_processes[name]
        pid = process_info.get("pid")
        
        if not pid:
            return True
        
        try:
            # 프로세스 존재 확인
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                
                # 3초 대기 후 강제 종료
                try:
                    process.wait(timeout=3)
                except psutil.TimeoutExpired:
                    process.kill()
                
                logger.info(f"🛑 프로세스 중지됨: {name} (PID: {pid})")
            
            process_info["pid"] = None
            return True
            
        except Exception as e:
            logger.error(f"프로세스 중지 실패 {name}: {e}")
            return False
    
    def is_process_running(self, name: str) -> bool:
        """프로세스 실행 상태 확인"""
        if name not in self.monitored_processes:
            return False
        
        pid = self.monitored_processes[name].get("pid")
        
        if not pid:
            return False
        
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except:
            return False
    
    def restart_process(self, name: str) -> bool:
        """프로세스 재시작"""
        if name not in self.monitored_processes:
            return False
        
        process_info = self.monitored_processes[name]
        
        # 최대 재시작 횟수 확인
        if process_info["restart_count"] >= self.max_restart_attempts:
            logger.error(f"최대 재시작 횟수 초과: {name} ({process_info['restart_count']}회)")
            return False
        
        logger.info(f"🔄 프로세스 재시작: {name}")
        
        # 중지 후 시작
        self.stop_process(name)
        time.sleep(2)
        return self.start_process(name)
    
    def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """프로세스 정보 조회"""
        if name not in self.monitored_processes or not self.is_process_running(name):
            return None
        
        pid = self.monitored_processes[name]["pid"]
        
        try:
            process = psutil.Process(pid)
            
            return ProcessInfo(
                pid=pid,
                name=process.name(),
                status=process.status(),
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                create_time=process.create_time(),
                is_running=process.is_running()
            )
            
        except Exception as e:
            logger.warning(f"프로세스 정보 조회 실패 {name}: {e}")
            return None
    
    def monitor_all_processes(self) -> Dict[str, bool]:
        """모든 등록된 프로세스 상태 확인"""
        status = {}
        
        for name in self.monitored_processes:
            is_running = self.is_process_running(name)
            status[name] = is_running
            
            if not is_running:
                logger.warning(f"⚠️ 프로세스 다운 감지: {name}")
        
        return status

class AutoRestartSystem:
    """자동 재시작 시스템"""
    
    def __init__(self, config_path: str = "monitor_config.json"):
        self.config_path = config_path
        self.system_monitor = SystemMonitor()
        self.process_monitor = ProcessMonitor()
        
        # 모니터링 설정
        self.monitoring_enabled = True
        self.auto_restart_enabled = True
        self.check_interval = 60  # 초
        
        # 알림 설정
        self.alert_cooldown = 300  # 5분
        self.last_alerts = {}
        
        # 통계
        self.stats = {
            "total_checks": 0,
            "health_issues": 0,
            "process_restarts": 0,
            "system_alerts": 0,
            "uptime_start": datetime.now()
        }
        
        # 설정 로드
        self.load_config()
        
        # 기본 프로세스 등록
        self.register_default_processes()
        
        logger.info("🎛️ 자동 재시작 시스템 초기화 완료")
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.check_interval = config.get("check_interval", 60)
                self.auto_restart_enabled = config.get("auto_restart_enabled", True)
                self.system_monitor.alert_thresholds.update(config.get("alert_thresholds", {}))
                
                logger.info(f"📋 설정 로드됨: {self.config_path}")
            else:
                self.save_config()
                
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
    
    def save_config(self):
        """설정 파일 저장"""
        config = {
            "check_interval": self.check_interval,
            "auto_restart_enabled": self.auto_restart_enabled,
            "alert_thresholds": self.system_monitor.alert_thresholds,
            "max_restart_attempts": self.process_monitor.max_restart_attempts
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 설정 저장됨: {self.config_path}")
            
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def register_default_processes(self):
        """기본 모니터링 프로세스 등록"""
        
        # AutoCI 학습 데몬
        self.process_monitor.register_process(
            "autoci_learning_daemon",
            "python3 continuous_neural_learning_daemon.py",
            os.getcwd()
        )
        
        # AutoCI 대화형 시스템
        self.process_monitor.register_process(
            "autoci_interactive",
            "python3 enhanced_autoci_korean.py",
            os.getcwd()
        )
        
        logger.info("📝 기본 프로세스 등록 완료")
    
    def start_monitoring(self):
        """모니터링 시작"""
        logger.info("🔍 모니터링 시작")
        
        # 등록된 프로세스들 시작
        for name in self.process_monitor.monitored_processes:
            if not self.process_monitor.is_process_running(name):
                self.process_monitor.start_process(name)
        
        # 모니터링 루프
        while self.monitoring_enabled:
            try:
                self.monitoring_cycle()
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 모니터링 중단 요청")
                break
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                logger.error(traceback.format_exc())
                time.sleep(30)  # 오류 시 30초 대기
        
        logger.info("👋 모니터링 종료")
    
    def monitoring_cycle(self):
        """모니터링 사이클"""
        self.stats["total_checks"] += 1
        
        # 시스템 상태 확인
        health = self.system_monitor.get_system_health()
        
        if not health.is_healthy:
            self.stats["health_issues"] += 1
            self.handle_system_health_issue(health)
        
        # 프로세스 상태 확인
        process_status = self.process_monitor.monitor_all_processes()
        
        for name, is_running in process_status.items():
            if not is_running and self.auto_restart_enabled:
                logger.warning(f"🚨 프로세스 다운 감지, 재시작 시도: {name}")
                if self.process_monitor.restart_process(name):
                    self.stats["process_restarts"] += 1
                    logger.info(f"✅ 프로세스 재시작 성공: {name}")
                else:
                    logger.error(f"❌ 프로세스 재시작 실패: {name}")
                    self.send_alert(f"프로세스 재시작 실패: {name}")
        
        # 주기적 상태 로그
        if self.stats["total_checks"] % 10 == 0:  # 10번마다
            self.log_status_summary(health, process_status)
    
    def handle_system_health_issue(self, health: SystemHealth):
        """시스템 상태 문제 처리"""
        
        issues = []
        
        if health.cpu_usage > self.system_monitor.alert_thresholds["cpu_usage"]:
            issues.append(f"높은 CPU 사용률: {health.cpu_usage:.1f}%")
        
        if health.memory_usage > self.system_monitor.alert_thresholds["memory_usage"]:
            issues.append(f"높은 메모리 사용률: {health.memory_usage:.1f}%")
        
        if health.disk_usage > self.system_monitor.alert_thresholds["disk_usage"]:
            issues.append(f"높은 디스크 사용률: {health.disk_usage:.1f}%")
        
        if health.network_latency > self.system_monitor.alert_thresholds["network_latency"]:
            issues.append(f"높은 네트워크 지연: {health.network_latency:.0f}ms")
        
        if issues:
            alert_message = "시스템 상태 경고: " + ", ".join(issues)
            self.send_alert(alert_message)
            
            # 메모리 부족 시 메모리 정리 시도
            if health.memory_usage > 90:
                self.cleanup_memory()
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            logger.info("🧹 메모리 정리 시작")
            
            # Python 가비지 컬렉션
            import gc
            collected = gc.collect()
            logger.info(f"가비지 컬렉션: {collected}개 객체 정리")
            
            # 시스템 캐시 정리 (Linux)
            if os.name == 'posix':
                try:
                    subprocess.run(['sync'], check=True)
                    subprocess.run(['sudo', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], 
                                 check=True, timeout=10)
                    logger.info("시스템 캐시 정리 완료")
                except:
                    logger.warning("시스템 캐시 정리 실패 (권한 부족)")
            
        except Exception as e:
            logger.error(f"메모리 정리 실패: {e}")
    
    def send_alert(self, message: str):
        """알림 전송"""
        current_time = time.time()
        
        # 쿨다운 확인
        if message in self.last_alerts:
            if current_time - self.last_alerts[message] < self.alert_cooldown:
                return
        
        self.last_alerts[message] = current_time
        self.stats["system_alerts"] += 1
        
        # 로그 알림
        logger.warning(f"🚨 ALERT: {message}")
        
        # 파일로 알림 저장
        alert_file = "system_alerts.log"
        with open(alert_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
        
        # 추가 알림 방법들 (이메일, 슬랙 등)을 여기에 구현 가능
    
    def log_status_summary(self, health: SystemHealth, process_status: Dict[str, bool]):
        """상태 요약 로그"""
        uptime = datetime.now() - self.stats["uptime_start"]
        
        running_processes = sum(1 for status in process_status.values() if status)
        total_processes = len(process_status)
        
        logger.info(f"📊 상태 요약: "
                   f"업타임={uptime.total_seconds()/3600:.1f}h, "
                   f"CPU={health.cpu_usage:.1f}%, "
                   f"메모리={health.memory_usage:.1f}%, "
                   f"프로세스={running_processes}/{total_processes}, "
                   f"재시작={self.stats['process_restarts']}회")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_enabled = False
        
        # 모든 프로세스 중지
        for name in self.process_monitor.monitored_processes:
            self.process_monitor.stop_process(name)
        
        logger.info("🛑 모니터링 및 모든 프로세스 중지")
    
    def get_system_status(self) -> Dict:
        """전체 시스템 상태 반환"""
        health = self.system_monitor.get_system_health()
        trend = self.system_monitor.get_health_trend()
        process_status = self.process_monitor.monitor_all_processes()
        
        return {
            "system_health": {
                "cpu_usage": health.cpu_usage,
                "memory_usage": health.memory_usage,
                "disk_usage": health.disk_usage,
                "gpu_usage": health.gpu_usage,
                "network_latency": health.network_latency,
                "is_healthy": health.is_healthy
            },
            "health_trend": trend,
            "processes": {
                name: {
                    "running": status,
                    "restart_count": self.process_monitor.monitored_processes[name]["restart_count"]
                }
                for name, status in process_status.items()
            },
            "monitoring_stats": self.stats,
            "uptime_hours": (datetime.now() - self.stats["uptime_start"]).total_seconds() / 3600
        }

def main():
    """메인 함수"""
    print("🚀 AutoCI 자동 재시작 모니터링 시스템")
    print("=" * 60)
    
    try:
        # 모니터링 시스템 초기화
        monitor_system = AutoRestartSystem()
        
        # 시그널 핸들러 설정
        def signal_handler(signum, frame):
            logger.info(f"종료 시그널 받음: {signum}")
            monitor_system.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 모니터링 시작
        monitor_system.start_monitoring()
        
    except Exception as e:
        logger.error(f"모니터링 시스템 실행 실패: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())