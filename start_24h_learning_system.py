#!/usr/bin/env python3
"""
AutoCI 24시간 학습 시스템 통합 런처
모든 학습 컴포넌트를 통합하여 24시간 무중단 학습 시스템 시작
"""

import os
import sys
import time
import json
import subprocess
import threading
import logging
import signal
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_24h_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoCI24HSystem:
    """AutoCI 24시간 학습 시스템"""
    
    def __init__(self):
        self.system_components = {
            "neural_learning_daemon": {
                "script": "continuous_neural_learning_daemon.py",
                "description": "24시간 신경망 학습 데몬",
                "process": None,
                "enabled": True,
                "critical": True
            },
            "learning_optimizer": {
                "script": "learning_scheduler_optimizer.py", 
                "description": "학습 스케줄러 및 최적화기",
                "process": None,
                "enabled": True,
                "critical": False
            },
            "auto_restart_monitor": {
                "script": "auto_restart_monitor.py",
                "description": "자동 재시작 모니터",
                "process": None,
                "enabled": True,
                "critical": True
            },
            "progress_tracker": {
                "script": "learning_progress_tracker.py",
                "description": "학습 진행률 추적기",
                "process": None,
                "enabled": True,
                "critical": False
            }
        }
        
        self.running = True
        self.start_time = datetime.now()
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 AutoCI 24시간 학습 시스템 초기화")
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"🛑 종료 시그널 받음 ({signum})")
        self.running = False
        self.stop_all_components()
    
    def check_dependencies(self) -> bool:
        """의존성 확인"""
        logger.info("🔍 시스템 의존성 확인 중...")
        
        required_packages = [
            "torch", "scikit-learn", "matplotlib", "pandas", 
            "numpy", "psutil", "schedule"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 설치됨")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"⚠️ {package} 누락")
        
        if missing_packages:
            logger.error(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
            logger.info("설치 명령: pip install " + " ".join(missing_packages))
            return False
        
        # 스크립트 파일 확인
        missing_scripts = []
        for component, info in self.system_components.items():
            script_path = info["script"]
            if not os.path.exists(script_path):
                missing_scripts.append(script_path)
                logger.warning(f"⚠️ 스크립트 누락: {script_path}")
        
        if missing_scripts:
            logger.error(f"❌ 누락된 스크립트: {', '.join(missing_scripts)}")
            return False
        
        logger.info("✅ 모든 의존성 확인 완료")
        return True
    
    def start_component(self, component_name: str) -> bool:
        """컴포넌트 시작"""
        if component_name not in self.system_components:
            logger.error(f"알 수 없는 컴포넌트: {component_name}")
            return False
        
        component = self.system_components[component_name]
        
        if not component["enabled"]:
            logger.info(f"⏸️ 컴포넌트 비활성화됨: {component_name}")
            return True
        
        if component["process"] and component["process"].poll() is None:
            logger.info(f"⚠️ 컴포넌트 이미 실행 중: {component_name}")
            return True
        
        try:
            logger.info(f"🚀 컴포넌트 시작: {component['description']}")
            
            # Python 스크립트 실행
            process = subprocess.Popen(
                [sys.executable, component["script"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            component["process"] = process
            logger.info(f"✅ {component_name} 시작됨 (PID: {process.pid})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {component_name} 시작 실패: {e}")
            return False
    
    def stop_component(self, component_name: str) -> bool:
        """컴포넌트 중지"""
        if component_name not in self.system_components:
            return False
        
        component = self.system_components[component_name]
        process = component["process"]
        
        if not process:
            return True
        
        try:
            logger.info(f"🛑 컴포넌트 중지: {component['description']}")
            
            # 정상 종료 시도
            process.terminate()
            
            # 3초 대기
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # 강제 종료
                process.kill()
                process.wait()
            
            component["process"] = None
            logger.info(f"✅ {component_name} 중지됨")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {component_name} 중지 실패: {e}")
            return False
    
    def check_component_health(self, component_name: str) -> bool:
        """컴포넌트 상태 확인"""
        if component_name not in self.system_components:
            return False
        
        component = self.system_components[component_name]
        process = component["process"]
        
        if not process:
            return False
        
        # 프로세스가 실행 중인지 확인
        return process.poll() is None
    
    def restart_component(self, component_name: str) -> bool:
        """컴포넌트 재시작"""
        logger.info(f"🔄 컴포넌트 재시작: {component_name}")
        
        self.stop_component(component_name)
        time.sleep(2)
        return self.start_component(component_name)
    
    def start_all_components(self) -> bool:
        """모든 컴포넌트 시작"""
        logger.info("🚀 모든 컴포넌트 시작 중...")
        
        success_count = 0
        total_count = len([c for c in self.system_components.values() if c["enabled"]])
        
        for component_name in self.system_components:
            if self.start_component(component_name):
                success_count += 1
            
            # 컴포넌트 간 시작 간격
            time.sleep(2)
        
        logger.info(f"✅ {success_count}/{total_count} 컴포넌트 시작 완료")
        return success_count > 0
    
    def stop_all_components(self):
        """모든 컴포넌트 중지"""
        logger.info("🛑 모든 컴포넌트 중지 중...")
        
        for component_name in self.system_components:
            self.stop_component(component_name)
    
    def monitor_components(self):
        """컴포넌트 모니터링"""
        logger.info("🔍 컴포넌트 모니터링 시작")
        
        while self.running:
            try:
                unhealthy_components = []
                
                for component_name, component in self.system_components.items():
                    if not component["enabled"]:
                        continue
                    
                    if not self.check_component_health(component_name):
                        unhealthy_components.append(component_name)
                        
                        if component["critical"]:
                            logger.warning(f"🚨 중요 컴포넌트 다운: {component_name}")
                            if self.restart_component(component_name):
                                logger.info(f"✅ {component_name} 재시작 성공")
                            else:
                                logger.error(f"❌ {component_name} 재시작 실패")
                        else:
                            logger.warning(f"⚠️ 비중요 컴포넌트 다운: {component_name}")
                
                # 상태 로그 (10분마다)
                if int(time.time()) % 600 == 0:
                    self.log_system_status()
                
                time.sleep(30)  # 30초마다 확인
                
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                time.sleep(60)
    
    def log_system_status(self):
        """시스템 상태 로그"""
        uptime = datetime.now() - self.start_time
        
        status = {
            "uptime_hours": uptime.total_seconds() / 3600,
            "components": {}
        }
        
        for component_name, component in self.system_components.items():
            status["components"][component_name] = {
                "enabled": component["enabled"],
                "running": self.check_component_health(component_name),
                "critical": component["critical"]
            }
        
        running_count = sum(1 for c in status["components"].values() if c["running"])
        total_count = len(status["components"])
        
        logger.info(f"📊 시스템 상태: 업타임={uptime.total_seconds()/3600:.1f}h, "
                   f"실행중={running_count}/{total_count}")
    
    def create_system_dashboard(self) -> str:
        """시스템 대시보드 생성"""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AutoCI 24시간 학습 시스템 대시보드</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .component {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .running {{ background: #d4edda; border-color: #c3e6cb; }}
                .stopped {{ background: #f8d7da; border-color: #f5c6cb; }}
                .disabled {{ background: #e2e3e5; border-color: #d6d8db; }}
                .status {{ font-weight: bold; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 AutoCI 24시간 학습 시스템</h1>
                <p>실시간 시스템 상태 대시보드</p>
            </div>
            
            <h2>📊 시스템 개요</h2>
            <p><strong>시작 시간:</strong> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>업타임:</strong> {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} 시간</p>
            
            <h2>🔧 컴포넌트 상태</h2>
        """
        
        for component_name, component in self.system_components.items():
            is_running = self.check_component_health(component_name)
            
            if not component["enabled"]:
                css_class = "disabled"
                status_text = "비활성화"
            elif is_running:
                css_class = "running"
                status_text = "실행중"
            else:
                css_class = "stopped"
                status_text = "중지됨"
            
            dashboard_html += f"""
            <div class="component {css_class}">
                <h3>{component['description']}</h3>
                <p><strong>스크립트:</strong> {component['script']}</p>
                <p><strong>상태:</strong> <span class="status">{status_text}</span></p>
                <p><strong>중요도:</strong> {'중요' if component['critical'] else '일반'}</p>
            </div>
            """
        
        dashboard_html += f"""
            <div class="timestamp">
                <p>마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        dashboard_file = "autoci_dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return dashboard_file
    
    def run_system(self):
        """시스템 실행"""
        logger.info("🎯 AutoCI 24시간 학습 시스템 시작")
        
        # 의존성 확인
        if not self.check_dependencies():
            logger.error("❌ 의존성 확인 실패, 시스템 종료")
            return False
        
        # 모든 컴포넌트 시작
        if not self.start_all_components():
            logger.error("❌ 컴포넌트 시작 실패, 시스템 종료")
            return False
        
        # 대시보드 생성
        dashboard_file = self.create_system_dashboard()
        logger.info(f"📊 대시보드 생성: {dashboard_file}")
        
        # 모니터링 시작
        monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
        monitor_thread.start()
        
        # 메인 루프
        try:
            logger.info("🔄 24시간 학습 시스템 메인 루프 시작")
            
            while self.running:
                time.sleep(60)  # 1분마다 체크
                
                # 주기적 대시보드 업데이트 (10분마다)
                if int(time.time()) % 600 == 0:
                    self.create_system_dashboard()
                
        except KeyboardInterrupt:
            logger.info("🛑 키보드 인터럽트로 시스템 종료")
        
        finally:
            self.stop_all_components()
            logger.info("👋 AutoCI 24시간 학습 시스템 종료")
        
        return True

def main():
    """메인 함수"""
    print("🚀 AutoCI 24시간 지속 학습 시스템")
    print("=" * 60)
    print("🧠 실제 신경망 기반 24시간 무중단 학습")
    print("📊 자동 데이터 수집 및 최적화")
    print("🔍 시스템 모니터링 및 자동 재시작")
    print("📈 실시간 학습 진행률 추적")
    print("=" * 60)
    
    try:
        # 시스템 초기화 및 실행
        system = AutoCI24HSystem()
        return 0 if system.run_system() else 1
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())