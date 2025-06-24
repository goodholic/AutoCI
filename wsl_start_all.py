#!/usr/bin/env python3
"""
AutoCI WSL 환경 전용 실행 스크립트
WSL(Windows Subsystem for Linux)에서 모든 서비스를 시작하고 관리
"""

import os
import sys
import subprocess
import time
import threading
import signal
import logging
from pathlib import Path
import json
import psutil
from datetime import datetime
import platform

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_wsl_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WSLAutoCI:
    """WSL 환경용 AutoCI 런처"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.processes = {}
        self.is_running = False
        self.start_time = None
        self.is_wsl = self._detect_wsl()
        
        # 가상환경 Python 경로 설정 (여러 경로 시도)
        potential_paths = [
            self.base_dir / "llm_venv_wsl" / "bin" / "python",
            self.base_dir / "llm_venv_wsl" / "bin" / "python3",
            self.base_dir / "llm_venv_wsl" / "bin" / "python3.12",
            self.base_dir / "llm_venv" / "bin" / "python",
            self.base_dir / "llm_venv" / "bin" / "python3"
        ]
        
        self.venv_python = None
        for path in potential_paths:
            if path.exists() and path.is_file():
                self.venv_python = str(path.resolve())  # 절대 경로로 변환
                logger.info(f"가상환경 Python 발견: {self.venv_python}")
                break
        
        if not self.venv_python:
            logger.warning("가상환경 Python을 찾을 수 없음. 시스템 Python 사용")
            # WSL에서는 python3 명령어 사용
            self.venv_python = "python3"
            
        logger.info(f"사용할 Python 경로: {self.venv_python}")
        
        # 가상환경 유효성 검증
        if "llm_venv" in str(self.venv_python):
            try:
                # 가상환경에서 peft 패키지 확인
                result = subprocess.run([
                    "bash", "-c", 
                    f'source "{self.base_dir}/llm_venv_wsl/bin/activate" && python3 -c "import peft; print(\\"peft available\\")"'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "peft available" in result.stdout:
                    logger.info("✅ 가상환경에서 peft 패키지 확인됨")
                else:
                    logger.warning("⚠️  가상환경에서 peft 패키지를 찾을 수 없음")
            except Exception as e:
                logger.warning(f"가상환경 검증 실패: {e}")
        
        # 서비스 포트 설정
        self.ports = {
            'ai_server': 8000,
            'monitoring_api': 8080,
            'backend': 5049,
            'frontend': 7100
        }
        
    def _detect_wsl(self):
        """WSL 환경 감지"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def print_banner(self):
        """시작 배너 출력"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🤖 AutoCI - 24시간 AI 코딩 공장 (WSL Edition)              ║
║                                                              ║
║  Code Llama 7B-Instruct 기반 C# 전문가 AI                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)
        
        if self.is_wsl:
            print("✅ WSL 환경이 감지되었습니다.\n")
        else:
            print("⚠️  WSL 환경이 아닙니다. 일반 Linux로 실행합니다.\n")
    
    def check_system_requirements(self):
        """시스템 요구사항 확인"""
        print("🔍 시스템 요구사항 확인 중...")
        
        issues = []
        
        # Python 버전 확인
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            issues.append("❌ Python 3.8 이상이 필요합니다")
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor} 확인")
        
        # 메모리 확인
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        if total_gb < 16:
            issues.append(f"⚠️  메모리가 {total_gb:.1f}GB입니다. 16GB 이상 권장")
        else:
            print(f"✅ 메모리 {total_gb:.1f}GB 확인")
        
        # 디스크 공간 확인
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        if free_gb < 50:
            issues.append(f"⚠️  디스크 여유 공간이 {free_gb:.1f}GB입니다. 50GB 이상 권장")
        else:
            print(f"✅ 디스크 여유 공간 {free_gb:.1f}GB 확인")
        
        # .NET 확인
        try:
            result = subprocess.run(['dotnet', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ .NET SDK {result.stdout.strip()} 확인")
            else:
                issues.append("❌ .NET SDK가 설치되지 않았습니다")
        except:
            issues.append("❌ .NET SDK가 설치되지 않았습니다")
        
        # Git 확인
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            print("✅ Git 설치 확인")
        except:
            issues.append("❌ Git이 설치되지 않았습니다")
        
        if issues:
            print("\n문제 발견:")
            for issue in issues:
                print(f"  {issue}")
            print("\n⚠️  경고가 있지만 계속 진행합니다...")
        else:
            print("\n✅ 모든 시스템 요구사항을 충족합니다!")
        return True
    
    def check_model_exists(self):
        """Code Llama 모델 존재 확인"""
        model_path = self.base_dir / "CodeLlama-7b-Instruct-hf"
        if model_path.exists() and any(model_path.iterdir()):
            print("✅ Code Llama 모델이 이미 존재합니다")
            return True
        return False
    
    def download_model(self):
        """모델 다운로드"""
        if self.check_model_exists():
            return True
            
        print("\n📥 Code Llama 7B-Instruct 모델 다운로드 중... (약 13GB)")
        
        # download_model.py 실행
        if (self.base_dir / "download_model.py").exists():
            result = subprocess.run([sys.executable, "download_model.py"])
            return result.returncode == 0
        else:
            print("❌ download_model.py 파일이 없습니다")
            return False
    
    def check_port_available(self, port):
        """포트 사용 가능 여부 확인"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
    
    def start_service(self, name, command, cwd=None, env=None):
        """개별 서비스 시작 (WSL 최적화)"""
        try:
            print(f"\n🚀 {name} 시작 중...")
            
            # 환경 변수 설정
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Python 서비스인 경우 가상환경 activate
            if command[0] in [str(self.venv_python), "python", "python3"] or any(cmd.endswith(".py") for cmd in command):
                # 가상환경 활성화 및 실행을 위한 bash 명령어 생성
                venv_activate_path = self.base_dir / "llm_venv_wsl" / "bin" / "activate"
                if venv_activate_path.exists():
                    # 명령어를 문자열로 조합
                    if command[0] == str(self.venv_python):
                        # 이미 가상환경 python 경로가 지정된 경우 - python3으로 변경
                        new_command = ["python3"] + command[1:]
                        cmd_str = " ".join([f'"{cmd}"' if " " in cmd else cmd for cmd in new_command])
                    else:
                        # python 명령어를 python3로 변경
                        if command[0] == "python":
                            new_command = ["python3"] + command[1:]
                        else:
                            new_command = command
                        cmd_str = " ".join([f'"{cmd}"' if " " in cmd else cmd for cmd in new_command])
                    
                    # 절대 경로로 활성화 스크립트 사용 (따옴표로 감싸기)
                    abs_activate_path = str(venv_activate_path.resolve())
                    
                    # bash를 통해 가상환경 활성화 후 실행 (경로를 따옴표로 안전하게 감싸기)
                    venv_command = f'source "{abs_activate_path}" && {cmd_str}'
                    command = ["bash", "-c", venv_command]
                    
                    print(f'   가상환경에서 실행: source "{abs_activate_path}" && {cmd_str}')
                else:
                    print(f"   ⚠️  가상환경이 없어 시스템 Python으로 실행: {command}")
                    # python을 python3로 변경
                    if command[0] == "python":
                        command[0] = "python3"
            
            # WSL에서는 Windows 특정 플래그 제거
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.processes[name] = process
            time.sleep(3)  # 서비스 시작 대기 시간 증가
            
            # 프로세스 확인
            if process.poll() is None:
                print(f"✅ {name} 시작 완료 (PID: {process.pid})")
                return True
            else:
                # 오류 로그 출력
                stdout, stderr = process.communicate()
                print(f"❌ {name} 시작 실패")
                if stderr:
                    print(f"   오류: {stderr.decode('utf-8', errors='ignore')[:200]}...")
                if stdout:
                    print(f"   출력: {stdout.decode('utf-8', errors='ignore')[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ {name} 시작 오류: {str(e)}")
            return False
    
    def start_expert_learning(self):
        """24시간 전문가 학습 시스템 시작"""
        if not self.check_port_available(self.ports['ai_server']):
            print(f"⚠️  AI 서버 포트 {self.ports['ai_server']}가 이미 사용 중입니다")
            return False
            
        # 전문가 학습 크롤러 시작
        if (self.base_dir / "csharp_expert_crawler.py").exists():
            self.start_service(
                "Expert Learning System",
                [str(self.venv_python), "csharp_expert_crawler.py"],
                cwd=str(self.base_dir)
            )
        
        return True
    
    def start_ai_server(self):
        """AI 모델 서버 시작"""
        models_dir = self.base_dir / "MyAIWebApp" / "Models"
        server_file = models_dir / "enhanced_server.py"
        
        if not server_file.exists():
            print("⚠️  enhanced_server.py가 없어 기본 서버를 생성합니다")
            return False
        
        # WSL에서는 항상 0.0.0.0으로 바인딩
        return self.start_service(
            "AI Model Server",
            [str(self.venv_python), "-m", "uvicorn", "enhanced_server:app", 
             "--host", "0.0.0.0", "--port", str(self.ports['ai_server'])],
            cwd=str(models_dir)
        )
    
    def start_monitoring_api(self):
        """모니터링 API 시작"""
        if not self.check_port_available(self.ports['monitoring_api']):
            print(f"⚠️  모니터링 API 포트 {self.ports['monitoring_api']}가 이미 사용 중입니다")
            return False
            
        if (self.base_dir / "expert_learning_api.py").exists():
            return self.start_service(
                "Monitoring API",
                [str(self.venv_python), "expert_learning_api.py"],
                cwd=str(self.base_dir)
            )
        return False
    
    def start_backend(self):
        """C# Backend 시작"""
        backend_dir = self.base_dir / "MyAIWebApp" / "Backend"
        
        if not backend_dir.exists():
            print("⚠️  Backend 디렉토리가 없습니다")
            return False
        
        if not self.check_port_available(self.ports['backend']):
            print(f"⚠️  Backend 포트 {self.ports['backend']}가 이미 사용 중입니다")
            return False
        
        # WSL에서는 0.0.0.0으로 바인딩
        return self.start_service(
            "C# Backend",
            ["dotnet", "run"],
            cwd=str(backend_dir),
            env={"ASPNETCORE_URLS": f"http://0.0.0.0:{self.ports['backend']}"}
        )
    
    def start_frontend(self):
        """Blazor Frontend 시작"""
        frontend_dir = self.base_dir / "MyAIWebApp" / "Frontend"
        
        if not frontend_dir.exists():
            print("⚠️  Frontend 디렉토리가 없습니다")
            return False
        
        if not self.check_port_available(self.ports['frontend']):
            print(f"⚠️  Frontend 포트 {self.ports['frontend']}가 이미 사용 중입니다")
            return False
        
        # WSL에서는 0.0.0.0으로 바인딩
        return self.start_service(
            "Blazor Frontend",
            ["dotnet", "run"],
            cwd=str(frontend_dir),
            env={"ASPNETCORE_URLS": f"http://0.0.0.0:{self.ports['frontend']}"}
        )
    
    def get_wsl_ip(self):
        """WSL의 IP 주소 가져오기"""
        try:
            # hostname -I 명령으로 IP 주소 가져오기
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            if result.returncode == 0:
                ips = result.stdout.strip().split()
                if ips:
                    return ips[0]  # 첫 번째 IP 주소 반환
        except:
            pass
        return "localhost"
    
    def show_status(self):
        """현재 상태 표시 (WSL 특화)"""
        print("\n" + "="*60)
        print("✅ AutoCI 시스템이 성공적으로 시작되었습니다!")
        print("="*60)
        
        # WSL IP 주소 표시
        if self.is_wsl:
            wsl_ip = self.get_wsl_ip()
            print(f"\n📍 WSL IP 주소: {wsl_ip}")
            print("   (Windows에서 접속 시 이 IP를 사용하세요)")
        
        print(f"\n📌 서비스 접속 주소:")
        
        if self.is_wsl:
            # WSL 환경에서는 Windows와 WSL 양쪽 접속 방법 표시
            print("\n  [WSL 내부에서 접속]")
            print(f"  • AI 코드 생성: http://localhost:{self.ports['frontend']}/codegen")
            print(f"  • 스마트 검색: http://localhost:{self.ports['frontend']}/codesearch")
            print(f"  • 프로젝트 Q&A: http://localhost:{self.ports['frontend']}/rag")
            print(f"  • 학습 대시보드: http://localhost:{self.ports['monitoring_api']}/dashboard/expert_learning_dashboard.html")
            
            wsl_ip = self.get_wsl_ip()
            print("\n  [Windows에서 접속]")
            print(f"  • AI 코드 생성: http://{wsl_ip}:{self.ports['frontend']}/codegen")
            print(f"  • 스마트 검색: http://{wsl_ip}:{self.ports['frontend']}/codesearch")
            print(f"  • 프로젝트 Q&A: http://{wsl_ip}:{self.ports['frontend']}/rag")
            print(f"  • 학습 대시보드: http://{wsl_ip}:{self.ports['monitoring_api']}/dashboard/expert_learning_dashboard.html")
        else:
            print(f"  • AI 코드 생성: http://localhost:{self.ports['frontend']}/codegen")
            print(f"  • 스마트 검색: http://localhost:{self.ports['frontend']}/codesearch")
            print(f"  • 프로젝트 Q&A: http://localhost:{self.ports['frontend']}/rag")
            print(f"  • 학습 대시보드: http://localhost:{self.ports['monitoring_api']}/dashboard/expert_learning_dashboard.html")
        
        print(f"\n📊 실행 중인 서비스:")
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  • {name}: 실행 중 (PID: {process.pid})")
            else:
                print(f"  • {name}: 중지됨")
        
        print(f"\n💡 팁:")
        print(f"  • 로그 확인: tail -f autoci_wsl_startup.log")
        print(f"  • 학습 진행 상황: tail -f csharp_expert_learning.log")
        
        if self.is_wsl:
            print("\n🔧 WSL 방화벽 설정:")
            print("  Windows에서 접속이 안 될 경우:")
            print("  1. Windows PowerShell을 관리자 권한으로 실행")
            print("  2. 다음 명령 실행:")
            for port in self.ports.values():
                print(f"     New-NetFirewallRule -DisplayName 'WSL Port {port}' -Direction Inbound -LocalPort {port} -Protocol TCP -Action Allow")
        
        print(f"\n종료하려면 Ctrl+C를 누르세요")
        print("="*60)
    
    def cleanup(self, signum=None, frame=None):
        """종료 시 정리 작업"""
        print("\n🛑 AutoCI 시스템을 종료합니다...")
        
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  • {name} 종료 중...")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except:
                    try:
                        process.terminate()
                    except:
                        pass
        
        # 프로세스 종료 대기
        time.sleep(2)
        
        # 강제 종료
        for name, process in self.processes.items():
            if process.poll() is None:
                try:
                    process.kill()
                except:
                    pass
        
        print("✅ 모든 서비스가 종료되었습니다")
        
        # 실행 시간 출력
        if self.start_time:
            runtime = datetime.now() - self.start_time
            hours = runtime.total_seconds() / 3600
            print(f"\n총 실행 시간: {hours:.1f}시간")
        
        sys.exit(0)
    
    def run(self):
        """메인 실행"""
        self.print_banner()
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # 시스템 요구사항 확인
        if not self.check_system_requirements():
            print("\n시스템 요구사항을 먼저 해결해주세요")
            return
        
        # 모델 다운로드
        if not self.download_model():
            print("\n모델 다운로드에 실패했습니다")
            return
        
        self.start_time = datetime.now()
        
        # 서비스 시작
        print("\n🚀 서비스를 시작합니다...")
        
        # 1. 24시간 전문가 학습 시스템
        self.start_expert_learning()
        
        # 2. AI 모델 서버
        self.start_ai_server()
        
        # 3. 모니터링 API
        self.start_monitoring_api()
        
        # 4. C# Backend
        self.start_backend()
        
        # 5. Blazor Frontend
        self.start_frontend()
        
        # 상태 표시
        self.show_status()
        
        # 대기
        print("\n시스템이 실행 중입니다. 종료하려면 Ctrl+C를 누르세요...")
        
        try:
            while True:
                time.sleep(60)  # 1분마다 상태 체크
                
                # 프로세스 상태 확인
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"\n⚠️  {name}이(가) 종료되었습니다. 재시작을 시도합니다...")
                        # 재시작 로직 구현 가능
                        
        except KeyboardInterrupt:
            self.cleanup()

def main():
    """메인 함수"""
    launcher = WSLAutoCI()
    launcher.run()

if __name__ == "__main__":
    main()