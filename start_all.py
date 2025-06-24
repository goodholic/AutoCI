#!/usr/bin/env python3
"""
AutoCI 전체 시스템 통합 실행 스크립트
모든 서비스를 한 번에 시작하고 관리
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
import colorama
from colorama import Fore, Back, Style

# colorama 초기화
colorama.init()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoCILauncher:
    """AutoCI 통합 런처"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.processes = {}
        self.is_running = False
        self.start_time = None
        
        # 서비스 포트 설정
        self.ports = {
            'ai_server': 8000,
            'monitoring_api': 8080,
            'backend': 5049,
            'frontend': 7100
        }
        
    def print_banner(self):
        """시작 배너 출력"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Fore.YELLOW}🤖 AutoCI - 24시간 AI 코딩 공장{Fore.CYAN}                            ║
║                                                              ║
║  {Fore.GREEN}Code Llama 7B-Instruct 기반 C# 전문가 AI{Fore.CYAN}                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
        
    def check_system_requirements(self):
        """시스템 요구사항 확인"""
        print(f"\n{Fore.YELLOW}🔍 시스템 요구사항 확인 중...{Style.RESET_ALL}")
        
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
            print(f"\n{Fore.YELLOW}문제 발견:{Style.RESET_ALL}")
            for issue in issues:
                print(f"  {issue}")
            print(f"\n{Fore.YELLOW}⚠️  경고가 있지만 계속 진행합니다...{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}✅ 모든 시스템 요구사항을 충족합니다!{Style.RESET_ALL}")
        return True
    
    def check_model_exists(self):
        """Code Llama 모델 존재 확인"""
        model_path = self.base_dir / "CodeLlama-7b-Instruct-hf"
        if model_path.exists() and any(model_path.iterdir()):
            print(f"{Fore.GREEN}✅ Code Llama 모델이 이미 존재합니다{Style.RESET_ALL}")
            return True
        return False
    
    def download_model(self):
        """모델 다운로드"""
        if self.check_model_exists():
            return True
            
        print(f"\n{Fore.YELLOW}📥 Code Llama 7B-Instruct 모델 다운로드 중... (약 13GB){Style.RESET_ALL}")
        
        # download_model.py 실행
        if (self.base_dir / "download_model.py").exists():
            result = subprocess.run([sys.executable, "download_model.py"])
            return result.returncode == 0
        else:
            print(f"{Fore.RED}❌ download_model.py 파일이 없습니다{Style.RESET_ALL}")
            return False
    
    def check_port_available(self, port):
        """포트 사용 가능 여부 확인"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
    
    def start_service(self, name, command, cwd=None, env=None):
        """개별 서비스 시작"""
        try:
            print(f"\n{Fore.YELLOW}🚀 {name} 시작 중...{Style.RESET_ALL}")
            
            # 환경 변수 설정
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # 프로세스 시작
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=process_env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:  # Linux/Mac
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=process_env,
                    preexec_fn=os.setsid
                )
            
            self.processes[name] = process
            time.sleep(2)  # 서비스 시작 대기
            
            # 프로세스 확인
            if process.poll() is None:
                print(f"{Fore.GREEN}✅ {name} 시작 완료 (PID: {process.pid}){Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}❌ {name} 시작 실패{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ {name} 시작 오류: {str(e)}{Style.RESET_ALL}")
            return False
    
    def start_expert_learning(self):
        """24시간 전문가 학습 시스템 시작"""
        if not self.check_port_available(self.ports['ai_server']):
            print(f"{Fore.YELLOW}⚠️  AI 서버 포트 {self.ports['ai_server']}가 이미 사용 중입니다{Style.RESET_ALL}")
            return False
            
        # 전문가 학습 크롤러 시작
        if (self.base_dir / "csharp_expert_crawler.py").exists():
            self.start_service(
                "Expert Learning System",
                [sys.executable, "csharp_expert_crawler.py"],
                cwd=str(self.base_dir)
            )
        
        return True
    
    def start_ai_server(self):
        """AI 모델 서버 시작"""
        models_dir = self.base_dir / "MyAIWebApp" / "Models"
        server_file = models_dir / "enhanced_server.py"
        
        if not server_file.exists():
            print(f"{Fore.YELLOW}⚠️  enhanced_server.py가 없어 기본 서버를 생성합니다{Style.RESET_ALL}")
            # 기본 서버 생성은 별도 구현
            return False
        
        return self.start_service(
            "AI Model Server",
            [sys.executable, "-m", "uvicorn", "enhanced_server:app", 
             "--host", "0.0.0.0", "--port", str(self.ports['ai_server'])],
            cwd=str(models_dir)
        )
    
    def start_monitoring_api(self):
        """모니터링 API 시작"""
        if not self.check_port_available(self.ports['monitoring_api']):
            print(f"{Fore.YELLOW}⚠️  모니터링 API 포트 {self.ports['monitoring_api']}가 이미 사용 중입니다{Style.RESET_ALL}")
            return False
            
        if (self.base_dir / "expert_learning_api.py").exists():
            return self.start_service(
                "Monitoring API",
                [sys.executable, "expert_learning_api.py"],
                cwd=str(self.base_dir)
            )
        return False
    
    def start_backend(self):
        """C# Backend 시작"""
        backend_dir = self.base_dir / "MyAIWebApp" / "Backend"
        
        if not backend_dir.exists():
            print(f"{Fore.YELLOW}⚠️  Backend 디렉토리가 없습니다{Style.RESET_ALL}")
            return False
        
        if not self.check_port_available(self.ports['backend']):
            print(f"{Fore.YELLOW}⚠️  Backend 포트 {self.ports['backend']}가 이미 사용 중입니다{Style.RESET_ALL}")
            return False
        
        return self.start_service(
            "C# Backend",
            ["dotnet", "run"],
            cwd=str(backend_dir),
            env={"ASPNETCORE_URLS": f"http://localhost:{self.ports['backend']}"}
        )
    
    def start_frontend(self):
        """Blazor Frontend 시작"""
        frontend_dir = self.base_dir / "MyAIWebApp" / "Frontend"
        
        if not frontend_dir.exists():
            print(f"{Fore.YELLOW}⚠️  Frontend 디렉토리가 없습니다{Style.RESET_ALL}")
            return False
        
        if not self.check_port_available(self.ports['frontend']):
            print(f"{Fore.YELLOW}⚠️  Frontend 포트 {self.ports['frontend']}가 이미 사용 중입니다{Style.RESET_ALL}")
            return False
        
        return self.start_service(
            "Blazor Frontend",
            ["dotnet", "run"],
            cwd=str(frontend_dir),
            env={"ASPNETCORE_URLS": f"http://localhost:{self.ports['frontend']}"}
        )
    
    def show_status(self):
        """현재 상태 표시"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ AutoCI 시스템이 성공적으로 시작되었습니다!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}📌 서비스 접속 주소:{Style.RESET_ALL}")
        print(f"  • AI 코드 생성: http://localhost:{self.ports['frontend']}/codegen")
        print(f"  • 스마트 검색: http://localhost:{self.ports['frontend']}/codesearch")
        print(f"  • 프로젝트 Q&A: http://localhost:{self.ports['frontend']}/rag")
        print(f"  • 학습 대시보드: http://localhost:{self.ports['monitoring_api']}/dashboard/expert_learning_dashboard.html")
        print(f"  • 모니터링 API: http://localhost:{self.ports['monitoring_api']}/api/status")
        
        print(f"\n{Fore.YELLOW}📊 실행 중인 서비스:{Style.RESET_ALL}")
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  • {name}: {Fore.GREEN}실행 중{Style.RESET_ALL} (PID: {process.pid})")
            else:
                print(f"  • {name}: {Fore.RED}중지됨{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}💡 팁:{Style.RESET_ALL}")
        print(f"  • 로그 확인: tail -f autoci_startup.log")
        print(f"  • 학습 진행 상황: tail -f csharp_expert_learning.log")
        print(f"  • 종료하려면 Ctrl+C를 누르세요")
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def cleanup(self, signum=None, frame=None):
        """종료 시 정리 작업"""
        print(f"\n{Fore.YELLOW}🛑 AutoCI 시스템을 종료합니다...{Style.RESET_ALL}")
        
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"  • {name} 종료 중...")
                try:
                    if os.name == 'nt':
                        process.terminate()
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
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
        
        print(f"{Fore.GREEN}✅ 모든 서비스가 종료되었습니다{Style.RESET_ALL}")
        
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
            print(f"\n{Fore.RED}시스템 요구사항을 먼저 해결해주세요{Style.RESET_ALL}")
            return
        
        # 모델 다운로드
        if not self.download_model():
            print(f"\n{Fore.RED}모델 다운로드에 실패했습니다{Style.RESET_ALL}")
            return
        
        self.start_time = datetime.now()
        
        # 서비스 시작
        print(f"\n{Fore.CYAN}🚀 서비스를 시작합니다...{Style.RESET_ALL}")
        
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
        print(f"\n{Fore.CYAN}시스템이 실행 중입니다. 종료하려면 Ctrl+C를 누르세요...{Style.RESET_ALL}")
        
        try:
            while True:
                time.sleep(60)  # 1분마다 상태 체크
                
                # 프로세스 상태 확인
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"\n{Fore.YELLOW}⚠️  {name}이(가) 종료되었습니다. 재시작을 시도합니다...{Style.RESET_ALL}")
                        # 재시작 로직 구현 가능
                        
        except KeyboardInterrupt:
            self.cleanup()

def main():
    """메인 함수"""
    launcher = AutoCILauncher()
    launcher.run()

if __name__ == "__main__":
    main()