#!/usr/bin/env python3
"""
AutoCI 24시간 AI Agent 시작 스크립트
Llama 7B + Gemini CLI + Godot 통합 시스템
"""

import os
import sys
import subprocess
import asyncio
import argparse
from pathlib import Path

# 색상 출력을 위한 ANSI 코드
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    """배너 출력"""
    print(f"""{Colors.MAGENTA}
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     {Colors.CYAN}🤖 AutoCI - 24시간 게임 제작 AI Agent 🎮{Colors.MAGENTA}                ║
    ║                                                               ║
    ║     {Colors.GREEN}Llama 7B + Gemini CLI + Godot Engine{Colors.MAGENTA}                    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    {Colors.END}""")

def check_requirements():
    """필수 요구사항 확인"""
    print(f"{Colors.YELLOW}시스템 요구사항 확인 중...{Colors.END}")
    
    requirements = {
        "Python": check_python_version(),
        "Node.js": check_node(),
        "Godot": check_godot(),
        "GPU (CUDA)": check_cuda(),
        "Llama Model": check_llama_model(),
        "Gemini CLI": check_gemini_cli()
    }
    
    all_ok = True
    for req, status in requirements.items():
        if status:
            print(f"  {Colors.GREEN}✓{Colors.END} {req}")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {req}")
            all_ok = False
            
    return all_ok

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    return version.major >= 3 and version.minor >= 10

def check_node():
    """Node.js 설치 확인"""
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_godot():
    """Godot 설치 확인"""
    try:
        result = subprocess.run(['godot', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_cuda():
    """CUDA 사용 가능 확인"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def check_llama_model():
    """Llama 모델 확인"""
    model_path = Path("CodeLlama-7b-Instruct-hf")
    return model_path.exists() and model_path.is_dir()

def check_gemini_cli():
    """Gemini CLI 확인"""
    cli_path = Path("gemini-cli/packages/cli/dist/index.js")
    return cli_path.exists()

def install_dependencies():
    """종속성 설치"""
    print(f"{Colors.YELLOW}종속성 설치 중...{Colors.END}")
    
    # Python 패키지
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Gemini CLI 빌드
    if Path("gemini-cli").exists():
        print(f"{Colors.YELLOW}Gemini CLI 빌드 중...{Colors.END}")
        subprocess.run(['npm', 'install'], cwd='gemini-cli')
        subprocess.run(['npm', 'run', 'build'], cwd='gemini-cli')

def setup_environment():
    """환경 설정"""
    print(f"{Colors.YELLOW}환경 설정 중...{Colors.END}")
    
    # 환경 변수 설정
    if not os.environ.get('GEMINI_API_KEY'):
        print(f"{Colors.YELLOW}경고: GEMINI_API_KEY가 설정되지 않았습니다.{Colors.END}")
        print(f"Gemini 기능을 사용하려면 API 키를 설정하세요:")
        print(f"  export GEMINI_API_KEY='your-api-key'")
        
    # 디렉토리 생성
    Path("projects").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

async def start_autoci(mode: str = "interactive"):
    """AutoCI 시작"""
    if mode == "interactive":
        # 대화형 모드
        print(f"{Colors.GREEN}대화형 모드로 시작합니다...{Colors.END}")
        from autoci_conversational_interface import main
        main()
    elif mode == "daemon":
        # 데몬 모드 (24시간 백그라운드)
        print(f"{Colors.GREEN}24시간 데몬 모드로 시작합니다...{Colors.END}")
        from autoci_24h_learning_system import main
        await main()
    elif mode == "web":
        # 웹 인터페이스 모드
        print(f"{Colors.GREEN}웹 인터페이스 모드로 시작합니다...{Colors.END}")
        # TODO: 웹 인터페이스 구현
        print(f"{Colors.YELLOW}웹 인터페이스는 개발 중입니다.{Colors.END}")
    else:
        print(f"{Colors.RED}알 수 없는 모드: {mode}{Colors.END}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="AutoCI 24시간 게임 제작 AI Agent"
    )
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'daemon', 'web'],
        default='interactive',
        help='실행 모드 선택'
    )
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='종속성 설치'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='요구사항만 확인'
    )
    
    args = parser.parse_args()
    
    # 배너 출력
    print_banner()
    
    # 요구사항 확인
    if not check_requirements():
        print(f"\n{Colors.RED}일부 요구사항이 충족되지 않았습니다.{Colors.END}")
        print(f"설치 가이드는 README.md를 참조하세요.")
        
        if not args.check_only:
            response = input(f"\n계속하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}모든 요구사항이 충족되었습니다!{Colors.END}")
        
    if args.check_only:
        sys.exit(0)
        
    # 종속성 설치
    if args.install_deps:
        install_dependencies()
        
    # 환경 설정
    setup_environment()
    
    # AutoCI 시작
    print(f"\n{Colors.CYAN}AutoCI를 시작합니다...{Colors.END}\n")
    
    try:
        if args.mode == "daemon":
            asyncio.run(start_autoci(args.mode))
        else:
            asyncio.run(start_autoci(args.mode))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}프로그램이 중단되었습니다.{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}오류 발생: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()