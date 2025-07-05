#!/usr/bin/env python3
"""
AutoCI 런처 - 가상 환경 자동 활성화 및 명령어 실행
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import click
import json
import time

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_DIR = PROJECT_ROOT / "autoci_env"
SYSTEM = platform.system()

# 가상 환경 Python 실행 파일 경로
if SYSTEM == "Windows":
    PYTHON_EXECUTABLE = VENV_DIR / "Scripts" / "python.exe"
    PIP_EXECUTABLE = VENV_DIR / "Scripts" / "pip.exe"
    ACTIVATE_SCRIPT = VENV_DIR / "Scripts" / "activate.bat"
else:
    PYTHON_EXECUTABLE = VENV_DIR / "bin" / "python"
    PIP_EXECUTABLE = VENV_DIR / "bin" / "pip"
    ACTIVATE_SCRIPT = VENV_DIR / "bin" / "activate"


def create_venv():
    """가상 환경 생성"""
    if not VENV_DIR.exists():
        print("🔧 가상 환경 생성 중...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("✅ 가상 환경 생성 완료")
        return True
    return False


def install_requirements():
    """필수 패키지 설치"""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    # requirements.txt가 없으면 생성
    if not requirements_file.exists():
        print("📝 requirements.txt 생성 중...")
        requirements = [
            "panda3d>=1.10.13",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "python-socketio[asyncio]>=5.7.0",
            "aiohttp>=3.8.0",
            "click>=8.0.0",
            "numpy>=1.21.0",
            "psutil>=5.9.0",
            "pyautogui>=0.9.53",
            "keyboard>=0.13.5",
            "mouse>=0.7.1",
            "Pillow>=9.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.40.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "tqdm>=4.62.0",
            "colorama>=0.4.4",
            "uvicorn>=0.18.0",
            "websockets>=10.0"
        ]
        requirements_file.write_text("\n".join(requirements))
    
    # 패키지 설치 상태 확인
    check_cmd = [str(PIP_EXECUTABLE), "list", "--format=json"]
    result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        installed_packages = {pkg["name"].lower() for pkg in json.loads(result.stdout)}
        
        # 핵심 패키지 확인
        core_packages = {"panda3d", "torch", "python-socketio", "click"}
        missing_packages = core_packages - installed_packages
        
        if missing_packages:
            print(f"📦 필수 패키지 설치 중: {', '.join(missing_packages)}")
            subprocess.run([str(PIP_EXECUTABLE), "install", "-r", str(requirements_file)], check=True)
            print("✅ 패키지 설치 완료")
            return True
    
    return False


def run_command(command, args):
    """가상 환경에서 명령어 실행"""
    # 가상 환경 확인 및 생성
    venv_created = create_venv()
    packages_installed = install_requirements()
    
    if venv_created or packages_installed:
        print("🚀 AutoCI 시스템 준비 완료\n")
    
    # 실행할 Python 스크립트 결정
    script_map = {
        "main": PROJECT_ROOT / "core_system" / "autoci_main.py",
        "learn": PROJECT_ROOT / "core_system" / "autoci_main.py",
        "fix": PROJECT_ROOT / "core_system" / "autoci_main.py",
        "create": PROJECT_ROOT / "core_system" / "autoci_main.py",
        "monitor": PROJECT_ROOT / "core_system" / "autoci_main.py",
        "evolve": PROJECT_ROOT / "core_system" / "autoci_main.py"
    }
    
    # 명령어에 따른 스크립트 선택
    if command == "learn":
        script = script_map["learn"]
        # learn 명령어 처리
        if args and args[0] == "low":
            cmd_args = ["learn", "low"]
        else:
            cmd_args = ["learn"]
    elif command == "fix":
        script = script_map["fix"]
        cmd_args = ["fix"] + list(args)
    elif command == "create":
        script = script_map["create"]
        # create platformer 형식으로 전달
        cmd_args = ["create"] + list(args)
    elif command == "evolve":
        script = script_map["evolve"]
        cmd_args = ["evolve"] + list(args)
    elif command in ["analyze", "monitor", "demo"]:
        script = script_map.get(command, script_map["main"])
        cmd_args = [command] + list(args)
    else:
        # 기본 autoci 명령어
        script = script_map["main"]
        cmd_args = list(args)
    
    # 명령어 실행
    cmd = [str(PYTHON_EXECUTABLE), str(script)] + cmd_args
    
    try:
        # 환경 변수 설정
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["AUTOCI_LAUNCHER"] = "true"  # 런처를 통해 실행됨을 표시
        
        # Windows에서 가상 환경 활성화
        if SYSTEM == "Windows":
            env["PATH"] = f"{VENV_DIR / 'Scripts'}{os.pathsep}{env['PATH']}"
        else:
            env["PATH"] = f"{VENV_DIR / 'bin'}{os.pathsep}{env['PATH']}"
        
        # 명령어 실행
        result = subprocess.run(cmd, env=env)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """AutoCI - AI 자동 게임 개발 시스템"""
    if ctx.invoked_subcommand is None:
        # 서브커맨드 없이 autoci만 실행한 경우
        print("""
╔══════════════════════════════════════════════════════════════╗
║                      AutoCI v5.0                             ║
║              AI 자동 게임 개발 시스템                        ║
╚══════════════════════════════════════════════════════════════╝

사용 가능한 명령어:
  autoci              - 대화형 모드로 AutoCI 시작
  autoci create       - 새 게임 자동 생성 (24시간)
  autoci learn        - AI 모델 기반 연속 학습
  autoci learn low    - 메모리 최적화 연속 학습
  autoci fix          - 학습 기반 게임 엔진 능력 업데이트
  autoci monitor      - 실시간 개발 모니터링
  autoci demo         - 5분 빠른 데모

예시:
  autoci create --name MyGame --type platformer
  autoci learn
  autoci monitor --port 5001
        """)
        # 기본 대화형 모드 실행
        run_command("main", [])


@cli.command()
@click.argument('name', required=False)
@click.option('--type', 'game_type', 
              type=click.Choice(['platformer', 'racing', 'rpg', 'puzzle', 'shooter', 'strategy', 'adventure', 'simulation']),
              required=False, help='게임 타입')
@click.option('--hours', default=24.0, help='개발 시간 (기본 24시간)')
def create(name, game_type, hours):
    """AI가 24시간 동안 자동으로 게임 개발"""
    # name과 game_type이 모두 제공된 경우에만 출력
    if name and game_type:
        print(f"\n🎮 AutoCI 게임 자동 생성")
        print(f"   프로젝트: {name}")
        print(f"   타입: {game_type}")
        print(f"   예상 시간: {hours}시간\n")
    
    # create 명령어를 autoci_main.py로 전달
    # name과 game_type을 함께 전달
    args = []
    if name:
        args.append(name)
    if game_type:
        args.append(game_type)
    run_command("create", args)


@cli.command()
@click.option('--low', is_flag=True, help='메모리 최적화 모드')
@click.option('--hours', type=float, help='학습 시간')
@click.option('--memory', type=float, help='메모리 제한 (GB)')
def learn(low, hours, memory):
    """AI 모델 기반 연속 학습"""
    print("\n🧠 AutoCI 연속 학습 시작")
    
    args = []
    if low:
        args.append("low")
    elif hours and memory:
        args.extend([str(hours), str(memory)])
    
    run_command("learn", args)


@cli.command()
@click.argument('args', nargs=-1)
def fix(args):
    """학습을 토대로 AI의 게임 엔진 능력 업데이트"""
    print("\n🔧 AutoCI 엔진 개선")
    run_command("fix", args)


@cli.command()
@click.option('--port', default=5001, help='모니터링 포트')
def monitor(port):
    """실시간 개발 모니터링"""
    print(f"\n📊 실시간 모니터링 (포트: {port})")
    run_command("monitor", ["--port", str(port)])


@cli.command()
def demo():
    """5분 빠른 데모"""
    print("\n🚀 AutoCI 빠른 데모 (5분)")
    run_command("demo", [])


@cli.command()
@click.argument('subcommand', default='insights')
def evolve(subcommand):
    """AI 진화 시스템 관리"""
    if subcommand == 'insights':
        print("\n🧬 진화 인사이트 조회 중...")
        run_command("evolve", ["insights"])
    else:
        print(f"❌ 알 수 없는 하위 명령어: {subcommand}")
        print("💡 사용법: autoci evolve insights")


@cli.command()
@click.argument('path')
def analyze(path):
    """게임 프로젝트 분석"""
    print(f"\n🔍 프로젝트 분석: {path}")
    run_command("analyze", ["--path", path])


if __name__ == "__main__":
    cli()