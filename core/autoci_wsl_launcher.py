#!/usr/bin/env python3
"""
AutoCI WSL 가상환경 자동 실행기
WSL 환경에서 가상환경을 자동으로 활성화하고 AutoCI를 실행
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def is_wsl():
    """WSL 환경인지 확인"""
    if platform.system() != "Linux":
        return False
    
    # WSL 특징적인 파일 확인
    wsl_indicators = [
        "/proc/sys/fs/binfmt_misc/WSLInterop",
        "/proc/version"
    ]
    
    for indicator in wsl_indicators:
        if os.path.exists(indicator):
            if indicator == "/proc/version":
                with open(indicator, 'r') as f:
                    if 'microsoft' in f.read().lower():
                        return True
            else:
                return True
    
    return False


def find_project_root():
    """프로젝트 루트 디렉토리 찾기"""
    current_path = Path(__file__).resolve().parent
    
    # 상위 디렉토리로 올라가며 프로젝트 루트 찾기
    while current_path != current_path.parent:
        if (current_path / "autoci_env").exists() or (current_path / "requirements.txt").exists():
            return current_path
        current_path = current_path.parent
    
    # 기본값: 현재 스크립트의 상위 디렉토리
    return Path(__file__).resolve().parent.parent


def activate_venv_and_run():
    """가상환경 활성화 후 AutoCI 실행"""
    project_root = find_project_root()
    venv_path = project_root / "autoci_env"
    
    # 가상환경 확인
    if not venv_path.exists():
        print("⚠️  가상환경이 없습니다. 생성 중...")
        create_venv(project_root)
    
    # 가상환경의 Python 실행 파일 경로
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # 가상환경 Python이 존재하는지 확인
    if not python_exe.exists():
        print(f"❌ 가상환경 Python을 찾을 수 없습니다: {python_exe}")
        sys.exit(1)
    
    # 환경 변수 설정
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)
    env["PATH"] = f"{venv_path / 'bin' if platform.system() != 'Windows' else venv_path / 'Scripts'}{os.pathsep}{env['PATH']}"
    
    # Python 경로에서 가상환경 제거 (충돌 방지)
    if "PYTHONHOME" in env:
        del env["PYTHONHOME"]
    
    print(f"🚀 AutoCI 시작 중...")
    print(f"📁 프로젝트 경로: {project_root}")
    print(f"🐍 Python 경로: {python_exe}")
    print(f"💻 환경: {'WSL' if is_wsl() else platform.system()}")
    print("-" * 50)
    
    # 실행 권한 설정 (WSL에서 필요)
    if platform.system() == "Linux":
        scripts_to_chmod = [
            project_root / "core" / "autoci.py",
            project_root / "core" / "autoci_main.py",
            project_root / "core" / "panda3d_terminal.py",
            project_root / "core" / "autoci_terminal.py",
            project_root / "core" / "autoci_command.py"
        ]
        
        for script in scripts_to_chmod:
            if script.exists():
                try:
                    script.chmod(0o755)
                except:
                    pass
    
    # AutoCI 실행
    try:
        # core/autoci.py 실행
        autoci_script = project_root / "core" / "autoci.py"
        if not autoci_script.exists():
            # 대체 경로 시도
            autoci_script = project_root / "autoci.py"
        
        if autoci_script.exists():
            # 명령줄 인수 전달
            cmd = [str(python_exe), str(autoci_script)] + sys.argv[1:]
            subprocess.run(cmd, env=env, cwd=str(project_root))
        else:
            print(f"❌ AutoCI 스크립트를 찾을 수 없습니다: {autoci_script}")
            # autoci_terminal.py 직접 실행 시도
            terminal_script = project_root / "core" / "autoci_terminal.py"
            if terminal_script.exists():
                cmd = [str(python_exe), str(terminal_script)] + sys.argv[1:]
                subprocess.run(cmd, env=env, cwd=str(project_root))
            else:
                print("❌ AutoCI 실행 파일을 찾을 수 없습니다.")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\n👋 AutoCI가 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


def create_venv(project_root):
    """가상환경 생성 및 설정"""
    venv_path = project_root / "autoci_env"
    
    print("🔧 가상환경 생성 중...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    
    # pip 업그레이드
    if platform.system() == "Windows":
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"
    
    print("📦 pip 업그레이드 중...")
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    
    # requirements.txt가 있으면 패키지 설치
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print("📚 패키지 설치 중...")
        subprocess.run([str(pip_exe), "install", "-r", str(requirements_file)], check=True)
    
    print("✅ 가상환경 설정 완료!")


def setup_wsl_autoci_command():
    """WSL에서 전역 autoci 명령어 설정"""
    if not is_wsl():
        print("ℹ️  WSL 환경이 아닙니다.")
        return
    
    # 현재 스크립트 경로
    launcher_path = Path(__file__).resolve()
    
    # /usr/local/bin에 심볼릭 링크 생성
    target_path = Path("/usr/local/bin/autoci")
    
    try:
        # 기존 링크 제거
        if target_path.exists() or target_path.is_symlink():
            subprocess.run(["sudo", "rm", "-f", str(target_path)], check=True)
        
        # 실행 스크립트 생성
        script_content = f"""#!/bin/bash
# AutoCI WSL Launcher
cd {launcher_path.parent.parent}
{sys.executable} {launcher_path} "$@"
"""
        
        # 임시 스크립트 파일 생성
        temp_script = Path("/tmp/autoci_launcher.sh")
        temp_script.write_text(script_content)
        temp_script.chmod(0o755)
        
        # /usr/local/bin으로 복사
        subprocess.run(["sudo", "cp", str(temp_script), str(target_path)], check=True)
        subprocess.run(["sudo", "chmod", "+x", str(target_path)], check=True)
        
        print(f"✅ 전역 'autoci' 명령어가 설정되었습니다!")
        print(f"📍 이제 어디서든 'autoci' 명령어를 사용할 수 있습니다.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 명령어 설정 실패: {e}")
        print("💡 수동으로 설정하려면:")
        print(f"   sudo ln -sf {launcher_path} /usr/local/bin/autoci")


def main():
    """메인 함수"""
    # --setup 옵션 처리
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        print("🔧 AutoCI WSL 설정 중...")
        setup_wsl_autoci_command()
        return
    
    # WSL 환경 확인
    if is_wsl():
        print("🐧 WSL 환경 감지됨")
    
    # 가상환경 활성화 및 AutoCI 실행
    activate_venv_and_run()


if __name__ == "__main__":
    main()