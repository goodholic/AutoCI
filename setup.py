#!/usr/bin/env python3
"""
AutoCI 설치 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """설치 메인 함수"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║         AutoCI 설치 프로그램 v5.0                     ║
    ║                                                       ║
    ║   AI가 직접 Panda3D로 게임을 만드는 시스템            ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Python 버전 체크
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        sys.exit(1)
    
    print("✅ Python 버전 확인 완료")
    
    # 가상환경 생성
    print("\n📦 가상환경 생성 중...")
    if not Path("autoci_env").exists():
        subprocess.run([sys.executable, "-m", "venv", "autoci_env"])
        print("✅ 가상환경 생성 완료")
    else:
        print("✅ 기존 가상환경 사용")
    
    # pip 업그레이드
    print("\n📦 pip 업그레이드 중...")
    if os.name == 'nt':  # Windows
        pip_path = Path("autoci_env/Scripts/pip")
    else:  # Linux/Mac
        pip_path = Path("autoci_env/bin/pip")
    
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    
    # 필수 패키지 설치
    print("\n📦 필수 패키지 설치 중...")
    print("(이 작업은 몇 분 정도 걸릴 수 있습니다)")
    
    # 기본 패키지 먼저 설치
    basic_packages = [
        "wheel",
        "setuptools",
        "numpy",
        "pillow",
        "flask",
        "psutil"
    ]
    
    for package in basic_packages:
        print(f"  - {package} 설치 중...")
        subprocess.run([str(pip_path), "install", package], capture_output=True)
    
    # Panda3D 설치
    print("\n🎮 Panda3D 엔진 설치 중...")
    subprocess.run([str(pip_path), "install", "panda3d"], capture_output=True)
    
    # requirements.txt 설치
    print("\n📦 전체 패키지 설치 중...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    # 폴더 구조 생성
    print("\n📁 폴더 구조 생성 중...")
    folders = [
        "game_projects",
        "logs_current",
        "data/learning",
        "data/evolution",
        "data/feedback",
        "models_ai",
        "archive/old_files",
        "archive/legacy_code",
        "archive/old_logs"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✅ 폴더 구조 생성 완료")
    
    # 실행 권한 설정 (Linux/Mac)
    if os.name != 'nt':
        os.chmod("autoci", 0o755)
    
    # 설치 완료
    print("\n✅ AutoCI 설치 완료!")
    print("\n사용 방법:")
    print("1. 가상환경 활성화:")
    if os.name == 'nt':
        print("   autoci_env\\Scripts\\activate.bat")
    else:
        print("   source autoci_env/bin/activate")
    print("\n2. AutoCI 실행:")
    print("   python autoci")
    print("   또는")
    print("   ./autoci  (Linux/Mac)")
    print("\n3. 도움말:")
    print("   python autoci --help")
    
    # AI 모델 설치 안내
    print("\n📌 AI 모델 설치 (선택사항):")
    print("   python install_llm_models.py")
    print("\n⚠️  AI 모델은 많은 디스크 공간(20-100GB)을 사용합니다.")
    print("   필요한 경우에만 설치하세요.")


if __name__ == "__main__":
    main()