#!/usr/bin/env python3
"""
continuous_learning_system.py 실행을 위한 래퍼 스크립트
가상환경을 자동으로 활성화하고 실행합니다.
"""
import sys
import os
import subprocess
from pathlib import Path

# AutoCI 프로젝트 루트 디렉토리 찾기
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent  # /mnt/d/AutoCI/AutoCI
autoci_env = project_root / "autoci_env"
core_dir = project_root / "core"

# 가상환경이 활성화되어 있는지 확인
def is_venv_active():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

# 가상환경에서 실행되고 있지 않다면, 가상환경을 활성화하여 재실행
if not is_venv_active() and autoci_env.exists():
    # 가상환경의 Python 실행 파일 경로
    if os.name == 'nt':  # Windows
        python_exe = autoci_env / "Scripts" / "python.exe"
    else:  # Linux/Unix
        python_exe = autoci_env / "bin" / "python"
    
    if python_exe.exists():
        # 가상환경의 Python으로 continuous_learning_system.py 실행
        cmd = [str(python_exe), str(core_dir / "continuous_learning_system.py")] + sys.argv[1:]
        result = subprocess.run(cmd, cwd=str(project_root))
        sys.exit(result.returncode)

# 이미 가상환경이 활성화되어 있거나 가상환경이 없는 경우
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(core_dir))

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(str(project_root))

# continuous_learning_system.py 실행
try:
    from core.continuous_learning_system import main
    main()
except ImportError:
    # 직접 스크립트 실행
    cmd = [sys.executable, str(core_dir / "continuous_learning_system.py")] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)