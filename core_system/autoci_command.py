#!/usr/bin/env python3
"""
AutoCI 명령 처리
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    # 프로젝트 루트 디렉토리로 이동
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Python 경로 확인
    python_cmd = sys.executable
    
    # 터미널 인터페이스 직접 실행
    try:
        terminal_script = script_dir / "autoci_terminal.py"
        subprocess.run([python_cmd, str(terminal_script)], check=True)
    except KeyboardInterrupt:
        print("\n\nAutoCI가 종료되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()