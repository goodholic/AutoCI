#!/usr/bin/env python3
"""
AI 수정된 Godot 설정 및 실행 도우미
AutoCI와 AI 수정된 Godot을 연결하는 스크립트
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIGodotSetup:
    """AI 수정된 Godot 설정"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / ".godot_config.json"
        self.godot_paths = {
            # Windows 실행 파일 (WSL 경로)
            "windows": [
                # 빌드된 경로
                str(self.project_root / "godot_modified" / "bin" / "godot.windows.editor.x86_64.exe"),
                # 대체 경로들
                "/mnt/d/godot-modified/bin/godot.windows.editor.x86_64.exe",
                "/mnt/c/godot-modified/bin/godot.windows.editor.x86_64.exe",
                # 일반 Godot (임시)
                "/mnt/c/Program Files/Godot/Godot.exe",
                "/mnt/d/Godot/Godot.exe",
            ],
            # Linux 실행 파일 (개발용)
            "linux": [
                str(self.project_root / "godot_engine" / "Godot_v4.3-stable_linux.x86_64"),
                str(self.project_root / "godot_engine" / "godot"),
            ]
        }
    
    def find_godot(self, platform="windows") -> str:
        """Godot 실행 파일 찾기"""
        # 저장된 경로 확인
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                saved_path = config.get("godot_path")
                if saved_path and Path(saved_path).exists():
                    return saved_path
        
        # 경로 목록에서 찾기
        for path in self.godot_paths.get(platform, []):
            if Path(path).exists():
                logger.info(f"✅ Godot 찾음: {path}")
                return path
        
        return None
    
    def save_godot_path(self, path: str):
        """Godot 경로 저장"""
        config = {"godot_path": path, "ai_modified": True}
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Godot 경로 저장됨: {path}")
    
    async def setup_windows_godot(self):
        """Windows용 Godot 설정"""
        godot_path = self.find_godot("windows")
        
        if not godot_path:
            print("\n❌ Windows용 Godot을 찾을 수 없습니다.")
            print("\n🔧 설정 방법:")
            print("1. AI 수정된 Godot 빌드:")
            print("   python3 setup_custom_godot.py")
            print("\n2. 또는 일반 Godot 사용 (임시):")
            print("   - https://godotengine.org/download 에서 Windows 버전 다운로드")
            print("   - C:\\Program Files\\Godot\\ 또는 D:\\Godot\\에 설치")
            
            # 수동 경로 입력
            manual_path = input("\nGodot 경로를 입력하세요 (취소: Enter): ").strip()
            if manual_path:
                # Windows 경로를 WSL 경로로 변환
                if manual_path.startswith("C:\\"):
                    wsl_path = "/mnt/c" + manual_path[2:].replace('\\', '/')
                elif manual_path.startswith("D:\\"):
                    wsl_path = "/mnt/d" + manual_path[2:].replace('\\', '/')
                else:
                    wsl_path = manual_path
                
                if Path(wsl_path).exists():
                    self.save_godot_path(wsl_path)
                    return wsl_path
                else:
                    print(f"❌ 파일을 찾을 수 없습니다: {wsl_path}")
                    return None
        
        return godot_path
    
    async def test_godot_launch(self, godot_path: str):
        """Godot 실행 테스트"""
        print(f"\n🚀 Godot 실행 테스트: {godot_path}")
        
        # Windows 경로로 변환
        win_path = godot_path
        if godot_path.startswith("/mnt/c/"):
            win_path = "C:\\" + godot_path[7:].replace('/', '\\')
        elif godot_path.startswith("/mnt/d/"):
            win_path = "D:\\" + godot_path[7:].replace('/', '\\')
        
        try:
            # cmd.exe를 통해 실행
            subprocess.Popen([
                "cmd.exe", "/c", "start", "", win_path, "--help"
            ])
            print("✅ Godot 실행 성공!")
            return True
        except Exception as e:
            print(f"❌ Godot 실행 실패: {e}")
            return False
    
    async def create_launcher_script(self):
        """Godot 실행 스크립트 생성"""
        launcher_path = self.project_root / "launch_ai_godot.sh"
        
        godot_path = self.find_godot("windows")
        if not godot_path:
            return
        
        # Windows 경로 변환
        win_path = godot_path
        if godot_path.startswith("/mnt/c/"):
            win_path = "C:\\" + godot_path[7:].replace('/', '\\')
        elif godot_path.startswith("/mnt/d/"):
            win_path = "D:\\" + godot_path[7:].replace('/', '\\')
        
        launcher_content = f"""#!/bin/bash
# AI 수정된 Godot 실행 스크립트

echo "🚀 AI 수정된 Godot 실행 중..."

# Godot 경로
GODOT_PATH="{win_path}"

# 프로젝트 경로 (선택사항)
PROJECT_PATH="$1"

if [ -z "$PROJECT_PATH" ]; then
    echo "프로젝트 없이 Godot 실행"
    cmd.exe /c start "" "$GODOT_PATH"
else
    echo "프로젝트와 함께 Godot 실행: $PROJECT_PATH"
    cmd.exe /c start "" "$GODOT_PATH" --path "$PROJECT_PATH"
fi
"""
        
        launcher_path.write_text(launcher_content)
        os.chmod(launcher_path, 0o755)
        print(f"✅ 실행 스크립트 생성됨: {launcher_path}")

async def main():
    """메인 함수"""
    print("🎮 AI 수정된 Godot 설정")
    print("=" * 60)
    
    setup = AIGodotSetup()
    
    # Windows Godot 설정
    godot_path = await setup.setup_windows_godot()
    
    if godot_path:
        print(f"\n✅ Godot 경로: {godot_path}")
        
        # 실행 테스트
        test = input("\nGodot 실행을 테스트하시겠습니까? (y/N): ")
        if test.lower() == 'y':
            await setup.test_godot_launch(godot_path)
        
        # 실행 스크립트 생성
        await setup.create_launcher_script()
        
        print("\n🎉 설정 완료!")
        print("이제 autoci를 실행하면 AI 수정된 Godot이 자동으로 열립니다.")
    else:
        print("\n❌ Godot 설정 실패")
        print("AI 수정된 Godot을 빌드하거나 일반 Godot을 설치해주세요.")

if __name__ == "__main__":
    asyncio.run(main())