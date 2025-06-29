#!/usr/bin/env python3
"""
간단한 AI Godot 빌드 테스트
"""
import os
import sys
import json
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime

class SimpleGodotBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "godot_ai_build"
        self.log_file = self.project_root / "simple_build.log"
        
    def log(self, message):
        """로그 메시지 출력 및 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    
    def check_environment(self):
        """환경 체크"""
        self.log("🔍 환경 체크 시작...")
        
        # Python 버전
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.log(f"Python 버전: {py_version}")
        
        # Git 체크
        has_git = shutil.which("git") is not None
        self.log(f"Git: {'✅ 설치됨' if has_git else '❌ 없음'}")
        
        # 디렉토리 생성
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"빌드 디렉토리: {self.build_dir}")
        
        return has_git
    
    def download_godot_sample(self):
        """Godot 샘플 다운로드 (테스트용)"""
        self.log("📥 Godot 바이너리 다운로드 시도...")
        
        # Windows용 Godot 4.3 다운로드 URL
        godot_url = "https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_win64.exe.zip"
        zip_path = self.build_dir / "godot.zip"
        
        try:
            # 이미 다운로드되어 있는지 확인
            output_exe = self.build_dir / "Godot_v4.3-stable_win64.exe"
            if output_exe.exists():
                self.log("✅ Godot이 이미 다운로드되어 있습니다.")
                return True
            
            self.log(f"다운로드 중: {godot_url}")
            
            # 다운로드 진행률 표시
            def download_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(block_num * block_size * 100 / total_size, 100)
                    mb_downloaded = block_num * block_size / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    sys.stdout.write(f"\r진행률: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(godot_url, zip_path, reporthook=download_progress)
            print()  # 줄바꿈
            
            self.log("📦 압축 해제 중...")
            
            # ZIP 파일 해제
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.build_dir)
            
            # 정리
            zip_path.unlink()
            
            self.log("✅ Godot 다운로드 완료!")
            return True
            
        except Exception as e:
            self.log(f"❌ 다운로드 실패: {e}")
            return False
    
    def create_ai_config(self):
        """AI 설정 파일 생성"""
        self.log("📝 AI 설정 파일 생성...")
        
        config = {
            "ai_enabled": True,
            "ai_port": 9999,
            "ai_features": {
                "script_injection": True,
                "scene_manipulation": True,
                "real_time_control": True,
                "command_execution": True
            },
            "build_info": {
                "date": datetime.now().isoformat(),
                "version": "1.0.0-ai",
                "builder": "AutoCI Simple Builder"
            }
        }
        
        config_path = self.build_dir / "ai_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        self.log(f"✅ AI 설정 저장: {config_path}")
        
        # .godot_config.json 업데이트
        godot_config_path = self.project_root / ".godot_config.json"
        godot_exe = self.build_dir / "Godot_v4.3-stable_win64.exe"
        
        godot_config = {
            "godot_path": str(godot_exe),
            "is_ai_modified": True,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(godot_config_path, "w", encoding="utf-8") as f:
            json.dump(godot_config, f, indent=2)
        
        self.log(f"✅ Godot 설정 저장: {godot_config_path}")
    
    def create_launch_script(self):
        """실행 스크립트 생성"""
        self.log("🚀 실행 스크립트 생성...")
        
        # Windows 배치 파일
        bat_content = f"""@echo off
echo AI Godot 실행 중...
cd /d "{self.build_dir}"
start Godot_v4.3-stable_win64.exe
"""
        
        bat_path = self.build_dir / "launch_ai_godot.bat"
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat_content)
        
        self.log(f"✅ 실행 스크립트: {bat_path}")
    
    def run(self):
        """빌드 실행"""
        self.log("=" * 50)
        self.log("🚀 간단한 AI Godot 설정 시작")
        self.log("=" * 50)
        
        try:
            # 1. 환경 체크
            if not self.check_environment():
                self.log("❌ 환경 체크 실패")
                return False
            
            # 2. Godot 다운로드
            if not self.download_godot_sample():
                self.log("❌ Godot 다운로드 실패")
                return False
            
            # 3. AI 설정 생성
            self.create_ai_config()
            
            # 4. 실행 스크립트 생성
            self.create_launch_script()
            
            self.log("=" * 50)
            self.log("✅ 설정 완료!")
            self.log(f"Godot 위치: {self.build_dir / 'Godot_v4.3-stable_win64.exe'}")
            self.log("이제 autoci 명령어를 사용할 수 있습니다.")
            self.log("=" * 50)
            
            return True
            
        except Exception as e:
            self.log(f"❌ 오류 발생: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

def main():
    builder = SimpleGodotBuilder()
    success = builder.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())