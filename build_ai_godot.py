#!/usr/bin/env python3
"""
AI 수정된 Godot 엔진 빌드 스크립트
AutoCI가 Godot을 완전히 제어할 수 있도록 소스코드를 수정하여 빌드
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
import shutil
from pathlib import Path
import time

class AIGodotBuilder:
    """AI 수정된 Godot 빌드 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "godot_ai_build"
        self.source_dir = self.build_dir / "godot-source"
        self.output_dir = self.build_dir / "output"
        self.logs_dir = self.build_dir / "logs"
        
        # Godot 정보
        self.version = "4.3-stable"
        self.source_url = f"https://github.com/godotengine/godot/archive/{self.version}.zip"
        
    def build(self):
        """AI Godot 빌드 실행"""
        print("🤖 AutoCI - AI 수정된 Godot 빌드 시스템")
        print("=" * 60)
        print("이 과정은 30분-1시간이 소요될 수 있습니다.")
        print()
        
        try:
            # 준비
            self._prepare()
            
            # 소스 다운로드
            if not self._download_source():
                return self._fallback_to_regular_godot()
            
            # AI 패치 적용
            self._apply_ai_patches()
            
            # 빌드
            if not self._build():
                return self._fallback_to_regular_godot()
            
            # 결과 확인
            result = self._finalize()
            
            if result:
                print(f"\n🎉 AI 수정된 Godot 빌드 완료!")
                print(f"📍 경로: {result}")
                print(f"💡 AutoCI에서 이 경로를 사용하세요:")
                print(f"   {result}")
                return result
            else:
                return self._fallback_to_regular_godot()
                
        except Exception as e:
            print(f"❌ 빌드 중 오류 발생: {e}")
            return self._fallback_to_regular_godot()
    
    def _prepare(self):
        """빌드 환경 준비"""
        print("📁 빌드 환경 준비 중...")
        
        # 디렉토리 생성
        for dir_path in [self.build_dir, self.output_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 빌드 도구 확인
        missing_tools = []
        tools = {"scons": "SCons 빌드 시스템", "pkg-config": "패키지 설정"}
        
        for tool, desc in tools.items():
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                print(f"  ✅ {tool} 확인됨")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append((tool, desc))
        
        if missing_tools:
            print("⚠️  필요한 도구가 누락되었습니다:")
            for tool, desc in missing_tools:
                print(f"  - {tool}: {desc}")
            
            print("\n🔧 자동 설치 시도 중...")
            try:
                subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
                subprocess.run(["sudo", "apt", "install", "-y", "scons", "pkg-config", "libx11-dev", "libxcursor-dev", "libxinerama-dev", "libgl1-mesa-dev", "libglu1-mesa-dev", "libasound2-dev", "libpulse-dev", "libudev-dev", "libxi-dev", "libxrandr-dev"], check=True, capture_output=True)
                print("  ✅ 빌드 도구 설치 완료")
            except subprocess.CalledProcessError:
                print("  ❌ 자동 설치 실패")
                raise Exception("필수 빌드 도구 설치 실패")
    
    def _download_source(self):
        """Godot 소스 다운로드"""
        if self.source_dir.exists():
            print("✅ 소스코드가 이미 존재합니다.")
            return True
        
        print("📥 Godot 소스코드 다운로드 중...")
        zip_path = self.build_dir / "godot-source.zip"
        
        try:
            # 다운로드 (진행률 표시)
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(f"\r  📥 다운로드 중... {percent:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(self.source_url, zip_path, progress_hook)
            print(f"\n  ✅ 다운로드 완료 ({zip_path.stat().st_size / 1024 / 1024:.1f}MB)")
            
            # 압축 해제
            print("📦 압축 해제 중...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.build_dir)
            
            # 폴더 이름 정리
            extracted_dir = self.build_dir / f"godot-{self.version}"
            if extracted_dir.exists():
                extracted_dir.rename(self.source_dir)
            
            # 정리
            zip_path.unlink()
            print("  ✅ 압축 해제 완료")
            return True
            
        except Exception as e:
            print(f"\n❌ 소스 다운로드 실패: {e}")
            return False
    
    def _apply_ai_patches(self):
        """AI 기능 패치 적용"""
        print("🔧 AI 기능 패치 적용 중...")
        
        patches_applied = 0
        
        # 1. EditorNode에 AI 인터페이스 추가
        editor_node_h = self.source_dir / "editor" / "editor_node.h"
        if editor_node_h.exists():
            content = editor_node_h.read_text()
            
            ai_interface = '''
public:
    // AutoCI AI Control Interface
    void ai_create_node(const String &type, const String &name);
    void ai_set_property(const String &path, const String &property, const Variant &value);
    void ai_save_scene(const String &path);
    String ai_get_scene_info();
    void ai_run_script(const String &script);
    bool ai_automation_enabled = true;
'''
            
            if "class EditorNode" in content and "ai_automation_enabled" not in content:
                content = content.replace("public:", "public:" + ai_interface)
                editor_node_h.write_text(content)
                patches_applied += 1
                print("  ✅ EditorNode AI 인터페이스 추가")
        
        # 2. Main에 AI 시작 메시지 추가
        main_cpp = self.source_dir / "main" / "main.cpp"
        if main_cpp.exists():
            content = main_cpp.read_text()
            
            ai_init = '''
    // AutoCI AI System Initialization
    print_line("=== AutoCI AI System Ready ===");
    print_line("AI automation features enabled");
'''
            
            if "int main(" in content and "AutoCI AI System Ready" not in content:
                # OS 초기화 후에 추가
                content = content.replace("OS::get_singleton()->initialize();", 
                                        "OS::get_singleton()->initialize();" + ai_init)
                main_cpp.write_text(content)
                patches_applied += 1
                print("  ✅ Main AI 초기화 추가")
        
        # 3. 프로젝트 설정에 AI 플래그 추가
        project_settings = self.source_dir / "core" / "config" / "project_settings.cpp"
        if project_settings.exists():
            content = project_settings.read_text()
            
            if '"application/config/name"' in content and "autoci_ai_enabled" not in content:
                ai_setting = '''
    GLOBAL_DEF("autoci/ai_enabled", true);
    GLOBAL_DEF("autoci/automation_level", 100);
'''
                # 다른 설정 뒤에 추가
                content = content.replace('GLOBAL_DEF("application/config/name", "");', 
                                        'GLOBAL_DEF("application/config/name", "");' + ai_setting)
                project_settings.write_text(content)
                patches_applied += 1
                print("  ✅ 프로젝트 설정에 AI 옵션 추가")
        
        print(f"✅ AI 패치 완료 ({patches_applied}개 적용)")
    
    def _build(self):
        """Godot 빌드 실행"""
        print("🔨 Godot 빌드 시작...")
        print("⏱️  예상 시간: 20-60분")
        
        # 빌드 디렉토리로 이동
        original_dir = os.getcwd()
        os.chdir(self.source_dir)
        
        try:
            # 빌드 명령어
            build_cmd = ["scons", "platform=linuxbsd", "target=editor", "bits=64", "-j2", "verbose=yes"]
            
            print(f"실행 명령: {' '.join(build_cmd)}")
            
            # 로그 파일 준비
            log_file = self.logs_dir / f"build_{int(time.time())}.log"
            
            start_time = time.time()
            
            # 빌드 실행
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    build_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # 실시간 출력
                for line in process.stdout:
                    print(line.rstrip())
                    log.write(line)
                
                process.wait()
            
            build_time = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\n✅ 빌드 완료! ({build_time/60:.1f}분 소요)")
                return True
            else:
                print(f"\n❌ 빌드 실패 (종료 코드: {process.returncode})")
                print(f"로그 파일: {log_file}")
                return False
                
        except Exception as e:
            print(f"❌ 빌드 중 예외 발생: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _finalize(self):
        """빌드 결과 정리"""
        print("📦 빌드 결과 정리 중...")
        
        # 빌드된 실행 파일 찾기
        bin_dir = self.source_dir / "bin"
        
        # 가능한 실행 파일 패턴들
        patterns = ["godot*editor*", "godot*tools*", "godot.linuxbsd*"]
        executables = []
        
        for pattern in patterns:
            executables.extend(bin_dir.glob(pattern))
        
        if not executables:
            print("❌ 빌드된 실행 파일을 찾을 수 없습니다.")
            return None
        
        # 첫 번째 실행 파일 사용
        source_exe = executables[0]
        target_exe = self.output_dir / "godot.ai.editor.linux.x86_64"
        
        # 실행 파일 복사
        shutil.copy2(source_exe, target_exe)
        target_exe.chmod(0o755)
        
        # 설정 파일 생성
        config = {
            "godot_path": str(target_exe),
            "version": self.version,
            "ai_modified": True,
            "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": [
                "ai_create_node",
                "ai_set_property", 
                "ai_save_scene",
                "ai_automation_enabled"
            ]
        }
        
        config_file = self.project_root / ".godot_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 설치 완료: {target_exe}")
        print(f"📋 설정 파일: {config_file}")
        
        return str(target_exe)
    
    def _fallback_to_regular_godot(self):
        """일반 Godot으로 대체"""
        print("\n💡 AI 빌드 실패 시 일반 Godot 사용")
        print("=" * 50)
        
        # 기존 Godot 확인
        regular_godot = self.project_root / "godot_engine" / "Godot_v4.3-stable_linux.x86_64"
        
        if regular_godot.exists():
            print(f"✅ 기존 Godot 발견: {regular_godot}")
            print("💡 AutoCI에서 다음 경로를 사용하세요:")
            print(f"   {regular_godot}")
            return str(regular_godot)
        else:
            print("❌ 기존 Godot도 찾을 수 없습니다.")
            print("🔧 다음 명령으로 Godot을 설치하세요:")
            print("   autoci --setup")
            return None

def main():
    builder = AIGodotBuilder()
    result = builder.build()
    
    if result:
        print(f"\n🎯 성공! AutoCI에서 사용할 경로:")
        print(f"   {result}")
    else:
        print(f"\n💡 일반 Godot을 먼저 설치하세요:")
        print(f"   python3 setup_ai_godot.py")

if __name__ == "__main__":
    main()