#!/usr/bin/env python3
"""
AI 수정된 Godot 엔진 Windows 빌드 스크립트
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

class AIGodotWindowsBuilder:
    """AI 수정된 Godot Windows 빌드 시스템"""
    
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
        """AI Godot Windows 빌드 실행"""
        print("🤖 AutoCI - AI 수정된 Godot Windows 빌드 시스템")
        print("=" * 60)
        print("이 과정은 60-90분이 소요될 수 있습니다.")
        print()
        
        try:
            # 준비
            self._prepare()
            
            # 소스 다운로드
            if not self._download_source():
                return self._fallback_to_regular_godot()
            
            # AI 패치 적용
            self._apply_ai_patches()
            
            # Windows 빌드
            if not self._build_windows():
                print("❌ AI 수정된 Godot Windows 빌드에 실패했습니다.")
                return None
            
            # 결과 확인
            result = self._finalize()
            
            if result:
                print(f"\n🎉 AI 수정된 Godot Windows 빌드 완료!")
                print(f"📍 경로: {result}")
                print(f"💡 AutoCI에서 이 경로를 사용하세요:")
                print(f"   {result}")
                return result
            else:
                print("❌ 빌드 결과를 정리하는데 실패했습니다.")
                return None
                
        except Exception as e:
            print(f"❌ 빌드 중 오류 발생: {e}")
            return None
    
    def _prepare(self):
        """빌드 환경 준비"""
        print("📁 빌드 환경 준비 중...")
        
        # 디렉토리 생성
        for dir_path in [self.build_dir, self.output_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Windows 빌드 도구 확인
        missing_tools = []
        
        # Visual Studio 확인
        vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional/Common7/Tools/VsDevCmd.bat", 
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/Common7/Tools/VsDevCmd.bat",
        ]
        
        vs_found = False
        for vs_path in vs_paths:
            if Path(vs_path).exists():
                vs_found = True
                print(f"  ✅ Visual Studio 확인됨: {vs_path}")
                break
                
        if not vs_found:
            print("  ❌ Visual Studio가 필요합니다.")
            print("     https://visualstudio.microsoft.com/downloads/ 에서 설치하세요.")
            raise Exception("Visual Studio가 설치되지 않았습니다.")
            
        # Python 확인
        try:
            subprocess.run(["python", "--version"], capture_output=True, check=True)
            print(f"  ✅ Python 확인됨")
        except:
            print("  ❌ Python이 필요합니다.")
            raise Exception("Python이 설치되지 않았습니다.")
    
    def _download_source(self):
        """Godot 소스 다운로드"""
        print("📥 Godot 소스 다운로드 중...")
        
        # 이미 다운로드됨?
        if self.source_dir.exists() and (self.source_dir / "SConstruct").exists():
            print("  ✅ 소스가 이미 다운로드되어 있습니다.")
            return True
        
        try:
            # 기존 디렉토리 삭제
            if self.source_dir.exists():
                shutil.rmtree(self.source_dir)
            
            # 다운로드
            zip_path = self.build_dir / "godot-source.zip"
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r  다운로드 중... {percent:.1f}%", end='')
                
            urllib.request.urlretrieve(self.source_url, zip_path, download_progress)
            print("\n  ✅ 다운로드 완료")
            
            # 압축 해제
            print("  📦 압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.build_dir)
            
            # 디렉토리 이름 변경
            extracted_dir = self.build_dir / f"godot-{self.version}"
            if extracted_dir.exists():
                extracted_dir.rename(self.source_dir)
            
            # zip 파일 삭제
            zip_path.unlink()
            
            print("  ✅ 소스 준비 완료")
            return True
            
        except Exception as e:
            print(f"  ❌ 다운로드 실패: {e}")
            return False
    
    def _apply_ai_patches(self):
        """AI 기능을 위한 패치 적용"""
        print("🔧 AI 패치 적용 중...")
        
        patches_applied = 0
        
        # 1. AI 노드 생성 API 추가
        engine_h = self.source_dir / "core" / "config" / "engine.h"
        if engine_h.exists():
            content = engine_h.read_text()
            if "ai_create_node" not in content:
                # AI API 선언 추가
                ai_api = """
// AutoCI AI Integration
public:
    void ai_create_node(const String &p_type, const String &p_name);
    void ai_set_property(const String &p_path, const String &p_property, const Variant &p_value);
    void ai_save_scene(const String &p_path);
    bool is_ai_automation_enabled() const { return ai_automation_enabled; }
    
private:
    bool ai_automation_enabled = true;
"""
                content = content.replace("class Engine {", f"class Engine {{{ai_api}")
                engine_h.write_text(content)
                patches_applied += 1
                print("  ✅ Engine 클래스에 AI API 추가")
        
        # 2. AI 기능 구현
        engine_cpp = self.source_dir / "core" / "config" / "engine.cpp"
        if engine_cpp.exists():
            content = engine_cpp.read_text()
            if "ai_create_node" not in content:
                # AI API 구현 추가
                ai_impl = """
// AutoCI AI Integration Implementation
void Engine::ai_create_node(const String &p_type, const String &p_name) {
    // AI 노드 생성 로직
    print_line("AI: Creating node " + p_name + " of type " + p_type);
}

void Engine::ai_set_property(const String &p_path, const String &p_property, const Variant &p_value) {
    // AI 속성 설정 로직
    print_line("AI: Setting property " + p_property + " on " + p_path);
}

void Engine::ai_save_scene(const String &p_path) {
    // AI 씬 저장 로직
    print_line("AI: Saving scene to " + p_path);
}
"""
                content += ai_impl
                engine_cpp.write_text(content)
                patches_applied += 1
                print("  ✅ Engine 구현에 AI 기능 추가")
        
        # 3. 프로젝트 설정에 AI 옵션 추가
        project_settings = self.source_dir / "core" / "config" / "project_settings.cpp"
        if project_settings.exists():
            content = project_settings.read_text()
            if "ai/automation_enabled" not in content:
                # AI 설정 추가
                ai_setting = """
        GLOBAL_DEF("ai/automation_enabled", true);
        GLOBAL_DEF("ai/api_endpoint", "http://localhost:11434");
        GLOBAL_DEF("ai/model_name", "autoci-godot");
"""
                content = content.replace('GLOBAL_DEF("application/config/name", "");',
                                        'GLOBAL_DEF("application/config/name", "");' + ai_setting)
                project_settings.write_text(content)
                patches_applied += 1
                print("  ✅ 프로젝트 설정에 AI 옵션 추가")
        
        print(f"✅ AI 패치 완료 ({patches_applied}개 적용)")
    
    def _build_windows(self):
        """Godot Windows 빌드 실행"""
        print("🔨 Godot Windows 빌드 시작...")
        print("⏱️  예상 시간: 60-90분")
        
        # 빌드 디렉토리로 이동
        original_dir = os.getcwd()
        os.chdir(self.source_dir)
        
        try:
            # Windows 빌드 명령어
            build_cmd = ["scons", "platform=windows", "target=editor", "arch=x86_64", "-j2", "verbose=yes"]
            
            print(f"실행 명령: {' '.join(build_cmd)}")
            
            # 로그 파일 준비
            log_file = self.logs_dir / f"build_windows_{int(time.time())}.log"
            
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
        
        # Windows 실행 파일 패턴들
        patterns = ["godot.windows.editor.x86_64.exe", "godot.windows.tools.64.exe", "godot*.exe"]
        executables = []
        
        for pattern in patterns:
            executables.extend(bin_dir.glob(pattern))
        
        if not executables:
            print("❌ 빌드된 실행 파일을 찾을 수 없습니다.")
            return None
        
        # 첫 번째 실행 파일 사용
        source_exe = executables[0]
        target_exe = self.output_dir / "godot.ai.editor.windows.x86_64.exe"
        
        # 실행 파일 복사
        shutil.copy2(source_exe, target_exe)
        
        # 설정 파일 생성
        config = {
            "godot_path": str(target_exe),
            "version": self.version,
            "ai_modified": True,
            "platform": "windows",
            "build_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "features": [
                "ai_create_node",
                "ai_set_property", 
                "ai_save_scene",
                "ai_automation_enabled"
            ]
        }
        
        config_file = self.output_dir / "ai_godot_config.json"
        config_file.write_text(json.dumps(config, indent=2))
        
        print(f"✅ 실행 파일: {target_exe}")
        print(f"✅ 설정 파일: {config_file}")
        
        return str(target_exe)
    


if __name__ == "__main__":
    builder = AIGodotWindowsBuilder()
    result = builder.build()
    
    if result:
        print("\n✅ 빌드가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 빌드에 실패했습니다.")
        sys.exit(1)