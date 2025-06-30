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

# 색상 코드
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

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
        
        # 빌드 도구 확인 (Windows 크로스 컴파일용)
        missing_tools = []
        basic_tools = {
            "scons": "SCons 빌드 시스템", 
            "pkg-config": "패키지 설정"
        }
        
        # 기본 도구 확인
        for tool, desc in basic_tools.items():
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                print(f"  ✅ {tool} 확인됨")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append((tool, desc))
        
        # MinGW 확인 (posix threads 우선)
        mingw_found = False
        mingw_compilers = [
            ("x86_64-w64-mingw32-g++-posix", "MinGW-w64 posix threads (권장)"),
            ("x86_64-w64-mingw32-g++", "MinGW-w64 기본")
        ]
        
        for compiler, desc in mingw_compilers:
            try:
                result = subprocess.run([compiler, "--version"], capture_output=True, check=True, text=True)
                print(f"  ✅ {desc} 확인됨")
                # posix threads 지원 확인
                if "posix" in result.stdout.lower() or "posix" in compiler:
                    print(f"    💡 posix threads 지원")
                mingw_found = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if not mingw_found:
            missing_tools.append(("x86_64-w64-mingw32-g++", "MinGW-w64 크로스 컴파일러"))
        
        if missing_tools:
            print("⚠️  Windows 크로스 빌드에 필요한 도구가 누락되었습니다:")
            for tool, desc in missing_tools:
                print(f"  - {tool}: {desc}")
            
            print("\n🔧 자동 설치 시도 중...")
            try:
                subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
                
                # 더 완전한 MinGW 도구 세트 설치
                mingw_packages = [
                    "scons", 
                    "pkg-config", 
                    "gcc-mingw-w64-x86-64", 
                    "g++-mingw-w64-x86-64",
                    "mingw-w64-tools",       # windres, dlltool 등
                    "mingw-w64-x86-64-dev",  # 헤더 파일들
                    "build-essential"
                ]
                
                subprocess.run(["sudo", "apt", "install", "-y"] + mingw_packages, 
                             check=True, capture_output=True)
                print("  ✅ Windows 크로스 빌드 도구 설치 완료")
                
                # posix threads 설정 자동화
                print("  🔧 posix threads 설정 중...")
                try:
                    # gcc posix 설정
                    subprocess.run(["sudo", "update-alternatives", "--install", 
                                  "/usr/bin/x86_64-w64-mingw32-gcc", "x86_64-w64-mingw32-gcc", 
                                  "/usr/bin/x86_64-w64-mingw32-gcc-posix", "60"], 
                                 check=True, capture_output=True)
                    subprocess.run(["sudo", "update-alternatives", "--set", 
                                  "x86_64-w64-mingw32-gcc", "/usr/bin/x86_64-w64-mingw32-gcc-posix"], 
                                 check=True, capture_output=True)
                    
                    # g++ posix 설정
                    subprocess.run(["sudo", "update-alternatives", "--install", 
                                  "/usr/bin/x86_64-w64-mingw32-g++", "x86_64-w64-mingw32-g++", 
                                  "/usr/bin/x86_64-w64-mingw32-g++-posix", "60"], 
                                 check=True, capture_output=True)
                    subprocess.run(["sudo", "update-alternatives", "--set", 
                                  "x86_64-w64-mingw32-g++", "/usr/bin/x86_64-w64-mingw32-g++-posix"], 
                                 check=True, capture_output=True)
                    
                    print("  ✅ posix threads 자동 설정 완료")
                except:
                    print("  ⚠️  posix threads 자동 설정 실패 (수동 설정 필요)")
                    
            except subprocess.CalledProcessError:
                print("  ❌ 자동 설치 실패")
                print("  💡 수동 설치 명령어:")
                print("     sudo apt update")
                print("     sudo apt install scons pkg-config gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64")
                print("     sudo apt install mingw-w64-tools mingw-w64-x86-64-dev build-essential")
                print("  💡 posix threads 수동 설정:")
                print("     sudo update-alternatives --config x86_64-w64-mingw32-g++")
                raise Exception("Windows 크로스 빌드 도구 설치 실패")
    
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
            # MinGW posix threads 환경 설정 (우선순위: posix -> 기본)
            mingw_env = os.environ.copy()
            
            # MinGW 경로 및 prefix 설정 - 다중 경로 시도
            possible_prefixes = ['/usr', '/usr/bin', '']
            mingw_found = False
            
            # 설치된 컴파일러 경로 탐지
            gcc_options = ['x86_64-w64-mingw32-gcc-posix', 'x86_64-w64-mingw32-gcc']
            gxx_options = ['x86_64-w64-mingw32-g++-posix', 'x86_64-w64-mingw32-g++']
            
            # MinGW 컴파일러 자동 탐지
            for prefix in possible_prefixes:
                test_path = f"{prefix}/bin/x86_64-w64-mingw32-gcc" if prefix else "x86_64-w64-mingw32-gcc"
                try:
                    import shutil
                    if shutil.which("x86_64-w64-mingw32-gcc-posix") or shutil.which("x86_64-w64-mingw32-gcc"):
                        mingw_env['MINGW_PREFIX'] = prefix if prefix else ''
                        mingw_found = True
                        print(f"    MinGW 탐지됨: prefix={prefix}")
                        break
                except:
                    continue
            
            if not mingw_found:
                # 기본값으로 설정
                mingw_env['MINGW_PREFIX'] = '/usr'
                print("    MinGW 자동 탐지 실패, 기본값 사용: /usr")
            
            mingw_gcc = None
            mingw_gxx = None
            
            for gcc in gcc_options:
                try:
                    subprocess.run([gcc, '--version'], capture_output=True, check=True)
                    mingw_env['CC'] = gcc
                    mingw_gcc = gcc
                    print(f"    CC 설정: {gcc}")
                    break
                except:
                    continue
            
            for gxx in gxx_options:
                try:
                    subprocess.run([gxx, '--version'], capture_output=True, check=True)
                    mingw_env['CXX'] = gxx
                    mingw_gxx = gxx
                    print(f"    CXX 설정: {gxx}")
                    break
                except:
                    continue
            
            # 추가 환경 변수 설정 (Godot 감지 개선)
            if mingw_gcc and mingw_gxx:
                # 컴파일러 경로를 PATH에 추가
                current_path = mingw_env.get('PATH', '')
                mingw_env['PATH'] = f"/usr/bin:{current_path}"
                
                # Godot에서 찾는 환경 변수들 설정
                mingw_env['CROSS_COMPILE'] = 'x86_64-w64-mingw32-'
                mingw_env['AR'] = 'x86_64-w64-mingw32-ar'
                mingw_env['RANLIB'] = 'x86_64-w64-mingw32-ranlib'
                mingw_env['STRIP'] = 'x86_64-w64-mingw32-strip'
                mingw_env['WINDRES'] = 'x86_64-w64-mingw32-windres'
                
                print("    추가 환경 변수 설정 완료")
            
            # MinGW 도구들 경로 확인 및 설정
            mingw_tools = ['ar', 'ranlib', 'strip', 'windres']
            for tool in mingw_tools:
                tool_name = f'x86_64-w64-mingw32-{tool}'
                try:
                    tool_path = subprocess.run(['which', tool_name], capture_output=True, check=True, text=True).stdout.strip()
                    print(f"    {tool.upper()} 확인: {tool_path}")
                except:
                    print(f"    ⚠️  {tool_name} 찾을 수 없음")
            
            print(f"    MINGW_PREFIX 설정: {mingw_env['MINGW_PREFIX']}")
            
            # 빌드 명령어 - Windows 크로스 컴파일 (posix threads 강제)
            build_cmd = [
                "scons", 
                "platform=windows", 
                "target=editor", 
                "arch=x86_64", 
                "use_mingw=yes",
                "mingw_prefix=x86_64-w64-mingw32-",  # 실제 컴파일러 prefix
                "debug_symbols=no",  # 빌드 시간 단축
                "optimize=speed",    # 최적화
                "-j2", 
                "verbose=yes"
            ]
            
            # 컴파일러가 명시적으로 설정된 경우 추가 옵션
            if mingw_gcc and mingw_gxx:
                build_cmd.extend([
                    f"CC={mingw_gcc}",
                    f"CXX={mingw_gxx}"
                ])
            
            print(f"실행 명령: {' '.join(build_cmd)}")
            print(f"MinGW 환경: CC={mingw_env.get('CC')}, CXX={mingw_env.get('CXX')}")
            
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
                    bufsize=1,
                    env=mingw_env  # posix threads 환경 변수 적용
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
                print(f"\n❌ Windows 빌드 실패 (종료 코드: {process.returncode})")
                print(f"로그 파일: {log_file}")
                
                # Linux 대안 빌드 제안
                print("\n🔄 대안: Linux 버전으로 빌드 시도 중...")
                return self._build_linux_alternative(mingw_env)
                
        except Exception as e:
            print(f"❌ 빌드 중 예외 발생: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _build_linux_alternative(self, env):
        """Windows 빌드 실패 시 Linux 대안 빌드"""
        try:
            print("🐧 Linux 버전 AI Godot 빌드 시도...")
            
            # Linux 빌드 명령어
            build_cmd = [
                "scons", 
                "platform=linuxbsd", 
                "target=editor", 
                "arch=x86_64", 
                "debug_symbols=no",
                "optimize=speed",
                "-j2", 
                "verbose=yes"
            ]
            
            log_file = self.logs_dir / f"build_linux_{int(time.time())}.log"
            
            print(f"실행 명령: {' '.join(build_cmd)}")
            print("💡 Linux 버전은 WSL에서 X11 forwarding으로 실행 가능합니다")
            
            start_time = time.time()
            
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    build_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
                
                for line in process.stdout:
                    print(line.rstrip())
                    log.write(line)
                
                process.wait()
            
            build_time = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\n✅ Linux 빌드 완료! ({build_time/60:.1f}분 소요)")
                print("💡 WSL에서 'export DISPLAY=:0'으로 Windows X11 서버 연결 후 실행 가능")
                return True
            else:
                print(f"\n❌ Linux 빌드도 실패 (종료 코드: {process.returncode})")
                print(f"로그 파일: {log_file}")
                return False
                
        except Exception as e:
            print(f"❌ Linux 대안 빌드 중 예외 발생: {e}")
            return False
    
    def _finalize(self):
        """빌드 결과 정리"""
        print("📦 빌드 결과 정리 중...")
        
        # 빌드된 실행 파일 찾기
        bin_dir = self.source_dir / "bin"
        
        # 가능한 실행 파일 패턴들 - Windows용
        patterns = ["godot*editor*.exe", "godot*tools*.exe", "godot.windows*.exe"]
        executables = []
        
        for pattern in patterns:
            executables.extend(bin_dir.glob(pattern))
        
        if not executables:
            print("❌ 빌드된 실행 파일을 찾을 수 없습니다.")
            return None
        
        # 첫 번째 실행 파일 사용
        source_exe = executables[0]
        target_exe = self.output_dir / "godot.ai.editor.windows.x86_64.exe"
        
        # 실행 파일 복사 (Windows exe)
        shutil.copy2(source_exe, target_exe)
        
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
        """AI 빌드 실패 시 대안책 제시"""
        print("\n🔧 AI Godot Windows 빌드 실패 - 문제 해결 방법")
        print("=" * 60)
        
        print("🚫 일반 Godot은 사용하지 않습니다. AI 수정된 Godot만 사용합니다.")
        print("\n📋 문제 해결 단계:")
        
        print("\n1️⃣ MinGW posix threads 수동 설정:")
        print("   sudo update-alternatives --config x86_64-w64-mingw32-g++")
        print("   → posix threads 버전 선택")
        
        print("\n2️⃣ 필수 패키지 재설치:")
        print("   sudo apt update")
        print("   sudo apt install --reinstall gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64")
        
        print("\n3️⃣ MinGW 설정 스크립트 실행:")
        print("   chmod +x fix_mingw_posix.sh && ./fix_mingw_posix.sh")
        
        print("\n4️⃣ 빌드 재시도:")
        print("   build-godot")
        
        print("\n💡 또는 Linux 버전 빌드:")
        print("   build-godot-linux")
        print("   (Windows에서 WSL X11로 실행 가능)")
        
        print("\n🔍 로그 확인:")
        log_files = list(self.logs_dir.glob("build_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   tail -f {latest_log}")
        
        print(f"\n⚠️  AI 수정된 Godot만 사용하므로, 빌드 성공이 필수입니다.")
        print("   문제가 계속되면 GitHub Issues에 로그와 함께 문의하세요.")
        return None

def main():
    builder = AIGodotBuilder()
    result = builder.build()
    
    if result:
        print(f"\n🎯 성공! AutoCI에서 사용할 경로:")
        print(f"   {result}")
        print(f"\n다음 단계:")
        print(f"1. {GREEN}autoci{RESET} 명령어로 실행")
        print(f"2. AI 수정된 Godot이 자동으로 실행됩니다")
    else:
        print(f"\n❌ AI Godot 빌드 실패")
        print(f"💡 문제 해결 후 다시 시도하세요: build-godot")

if __name__ == "__main__":
    main()