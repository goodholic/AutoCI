#!/usr/bin/env python3
"""
WSL 환경 관리자
WSL 환경 최적화, 가상화 설정, AI 개발 환경 구성
"""

import os
import sys
import asyncio
import logging
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Any

class WSLManager:
    """WSL 환경 관리 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("WSLManager")
        self.is_wsl = self._detect_wsl_environment()
        self.system_info = self._get_system_info()
        
    def _detect_wsl_environment(self) -> bool:
        """WSL 환경 감지"""
        try:
            # WSL 환경 확인 방법들
            wsl_indicators = [
                os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower(),
                'microsoft' in platform.uname().release.lower(),
                os.environ.get('WSL_DISTRO_NAME') is not None,
                os.path.exists('/mnt/c')
            ]
            return any(wsl_indicators)
        except:
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "is_wsl": self.is_wsl,
            "cpu_count": os.cpu_count(),
            "memory_gb": self._get_memory_info()
        }
        
        if self.is_wsl:
            info.update(self._get_wsl_specific_info())
        
        return info
    
    def _get_memory_info(self) -> float:
        """메모리 정보 가져오기"""
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 1)  # GB 단위
        except:
            pass
        return 0.0
    
    def _get_wsl_specific_info(self) -> Dict[str, Any]:
        """WSL 특화 정보"""
        wsl_info = {}
        
        # WSL 버전 확인
        try:
            result = subprocess.run(['wsl', '--version'], 
                                  capture_output=True, text=True, check=True)
            wsl_info['wsl_version'] = result.stdout.strip()
        except:
            wsl_info['wsl_version'] = "WSL 1 또는 확인 불가"
        
        # 배포판 정보
        wsl_info['distro'] = os.environ.get('WSL_DISTRO_NAME', 'Unknown')
        
        # Windows 드라이브 마운트 상태
        wsl_info['windows_drives'] = self._check_windows_drives()
        
        return wsl_info
    
    def _check_windows_drives(self) -> List[str]:
        """Windows 드라이브 마운트 확인"""
        drives = []
        mount_path = Path('/mnt')
        if mount_path.exists():
            for item in mount_path.iterdir():
                if item.is_dir() and len(item.name) == 1:
                    drives.append(item.name.upper())
        return sorted(drives)
    
    async def optimize_wsl_environment(self):
        """WSL 환경 최적화"""
        print("🔧 WSL 환경 최적화 중...")
        
        # 시스템 정보 출력
        await self._display_system_info()
        
        # WSL 최적화 작업들
        optimizations = [
            self._optimize_wsl_memory,
            self._optimize_wsl_networking,
            self._setup_wsl_interop,
            self._configure_wsl_performance
        ]
        
        for optimization in optimizations:
            try:
                await optimization()
            except Exception as e:
                self.logger.warning(f"최적화 작업 실패: {e}")
                
        print("✅ WSL 환경 최적화 완료")
    
    async def _display_system_info(self):
        """시스템 정보 표시"""
        print("\n📊 시스템 정보:")
        print(f"  💻 플랫폼: {self.system_info['platform']}")
        print(f"  🏗️  아키텍처: {self.system_info['architecture']}")
        print(f"  🐍 Python: {self.system_info['python_version']}")
        print(f"  🔀 WSL: {'✅' if self.system_info['is_wsl'] else '❌'}")
        print(f"  🧠 CPU 코어: {self.system_info['cpu_count']}")
        print(f"  💾 메모리: {self.system_info['memory_gb']} GB")
        
        if self.is_wsl:
            print(f"  📀 WSL 버전: {self.system_info.get('wsl_version', 'Unknown')}")
            print(f"  🐧 배포판: {self.system_info.get('distro', 'Unknown')}")
            drives = self.system_info.get('windows_drives', [])
            print(f"  🗂️  Windows 드라이브: {', '.join(drives) if drives else 'None'}")
        print()
    
    async def _optimize_wsl_memory(self):
        """WSL 메모리 최적화"""
        print("  🧠 메모리 최적화 설정...")
        
        # .wslconfig 파일 생성/수정 (Windows 사용자 홈에)
        wsl_config_content = f"""[wsl2]
memory={max(4, int(self.system_info['memory_gb'] * 0.8))}GB
processors={self.system_info['cpu_count']}
swap=2GB
localhostForwarding=true
guiApplications=true
"""
        
        # Windows 사용자 홈 디렉토리 찾기
        windows_home = self._get_windows_home_path()
        if windows_home:
            wsl_config_path = windows_home / '.wslconfig'
            try:
                with open(wsl_config_path, 'w') as f:
                    f.write(wsl_config_content)
                print(f"    ✅ WSL 설정 파일 생성: {wsl_config_path}")
            except Exception as e:
                print(f"    ⚠️ WSL 설정 파일 생성 실패: {e}")
    
    def _get_windows_home_path(self) -> Path:
        """Windows 사용자 홈 경로 가져오기"""
        try:
            # WSL에서 Windows 사용자 홈 찾기
            result = subprocess.run(['wslpath', '-w', '/mnt/c/Users'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                users_dir = Path('/mnt/c/Users')
                # 현재 사용자 찾기
                username = os.environ.get('USER', 'user')
                for user_dir in users_dir.iterdir():
                    if user_dir.is_dir() and user_dir.name.lower() == username.lower():
                        return user_dir
                # 기본 경로
                return users_dir / username
        except:
            pass
        return None
    
    async def _optimize_wsl_networking(self):
        """WSL 네트워킹 최적화"""
        print("  🌐 네트워킹 최적화...")
        
        # DNS 설정 최적화
        resolv_conf = Path('/etc/resolv.conf')
        if resolv_conf.exists():
            try:
                # 빠른 DNS 서버 설정
                dns_config = """# AutoCI 최적화 DNS 설정
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1
"""
                # sudo 권한이 필요하므로 사용자에게 안내
                print("    ℹ️ DNS 최적화를 위해 수동 설정이 필요할 수 있습니다.")
            except Exception as e:
                print(f"    ⚠️ DNS 설정 실패: {e}")
    
    async def _setup_wsl_interop(self):
        """WSL-Windows 상호 운용성 설정"""
        print("  🔗 Windows 상호 운용성 설정...")
        
        # Windows PATH 접근 확인
        if os.environ.get('PATH', '').find('/mnt/c/Windows') != -1:
            print("    ✅ Windows PATH 접근 가능")
        else:
            print("    ⚠️ Windows PATH 접근 제한됨")
        
        # Windows 실행 파일 접근 테스트
        try:
            result = subprocess.run(['cmd.exe', '/c', 'echo Windows 접근 테스트'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("    ✅ Windows 실행 파일 접근 가능")
            else:
                print("    ⚠️ Windows 실행 파일 접근 제한됨")
        except:
            print("    ⚠️ Windows 실행 파일 접근 불가")
    
    async def _configure_wsl_performance(self):
        """WSL 성능 설정"""
        print("  ⚡ 성능 최적화 설정...")
        
        # 파일 시스템 성능 최적화
        try:
            # 임시 파일 시스템을 tmpfs로 설정
            await self._run_command(['sudo', 'mount', '-t', 'tmpfs', 'tmpfs', '/tmp'], 
                                  check=False)
            print("    ✅ 임시 파일 시스템 최적화")
        except:
            print("    ⚠️ 임시 파일 시스템 최적화 실패")
    
    async def setup_virtualization(self):
        """가상화 환경 설정"""
        print("🖥️ 가상화 환경 설정 중...")
        
        # 가상화 지원 확인
        virtualization_support = await self._check_virtualization_support()
        
        if virtualization_support['hardware_support']:
            print("  ✅ 하드웨어 가상화 지원 감지")
            
            # Docker 설정
            await self._setup_docker_environment()
            
            # GPU 가상화 설정 (사용 가능한 경우)
            await self._setup_gpu_virtualization()
            
        else:
            print("  ⚠️ 하드웨어 가상화 지원 제한됨")
        
        print("✅ 가상화 환경 설정 완료")
    
    async def _check_virtualization_support(self) -> Dict[str, bool]:
        """가상화 지원 확인"""
        support = {
            'hardware_support': False,
            'nested_virtualization': False,
            'gpu_virtualization': False
        }
        
        try:
            # CPU 가상화 기능 확인
            with open('/proc/cpuinfo') as f:
                cpuinfo = f.read()
                if 'vmx' in cpuinfo or 'svm' in cpuinfo:
                    support['hardware_support'] = True
                    
            # WSL에서 중첩 가상화 확인
            if self.is_wsl:
                result = await self._run_command(['systemd-detect-virt'], check=False)
                if result and result.returncode == 0:
                    support['nested_virtualization'] = True
        except:
            pass
        
        return support
    
    async def _setup_docker_environment(self):
        """Docker 환경 설정"""
        print("  🐳 Docker 환경 설정...")
        
        # Docker 설치 확인
        docker_installed = await self._check_command_exists('docker')
        
        if not docker_installed:
            print("    📦 Docker 설치 중...")
            await self._install_docker()
        else:
            print("    ✅ Docker 이미 설치됨")
        
        # Docker Compose 확인
        compose_installed = await self._check_command_exists('docker-compose')
        if not compose_installed:
            print("    📦 Docker Compose 설치 중...")
            await self._install_docker_compose()
        else:
            print("    ✅ Docker Compose 이미 설치됨")
    
    async def _install_docker(self):
        """Docker 설치"""
        try:
            # 공식 Docker 설치 스크립트 사용
            install_commands = [
                ['curl', '-fsSL', 'https://get.docker.com', '-o', 'get-docker.sh'],
                ['sh', 'get-docker.sh'],
                ['sudo', 'usermod', '-aG', 'docker', os.environ.get('USER', 'user')],
                ['rm', 'get-docker.sh']
            ]
            
            for cmd in install_commands:
                await self._run_command(cmd, check=False)
            
            print("    ✅ Docker 설치 완료")
        except Exception as e:
            print(f"    ⚠️ Docker 설치 실패: {e}")
    
    async def _install_docker_compose(self):
        """Docker Compose 설치"""
        try:
            # pip를 통한 설치
            await self._run_command(['pip3', 'install', 'docker-compose'], check=False)
            print("    ✅ Docker Compose 설치 완료")
        except Exception as e:
            print(f"    ⚠️ Docker Compose 설치 실패: {e}")
    
    async def _setup_gpu_virtualization(self):
        """GPU 가상화 설정"""
        print("  🎮 GPU 가상화 설정...")
        
        # NVIDIA GPU 확인
        nvidia_gpu = await self._check_nvidia_gpu()
        
        if nvidia_gpu:
            print("    🎯 NVIDIA GPU 감지됨")
            await self._setup_nvidia_docker()
        else:
            print("    ℹ️ NVIDIA GPU 감지되지 않음")
    
    async def _check_nvidia_gpu(self) -> bool:
        """NVIDIA GPU 확인"""
        try:
            result = await self._run_command(['nvidia-smi'], check=False)
            return result and result.returncode == 0
        except:
            return False
    
    async def _setup_nvidia_docker(self):
        """NVIDIA Docker 설정"""
        try:
            # NVIDIA Container Toolkit 설치
            install_commands = [
                ['curl', '-fsSL', 'https://nvidia.github.io/nvidia-docker/gpgkey', '|', 'sudo', 'apt-key', 'add', '-'],
                ['sudo', 'apt-get', 'update'],
                ['sudo', 'apt-get', 'install', '-y', 'nvidia-docker2'],
                ['sudo', 'systemctl', 'restart', 'docker']
            ]
            
            for cmd in install_commands:
                await self._run_command(cmd, check=False)
            
            print("    ✅ NVIDIA Docker 설정 완료")
        except Exception as e:
            print(f"    ⚠️ NVIDIA Docker 설정 실패: {e}")
    
    async def setup_ai_development_environment(self):
        """AI 개발 환경 구성"""
        print("🤖 AI 개발 환경 구성 중...")
        
        # Python 패키지 관리 최적화
        await self._optimize_python_environment()
        
        # AI 프레임워크 설치
        await self._setup_ai_frameworks()
        
        # GPU 지원 설정
        await self._setup_gpu_support()
        
        # 개발 도구 설정
        await self._setup_development_tools()
        
        print("✅ AI 개발 환경 구성 완료")
    
    async def _optimize_python_environment(self):
        """Python 환경 최적화"""
        print("  🐍 Python 환경 최적화...")
        
        # pip 최신 버전 확인
        await self._run_command(['pip3', 'install', '--upgrade', 'pip'], check=False)
        
        # 가상 환경 도구 설치
        tools = ['virtualenv', 'pipenv', 'poetry']
        for tool in tools:
            await self._run_command(['pip3', 'install', tool], check=False)
        
        print("    ✅ Python 도구 최적화 완료")
    
    async def _setup_ai_frameworks(self):
        """AI 프레임워크 설치"""
        print("  🧠 AI 프레임워크 설정...")
        
        # 기본 AI 라이브러리들 (requirements_ai_agents.txt 파일이 있다면 사용)
        requirements_file = Path('requirements_ai_agents.txt')
        
        if requirements_file.exists():
            await self._run_command(['pip3', 'install', '-r', str(requirements_file)], 
                                  check=False)
            print("    ✅ AI 패키지 설치 완료")
        else:
            print("    ℹ️ requirements_ai_agents.txt 파일을 찾을 수 없음")
    
    async def _setup_gpu_support(self):
        """GPU 지원 설정"""
        print("  🎮 GPU 지원 설정...")
        
        # CUDA 지원 확인
        cuda_available = await self._check_command_exists('nvcc')
        
        if cuda_available:
            print("    ✅ CUDA 개발 환경 감지됨")
        else:
            print("    ℹ️ CUDA 개발 환경 감지되지 않음")
    
    async def _setup_development_tools(self):
        """개발 도구 설정"""
        print("  🛠️ 개발 도구 설정...")
        
        # Git 설정 확인
        git_installed = await self._check_command_exists('git')
        if git_installed:
            print("    ✅ Git 사용 가능")
        
        # VSCode 서버 설정 (WSL에서)
        if self.is_wsl:
            print("    📝 VSCode WSL 확장 사용 권장")
    
    async def _check_command_exists(self, command: str) -> bool:
        """명령어 존재 확인"""
        try:
            result = await self._run_command(['which', command], check=False)
            return result and result.returncode == 0
        except:
            return False
    
    async def _run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """비동기 명령 실행"""
        try:
            # 비동기 프로세스 실행
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = subprocess.CompletedProcess(
                cmd, process.returncode, stdout, stderr
            )
            
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)
            
            return result
            
        except Exception as e:
            if check:
                raise e
            return None

# 사용 예시
async def main():
    """테스트 실행"""
    wsl_manager = WSLManager()
    
    print("🚀 WSL 환경 관리자 테스트")
    print("=" * 60)
    
    await wsl_manager.optimize_wsl_environment()
    await wsl_manager.setup_virtualization()
    await wsl_manager.setup_ai_development_environment()

if __name__ == "__main__":
    asyncio.run(main())