#!/usr/bin/env python3
"""
AutoCI 시스템 상태 확인
전체 시스템의 상태를 종합적으로 분석하고 표시
"""

import asyncio
import platform
import psutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

class SystemStatus:
    """시스템 상태 확인기"""
    
    def __init__(self):
        self.logger = logging.getLogger("SystemStatus")
        self.autoci_root = Path(__file__).parent.parent
        
    async def display_comprehensive_status(self):
        """종합 시스템 상태 표시"""
        print("📊 AutoCI 시스템 종합 상태 분석")
        print("=" * 80)
        
        # 각 섹션별로 상태 확인
        sections = [
            ("🖥️ 시스템 환경", self._check_system_environment),
            ("🐍 Python 환경", self._check_python_environment),
            ("🤖 AI 모델", self._check_ai_models),
            ("🎮 Godot 통합", self._check_godot_integration),
            ("🔧 개발 도구", self._check_development_tools),
            ("📦 의존성", self._check_dependencies),
            ("🔒 보안", self._check_security_status),
            ("📊 성능", self._check_performance_status),
            ("💾 저장소", self._check_storage_status),
            ("🌐 네트워크", self._check_network_status)
        ]
        
        status_results = {}
        
        for section_name, check_function in sections:
            print(f"\n{section_name}")
            print("-" * 40)
            
            try:
                result = await check_function()
                status_results[section_name] = result
                await self._display_section_result(result)
            except Exception as e:
                print(f"❌ 확인 실패: {e}")
                status_results[section_name] = {"status": "error", "error": str(e)}
        
        # 종합 요약
        await self._display_overall_summary(status_results)
        
        # 권장 사항
        await self._display_recommendations(status_results)
    
    async def _check_system_environment(self) -> Dict[str, Any]:
        """시스템 환경 확인"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
            "disk_total": psutil.disk_usage('/').total // (1024**3),  # GB
        }
        
        # WSL 감지
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                info["is_wsl"] = 'microsoft' in version_info
        except:
            info["is_wsl"] = False
        
        # Docker 확인
        info["docker_available"] = await self._check_command_exists("docker")
        
        print(f"  OS: {info['os']} {info['architecture']}")
        print(f"  Python: {info['python_version']}")
        print(f"  CPU: {info['cpu_count']}코어")
        print(f"  메모리: {info['memory_total']}GB")
        print(f"  디스크: {info['disk_total']}GB")
        print(f"  WSL: {'✅' if info['is_wsl'] else '❌'}")
        print(f"  Docker: {'✅' if info['docker_available'] else '❌'}")
        
        return {"status": "ok", "details": info}
    
    async def _check_python_environment(self) -> Dict[str, Any]:
        """Python 환경 확인"""
        import sys
        import pkg_resources
        
        # 가상 환경 확인
        in_venv = hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix
        
        # 설치된 패키지 수
        installed_packages = list(pkg_resources.working_set)
        
        # 중요 패키지들 확인
        important_packages = {
            "asyncio": True,  # 내장 모듈
            "pathlib": True,  # 내장 모듈
            "json": True,     # 내장 모듈
        }
        
        # 선택적 패키지들 확인
        optional_packages = ["torch", "transformers", "numpy", "scipy", "PIL", "psutil"]
        for package in optional_packages:
            try:
                __import__(package)
                important_packages[package] = True
            except ImportError:
                important_packages[package] = False
        
        print(f"  Python 버전: {sys.version.split()[0]}")
        print(f"  가상 환경: {'✅' if in_venv else '❌'}")
        print(f"  설치된 패키지: {len(installed_packages)}개")
        print(f"  주요 패키지:")
        
        for package, available in important_packages.items():
            status = '✅' if available else '❌'
            print(f"    {package}: {status}")
        
        return {
            "status": "ok",
            "details": {
                "python_version": sys.version.split()[0],
                "virtual_env": in_venv,
                "packages_count": len(installed_packages),
                "important_packages": important_packages
            }
        }
    
    async def _check_ai_models(self) -> Dict[str, Any]:
        """AI 모델 상태 확인"""
        models_dir = self.autoci_root / "models"
        
        available_models = []
        if models_dir.exists():
            for model_path in models_dir.iterdir():
                if model_path.is_dir():
                    # 모델 크기 계산
                    size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    size_gb = size / (1024**3)
                    
                    available_models.append({
                        "name": model_path.name,
                        "size_gb": round(size_gb, 1),
                        "path": str(model_path)
                    })
        
        # 메모리 기반 추천 모델
        memory_gb = psutil.virtual_memory().total // (1024**3)
        if memory_gb >= 32:
            recommended = "Qwen2.5-Coder-32B"
        elif memory_gb >= 16:
            recommended = "CodeLlama-13B"
        else:
            recommended = "Llama-3.1-8B"
        
        print(f"  사용 가능한 모델: {len(available_models)}개")
        for model in available_models:
            print(f"    📦 {model['name']} ({model['size_gb']}GB)")
        
        print(f"  권장 모델: {recommended} (메모리 {memory_gb}GB 기준)")
        
        return {
            "status": "ok",
            "details": {
                "available_models": available_models,
                "recommended_model": recommended,
                "memory_gb": memory_gb
            }
        }
    
    async def _check_godot_integration(self) -> Dict[str, Any]:
        """Godot 통합 상태 확인"""
        try:
            from modules.godot_ai_integration import GodotAIIntegration
            integration = GodotAIIntegration()
            status = integration.get_integration_status()
            
            print(f"  Godot 설치: {'✅' if status['godot_installed'] else '❌'}")
            print(f"  AI 플러그인: {status['plugins_installed']}개")
            print(f"  프로젝트 템플릿: {status['templates_available']}개")
            print(f"  개발 도구: {status['tools_available']}개")
            
            # Godot 경로 확인
            godot_dir = self.autoci_root / "godot_ai"
            if godot_dir.exists():
                print(f"  설치 경로: {godot_dir}")
            
            return {"status": "ok", "details": status}
            
        except Exception as e:
            print(f"  ❌ Godot 통합 확인 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _check_development_tools(self) -> Dict[str, Any]:
        """개발 도구 확인"""
        tools = {
            "git": await self._check_command_exists("git"),
            "code": await self._check_command_exists("code"),  # VSCode
            "docker": await self._check_command_exists("docker"),
            "docker-compose": await self._check_command_exists("docker-compose"),
            "pip": await self._check_command_exists("pip3"),
            "npm": await self._check_command_exists("npm"),
            "curl": await self._check_command_exists("curl"),
            "wget": await self._check_command_exists("wget")
        }
        
        print("  개발 도구:")
        for tool, available in tools.items():
            status = '✅' if available else '❌'
            print(f"    {tool}: {status}")
        
        return {"status": "ok", "details": tools}
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """의존성 확인"""
        # requirements 파일들 확인
        requirements_files = [
            "requirements_ai_agents.txt",
            "requirements_enhanced.txt",
            "requirements_expert.txt"
        ]
        
        found_requirements = []
        for req_file in requirements_files:
            req_path = self.autoci_root / req_file
            if req_path.exists():
                with open(req_path) as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    found_requirements.append({
                        "file": req_file,
                        "packages": len(lines)
                    })
        
        print(f"  Requirements 파일: {len(found_requirements)}개")
        for req in found_requirements:
            print(f"    📄 {req['file']}: {req['packages']}개 패키지")
        
        # 핵심 디렉토리 확인
        core_dirs = ["modules", "tools", "tests", "config"]
        missing_dirs = []
        
        for dir_name in core_dirs:
            dir_path = self.autoci_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"  ❌ 누락된 디렉토리: {', '.join(missing_dirs)}")
        else:
            print(f"  ✅ 핵심 디렉토리 모두 존재")
        
        return {
            "status": "ok" if not missing_dirs else "warning",
            "details": {
                "requirements_files": found_requirements,
                "missing_dirs": missing_dirs
            }
        }
    
    async def _check_security_status(self) -> Dict[str, Any]:
        """보안 상태 확인"""
        import os
        security_checks = {}
        
        # 파일 권한 확인
        sensitive_files = [
            self.autoci_root / "config",
            self.autoci_root / "logs",
            self.autoci_root / "data"
        ]
        
        permission_issues = []
        for file_path in sensitive_files:
            if file_path.exists():
                stat_info = file_path.stat()
                # 간단한 권한 확인 (실제로는 더 복잡한 검사 필요)
                if stat_info.st_mode & 0o077:  # 다른 사용자가 읽기/쓰기 가능
                    permission_issues.append(str(file_path))
        
        # 환경 변수에서 민감한 정보 확인
        env_secrets = []
        sensitive_env_vars = ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
        for var in sensitive_env_vars:
            for env_var in os.environ:
                if any(sensitive in env_var.upper() for sensitive in sensitive_env_vars):
                    env_secrets.append(env_var)
        
        print(f"  파일 권한: {'⚠️' if permission_issues else '✅'}")
        if permission_issues:
            print(f"    권한 확인 필요: {len(permission_issues)}개 파일")
        
        print(f"  환경 변수: {'⚠️' if env_secrets else '✅'}")
        if env_secrets:
            print(f"    민감한 환경 변수: {len(env_secrets)}개")
        
        return {
            "status": "warning" if permission_issues or env_secrets else "ok",
            "details": {
                "permission_issues": permission_issues,
                "sensitive_env_vars": len(env_secrets)
            }
        }
    
    async def _check_performance_status(self) -> Dict[str, Any]:
        """성능 상태 확인"""
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        
        # 네트워크 상태
        network = psutil.net_io_counters()
        
        # 실행 중인 프로세스 수
        process_count = len(psutil.pids())
        
        performance_score = 100
        issues = []
        
        if cpu_percent > 80:
            performance_score -= 20
            issues.append("높은 CPU 사용률")
        
        if memory.percent > 85:
            performance_score -= 25
            issues.append("높은 메모리 사용률")
        
        if disk.percent > 90:
            performance_score -= 15
            issues.append("낮은 디스크 여유 공간")
        
        print(f"  CPU 사용률: {cpu_percent:.1f}%")
        print(f"  메모리 사용률: {memory.percent:.1f}%")
        print(f"  디스크 사용률: {disk.percent:.1f}%")
        print(f"  실행 중인 프로세스: {process_count}개")
        print(f"  성능 점수: {performance_score}/100")
        
        if issues:
            print(f"  ⚠️ 성능 이슈: {', '.join(issues)}")
        
        return {
            "status": "ok" if performance_score >= 70 else "warning",
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "process_count": process_count,
                "performance_score": performance_score,
                "issues": issues
            }
        }
    
    async def _check_storage_status(self) -> Dict[str, Any]:
        """저장소 상태 확인"""
        # 주요 디렉토리 크기 확인
        directories = {
            "logs": self.autoci_root / "logs",
            "models": self.autoci_root / "models", 
            "game_projects": self.autoci_root / "game_projects",
            "godot_ai": self.autoci_root / "godot_ai"
        }
        
        dir_sizes = {}
        total_size = 0
        
        for name, path in directories.items():
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                dir_sizes[name] = size_gb
                total_size += size_gb
            else:
                dir_sizes[name] = 0
        
        # 디스크 여유 공간
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        print(f"  디렉토리 크기:")
        for name, size in dir_sizes.items():
            print(f"    {name}: {size:.1f}GB")
        print(f"  총 사용량: {total_size:.1f}GB")
        print(f"  여유 공간: {free_gb:.1f}GB")
        
        return {
            "status": "ok" if free_gb > 5 else "warning",
            "details": {
                "directory_sizes": dir_sizes,
                "total_size_gb": total_size,
                "free_space_gb": free_gb
            }
        }
    
    async def _check_network_status(self) -> Dict[str, Any]:
        """네트워크 상태 확인"""
        # 인터넷 연결 확인
        connectivity_tests = [
            ("Google DNS", "8.8.8.8"),
            ("Cloudflare DNS", "1.1.1.1"),
            ("GitHub", "github.com")
        ]
        
        connectivity_results = {}
        
        for name, target in connectivity_tests:
            try:
                if target.count('.') == 3:  # IP 주소
                    result = await self._ping_host(target)
                else:  # 도메인 이름
                    result = await self._check_domain_connectivity(target)
                connectivity_results[name] = result
            except Exception as e:
                connectivity_results[name] = False
        
        # 네트워크 인터페이스 확인
        network_interfaces = psutil.net_if_addrs()
        active_interfaces = len([iface for iface in network_interfaces.keys() 
                               if iface != 'lo'])  # loopback 제외
        
        print(f"  연결 테스트:")
        for name, result in connectivity_results.items():
            status = '✅' if result else '❌'
            print(f"    {name}: {status}")
        
        print(f"  네트워크 인터페이스: {active_interfaces}개")
        
        connected_count = sum(1 for result in connectivity_results.values() if result)
        
        return {
            "status": "ok" if connected_count >= 2 else "warning",
            "details": {
                "connectivity_tests": connectivity_results,
                "active_interfaces": active_interfaces,
                "connection_score": f"{connected_count}/{len(connectivity_tests)}"
            }
        }
    
    async def _ping_host(self, host: str, timeout: int = 3) -> bool:
        """호스트 ping 테스트"""
        try:
            result = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', str(timeout), host,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await result.wait()
            return result.returncode == 0
        except:
            return False
    
    async def _check_domain_connectivity(self, domain: str) -> bool:
        """도메인 연결 확인"""
        try:
            result = await asyncio.create_subprocess_exec(
                'curl', '-s', '--connect-timeout', '5', '--max-time', '10', 
                f'https://{domain}',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await result.wait()
            return result.returncode == 0
        except:
            return False
    
    async def _check_command_exists(self, command: str) -> bool:
        """명령어 존재 확인"""
        try:
            result = await asyncio.create_subprocess_exec(
                'which', command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await result.wait()
            return result.returncode == 0
        except:
            return False
    
    async def _display_section_result(self, result: Dict[str, Any]):
        """섹션 결과 표시"""
        # 결과는 이미 각 체크 함수에서 출력됨
        pass
    
    async def _display_overall_summary(self, status_results: Dict[str, Any]):
        """전체 요약 표시"""
        print("\n" + "=" * 80)
        print("📋 종합 상태 요약")
        print("=" * 80)
        
        # 상태별 분류
        ok_count = 0
        warning_count = 0
        error_count = 0
        
        for section, result in status_results.items():
            status = result.get('status', 'unknown')
            if status == 'ok':
                ok_count += 1
                status_icon = '✅'
            elif status == 'warning':
                warning_count += 1
                status_icon = '⚠️'
            elif status == 'error':
                error_count += 1
                status_icon = '❌'
            else:
                status_icon = '❓'
            
            print(f"  {status_icon} {section}")
        
        print(f"\n📊 전체 상태: ✅ {ok_count}개 | ⚠️ {warning_count}개 | ❌ {error_count}개")
        
        # 전체 시스템 상태 판정
        if error_count == 0 and warning_count <= 2:
            overall_status = "🟢 양호"
        elif error_count <= 1 and warning_count <= 4:
            overall_status = "🟡 주의"
        else:
            overall_status = "🔴 개선 필요"
        
        print(f"🎯 종합 판정: {overall_status}")
    
    async def _display_recommendations(self, status_results: Dict[str, Any]):
        """권장 사항 표시"""
        print("\n💡 권장 사항:")
        
        recommendations = []
        
        # 각 섹션별 권장 사항 생성
        for section, result in status_results.items():
            if result.get('status') == 'warning' or result.get('status') == 'error':
                if 'Python 환경' in section:
                    recommendations.append("🐍 필요한 Python 패키지를 설치하세요: pip install -r requirements_ai_agents.txt")
                elif 'Godot 통합' in section:
                    recommendations.append("🎮 Godot AI 통합을 설정하세요: autoci --godot")
                elif '성능' in section:
                    recommendations.append("⚡ 시스템 리소스 사용량을 확인하고 최적화하세요")
                elif '저장소' in section:
                    recommendations.append("💾 디스크 정리를 수행하여 여유 공간을 확보하세요")
                elif '네트워크' in section:
                    recommendations.append("🌐 네트워크 연결을 확인하고 방화벽 설정을 점검하세요")
                elif '보안' in section:
                    recommendations.append("🔒 파일 권한을 검토하고 민감한 정보를 보호하세요")
        
        if not recommendations:
            recommendations.append("✨ 모든 시스템이 정상 상태입니다! AutoCI를 시작할 준비가 되었습니다.")
            recommendations.append("🚀 'autoci --production' 명령으로 24시간 AI 개발을 시작하세요.")
        
        for i, recommendation in enumerate(recommendations, 1):
            print(f"  {i}. {recommendation}")
        
        print("\n" + "=" * 80)

# 독립 실행용
async def main():
    """테스트 실행"""
    status_checker = SystemStatus()
    await status_checker.display_comprehensive_status()

if __name__ == "__main__":
    asyncio.run(main())