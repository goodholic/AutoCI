#!/usr/bin/env python3
"""
AutoCI 24시간 학습 시스템 의존성 설치 스크립트
필요한 모든 패키지를 자동으로 설치
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyInstaller:
    """의존성 설치기"""
    
    def __init__(self):
        self.required_packages = {
            # 머신러닝 및 딥러닝
            "torch": "PyTorch - 신경망 프레임워크",
            "scikit-learn": "머신러닝 라이브러리",
            "numpy": "수치 계산",
            "pandas": "데이터 분석",
            
            # 시각화
            "matplotlib": "그래프 및 차트",
            "seaborn": "통계 시각화",
            
            # 시스템 모니터링
            "psutil": "시스템 리소스 모니터링",
            "GPUtil": "GPU 모니터링 (선택사항)",
            
            # 스케줄링 및 유틸리티
            "schedule": "작업 스케줄링",
            "requests": "HTTP 요청",
            
            # 개발 도구
            "tqdm": "진행률 표시",
            "colorama": "터미널 컬러",
            
            # 데이터베이스
            "sqlite3": "내장 데이터베이스 (Python 기본)",
        }
        
        # 선택적 패키지 (설치 실패해도 계속 진행)
        self.optional_packages = {
            "transformers": "Hugging Face 트랜스포머",
            "accelerate": "PyTorch 가속화",
            "datasets": "데이터셋 라이브러리"
        }
        
    def check_python_version(self) -> bool:
        """Python 버전 확인"""
        major, minor = sys.version_info[:2]
        
        if major < 3 or (major == 3 and minor < 8):
            logger.error(f"Python 3.8 이상이 필요합니다. 현재 버전: {major}.{minor}")
            return False
        
        logger.info(f"✅ Python 버전 확인: {major}.{minor}")
        return True
    
    def check_pip(self) -> bool:
        """pip 설치 확인"""
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            logger.info("✅ pip 설치 확인")
            return True
        except subprocess.CalledProcessError:
            logger.error("❌ pip가 설치되지 않았습니다")
            return False
    
    def upgrade_pip(self) -> bool:
        """pip 업그레이드"""
        try:
            logger.info("🔄 pip 업그레이드 중...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            logger.info("✅ pip 업그레이드 완료")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ pip 업그레이드 실패: {e}")
            return False
    
    def check_package_installed(self, package_name: str) -> bool:
        """패키지 설치 여부 확인"""
        try:
            # sqlite3는 Python 내장 모듈
            if package_name == "sqlite3":
                import sqlite3
                return True
            
            __import__(package_name.replace("-", "_"))
            return True
        except ImportError:
            return False
    
    def install_package(self, package_name: str, description: str = "") -> bool:
        """개별 패키지 설치"""
        if self.check_package_installed(package_name):
            logger.info(f"✅ {package_name} 이미 설치됨")
            return True
        
        logger.info(f"📦 {package_name} 설치 중... ({description})")
        
        try:
            # 특별한 설치 명령이 필요한 패키지들
            if package_name == "torch":
                # PyTorch는 플랫폼에 따라 다른 설치 명령 필요
                cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package_name]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ {package_name} 설치 완료")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {package_name} 설치 실패: {e}")
            logger.error(f"오류 출력: {e.stderr}")
            return False
    
    def install_all_packages(self) -> Dict[str, bool]:
        """모든 패키지 설치"""
        results = {}
        
        logger.info("🚀 필수 패키지 설치 시작")
        
        # 필수 패키지 설치
        for package, description in self.required_packages.items():
            results[package] = self.install_package(package, description)
        
        logger.info("🔧 선택적 패키지 설치 시작")
        
        # 선택적 패키지 설치
        for package, description in self.optional_packages.items():
            try:
                success = self.install_package(package, description)
                results[package] = success
                if not success:
                    logger.info(f"⚠️ {package} 설치 실패 (선택사항이므로 계속 진행)")
            except Exception as e:
                logger.warning(f"⚠️ {package} 설치 중 오류 (선택사항): {e}")
                results[package] = False
        
        return results
    
    def create_requirements_file(self):
        """requirements.txt 파일 생성"""
        requirements_content = """# AutoCI 24시간 학습 시스템 의존성

# 머신러닝 및 딥러닝
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.21.0
pandas>=1.5.0

# 시각화
matplotlib>=3.5.0
seaborn>=0.11.0

# 시스템 모니터링
psutil>=5.9.0
GPUtil>=1.4.0

# 스케줄링 및 유틸리티
schedule>=1.2.0
requests>=2.28.0

# 개발 도구
tqdm>=4.64.0
colorama>=0.4.5

# 선택적 패키지
transformers>=4.20.0
accelerate>=0.20.0
datasets>=2.10.0
"""
        
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        logger.info("📝 requirements.txt 파일 생성 완료")
    
    def verify_installation(self) -> bool:
        """설치 검증"""
        logger.info("🔍 설치 검증 중...")
        
        essential_packages = ["torch", "scikit-learn", "numpy", "matplotlib", "psutil", "schedule"]
        failed_packages = []
        
        for package in essential_packages:
            if not self.check_package_installed(package):
                failed_packages.append(package)
        
        if failed_packages:
            logger.error(f"❌ 다음 필수 패키지 설치 실패: {', '.join(failed_packages)}")
            return False
        
        logger.info("✅ 모든 필수 패키지 설치 확인 완료")
        return True
    
    def create_virtual_environment(self, venv_name: str = "autoci_venv") -> bool:
        """가상환경 생성"""
        try:
            logger.info(f"🏗️ 가상환경 생성 중: {venv_name}")
            
            # 가상환경 생성
            subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
            
            logger.info(f"✅ 가상환경 생성 완료: {venv_name}")
            logger.info(f"활성화 명령 (Linux/Mac): source {venv_name}/bin/activate")
            logger.info(f"활성화 명령 (Windows): {venv_name}\\Scripts\\activate")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 가상환경 생성 실패: {e}")
            return False
    
    def run_installation(self, create_venv: bool = False, venv_name: str = "autoci_venv") -> bool:
        """전체 설치 프로세스 실행"""
        logger.info("🚀 AutoCI 24시간 학습 시스템 의존성 설치 시작")
        
        # Python 버전 확인
        if not self.check_python_version():
            return False
        
        # pip 확인
        if not self.check_pip():
            return False
        
        # 가상환경 생성 (선택사항)
        if create_venv:
            if not self.create_virtual_environment(venv_name):
                logger.warning("⚠️ 가상환경 생성 실패, 시스템 Python 사용")
        
        # pip 업그레이드
        self.upgrade_pip()
        
        # requirements.txt 생성
        self.create_requirements_file()
        
        # 패키지 설치
        results = self.install_all_packages()
        
        # 설치 결과 요약
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"📊 설치 결과: {successful}/{total} 패키지 성공")
        
        # 실패한 패키지 목록
        failed_packages = [pkg for pkg, success in results.items() if not success]
        if failed_packages:
            logger.warning(f"⚠️ 설치 실패한 패키지: {', '.join(failed_packages)}")
        
        # 설치 검증
        if self.verify_installation():
            logger.info("🎉 AutoCI 24시간 학습 시스템 의존성 설치 완료!")
            logger.info("시스템 시작 명령: python3 start_24h_learning_system.py")
            return True
        else:
            logger.error("❌ 설치 검증 실패")
            return False

def main():
    """메인 함수"""
    print("🚀 AutoCI 24시간 학습 시스템 의존성 설치")
    print("=" * 60)
    
    installer = DependencyInstaller()
    
    # 명령행 인자 처리
    create_venv = "--venv" in sys.argv or "-v" in sys.argv
    
    if create_venv:
        print("🏗️ 가상환경을 생성하여 설치합니다")
    else:
        print("📦 시스템 Python에 직접 설치합니다")
        print("가상환경 사용을 원하면: python3 install_dependencies.py --venv")
    
    print("=" * 60)
    
    try:
        success = installer.run_installation(create_venv=create_venv)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 설치 중단됨")
        return 1
    except Exception as e:
        logger.error(f"❌ 설치 중 예상치 못한 오류: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())