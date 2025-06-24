#!/usr/bin/env python3
"""
README 구현 검증 스크립트
README.md에 명시된 모든 요구사항이 제대로 구현되었는지 확인
"""

import os
import json
from pathlib import Path
import sys

class ImplementationVerifier:
    def __init__(self):
        self.base_dir = Path(".")
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def verify_all(self):
        """모든 검증 수행"""
        print("🔍 AutoCI 구현 검증 시작...\n")
        
        # 1. 디렉토리 구조 검증
        self.verify_directory_structure()
        
        # 2. 필수 파일 검증
        self.verify_required_files()
        
        # 3. Python 파일 구문 검증
        self.verify_python_syntax()
        
        # 4. 설정 파일 검증
        self.verify_config_files()
        
        # 5. 포트 설정 검증
        self.verify_port_configuration()
        
        # 결과 출력
        self.print_results()
        
    def verify_directory_structure(self):
        """README에 명시된 디렉토리 구조 검증"""
        print("📁 디렉토리 구조 검증...")
        
        required_dirs = [
            "MyAIWebApp",
            "MyAIWebApp/Backend",
            "MyAIWebApp/Backend/Services",
            "MyAIWebApp/Backend/Controllers",
            "MyAIWebApp/Frontend",
            "MyAIWebApp/Frontend/Pages",
            "MyAIWebApp/Frontend/wwwroot",
            "MyAIWebApp/Models",
            "expert_training_data"
        ]
        
        for dir_path in required_dirs:
            path = self.base_dir / dir_path
            if path.exists() and path.is_dir():
                self.successes.append(f"✅ 디렉토리 존재: {dir_path}")
            else:
                self.errors.append(f"❌ 디렉토리 없음: {dir_path}")
                
    def verify_required_files(self):
        """README에 명시된 필수 파일 검증"""
        print("\n📄 필수 파일 검증...")
        
        required_files = {
            # 루트 레벨 파일
            "csharp_expert_crawler.py": "24시간 학습 엔진",
            "start_expert_learning.py": "설치 스크립트",
            "expert_learning_api.py": "모니터링 API",
            "start_all.py": "통합 실행",
            "download_model.py": "모델 다운로드",
            "requirements_expert.txt": "전체 패키지 목록",
            "auto_train_collector.py": "자동 학습 데이터 수집",
            "save_feedback.py": "피드백 저장",
            
            # Models 디렉토리
            "MyAIWebApp/Models/enhanced_server.py": "AI 모델 서버",
            "MyAIWebApp/Models/fine_tune.py": "파인튜닝 스크립트",
            "MyAIWebApp/Models/requirements.txt": "Models 패키지 목록",
            
            # Backend Services
            "MyAIWebApp/Backend/Services/AIService.cs": "AI 서비스",
            "MyAIWebApp/Backend/Services/SearchService.cs": "검색 서비스",
            "MyAIWebApp/Backend/Services/LlamaService.cs": "Llama 서비스",
            "MyAIWebApp/Backend/Services/RAGService.cs": "RAG 서비스",
            
            # Frontend Pages
            "MyAIWebApp/Frontend/Pages/CodeGenerator.razor": "코드 생성 페이지",
            "MyAIWebApp/Frontend/Pages/CodeSearch.razor": "코드 검색 페이지",
            "MyAIWebApp/Frontend/Pages/RAG.razor": "RAG Q&A 페이지"
        }
        
        for file_path, description in required_files.items():
            path = self.base_dir / file_path
            if path.exists() and path.is_file():
                self.successes.append(f"✅ {description}: {file_path}")
            else:
                self.errors.append(f"❌ {description} 없음: {file_path}")
                
    def verify_python_syntax(self):
        """Python 파일 구문 검증"""
        print("\n🐍 Python 파일 구문 검증...")
        
        python_files = [
            "csharp_expert_crawler.py",
            "start_expert_learning.py",
            "expert_learning_api.py",
            "start_all.py",
            "download_model.py",
            "auto_train_collector.py",
            "save_feedback.py",
            "MyAIWebApp/Models/enhanced_server.py",
            "MyAIWebApp/Models/fine_tune.py"
        ]
        
        for file_path in python_files:
            path = self.base_dir / file_path
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, file_path, 'exec')
                    self.successes.append(f"✅ Python 구문 정상: {file_path}")
                except SyntaxError as e:
                    self.errors.append(f"❌ Python 구문 오류 {file_path}: {e}")
                except Exception as e:
                    self.warnings.append(f"⚠️  Python 파일 읽기 오류 {file_path}: {e}")
                    
    def verify_config_files(self):
        """설정 파일 검증"""
        print("\n⚙️  설정 파일 검증...")
        
        # launchSettings.json 확인
        launch_settings_paths = [
            "MyAIWebApp/Backend/Properties/launchSettings.json",
            "MyAIWebApp/Frontend/Properties/launchSettings.json"
        ]
        
        for settings_path in launch_settings_paths:
            path = self.base_dir / settings_path
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.successes.append(f"✅ 설정 파일 정상: {settings_path}")
                except json.JSONDecodeError:
                    self.errors.append(f"❌ JSON 파싱 오류: {settings_path}")
            else:
                self.warnings.append(f"⚠️  설정 파일 없음: {settings_path}")
                
    def verify_port_configuration(self):
        """포트 설정 검증"""
        print("\n🔌 포트 설정 검증...")
        
        expected_ports = {
            "AI Server": 8000,
            "Monitoring API": 8080,
            "Backend": 5049,
            "Frontend": 7100
        }
        
        # start_all.py에서 포트 설정 확인
        start_all_path = self.base_dir / "start_all.py"
        if start_all_path.exists():
            with open(start_all_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for service, port in expected_ports.items():
                if str(port) in content:
                    self.successes.append(f"✅ {service} 포트 설정 확인: {port}")
                else:
                    self.warnings.append(f"⚠️  {service} 포트 설정 미확인: {port}")
                    
    def print_results(self):
        """검증 결과 출력"""
        print("\n" + "="*60)
        print("📊 검증 결과")
        print("="*60)
        
        # 성공 항목
        if self.successes:
            print(f"\n✅ 성공: {len(self.successes)}개")
            for success in self.successes[:5]:  # 처음 5개만 표시
                print(f"   {success}")
            if len(self.successes) > 5:
                print(f"   ... 외 {len(self.successes)-5}개")
                
        # 경고 항목
        if self.warnings:
            print(f"\n⚠️  경고: {len(self.warnings)}개")
            for warning in self.warnings:
                print(f"   {warning}")
                
        # 오류 항목
        if self.errors:
            print(f"\n❌ 오류: {len(self.errors)}개")
            for error in self.errors:
                print(f"   {error}")
                
        # 최종 판정
        print("\n" + "="*60)
        if not self.errors:
            print("🎉 모든 필수 구현 사항이 README와 일치합니다!")
        else:
            print("⚠️  일부 구현 사항이 누락되었습니다. 위의 오류를 확인하세요.")
            
        # 실행 가능 여부
        print("\n💡 다음 단계:")
        if not self.errors:
            print("1. python start_expert_learning.py - 전문가 학습 시스템 설치")
            print("2. python start_all.py - 전체 시스템 시작")
        else:
            print("1. 위의 오류를 먼저 해결하세요")
            print("2. 다시 검증 스크립트를 실행하세요")

def main():
    verifier = ImplementationVerifier()
    verifier.verify_all()

if __name__ == "__main__":
    main()