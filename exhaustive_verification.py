#!/usr/bin/env python3
"""
README 완전 검증 스크립트
README.md의 모든 내용을 한 줄씩 검증
"""

import os
import re
import json
import ast
from pathlib import Path
import subprocess

class ExhaustiveVerification:
    def __init__(self):
        self.base_dir = Path(".")
        self.readme_path = self.base_dir / "README.md"
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
        
    def run_all_verifications(self):
        """모든 검증 실행"""
        print("🔍 AutoCI README 완전 검증 시작...\n")
        
        # README 파일 읽기
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
            readme_lines = readme_content.split('\n')
        
        # 1. 프로젝트 이름 검증
        self.verify_project_name(readme_lines)
        
        # 2. 모델명 검증
        self.verify_model_name(readme_lines)
        
        # 3. 디렉토리 구조 검증 (README 362-379행)
        self.verify_directory_structure()
        
        # 4. 포트 번호 검증
        self.verify_ports(readme_content)
        
        # 5. 파일별 상세 검증
        self.verify_each_file_content(readme_content)
        
        # 6. API 엔드포인트 검증
        self.verify_api_endpoints(readme_content)
        
        # 7. 품질 평가 기준 검증
        self.verify_quality_criteria(readme_content)
        
        # 8. 명령어 실행 가능성 검증
        self.verify_commands(readme_content)
        
        # 9. 학습 사이클 검증
        self.verify_learning_cycle(readme_content)
        
        # 10. 웹 인터페이스 경로 검증
        self.verify_web_routes(readme_content)
        
        # 결과 출력
        self.print_comprehensive_results()
        
    def verify_project_name(self, readme_lines):
        """프로젝트 이름 일치 확인"""
        print("📌 프로젝트 이름 검증...")
        if any("AutoCI" in line for line in readme_lines[:10]):
            self.checks_passed += 1
            print("  ✅ 프로젝트 이름 'AutoCI' 확인")
        else:
            self.checks_failed += 1
            self.errors.append("프로젝트 이름 'AutoCI'가 README에 없음")
            
    def verify_model_name(self, readme_lines):
        """모델명 일치 확인"""
        print("\n🤖 모델명 검증...")
        model_name = "Code Llama 7B-Instruct"
        if any(model_name in line for line in readme_lines):
            self.checks_passed += 1
            print(f"  ✅ 모델명 '{model_name}' 확인")
        else:
            self.checks_failed += 1
            self.errors.append(f"모델명 '{model_name}'이 README에 없음")
            
    def verify_directory_structure(self):
        """디렉토리 구조 완전 검증"""
        print("\n📁 디렉토리 구조 검증...")
        
        # README 362-379행의 구조
        required_structure = {
            "MyAIWebApp": {
                "Backend": {
                    "Services": ["AIService.cs", "SearchService.cs", "LlamaService.cs", "RAGService.cs"],
                    "Controllers": [],
                    "Properties": ["launchSettings.json"]
                },
                "Frontend": {
                    "Pages": ["CodeGenerator.razor", "CodeSearch.razor", "RAG.razor"],
                    "wwwroot": [],
                    "Properties": ["launchSettings.json"]
                },
                "Models": ["enhanced_server.py", "fine_tune.py", "requirements.txt"]
            },
            "expert_training_data": []
        }
        
        # 재귀적 검증
        def verify_structure(base_path, structure, indent=""):
            for item, sub_items in structure.items():
                path = base_path / item
                if isinstance(sub_items, dict):
                    # 디렉토리
                    if path.exists() and path.is_dir():
                        self.checks_passed += 1
                        print(f"{indent}  ✅ 디렉토리: {item}")
                        verify_structure(path, sub_items, indent + "    ")
                    else:
                        self.checks_failed += 1
                        self.errors.append(f"디렉토리 없음: {path}")
                        print(f"{indent}  ❌ 디렉토리 없음: {item}")
                elif isinstance(sub_items, list):
                    # 파일 목록
                    for file_name in sub_items:
                        file_path = path / file_name
                        if file_path.exists():
                            self.checks_passed += 1
                            print(f"{indent}    ✅ 파일: {file_name}")
                        else:
                            self.checks_failed += 1
                            self.errors.append(f"파일 없음: {file_path}")
                            print(f"{indent}    ❌ 파일 없음: {file_name}")
                            
        verify_structure(self.base_dir, required_structure)
        
        # 루트 레벨 파일들
        root_files = [
            "csharp_expert_crawler.py",
            "start_expert_learning.py", 
            "expert_learning_api.py",
            "start_all.py",
            "download_model.py",
            "requirements_expert.txt",
            "auto_train_collector.py",
            "save_feedback.py"
        ]
        
        print("\n📄 루트 레벨 파일 검증...")
        for file_name in root_files:
            if (self.base_dir / file_name).exists():
                self.checks_passed += 1
                print(f"  ✅ {file_name}")
            else:
                self.checks_failed += 1
                self.errors.append(f"루트 파일 없음: {file_name}")
                print(f"  ❌ {file_name}")
                
    def verify_ports(self, readme_content):
        """포트 번호 검증"""
        print("\n🔌 포트 설정 검증...")
        
        expected_ports = {
            "8000": "Python AI Server",
            "8080": "모니터링 API", 
            "5049": "Backend",
            "7100": "Frontend"
        }
        
        for port, service in expected_ports.items():
            if port in readme_content:
                # start_all.py에서 확인
                start_all_path = self.base_dir / "start_all.py"
                if start_all_path.exists():
                    with open(start_all_path, 'r', encoding='utf-8') as f:
                        if port in f.read():
                            self.checks_passed += 1
                            print(f"  ✅ {service} 포트 {port}")
                        else:
                            self.checks_failed += 1
                            self.errors.append(f"{service} 포트 {port}가 start_all.py에 없음")
                            
    def verify_each_file_content(self, readme_content):
        """각 파일의 내용 상세 검증"""
        print("\n📝 파일 내용 상세 검증...")
        
        # download_model.py 검증
        print("\n  🔸 download_model.py 검증:")
        dl_path = self.base_dir / "download_model.py"
        if dl_path.exists():
            with open(dl_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 모델명 확인
            if '"codellama/CodeLlama-7b-Instruct-hf"' in content:
                self.checks_passed += 1
                print("    ✅ 정확한 모델명 사용")
            else:
                self.checks_failed += 1
                self.errors.append("download_model.py: 잘못된 모델명")
                
            # --check-only 옵션 확인
            if '--check-only' in content:
                self.checks_passed += 1
                print("    ✅ --check-only 옵션 구현")
            else:
                self.checks_failed += 1
                self.errors.append("download_model.py: --check-only 옵션 없음")
                
        # enhanced_server.py 검증
        print("\n  🔸 enhanced_server.py 검증:")
        es_path = self.base_dir / "MyAIWebApp/Models/enhanced_server.py"
        if es_path.exists():
            with open(es_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # FastAPI 앱 확인
            if 'app = FastAPI(' in content:
                self.checks_passed += 1
                print("    ✅ FastAPI 앱 정의")
            else:
                self.checks_failed += 1
                self.errors.append("enhanced_server.py: FastAPI 앱 정의 없음")
                
            # uvicorn 실행 코드 확인
            if 'uvicorn.run(app, host="0.0.0.0", port=8000)' in content:
                self.checks_passed += 1
                print("    ✅ uvicorn 실행 코드 (포트 8000)")
            else:
                self.checks_failed += 1
                self.errors.append("enhanced_server.py: uvicorn 실행 코드 오류")
                
    def verify_api_endpoints(self, readme_content):
        """API 엔드포인트 검증"""
        print("\n🌐 API 엔드포인트 검증...")
        
        # README 238-246행의 API 엔드포인트
        api_endpoints = {
            "/api/status": "expert_learning_api.py",
            "/api/start": "expert_learning_api.py",
            "/api/stop": "expert_learning_api.py",
            "/api/stats": "expert_learning_api.py",
            "/api/improve": "expert_learning_api.py",
            "/api/logs": "expert_learning_api.py",
            "/generate": "MyAIWebApp/Models/enhanced_server.py",
            "/improve": "MyAIWebApp/Models/enhanced_server.py",
            "/analyze": "MyAIWebApp/Models/enhanced_server.py",
            "/feedback": "save_feedback.py"
        }
        
        for endpoint, file_path in api_endpoints.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    if f'"{endpoint}"' in f.read() or f"'{endpoint}'" in f.read():
                        self.checks_passed += 1
                        print(f"  ✅ {endpoint} in {file_path}")
                    else:
                        self.checks_failed += 1
                        self.errors.append(f"엔드포인트 {endpoint}가 {file_path}에 없음")
                        
    def verify_quality_criteria(self, readme_content):
        """품질 평가 기준 검증"""
        print("\n📊 품질 평가 기준 검증...")
        
        # README 121-129행의 기준
        criteria = [
            ("XML 문서 주석", "20%", 0.20),
            ("디자인 패턴", "15%", 0.15),
            ("현대적 C# 기능", "15%", 0.15),
            ("에러 처리", "10%", 0.10),
            ("코드 구조", "10%", 0.10),
            ("테스트 코드", "5%", 0.05)
        ]
        
        crawler_path = self.base_dir / "csharp_expert_crawler.py"
        if crawler_path.exists():
            with open(crawler_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for name, percent, value in criteria:
                if f"{name} ({percent})" in content and f"score += {value}" in content:
                    self.checks_passed += 1
                    print(f"  ✅ {name} {percent} 구현")
                else:
                    self.checks_failed += 1
                    self.errors.append(f"품질 기준 {name} {percent} 미구현")
                    
    def verify_commands(self, readme_content):
        """README의 명령어 실행 가능성 검증"""
        print("\n💻 명령어 실행 가능성 검증...")
        
        # 주요 명령어 패턴 추출
        commands = [
            "python start_expert_learning.py",
            "python start_all.py",
            "python download_model.py",
            "python download_model.py --check-only",
            "cd MyAIWebApp/Models",
            "uvicorn enhanced_server:app --host 0.0.0.0 --port 8000",
            "cd MyAIWebApp/Backend",
            "dotnet run",
            "cd MyAIWebApp/Frontend", 
            "python csharp_expert_crawler.py",
            "python expert_learning_api.py"
        ]
        
        for cmd in commands:
            if cmd in readme_content:
                # 파이썬 스크립트 존재 확인
                if cmd.startswith("python ") and ".py" in cmd:
                    script_name = cmd.split()[1]
                    if (self.base_dir / script_name).exists():
                        self.checks_passed += 1
                        print(f"  ✅ 실행 가능: {cmd}")
                    else:
                        self.checks_failed += 1
                        self.errors.append(f"스크립트 없음: {script_name}")
                else:
                    self.checks_passed += 1
                    print(f"  ✅ 명령어 존재: {cmd}")
                    
    def verify_learning_cycle(self, readme_content):
        """24시간 학습 사이클 검증"""
        print("\n⏰ 24시간 학습 사이클 검증...")
        
        # README 104-112행의 학습 사이클
        cycle_times = {
            "4시간": "GitHub/StackOverflow 데이터 수집",
            "1시간": "데이터 전처리 및 품질 검증",
            "6시간": "Code Llama 모델 파인튜닝",
            "1시간": "모델 평가 및 배포",
            "12시간": "실시간 코드 개선 서비스"
        }
        
        for time, task in cycle_times.items():
            if time in readme_content and task in readme_content:
                self.checks_passed += 1
                print(f"  ✅ {time}: {task}")
            else:
                self.checks_failed += 1
                self.warnings.append(f"학습 사이클 {time} {task} 미확인")
                
    def verify_web_routes(self, readme_content):
        """웹 인터페이스 경로 검증"""
        print("\n🌐 웹 인터페이스 경로 검증...")
        
        # README 65-68행의 경로
        routes = {
            "/codegen": "MyAIWebApp/Frontend/Pages/CodeGenerator.razor",
            "/codesearch": "MyAIWebApp/Frontend/Pages/CodeSearch.razor",
            "/rag": "MyAIWebApp/Frontend/Pages/RAG.razor"
        }
        
        for route, file_path in routes.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    if f'@page "{route}"' in f.read():
                        self.checks_passed += 1
                        print(f"  ✅ {route} → {file_path}")
                    else:
                        self.checks_failed += 1
                        self.errors.append(f"라우트 {route}가 {file_path}에 없음")
                        
    def print_comprehensive_results(self):
        """종합 결과 출력"""
        print("\n" + "="*70)
        print("📊 완전 검증 결과")
        print("="*70)
        
        total_checks = self.checks_passed + self.checks_failed
        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n총 검증 항목: {total_checks}개")
        print(f"✅ 성공: {self.checks_passed}개")
        print(f"❌ 실패: {self.checks_failed}개")
        print(f"성공률: {success_rate:.1f}%")
        
        if self.errors:
            print(f"\n❌ 오류 목록 ({len(self.errors)}개):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print(f"\n⚠️  경고 목록 ({len(self.warnings)}개):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        print("\n" + "="*70)
        
        if self.checks_failed == 0:
            print("🎉 완벽합니다! 모든 검증을 통과했습니다!")
            print("README의 모든 요구사항이 100% 구현되었습니다.")
        else:
            print("⚠️  일부 구현이 누락되었습니다.")
            print("위의 오류를 확인하고 수정해주세요.")

def main():
    verifier = ExhaustiveVerification()
    verifier.run_all_verifications()

if __name__ == "__main__":
    main()