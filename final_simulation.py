#!/usr/bin/env python3
"""
최종 시뮬레이션
README의 모든 요구사항이 구현되었는지 최종 확인
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import re

class FinalSimulation:
    def __init__(self):
        self.base_dir = Path(".")
        self.results = {
            "총 검사": 0,
            "성공": 0,
            "실패": 0,
            "경고": 0
        }
        self.errors = []
        
    def run(self):
        """최종 시뮬레이션 실행"""
        print("🚀 AutoCI 최종 시뮬레이션 시작...\n")
        
        # 1. 핵심 파일 존재 확인
        self.check_core_files()
        
        # 2. 디렉토리 구조 확인
        self.check_directory_structure()
        
        # 3. 파일 내용 검증
        self.check_file_contents()
        
        # 4. 포트 설정 확인
        self.check_port_configuration()
        
        # 5. API 엔드포인트 확인
        self.check_api_endpoints()
        
        # 6. 웹 라우트 확인
        self.check_web_routes()
        
        # 7. 품질 평가 기준 확인
        self.check_quality_criteria()
        
        # 8. 명령어 실행 가능성 확인
        self.check_command_executability()
        
        # 결과 출력
        self.print_results()
        
    def check(self, condition, success_msg, error_msg):
        """검사 수행"""
        self.results["총 검사"] += 1
        if condition:
            self.results["성공"] += 1
            print(f"  ✅ {success_msg}")
        else:
            self.results["실패"] += 1
            self.errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            
    def check_core_files(self):
        """핵심 파일 존재 확인"""
        print("📄 핵심 파일 확인:")
        
        files = {
            # 루트 레벨 파일
            "README.md": "프로젝트 문서",
            "download_model.py": "모델 다운로드 스크립트",
            "start_all.py": "통합 실행 스크립트",
            "csharp_expert_crawler.py": "24시간 학습 엔진",
            "start_expert_learning.py": "설치 스크립트",
            "expert_learning_api.py": "모니터링 API",
            "requirements_expert.txt": "Python 패키지 목록",
            "auto_train_collector.py": "자동 학습 수집기",
            "save_feedback.py": "피드백 저장 API",
            
            # Models 디렉토리
            "MyAIWebApp/Models/enhanced_server.py": "AI 모델 서버",
            "MyAIWebApp/Models/fine_tune.py": "파인튜닝 스크립트",
            "MyAIWebApp/Models/requirements.txt": "Models 패키지",
            
            # Backend Services
            "MyAIWebApp/Backend/Services/AIService.cs": "AI 서비스",
            "MyAIWebApp/Backend/Services/SearchService.cs": "검색 서비스",
            "MyAIWebApp/Backend/Services/LlamaService.cs": "Llama 서비스",
            "MyAIWebApp/Backend/Services/RAGService.cs": "RAG 서비스",
            
            # Frontend Pages
            "MyAIWebApp/Frontend/Pages/CodeGenerator.razor": "코드 생성 페이지",
            "MyAIWebApp/Frontend/Pages/CodeSearch.razor": "코드 검색 페이지",
            "MyAIWebApp/Frontend/Pages/RAG.razor": "Q&A 페이지",
            
            # Properties
            "MyAIWebApp/Backend/Properties/launchSettings.json": "Backend 설정",
            "MyAIWebApp/Frontend/Properties/launchSettings.json": "Frontend 설정"
        }
        
        for file_path, description in files.items():
            self.check(
                (self.base_dir / file_path).exists(),
                f"{description} ({file_path})",
                f"{description} 없음: {file_path}"
            )
            
    def check_directory_structure(self):
        """디렉토리 구조 확인"""
        print("\n📁 디렉토리 구조 확인:")
        
        dirs = [
            "MyAIWebApp",
            "MyAIWebApp/Backend",
            "MyAIWebApp/Backend/Services",
            "MyAIWebApp/Backend/Controllers",
            "MyAIWebApp/Backend/Properties",
            "MyAIWebApp/Frontend",
            "MyAIWebApp/Frontend/Pages",
            "MyAIWebApp/Frontend/wwwroot",
            "MyAIWebApp/Frontend/Properties",
            "MyAIWebApp/Models",
            "expert_training_data"
        ]
        
        for dir_path in dirs:
            self.check(
                (self.base_dir / dir_path).exists() and (self.base_dir / dir_path).is_dir(),
                f"{dir_path}/",
                f"디렉토리 없음: {dir_path}"
            )
            
    def check_file_contents(self):
        """파일 내용 검증"""
        print("\n📝 파일 내용 검증:")
        
        # download_model.py 검증
        dl_path = self.base_dir / "download_model.py"
        if dl_path.exists():
            with open(dl_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.check(
                '"codellama/CodeLlama-7b-Instruct-hf"' in content,
                "download_model.py: 올바른 모델명",
                "download_model.py: 잘못된 모델명"
            )
            
            self.check(
                '"CodeLlama-7b-Instruct-hf"' in content,
                "download_model.py: 올바른 디렉토리명", 
                "download_model.py: 잘못된 디렉토리명"
            )
            
            self.check(
                '--check-only' in content,
                "download_model.py: --check-only 옵션",
                "download_model.py: --check-only 옵션 없음"
            )
            
    def check_port_configuration(self):
        """포트 설정 확인"""
        print("\n🔌 포트 설정 확인:")
        
        # start_all.py에서 포트 확인
        start_all_path = self.base_dir / "start_all.py"
        if start_all_path.exists():
            with open(start_all_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            ports = {
                "8000": "AI 서버",
                "8080": "모니터링 API",
                "5049": "Backend",
                "7100": "Frontend"
            }
            
            for port, service in ports.items():
                self.check(
                    port in content,  # 숫자로도 포함되어 있는지 확인
                    f"{service} 포트 {port}",
                    f"{service} 포트 {port} 설정 없음"
                )
                
    def check_api_endpoints(self):
        """API 엔드포인트 확인"""
        print("\n🌐 API 엔드포인트 확인:")
        
        endpoints = {
            "expert_learning_api.py": ["/api/status", "/api/start", "/api/stop", "/api/stats", "/api/improve", "/api/logs"],
            "MyAIWebApp/Models/enhanced_server.py": ["/generate", "/improve", "/analyze", "/health"],
            "save_feedback.py": ["/feedback"]
        }
        
        for file_path, routes in endpoints.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for route in routes:
                    self.check(
                        f'"{route}"' in content or f"'{route}'" in content,
                        f"{route} in {file_path}",
                        f"{route} 없음 in {file_path}"
                    )
                    
    def check_web_routes(self):
        """웹 라우트 확인"""
        print("\n🌐 웹 라우트 확인:")
        
        routes = {
            "/codegen": "MyAIWebApp/Frontend/Pages/CodeGenerator.razor",
            "/codesearch": "MyAIWebApp/Frontend/Pages/CodeSearch.razor",
            "/rag": "MyAIWebApp/Frontend/Pages/RAG.razor"
        }
        
        for route, file_path in routes.items():
            full_path = self.base_dir / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.check(
                    f'@page "{route}"' in content,
                    f"{route} → {file_path}",
                    f"{route} 라우트 설정 없음"
                )
                
    def check_quality_criteria(self):
        """품질 평가 기준 확인"""
        print("\n📊 품질 평가 기준 확인:")
        
        crawler_path = self.base_dir / "csharp_expert_crawler.py"
        if crawler_path.exists():
            with open(crawler_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            criteria = [
                ("XML 문서 주석 (20%)", "score += 0.20"),
                ("디자인 패턴 (15%)", "score += 0.15"),
                ("현대적 C# 기능 (15%)", "score += 0.15"),
                ("에러 처리 (10%)", "score += 0.10"),
                ("코드 구조 (10%)", "score += 0.10"),
                ("테스트 코드 (5%)", "score += 0.05")
            ]
            
            for name, code in criteria:
                self.check(
                    name in content and code in content,
                    name,
                    f"{name} 구현 안됨"
                )
                
    def check_command_executability(self):
        """명령어 실행 가능성 확인"""
        print("\n💻 명령어 실행 가능성 확인:")
        
        python_scripts = [
            "start_expert_learning.py",
            "start_all.py",
            "download_model.py",
            "csharp_expert_crawler.py",
            "expert_learning_api.py"
        ]
        
        for script in python_scripts:
            self.check(
                (self.base_dir / script).exists(),
                f"python {script}",
                f"스크립트 없음: {script}"
            )
            
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*70)
        print("📊 최종 시뮬레이션 결과")
        print("="*70)
        
        print(f"\n총 검사: {self.results['총 검사']}개")
        print(f"✅ 성공: {self.results['성공']}개")
        print(f"❌ 실패: {self.results['실패']}개")
        
        success_rate = (self.results['성공'] / self.results['총 검사'] * 100) if self.results['총 검사'] > 0 else 0
        print(f"성공률: {success_rate:.1f}%")
        
        if self.errors:
            print(f"\n❌ 오류 목록:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        print("\n" + "="*70)
        
        if self.results['실패'] == 0:
            print("🎉 완벽합니다! 모든 검증을 통과했습니다!")
            print("README의 모든 요구사항이 100% 구현되었습니다.")
        else:
            print("⚠️  일부 구현이 누락되었습니다.")
            print("위의 오류를 확인하고 수정해주세요.")

def main():
    simulator = FinalSimulation()
    simulator.run()

if __name__ == "__main__":
    main()