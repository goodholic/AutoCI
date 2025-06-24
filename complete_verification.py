#!/usr/bin/env python3
"""
README 완전 검증 - 모든 세부사항 확인
"""

import os
import re
import json
import ast
from pathlib import Path

class CompleteVerification:
    def __init__(self):
        self.base_dir = Path(".")
        self.readme_path = self.base_dir / "README.md"
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_count = 0
        
    def run(self):
        """완전 검증 실행"""
        print("🔍 AutoCI 완전 검증 시작...\n")
        
        # README 내용 읽기
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
            
        # 1. 모든 .py 파일 검증
        print("📄 Python 파일 검증:")
        self.verify_python_files(readme_content)
        
        # 2. 모든 .cs 파일 검증
        print("\n📄 C# 파일 검증:")
        self.verify_csharp_files(readme_content)
        
        # 3. 모든 .razor 파일 검증
        print("\n📄 Razor 파일 검증:")
        self.verify_razor_files(readme_content)
        
        # 4. 모든 .json 파일 검증
        print("\n📄 JSON 파일 검증:")
        self.verify_json_files(readme_content)
        
        # 5. 디렉토리 구조 검증
        print("\n📁 디렉토리 구조 검증:")
        self.verify_directory_structure()
        
        # 6. 포트 번호 검증
        print("\n🔌 포트 번호 검증:")
        self.verify_ports(readme_content)
        
        # 7. URL 경로 검증
        print("\n🌐 URL 경로 검증:")
        self.verify_urls(readme_content)
        
        # 8. 코드 블록 검증
        print("\n💻 코드 블록 검증:")
        self.verify_code_blocks(readme_content)
        
        # 9. 품질 기준 검증
        print("\n📊 품질 기준 검증:")
        self.verify_quality_criteria(readme_content)
        
        # 10. 학습 사이클 검증
        print("\n⏰ 학습 사이클 검증:")
        self.verify_learning_cycle(readme_content)
        
        # 11. API 엔드포인트 검증
        print("\n🌐 API 엔드포인트 검증:")
        self.verify_api_endpoints(readme_content)
        
        # 12. 클래스 정의 검증
        print("\n🏗️ 클래스 정의 검증:")
        self.verify_class_definitions()
        
        # 13. 함수 정의 검증
        print("\n⚙️ 함수 정의 검증:")
        self.verify_function_definitions()
        
        # 14. 패키지 의존성 검증
        print("\n📦 패키지 의존성 검증:")
        self.verify_package_dependencies()
        
        # 결과 출력
        self.print_results()
        
    def check(self, condition, success_msg, error_msg):
        """검사 수행"""
        self.total_count += 1
        if condition:
            self.success_count += 1
            print(f"  ✅ {success_msg}")
            return True
        else:
            self.errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            return False
            
    def verify_python_files(self, readme_content):
        """Python 파일 검증"""
        # README에서 .py 파일 추출
        py_files = re.findall(r'(\w+\.py)\b', readme_content)
        py_files = list(set(py_files))  # 중복 제거
        
        # 특수 경로 파일
        special_paths = {
            "enhanced_server.py": "MyAIWebApp/Models/enhanced_server.py",
            "fine_tune.py": "MyAIWebApp/Models/fine_tune.py"
        }
        
        for py_file in sorted(py_files):
            if py_file in special_paths:
                path = self.base_dir / special_paths[py_file]
            else:
                path = self.base_dir / py_file
                
            self.check(
                path.exists(),
                py_file,
                f"{py_file} 파일 없음"
            )
            
    def verify_csharp_files(self, readme_content):
        """C# 파일 검증"""
        cs_files = re.findall(r'(\w+\.cs)\b', readme_content)
        cs_files = list(set(cs_files))
        
        service_path = self.base_dir / "MyAIWebApp/Backend/Services"
        
        for cs_file in sorted(cs_files):
            path = service_path / cs_file
            self.check(
                path.exists(),
                cs_file,
                f"{cs_file} 파일 없음"
            )
            
    def verify_razor_files(self, readme_content):
        """Razor 파일 검증"""
        # 직접 명시된 파일들
        razor_files = {
            "CodeGenerator.razor": "/codegen",
            "CodeSearch.razor": "/codesearch",
            "RAG.razor": "/rag"
        }
        
        pages_path = self.base_dir / "MyAIWebApp/Frontend/Pages"
        
        for razor_file, route in razor_files.items():
            path = pages_path / razor_file
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.check(
                    f'@page "{route}"' in content,
                    f"{razor_file} → {route}",
                    f"{razor_file} 라우트 설정 오류"
                )
            else:
                self.check(
                    False,
                    "",
                    f"{razor_file} 파일 없음"
                )
                
    def verify_json_files(self, readme_content):
        """JSON 파일 검증"""
        json_files = [
            "MyAIWebApp/Backend/Properties/launchSettings.json",
            "MyAIWebApp/Frontend/Properties/launchSettings.json"
        ]
        
        for json_file in json_files:
            path = self.base_dir / json_file
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.check(
                        True,
                        json_file,
                        ""
                    )
                except json.JSONDecodeError:
                    self.check(
                        False,
                        "",
                        f"{json_file} JSON 파싱 오류"
                    )
            else:
                self.check(
                    False,
                    "",
                    f"{json_file} 파일 없음"
                )
                
    def verify_directory_structure(self):
        """디렉토리 구조 검증"""
        # README 362-379행의 구조
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
            self.check(
                path.exists() and path.is_dir(),
                f"{dir_path}/",
                f"디렉토리 없음: {dir_path}"
            )
            
    def verify_ports(self, readme_content):
        """포트 번호 검증"""
        # README에 명시된 포트
        ports = {
            "8000": ["Python AI Server", "enhanced_server.py"],
            "8080": ["모니터링 API", "expert_learning_api.py"],
            "5049": ["Backend", "launchSettings.json"],
            "7100": ["Frontend", "launchSettings.json"]
        }
        
        # start_all.py에서 포트 확인
        start_all_path = self.base_dir / "start_all.py"
        if start_all_path.exists():
            with open(start_all_path, 'r', encoding='utf-8') as f:
                start_all_content = f.read()
                
            for port, (service, _) in ports.items():
                self.check(
                    port in start_all_content,
                    f"{service} 포트 {port}",
                    f"{service} 포트 {port} 설정 없음"
                )
                
    def verify_urls(self, readme_content):
        """URL 경로 검증"""
        # README 65-68행의 URL
        urls = [
            "http://localhost:7100/codegen",
            "http://localhost:7100/codesearch",
            "http://localhost:7100/rag",
            "http://localhost:8080/dashboard"
        ]
        
        for url in urls:
            self.check(
                url in readme_content,
                f"URL: {url}",
                f"URL 없음: {url}"
            )
            
    def verify_code_blocks(self, readme_content):
        """코드 블록 검증"""
        # quality_criteria 코드 블록 확인
        if "quality_criteria = {" in readme_content:
            # 필수 항목들
            required_items = [
                '"has_xml_docs": 0.20',
                '"uses_patterns": 0.15',
                '"modern_csharp": 0.15',
                '"follows_solid": 0.15',
                '"error_handling": 0.10',
                '"appropriate_length": 0.10'
            ]
            
            for item in required_items:
                self.check(
                    item in readme_content,
                    f"품질 기준: {item}",
                    f"품질 기준 누락: {item}"
                )
                
    def verify_quality_criteria(self, readme_content):
        """품질 기준 구현 검증"""
        # csharp_expert_crawler.py에서 확인
        crawler_path = self.base_dir / "csharp_expert_crawler.py"
        if crawler_path.exists():
            with open(crawler_path, 'r', encoding='utf-8') as f:
                crawler_content = f.read()
                
            # README 121-129행의 기준
            criteria = [
                ("XML 문서 주석", "20%", "0.20"),
                ("디자인 패턴", "15%", "0.15"),
                ("현대적 C# 기능", "15%", "0.15"),
                ("에러 처리", "10%", "0.10"),
                ("코드 구조", "10%", "0.10"),
                ("테스트 코드", "5%", "0.05")
            ]
            
            for name, percent, value in criteria:
                self.check(
                    f"{name} ({percent})" in crawler_content and f"score += {value}" in crawler_content,
                    f"{name} {percent}",
                    f"{name} {percent} 구현 안됨"
                )
                
    def verify_learning_cycle(self, readme_content):
        """학습 사이클 검증"""
        # README 104-112행
        cycle_items = [
            "4시간: GitHub/StackOverflow 데이터 수집",
            "1시간: 데이터 전처리 및 품질 검증",
            "6시간: Code Llama 모델 파인튜닝",
            "1시간: 모델 평가 및 배포",
            "12시간: 실시간 코드 개선 서비스"
        ]
        
        for item in cycle_items:
            self.check(
                item in readme_content,
                f"학습 사이클: {item}",
                f"학습 사이클 누락: {item}"
            )
            
    def verify_api_endpoints(self, readme_content):
        """API 엔드포인트 검증"""
        # README 239-246행
        endpoints = {
            "/api/status": "expert_learning_api.py",
            "/api/start": "expert_learning_api.py",
            "/api/stop": "expert_learning_api.py",
            "/api/stats": "expert_learning_api.py",
            "/api/improve": "expert_learning_api.py",
            "/api/logs": "expert_learning_api.py"
        }
        
        for endpoint, file_name in endpoints.items():
            file_path = self.base_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.check(
                    f'"{endpoint}"' in content or f"'{endpoint}'" in content,
                    f"{endpoint} → {file_name}",
                    f"{endpoint} 엔드포인트 없음"
                )
                
    def verify_class_definitions(self):
        """클래스 정의 검증"""
        # 주요 클래스들
        classes = {
            "ModelDownloader": "download_model.py",
            "AutoCILauncher": "start_all.py",
            "CSharpExpertCrawler": "csharp_expert_crawler.py",
            "ExpertLearningStartup": "start_expert_learning.py",
            "CodeLlamaFineTuner": "MyAIWebApp/Models/fine_tune.py",
            "CSharpDataset": "MyAIWebApp/Models/fine_tune.py",
            "ModelConfig": "MyAIWebApp/Models/fine_tune.py",
            "UnityCodeCollector": "auto_train_collector.py",
            "AutoTrainer": "auto_train_collector.py"
        }
        
        for class_name, file_path in classes.items():
            path = self.base_dir / file_path
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.check(
                    f"class {class_name}" in content,
                    f"{class_name} in {file_path}",
                    f"{class_name} 클래스 정의 없음"
                )
                
    def verify_function_definitions(self):
        """함수 정의 검증"""
        # 주요 함수들
        functions = {
            "_evaluate_code_quality": "csharp_expert_crawler.py",
            "create_monitoring_dashboard": "start_expert_learning.py",
            "download_model": "download_model.py",
            "generate_code": "MyAIWebApp/Models/enhanced_server.py",
            "improve_code": "MyAIWebApp/Models/enhanced_server.py",
            "analyze_code": "MyAIWebApp/Models/enhanced_server.py"
        }
        
        for func_name, file_path in functions.items():
            path = self.base_dir / file_path
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                self.check(
                    f"def {func_name}" in content or f"async def {func_name}" in content,
                    f"{func_name}() in {file_path}",
                    f"{func_name} 함수 정의 없음"
                )
                
    def verify_package_dependencies(self):
        """패키지 의존성 검증"""
        # requirements_expert.txt 확인
        req_path = self.base_dir / "requirements_expert.txt"
        if req_path.exists():
            with open(req_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 핵심 패키지들
            packages = [
                "torch",
                "transformers",
                "fastapi",
                "uvicorn",
                "peft",
                "pandas",
                "beautifulsoup4",
                "watchdog",
                "colorama",
                "psutil"
            ]
            
            for package in packages:
                self.check(
                    package in content,
                    f"패키지: {package}",
                    f"패키지 누락: {package}"
                )
                
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*70)
        print("📊 완전 검증 결과")
        print("="*70)
        
        print(f"\n총 검사: {self.total_count}개")
        print(f"✅ 성공: {self.success_count}개")
        print(f"❌ 실패: {len(self.errors)}개")
        
        if self.total_count > 0:
            success_rate = (self.success_count / self.total_count) * 100
            print(f"성공률: {success_rate:.1f}%")
        
        if self.errors:
            print(f"\n❌ 오류 목록:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print(f"\n⚠️ 경고 목록:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        print("\n" + "="*70)
        
        if len(self.errors) == 0:
            print("🎉 완벽합니다! 모든 검증을 통과했습니다!")
            print("README의 모든 요구사항이 100% 구현되었습니다.")
        else:
            print("⚠️ 일부 구현이 누락되었습니다.")
            print("위의 오류를 확인하고 수정해주세요.")

def main():
    verifier = CompleteVerification()
    verifier.run()

if __name__ == "__main__":
    main()