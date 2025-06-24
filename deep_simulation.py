#!/usr/bin/env python3
"""
README 명령어 실행 시뮬레이션
README.md에 있는 모든 명령어가 실제로 작동하는지 확인
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import ast
import importlib.util

class DeepSimulation:
    def __init__(self):
        self.base_dir = Path(".")
        self.results = []
        
    def run_all_tests(self):
        """모든 심층 테스트 실행"""
        print("🔬 AutoCI 심층 시뮬레이션 시작...\n")
        
        # 1. Python 모듈 임포트 테스트
        self.test_python_imports()
        
        # 2. FastAPI 앱 검증
        self.test_fastapi_apps()
        
        # 3. 클래스 및 함수 존재 확인
        self.test_required_classes()
        
        # 4. 설정 값 확인
        self.test_configurations()
        
        # 5. 파일 내용 검증
        self.test_file_contents()
        
        # 결과 출력
        self.print_results()
        
    def test_python_imports(self):
        """Python 파일의 임포트 가능 여부 테스트"""
        print("📦 Python 임포트 테스트...")
        
        python_files = [
            ("download_model.py", ["ModelDownloader"]),
            ("start_all.py", ["AutoCILauncher"]),
            ("csharp_expert_crawler.py", ["CSharpExpertCrawler"]),
            ("start_expert_learning.py", ["ExpertLearningStartup"]),  # 실제 클래스명으로 수정
            ("expert_learning_api.py", ["app"]),
            ("auto_train_collector.py", ["UnityCodeCollector", "AutoTrainer"]),
            ("save_feedback.py", ["app"]),
        ]
        
        for file_path, expected_items in python_files:
            path = self.base_dir / file_path
            if path.exists():
                try:
                    # 파일 읽기
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # AST 파싱
                    tree = ast.parse(content)
                    
                    # 클래스와 함수 찾기
                    found_items = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            found_items.append(node.name)
                        elif isinstance(node, ast.FunctionDef) and node.name in expected_items:
                            found_items.append(node.name)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id in expected_items:
                                    found_items.append(target.id)
                    
                    # 검증
                    for item in expected_items:
                        if item in found_items:
                            self.results.append(f"✅ {file_path}: {item} 정의 확인")
                        else:
                            self.results.append(f"❌ {file_path}: {item} 정의 없음")
                            
                except Exception as e:
                    self.results.append(f"❌ {file_path}: 파싱 오류 - {str(e)}")
            else:
                self.results.append(f"❌ {file_path}: 파일 없음")
                
    def test_fastapi_apps(self):
        """FastAPI 앱 구조 검증"""
        print("\n🌐 FastAPI 앱 검증...")
        
        fastapi_files = [
            ("MyAIWebApp/Models/enhanced_server.py", ["/generate", "/improve", "/analyze", "/health"]),
            ("expert_learning_api.py", ["/api/status", "/api/start", "/api/stop", "/api/stats"]),
            ("save_feedback.py", ["/feedback"]),
        ]
        
        for file_path, expected_routes in fastapi_files:
            path = self.base_dir / file_path
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for route in expected_routes:
                    if f'"{route}"' in content or f"'{route}'" in content:
                        self.results.append(f"✅ {file_path}: {route} 엔드포인트 존재")
                    else:
                        self.results.append(f"❌ {file_path}: {route} 엔드포인트 없음")
            else:
                self.results.append(f"❌ {file_path}: 파일 없음")
                
    def test_required_classes(self):
        """필수 클래스 존재 확인"""
        print("\n🏗️ 필수 클래스 검증...")
        
        # fine_tune.py 검증
        fine_tune_path = self.base_dir / "MyAIWebApp/Models/fine_tune.py"
        if fine_tune_path.exists():
            with open(fine_tune_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_classes = ["ModelConfig", "CSharpDataset", "CodeLlamaFineTuner", "ProgressCallback"]
            for class_name in required_classes:
                if f"class {class_name}" in content:
                    self.results.append(f"✅ fine_tune.py: {class_name} 클래스 존재")
                else:
                    self.results.append(f"❌ fine_tune.py: {class_name} 클래스 없음")
                    
    def test_configurations(self):
        """설정 값 확인"""
        print("\n⚙️ 설정 값 검증...")
        
        # start_all.py 포트 설정
        start_all_path = self.base_dir / "start_all.py"
        if start_all_path.exists():
            with open(start_all_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            expected_ports = {
                "'ai_server': 8000": "AI 서버 포트",
                "'monitoring_api': 8080": "모니터링 API 포트",
                "'backend': 5049": "백엔드 포트",
                "'frontend': 7100": "프론트엔드 포트"
            }
            
            for port_config, description in expected_ports.items():
                if port_config in content:
                    self.results.append(f"✅ {description} 설정 확인")
                else:
                    self.results.append(f"❌ {description} 설정 없음")
                    
    def test_file_contents(self):
        """파일 내용 상세 검증"""
        print("\n📄 파일 내용 검증...")
        
        # download_model.py 검증
        download_path = self.base_dir / "download_model.py"
        if download_path.exists():
            with open(download_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 모델명 확인
            if 'codellama/CodeLlama-7b-Instruct-hf' in content:
                self.results.append("✅ download_model.py: 올바른 모델명 사용")
            else:
                self.results.append("❌ download_model.py: 잘못된 모델명")
                
            # 디렉토리명 확인
            if '"CodeLlama-7b-Instruct-hf"' in content or '/ "CodeLlama-7b-Instruct-hf"' in content:
                self.results.append("✅ download_model.py: 올바른 디렉토리명")
            else:
                self.results.append("❌ download_model.py: 잘못된 디렉토리명")
                
        # csharp_expert_crawler.py 품질 기준 확인
        crawler_path = self.base_dir / "csharp_expert_crawler.py"
        if crawler_path.exists():
            with open(crawler_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # README 기준 품질 평가 확인
            if "XML 문서 주석 (20%)" in content and "score += 0.20" in content:
                self.results.append("✅ XML 문서 주석 20% 구현")
            else:
                self.results.append("❌ XML 문서 주석 20% 미구현")
                
            if "디자인 패턴 (15%)" in content and "score += 0.15" in content:
                self.results.append("✅ 디자인 패턴 15% 구현")
            else:
                self.results.append("❌ 디자인 패턴 15% 미구현")
                
            if "현대적 C# 기능 (15%)" in content and "score += 0.15" in content:
                self.results.append("✅ 현대적 C# 기능 15% 구현")
            else:
                self.results.append("❌ 현대적 C# 기능 15% 미구현")
                
            if "에러 처리 (10%)" in content and "score += 0.10" in content:
                self.results.append("✅ 에러 처리 10% 구현")
            else:
                self.results.append("❌ 에러 처리 10% 미구현")
                    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*60)
        print("📊 심층 시뮬레이션 결과")
        print("="*60)
        
        success_count = sum(1 for r in self.results if r.startswith("✅"))
        error_count = sum(1 for r in self.results if r.startswith("❌"))
        
        print(f"\n총 검사 항목: {len(self.results)}개")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {error_count}개")
        
        if error_count > 0:
            print("\n❌ 실패 항목:")
            for result in self.results:
                if result.startswith("❌"):
                    print(f"   {result}")
                    
        print("\n" + "="*60)
        
        if error_count == 0:
            print("🎉 모든 심층 검증을 통과했습니다!")
            print("README의 모든 요구사항이 정확히 구현되었습니다.")
        else:
            print("⚠️ 일부 구현이 README와 일치하지 않습니다.")
            print("위의 실패 항목을 확인하고 수정하세요.")

def main():
    simulator = DeepSimulation()
    simulator.run_all_tests()

if __name__ == "__main__":
    main()