#!/usr/bin/env python3
"""
README 상호 참조 검증
README에서 언급된 모든 파일, 함수, 클래스가 실제로 존재하는지 검증
"""

import re
import ast
from pathlib import Path

class CrossReferenceVerification:
    def __init__(self):
        self.base_dir = Path(".")
        self.readme_path = self.base_dir / "README.md"
        self.mismatches = []
        
    def run_verification(self):
        """상호 참조 검증 실행"""
        print("🔗 AutoCI 상호 참조 검증 시작...\n")
        
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # 1. README에서 언급된 모든 파일명 추출 및 검증
        self.verify_mentioned_files(readme_content)
        
        # 2. README에서 언급된 클래스/함수 검증
        self.verify_mentioned_code_elements(readme_content)
        
        # 3. README의 코드 예제 검증
        self.verify_code_examples(readme_content)
        
        # 4. URL 경로 검증
        self.verify_url_paths(readme_content)
        
        # 5. 설정값 일치성 검증
        self.verify_configuration_values(readme_content)
        
        # 결과 출력
        self.print_results()
        
    def verify_mentioned_files(self, readme_content):
        """README에서 언급된 파일 검증"""
        print("📄 언급된 파일 검증...")
        
        # .py, .cs, .json, .txt, .md 파일 패턴
        file_patterns = [
            r'(\w+\.py)',
            r'(\w+\.cs)', 
            r'(\w+\.json)',
            r'(\w+\.txt)',
            r'(\w+\.razor)',
            r'(\w+\.md)'
        ]
        
        mentioned_files = set()
        for pattern in file_patterns:
            mentioned_files.update(re.findall(pattern, readme_content))
        
        # 특수 경로 파일들
        special_files = {
            "enhanced_server.py": "MyAIWebApp/Models/enhanced_server.py",
            "fine_tune.py": "MyAIWebApp/Models/fine_tune.py",
            "AIService.cs": "MyAIWebApp/Backend/Services/AIService.cs",
            "SearchService.cs": "MyAIWebApp/Backend/Services/SearchService.cs",
            "LlamaService.cs": "MyAIWebApp/Backend/Services/LlamaService.cs",
            "RAGService.cs": "MyAIWebApp/Backend/Services/RAGService.cs",
            "CodeGenerator.razor": "MyAIWebApp/Frontend/Pages/CodeGenerator.razor",
            "CodeSearch.razor": "MyAIWebApp/Frontend/Pages/CodeSearch.razor",
            "RAG.razor": "MyAIWebApp/Frontend/Pages/RAG.razor"
        }
        
        for file_name in mentioned_files:
            if file_name in ["README.md", "CLAUDE.md", "TroubleShooting.md"]:
                continue  # 문서 파일은 스킵
                
            # 파일 경로 찾기
            if file_name in special_files:
                file_path = self.base_dir / special_files[file_name]
            else:
                file_path = self.base_dir / file_name
                
            if file_path.exists():
                print(f"  ✅ {file_name}")
            else:
                # 하위 디렉토리에서 검색
                found = False
                for p in self.base_dir.rglob(file_name):
                    if p.exists():
                        print(f"  ✅ {file_name} (경로: {p.relative_to(self.base_dir)})")
                        found = True
                        break
                        
                if not found:
                    print(f"  ❌ {file_name} - 파일을 찾을 수 없음")
                    self.mismatches.append(f"README에 언급된 {file_name} 파일이 없음")
                    
    def verify_mentioned_code_elements(self, readme_content):
        """README에서 언급된 클래스/함수 검증"""
        print("\n🔧 언급된 코드 요소 검증...")
        
        # 클래스명 패턴
        class_patterns = [
            r'class\s+(\w+)',
            r'(\w+Service)',
            r'(\w+Crawler)',
            r'(\w+Launcher)',
            r'(\w+Config)'
        ]
        
        mentioned_classes = set()
        for pattern in class_patterns:
            mentioned_classes.update(re.findall(pattern, readme_content))
        
        # 주요 클래스 검증
        important_classes = {
            "ModelDownloader": "download_model.py",
            "AutoCILauncher": "start_all.py",
            "CSharpExpertCrawler": "csharp_expert_crawler.py",
            "CodeLlamaFineTuner": "MyAIWebApp/Models/fine_tune.py",
            "UnityCodeCollector": "auto_train_collector.py",
            "AutoTrainer": "auto_train_collector.py"
        }
        
        for class_name, file_name in important_classes.items():
            file_path = self.base_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    if f"class {class_name}" in f.read():
                        print(f"  ✅ {class_name} in {file_name}")
                    else:
                        print(f"  ❌ {class_name} not found in {file_name}")
                        self.mismatches.append(f"클래스 {class_name}가 {file_name}에 없음")
                        
    def verify_code_examples(self, readme_content):
        """README의 코드 예제 검증"""
        print("\n💻 코드 예제 검증...")
        
        # 코드 블록 추출
        code_blocks = re.findall(r'```(?:python|bash|csharp)?\n(.*?)\n```', readme_content, re.DOTALL)
        
        # 특정 코드 패턴 검증
        patterns_to_check = [
            ("python start_expert_learning.py", "start_expert_learning.py 존재"),
            ("python start_all.py", "start_all.py 존재"),
            ("uvicorn enhanced_server:app", "enhanced_server.py의 FastAPI 앱"),
            ("quality_criteria =", "품질 평가 기준 예제")
        ]
        
        for pattern, description in patterns_to_check:
            found = any(pattern in block for block in code_blocks)
            if found:
                print(f"  ✅ {description}")
            else:
                print(f"  ⚠️  {description} - README에 예제 없음")
                
    def verify_url_paths(self, readme_content):
        """URL 경로 검증"""
        print("\n🌐 URL 경로 검증...")
        
        # URL 패턴 추출
        urls = re.findall(r'http://localhost:(\d+)(/[\w/]*)?', readme_content)
        
        expected_urls = {
            ("7100", "/codegen"): "AI 코드 생성",
            ("7100", "/codesearch"): "스마트 검색",
            ("7100", "/rag"): "프로젝트 Q&A",
            ("8080", "/dashboard"): "학습 대시보드"
        }
        
        for (port, path), description in expected_urls.items():
            url = f"http://localhost:{port}{path or ''}"
            if any(p == port and (path is None or path == p) for p, p in urls):
                print(f"  ✅ {url} - {description}")
            else:
                if url in readme_content:
                    print(f"  ✅ {url} - {description}")
                else:
                    print(f"  ❌ {url} - {description} URL이 README에 없음")
                    self.mismatches.append(f"URL {url} 누락")
                    
    def verify_configuration_values(self, readme_content):
        """설정값 일치성 검증"""
        print("\n⚙️ 설정값 일치성 검증...")
        
        # 포트 번호 일치성
        print("\n  포트 번호:")
        port_configs = {
            "8000": ["enhanced_server.py", "start_all.py"],
            "8080": ["expert_learning_api.py", "start_all.py"],
            "5049": ["launchSettings.json", "start_all.py"],
            "7100": ["launchSettings.json", "start_all.py"]
        }
        
        for port, files in port_configs.items():
            consistent = True
            for file_name in files:
                # 파일 찾기
                if file_name == "launchSettings.json":
                    paths = list(self.base_dir.rglob(file_name))
                else:
                    paths = [self.base_dir / file_name]
                    
                for path in paths:
                    if path.exists():
                        with open(path, 'r', encoding='utf-8') as f:
                            if port not in f.read():
                                consistent = False
                                break
                                
            if consistent:
                print(f"    ✅ 포트 {port} 일치")
            else:
                print(f"    ❌ 포트 {port} 불일치")
                self.mismatches.append(f"포트 {port} 설정 불일치")
                
        # 모델명 일치성
        print("\n  모델명:")
        model_name = "codellama/CodeLlama-7b-Instruct-hf"
        model_files = ["download_model.py", "enhanced_server.py"]
        
        model_consistent = True
        for file_name in model_files:
            file_path = self.base_dir / file_name if file_name == "download_model.py" else self.base_dir / "MyAIWebApp/Models" / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    if model_name not in f.read():
                        model_consistent = False
                        break
                        
        if model_consistent:
            print(f"    ✅ 모델명 '{model_name}' 일치")
        else:
            print(f"    ❌ 모델명 불일치")
            self.mismatches.append("모델명 설정 불일치")
            
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*60)
        print("📊 상호 참조 검증 결과")
        print("="*60)
        
        if not self.mismatches:
            print("\n✅ 모든 상호 참조가 일치합니다!")
            print("README와 실제 구현이 완벽하게 동기화되어 있습니다.")
        else:
            print(f"\n❌ 불일치 항목 {len(self.mismatches)}개:")
            for i, mismatch in enumerate(self.mismatches, 1):
                print(f"  {i}. {mismatch}")
                
        print("\n" + "="*60)

def main():
    verifier = CrossReferenceVerification()
    verifier.run_verification()

if __name__ == "__main__":
    main()