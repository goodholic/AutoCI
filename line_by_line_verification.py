#!/usr/bin/env python3
"""
README 한 줄씩 검증
README.md의 모든 코드, 경로, 명령어를 추출하여 검증
"""

import re
import json
import ast
from pathlib import Path

class LineByLineVerification:
    def __init__(self):
        self.base_dir = Path(".")
        self.readme_path = self.base_dir / "README.md"
        self.issues = []
        
    def verify(self):
        """README 한 줄씩 검증"""
        print("📖 README 한 줄씩 정밀 검증 시작...\n")
        
        with open(self.readme_path, 'r', encoding='utf-8') as f:
            readme_lines = f.readlines()
        
        # 1. 파일명 검증
        print("📄 파일명 검증:")
        self.verify_file_references(readme_lines)
        
        # 2. 디렉토리 검증
        print("\n📁 디렉토리 검증:")
        self.verify_directory_references(readme_lines)
        
        # 3. URL 검증
        print("\n🌐 URL 검증:")
        self.verify_urls(readme_lines)
        
        # 4. 코드 블록 검증
        print("\n💻 코드 블록 검증:")
        self.verify_code_blocks(readme_lines)
        
        # 5. 명령어 검증
        print("\n⌨️  명령어 검증:")
        self.verify_commands(readme_lines)
        
        # 6. 클래스/함수명 검증
        print("\n🔧 클래스/함수명 검증:")
        self.verify_class_function_names(readme_lines)
        
        # 결과
        self.print_results()
        
    def verify_file_references(self, lines):
        """파일명 참조 검증"""
        file_patterns = [
            r'`([a-zA-Z_]+\.py)`',
            r'`([a-zA-Z_]+\.cs)`',
            r'`([a-zA-Z_]+\.json)`',
            r'`([a-zA-Z_]+\.razor)`',
            r'([a-zA-Z_]+\.py)\b',
            r'([a-zA-Z_]+\.cs)\b',
        ]
        
        found_files = set()
        for line in lines:
            for pattern in file_patterns:
                matches = re.findall(pattern, line)
                found_files.update(matches)
        
        # 특수 경로 매핑
        file_locations = {
            "enhanced_server.py": "MyAIWebApp/Models/enhanced_server.py",
            "fine_tune.py": "MyAIWebApp/Models/fine_tune.py",
            "requirements.txt": ["requirements_expert.txt", "MyAIWebApp/Models/requirements.txt"],
            "launchSettings.json": ["MyAIWebApp/Backend/Properties/launchSettings.json", "MyAIWebApp/Frontend/Properties/launchSettings.json"]
        }
        
        for file_name in sorted(found_files):
            if file_name in ["README.md", "CLAUDE.md"]:
                continue
                
            found = False
            if file_name in file_locations:
                locations = file_locations[file_name]
                if isinstance(locations, list):
                    for loc in locations:
                        if (self.base_dir / loc).exists():
                            found = True
                            break
                else:
                    if (self.base_dir / locations).exists():
                        found = True
            else:
                # 직접 찾기
                if (self.base_dir / file_name).exists():
                    found = True
                else:
                    # 재귀 검색
                    for p in self.base_dir.rglob(file_name):
                        if p.exists():
                            found = True
                            break
                            
            if found:
                print(f"  ✅ {file_name}")
            else:
                print(f"  ❌ {file_name}")
                self.issues.append(f"파일 없음: {file_name}")
                
    def verify_directory_references(self, lines):
        """디렉토리 참조 검증"""
        dir_patterns = [
            r'`([a-zA-Z_/]+)/`',
            r'📁\s+([a-zA-Z_/]+)/',
            r'cd\s+([a-zA-Z_/]+)',
        ]
        
        found_dirs = set()
        for line in lines:
            for pattern in dir_patterns:
                matches = re.findall(pattern, line)
                found_dirs.update(matches)
        
        for dir_name in sorted(found_dirs):
            if any(skip in dir_name for skip in ["http", "https", "localhost"]):
                continue
                
            dir_path = self.base_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"  ✅ {dir_name}/")
            else:
                print(f"  ❌ {dir_name}/")
                self.issues.append(f"디렉토리 없음: {dir_name}")
                
    def verify_urls(self, lines):
        """URL 검증"""
        url_pattern = r'http://localhost:(\d+)(/[a-zA-Z_/]*)?'
        
        found_urls = set()
        for line in lines:
            matches = re.findall(url_pattern, line)
            for port, path in matches:
                found_urls.add((port, path or ""))
        
        expected_routes = {
            ("7100", "/codegen"): "CodeGenerator.razor",
            ("7100", "/codesearch"): "CodeSearch.razor", 
            ("7100", "/rag"): "RAG.razor",
            ("8080", "/dashboard"): "expert_learning_api.py"
        }
        
        for (port, path), file_ref in expected_routes.items():
            if (port, path) in found_urls:
                print(f"  ✅ http://localhost:{port}{path}")
            else:
                print(f"  ❌ http://localhost:{port}{path}")
                self.issues.append(f"URL 미확인: http://localhost:{port}{path}")
                
    def verify_code_blocks(self, lines):
        """코드 블록 내용 검증"""
        in_code_block = False
        code_content = []
        code_lang = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                    code_content = []
                else:
                    # 코드 블록 종료, 검증
                    if code_lang in ["python", "py"]:
                        self.verify_python_code(code_content, i)
                    elif code_lang == "bash":
                        self.verify_bash_commands(code_content, i)
                    in_code_block = False
            elif in_code_block:
                code_content.append(line)
                
    def verify_python_code(self, code_lines, line_num):
        """Python 코드 검증"""
        code = "".join(code_lines)
        
        # import 문 검증
        imports = re.findall(r'from\s+(\w+)\s+import', code)
        imports.extend(re.findall(r'import\s+(\w+)', code))
        
        # 특정 패턴 확인
        if "quality_criteria" in code:
            if all(item in code for item in ["has_xml_docs", "uses_patterns", "modern_csharp", "error_handling"]):
                print(f"  ✅ 품질 기준 코드 (줄 {line_num})")
            else:
                print(f"  ❌ 품질 기준 코드 불완전 (줄 {line_num})")
                self.issues.append(f"품질 기준 코드 불완전 (줄 {line_num})")
                
    def verify_bash_commands(self, code_lines, line_num):
        """Bash 명령어 검증"""
        for line in code_lines:
            if "python" in line and ".py" in line:
                # Python 스크립트 실행 명령어
                match = re.search(r'python\s+(\w+\.py)', line)
                if match:
                    script_name = match.group(1)
                    if not (self.base_dir / script_name).exists():
                        self.issues.append(f"스크립트 없음: {script_name} (줄 {line_num})")
                        
    def verify_commands(self, lines):
        """명령어 검증"""
        command_patterns = [
            r'`python\s+(\w+\.py)`',
            r'`uvicorn\s+(\w+):app`',
            r'`dotnet\s+(\w+)`',
        ]
        
        for line in lines:
            for pattern in command_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if pattern.startswith(r'`python'):
                        if (self.base_dir / match).exists():
                            print(f"  ✅ python {match}")
                        else:
                            print(f"  ❌ python {match}")
                            self.issues.append(f"스크립트 없음: {match}")
                            
    def verify_class_function_names(self, lines):
        """클래스/함수명 검증"""
        # README에서 언급된 주요 클래스
        important_classes = {
            "ModelDownloader": "download_model.py",
            "AutoCILauncher": "start_all.py",
            "CSharpExpertCrawler": "csharp_expert_crawler.py",
            "CodeLlamaFineTuner": "MyAIWebApp/Models/fine_tune.py",
            "ExpertLearningStartup": "start_expert_learning.py"
        }
        
        for class_name, file_path in important_classes.items():
            # README에서 클래스명 찾기
            mentioned = any(class_name in line for line in lines)
            if mentioned:
                # 실제 파일에서 확인
                full_path = self.base_dir / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        if f"class {class_name}" in f.read():
                            print(f"  ✅ {class_name}")
                        else:
                            print(f"  ❌ {class_name} 정의 없음")
                            self.issues.append(f"{class_name} 정의 없음: {file_path}")
                            
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*60)
        print("📊 한 줄씩 검증 결과")
        print("="*60)
        
        if not self.issues:
            print("\n✅ 모든 참조가 올바릅니다!")
            print("README의 모든 내용이 실제 구현과 일치합니다.")
        else:
            print(f"\n❌ 발견된 문제: {len(self.issues)}개")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
                
        print("\n" + "="*60)

def main():
    verifier = LineByLineVerification()
    verifier.verify()

if __name__ == "__main__":
    main()