#!/usr/bin/env python3
"""
README 빠른 최종 점검
"""

import os
from pathlib import Path

def main():
    base_dir = Path(".")
    errors = []
    
    print("🔍 AutoCI 빠른 최종 점검...\n")
    
    # 1. 핵심 파일 존재 확인
    print("📄 핵심 파일 확인:")
    core_files = [
        "download_model.py",
        "start_all.py", 
        "csharp_expert_crawler.py",
        "start_expert_learning.py",
        "expert_learning_api.py",
        "requirements_expert.txt",
        "MyAIWebApp/Models/enhanced_server.py",
        "MyAIWebApp/Models/fine_tune.py"
    ]
    
    for file_path in core_files:
        if (base_dir / file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            errors.append(file_path)
            
    # 2. 디렉토리 구조 확인
    print("\n📁 디렉토리 구조:")
    dirs = [
        "MyAIWebApp/Backend/Services",
        "MyAIWebApp/Frontend/Pages", 
        "expert_training_data"
    ]
    
    for dir_path in dirs:
        if (base_dir / dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            errors.append(dir_path)
            
    # 3. 포트 설정 확인
    print("\n🔌 포트 설정 확인:")
    start_all_path = base_dir / "start_all.py"
    if start_all_path.exists():
        with open(start_all_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        ports = ["8000", "8080", "5049", "7100"]
        for port in ports:
            if port in content:
                print(f"  ✅ 포트 {port}")
            else:
                print(f"  ❌ 포트 {port}")
                errors.append(f"포트 {port}")
                
    # 4. 품질 평가 기준 확인
    print("\n📊 품질 평가 기준:")
    crawler_path = base_dir / "csharp_expert_crawler.py"
    if crawler_path.exists():
        with open(crawler_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        criteria = [
            ("XML 문서 주석 (20%)", "score += 0.20"),
            ("디자인 패턴 (15%)", "score += 0.15"),
            ("현대적 C# 기능 (15%)", "score += 0.15"),
            ("에러 처리 (10%)", "score += 0.10")
        ]
        
        for name, code in criteria:
            if name in content and code in content:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
                errors.append(name)
                
    # 5. 웹 라우트 확인
    print("\n🌐 웹 라우트:")
    routes = {
        "/codegen": "MyAIWebApp/Frontend/Pages/CodeGenerator.razor",
        "/codesearch": "MyAIWebApp/Frontend/Pages/CodeSearch.razor",
        "/rag": "MyAIWebApp/Frontend/Pages/RAG.razor"
    }
    
    for route, file_path in routes.items():
        full_path = base_dir / file_path
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                if f'@page "{route}"' in f.read():
                    print(f"  ✅ {route}")
                else:
                    print(f"  ❌ {route}")
                    errors.append(route)
                    
    # 결과
    print("\n" + "="*50)
    if not errors:
        print("✅ 모든 검증 통과! README와 100% 일치합니다.")
    else:
        print(f"❌ {len(errors)}개 항목 불일치:")
        for error in errors:
            print(f"  - {error}")
    print("="*50)

if __name__ == "__main__":
    main()