#!/usr/bin/env python3
"""
Panda3D 업데이트 검증 스크립트
"""

import os
import re

def check_file_for_patterns(file_path, old_patterns, new_patterns):
    """파일에서 패턴 검증"""
    if not os.path.exists(file_path):
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_old = False
    has_new = False
    
    # 구 패턴 확인
    for pattern in old_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            print(f"⚠️  {file_path}에 아직 '{pattern}' 패턴이 남아있습니다")
            has_old = True
    
    # 신 패턴 확인  
    for pattern in new_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            has_new = True
    
    if not has_old and has_new:
        print(f"✅ {file_path} - Panda3D로 업데이트 완료")
        return True
    elif not has_new:
        print(f"❌ {file_path} - Panda3D 패턴이 없습니다")
        return False
    else:
        return False

def main():
    print("🔍 Panda3D 업데이트 검증 시작...\n")
    
    # Godot/C# 패턴
    old_patterns = [
        r"godot(?!.*panda3d)",  # godot (panda3d가 같은 줄에 없는 경우)
        r"C#(?!.*Python)",      # C# (Python이 같은 줄에 없는 경우)
        r"csharp(?!.*python)"   # csharp (python이 같은 줄에 없는 경우)
    ]
    
    # Panda3D/Python 패턴
    new_patterns = [
        r"panda3d",
        r"Python",
        r"python"
    ]
    
    # 검증할 파일들
    files_to_check = [
        "core_system/continuous_learning_system.py",
        "modules/intelligent_information_gatherer.py"
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        if not check_file_for_patterns(file_path, old_patterns, new_patterns):
            all_good = False
        print()
    
    # 특수 검증: learning topics
    print("📚 학습 토픽 검증...")
    with open("core_system/continuous_learning_system.py", 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 학습 토픽이 올바르게 설정되었는지 확인
    if "Python 프로그래밍" in content and "Panda3D 엔진" in content:
        print("✅ 학습 토픽이 Panda3D로 올바르게 설정됨")
    else:
        print("❌ 학습 토픽이 아직 업데이트되지 않음")
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✨ 모든 파일이 Panda3D로 성공적으로 업데이트되었습니다!")
    else:
        print("⚠️  일부 파일에서 업데이트가 필요합니다")
    
    print("\n💡 다음 명령어로 학습을 시작할 수 있습니다:")
    print("   autoci learn")
    print("   autoci learn low")

if __name__ == "__main__":
    main()