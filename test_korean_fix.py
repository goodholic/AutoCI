#!/usr/bin/env python3
"""
AutoCI 한국어 AI 오류 해결 테스트
"""

import sys
import os

def test_dependencies():
    """의존성 테스트"""
    print("🔧 의존성 테스트 중...")
    
    deps = ['rich', 'colorama', 'psutil']
    missing = []
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep} - 설치됨")
        except ImportError:
            print(f"  ❌ {dep} - 누락")
            missing.append(dep)
    
    return missing

def test_korean_ai():
    """한국어 AI 테스트"""
    print("\n🇰🇷 한국어 AI 테스트 중...")
    
    try:
        # autoci_simple_interactive 모듈 테스트
        sys.path.append('.')
        from autoci_simple_interactive import KoreanAI
        
        ai = KoreanAI()
        
        # 테스트 입력
        test_inputs = [
            "안녕하세요!",
            "너 나랑 대화할 수 있어?",
            "Unity 도와줘",
            "고마워"
        ]
        
        for test_input in test_inputs:
            analysis = ai.analyze_text(test_input)
            response = ai.generate_response(test_input, analysis)
            print(f"  📝 '{test_input}' → '{response[:50]}...'")
        
        print("  ✅ 한국어 AI 정상 작동")
        return True
        
    except Exception as e:
        print(f"  ❌ 한국어 AI 오류: {e}")
        return False

def main():
    """메인 테스트"""
    print("🤖 AutoCI 한국어 AI 오류 해결 테스트")
    print("=" * 50)
    
    # 의존성 확인
    missing_deps = test_dependencies()
    
    # 한국어 AI 테스트
    korean_ok = test_korean_ai()
    
    print("\n📊 테스트 결과:")
    print("=" * 50)
    
    if missing_deps:
        print(f"⚠️  누락된 의존성: {', '.join(missing_deps)}")
        print(f"🔧 해결 방법: bash install_dependencies_wsl.sh")
        print(f"💡 또는 간단한 버전 사용 가능")
    else:
        print("✅ 모든 의존성 설치됨")
    
    if korean_ok:
        print("✅ 한국어 AI 정상 작동")
    else:
        print("❌ 한국어 AI 오류")
    
    print("\n🚀 실행 방법:")
    if missing_deps:
        print("  의존성 오류 시: ./autoci (자동으로 간단한 버전 실행)")
    else:
        print("  정상: ./autoci (고급 버전 실행)")
    
    print("  한국어 모드: ./autoci korean")
    print("  또는: python3 autoci_simple_interactive.py")

if __name__ == "__main__":
    main() 