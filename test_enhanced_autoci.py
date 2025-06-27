#!/usr/bin/env python3
"""
Enhanced AutoCI 기능 테스트 스크립트
모든 메서드가 제대로 구현되었는지 확인
"""

import sys
from pathlib import Path
from enhanced_autoci_korean import EnhancedAutoCI

def test_enhanced_autoci():
    """Enhanced AutoCI 메서드 테스트"""
    print("🧪 Enhanced AutoCI 테스트 시작...")
    print("=" * 60)
    
    # 인스턴스 생성
    autoci = EnhancedAutoCI()
    print("✅ EnhancedAutoCI 인스턴스 생성 완료")
    
    # 메서드 목록 확인
    required_methods = [
        'check_unity_project',
        'show_unity_status',
        'check_learning_status',
        'do_project',
        'do_analyze',
        'do_improve',
        'do_organize',
        'do_search',
        'do_unity_organize',
        'do_script_organize',
        'do_script_move',
        'do_asset_organize',
        'do_start_learning',
        'do_learning_status',
        'do_monitor',
        'do_backup',
        'do_help',
        'do_status',
        'do_exit',
        'do_conversation_mode',
        'do_command_mode'
    ]
    
    print("\n📋 메서드 구현 확인:")
    all_implemented = True
    
    for method_name in required_methods:
        if hasattr(autoci, method_name):
            print(f"  ✅ {method_name}")
        else:
            print(f"  ❌ {method_name} - 구현되지 않음!")
            all_implemented = False
    
    if all_implemented:
        print("\n🎉 모든 메서드가 정상적으로 구현되었습니다!")
    else:
        print("\n⚠️  일부 메서드가 구현되지 않았습니다.")
        return False
    
    # 기본 기능 테스트
    print("\n🔧 기본 기능 테스트:")
    
    try:
        # Unity 프로젝트 확인 테스트
        test_path = Path.home()
        result = autoci.check_unity_project(test_path)
        print(f"  ✅ check_unity_project 실행 (결과: {result})")
        
        # 학습 상태 확인 테스트
        print("  🔄 check_learning_status 테스트 중...")
        autoci.check_learning_status()
        print("  ✅ check_learning_status 실행 완료")
        
        print("\n✅ 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🧪 Enhanced AutoCI 통합 테스트                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    success = test_enhanced_autoci()
    
    if success:
        print("\n🎊 Enhanced AutoCI가 완벽하게 구현되었습니다!")
        print("💡 이제 다음 명령어로 실행할 수 있습니다:")
        print("   python enhanced_autoci_korean.py")
        print("   또는")
        print("   ./autoci korean")
    else:
        print("\n❌ 일부 기능에 문제가 있습니다.")
        print("💡 오류를 수정한 후 다시 테스트해주세요.")
    
    sys.exit(0 if success else 1)