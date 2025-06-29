#!/usr/bin/env python3
"""
AI Godot 빌드 실행 스크립트
"""
import asyncio
import sys
import os
from pathlib import Path

# 현재 디렉토리를 모듈 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    print("🚀 AI Godot 빌드를 시작합니다...")
    print("=" * 50)
    
    try:
        # build_ai_godot 모듈 임포트
        from build_ai_godot import AIGodotBuilder
        
        # 빌더 인스턴스 생성
        builder = AIGodotBuilder()
        
        # 빌드 실행
        success = await builder.run()
        
        if success:
            print("\n✅ 빌드가 성공적으로 완료되었습니다!")
            print("빌드된 파일 위치: godot_ai_build/output/")
        else:
            print("\n❌ 빌드가 실패했습니다.")
            print("자세한 내용은 godot_build.log를 확인하세요.")
        
        return 0 if success else 1
        
    except ImportError as e:
        print(f"❌ 모듈 임포트 오류: {e}")
        print("build_ai_godot.py 파일이 같은 디렉토리에 있는지 확인하세요.")
        return 1
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Windows에서 asyncio 이벤트 루프 정책 설정
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 빌드 실행
    exit_code = asyncio.run(main())
    sys.exit(exit_code)