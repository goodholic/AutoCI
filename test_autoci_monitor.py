#!/usr/bin/env python3
"""
AutoCI 통합 모니터링 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 설정
AUTOCI_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(AUTOCI_ROOT))

from autoci_integrated import AutoCIIntegrated

async def main():
    """메인 테스트 함수"""
    print("AutoCI 통합 모니터링 테스트를 시작합니다...")
    
    # AutoCI 통합 시스템 생성
    autoci = AutoCIIntegrated()
    
    # 실행
    await autoci.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n테스트를 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")