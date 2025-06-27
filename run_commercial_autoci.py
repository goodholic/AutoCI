#!/usr/bin/env python3
"""
AutoCI 상용화 수준 시스템 실행 스크립트
모든 구성 요소를 통합하여 상용화 품질의 AI 서비스 제공
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from commercial_autoci_master import CommercialAutoCI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('autoci_run.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_welcome_banner():
    """환영 배너 출력"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🤖 AutoCI - 상용화 수준 AI 코딩 어시스턴트                                      ║
║                                                                              ║
║  💼 상용화 품질의 전문적인 AI 대화                                              ║
║  🎓 C# 전문가 수준의 코딩 지원                                                 ║
║  🔄 24시간 지속적인 자동 학습                                                  ║
║  ✅ 실시간 품질 검증 및 최적화                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 시스템 구성 요소:
   • Commercial AI Dialogue Engine      (상용화 수준 대화 엔진)
   • C# Expert Learning System          (C# 전문가 학습 시스템)  
   • Continuous Learning Pipeline       (24시간 자동 학습 파이프라인)
   • Commercial Quality Validator       (상용화 품질 검증 시스템)
   • Real Learning System               (실제 학습 시스템)
   • AI Learning Monitor                (AI 학습 모니터링)

📊 품질 기준:
   • 대화 자연스러움: 90% 이상
   • 기술 정확도: 95% 이상  
   • 응답 시간: 0.5초 이내
   • 사용자 만족도: 90% 이상

🎯 전문 분야:
   • C# 언어 및 고급 기능
   • Unity 게임 개발
   • 디자인 패턴 및 아키텍처
   • 성능 최적화
   • 에러 해결 및 디버깅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 사용법:
   1. 자연스러운 한국어로 질문하세요
   2. C#이나 Unity 관련 기술적 도움이 필요하면 구체적으로 물어보세요
   3. 코드 리뷰, 최적화, 에러 해결 등 모든 것을 도와드립니다
   4. 대화할 때마다 AI가 더 똑똑해집니다

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(banner)


def print_system_info():
    """시스템 정보 출력"""
    info = """
📋 시스템 정보:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 주요 명령어:
   • '상태'         - 시스템 현재 상태 확인
   • '품질보고서'     - 상세한 품질 검증 보고서
   • '통계'         - 학습 및 성능 통계
   • 'exit' / '종료' - 시스템 종료

💡 특별 기능:
   • 전문가 지식 자동 적용 (C#/Unity 키워드 감지시)
   • 실시간 학습 및 패턴 인식
   • 다중 품질 검증 (자연스러움, 정확성, 유용성)
   • 백그라운드 지속 학습

🎯 최적화된 응답:
   • 코드 예제 포함
   • 단계별 설명
   • 모범 사례 제시
   • 에러 예방 팁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(info)


async def main():
    """메인 실행 함수"""
    try:
        # 환영 배너 출력
        print_welcome_banner()
        
        # 시스템 정보 출력
        print_system_info()
        
        # AutoCI 시스템 초기화 및 실행
        logger.info("AutoCI 상용화 시스템 시작...")
        
        autoci = CommercialAutoCI()
        await autoci.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n🛑 사용자 요청으로 시스템을 종료합니다...")
        logger.info("사용자 인터럽트로 종료")
        
    except Exception as e:
        print(f"\n❌ 시스템 오류가 발생했습니다: {e}")
        logger.error(f"시스템 오류: {e}", exc_info=True)
        
    finally:
        print("\n👋 AutoCI를 사용해 주셔서 감사합니다!")
        logger.info("AutoCI 시스템 종료 완료")


if __name__ == "__main__":
    # 가상환경 체크
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  가상환경에서 실행하는 것을 권장합니다.")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")
        print()
    
    # 필수 라이브러리 체크
    required_packages = ['numpy', 'sqlite3', 'asyncio', 'pathlib', 'datetime', 'logging']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package not in ['sqlite3', 'asyncio', 'pathlib', 'datetime', 'logging']:  # 내장 모듈 제외
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 필수 패키지가 누락되었습니다: {', '.join(missing_packages)}")
        print("설치 명령어: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    # 이벤트 루프 실행
    asyncio.run(main())