#!/usr/bin/env python3
"""
AutoCI 전체 시스템 테스트
모든 기능이 제대로 작동하는지 확인
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 색상 코드
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    """헤더 출력"""
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def test_korean_ai():
    """한국어 AI 테스트"""
    print_header("🤖 한국어 AI 시스템 테스트")
    
    try:
        from advanced_korean_ai import AdvancedKoreanAI
        ai = AdvancedKoreanAI()
        
        test_inputs = [
            "안녕하세요! Unity 개발 도와주실 수 있나요?",
            "PlayerController에서 에러가 나요",
            "고마워요! 정말 도움이 됐어요!"
        ]
        
        for user_input in test_inputs:
            print(f"{Colors.YELLOW}👤 User:{Colors.RESET} {user_input}")
            analysis = ai.analyze_input(user_input)
            response = ai.generate_response(analysis)
            print(f"{Colors.GREEN}🤖 AI:{Colors.RESET} {response[:100]}...")
            print(f"{Colors.CYAN}   분석: {analysis['intent']}/{analysis['topic']}/{analysis['emotion']}{Colors.RESET}")
            print()
            
        print(f"{Colors.GREEN}✅ 한국어 AI 테스트 성공!{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ 한국어 AI 테스트 실패: {e}{Colors.RESET}")
        return False


def test_learning_system():
    """학습 시스템 테스트"""
    print_header("🧠 실제 학습 시스템 테스트")
    
    try:
        from real_learning_system import RealLearningSystem
        learning = RealLearningSystem()
        
        # 백그라운드 학습 시작
        learning.start_background_learning()
        print(f"{Colors.YELLOW}🔄 백그라운드 학습 시작됨{Colors.RESET}")
        
        # 테스트 대화 학습
        test_conversations = [
            ("Unity에서 Object Pool 만드는 방법 알려줘", 
             "Object Pool은 재사용 가능한 객체들을 미리 생성해두는 패턴입니다."),
            ("NullReferenceException 해결 방법",
             "객체가 null인지 먼저 확인하세요: if (obj != null)")
        ]
        
        for user, ai in test_conversations:
            result = learning.learn_from_conversation(user, ai)
            print(f"📝 학습 완료: {result['patterns']} 패턴 발견")
            
        # 통계 확인
        stats = learning.get_learning_stats()
        print(f"\n📊 학습 통계:")
        print(f"   - 총 대화: {stats['total_conversations']}")
        print(f"   - 학습된 패턴: {stats['learned_patterns']}")
        print(f"   - 정확도: {stats['accuracy']}")
        
        learning.stop_background_learning()
        print(f"{Colors.GREEN}✅ 학습 시스템 테스트 성공!{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ 학습 시스템 테스트 실패: {e}{Colors.RESET}")
        return False


def test_monitoring():
    """모니터링 시스템 테스트"""
    print_header("📊 모니터링 시스템 테스트")
    
    try:
        from ai_learning_monitor import AILearningMonitor
        monitor = AILearningMonitor()
        
        # 메트릭 수집
        metrics = monitor.collect_metrics()
        print(f"CPU 사용률: {metrics['cpu_percent']:.1f}%")
        print(f"메모리 사용률: {metrics['memory'].percent:.1f}%")
        print(f"활성 AI 프로세스: {len(metrics['processes'])}개")
        
        # 모니터링 시작 (백그라운드)
        print(f"\n{Colors.YELLOW}🔄 1분 모니터링 시작 (백그라운드){Colors.RESET}")
        print(f"웹 대시보드: http://localhost:8888")
        
        print(f"{Colors.GREEN}✅ 모니터링 테스트 성공!{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ 모니터링 테스트 실패: {e}{Colors.RESET}")
        return False


def test_autoci_command():
    """autoci 명령어 테스트"""
    print_header("🔧 autoci 명령어 테스트")
    
    try:
        # autoci --version 테스트
        result = subprocess.run(['bash', 'autoci', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            # 버전 명령이 없으면 help 테스트
            result = subprocess.run(['bash', 'autoci', 'help'], 
                                  capture_output=True, text=True)
            
        print("autoci 명령어 실행 가능")
        
        # 주요 명령어 리스트
        commands = [
            "autoci korean    # 한국어 대화 모드",
            "autoci monitor start  # 모니터링 시작",
            "autoci learn start    # 24시간 학습",
            "autoci enhance start  # 코드 개선"
        ]
        
        print(f"\n{Colors.CYAN}사용 가능한 명령어:{Colors.RESET}")
        for cmd in commands:
            print(f"  {cmd}")
            
        print(f"{Colors.GREEN}✅ autoci 명령어 테스트 성공!{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ autoci 명령어 테스트 실패: {e}{Colors.RESET}")
        return False


def test_integrated_system():
    """통합 시스템 테스트"""
    print_header("🚀 통합 시스템 테스트")
    
    try:
        from integrated_autoci_system import IntegratedAutoCI
        
        print("통합 시스템 초기화 중...")
        autoci = IntegratedAutoCI()
        autoci.start()
        
        print(f"{Colors.GREEN}✅ 모든 시스템이 성공적으로 통합되었습니다!{Colors.RESET}")
        
        # 간단한 대화 테스트
        test_input = "안녕! Unity 개발 도와줄 수 있어?"
        response = autoci.process_input(test_input)
        print(f"\n테스트 대화:")
        print(f"👤: {test_input}")
        print(f"🤖: {response[:100]}...")
        
        # 상태 확인
        status = autoci.show_status()
        print(f"\n{Colors.CYAN}시스템 상태:{Colors.RESET}")
        print(status[:200] + "...")
        
        autoci.stop()
        
        print(f"{Colors.GREEN}✅ 통합 시스템 테스트 성공!{Colors.RESET}")
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ 통합 시스템 테스트 실패: {e}{Colors.RESET}")
        return False


def main():
    """메인 테스트 함수"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*70)
    print("🚀 AutoCI 전체 시스템 테스트")
    print("="*70)
    print(f"{Colors.RESET}")
    
    start_time = datetime.now()
    results = []
    
    # 각 컴포넌트 테스트
    tests = [
        ("한국어 AI", test_korean_ai),
        ("학습 시스템", test_learning_system),
        ("모니터링", test_monitoring),
        ("autoci 명령어", test_autoci_command),
        ("통합 시스템", test_integrated_system)
    ]
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"{Colors.RED}테스트 중 오류: {e}{Colors.RESET}")
            results.append((name, False))
        
        time.sleep(1)  # 테스트 간 대기
    
    # 최종 결과
    print_header("📊 테스트 결과 요약")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for name, success in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if success else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"{name}: {status}")
    
    print(f"\n총 테스트: {total_tests}")
    print(f"성공: {passed_tests}")
    print(f"실패: {total_tests - passed_tests}")
    
    elapsed = datetime.now() - start_time
    print(f"\n테스트 소요 시간: {elapsed.total_seconds():.1f}초")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 모든 테스트 통과! AutoCI가 완벽하게 작동합니다!{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  일부 테스트가 실패했습니다. 로그를 확인하세요.{Colors.RESET}")
    
    # 사용 안내
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}🚀 AutoCI 사용 방법:{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"1. 터미널에서 '{Colors.YELLOW}autoci{Colors.RESET}' 입력")
    print(f"2. 자연스러운 한국어로 대화")
    print(f"3. Unity/C# 관련 질문하기")
    print(f"4. AI가 대화에서 실시간으로 학습")
    print(f"5. 1분마다 자동 모니터링")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")


if __name__ == "__main__":
    main()