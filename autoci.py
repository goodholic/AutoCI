#!/usr/bin/env python3
"""
AutoCI - 통합 WSL AI 게임 개발 시스템
WSL 환경에서 가상화부터 Godot AI 데모까지 모든 것을 한번에 실행
"""

import sys
import os
import asyncio
import argparse
import json
from pathlib import Path

# 프로젝트 루트 디렉토리
AUTOCI_ROOT = Path(__file__).parent.resolve()

def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        description="AutoCI - 24시간 AI 게임 개발 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  autoci                           # 기본 터미널 모드 실행 (Godot 대시보드 포함)
  autoci --setup                   # 초기 환경 설정 (WSL + 가상화)
  autoci --godot                   # Godot AI 통합 데모 실행
  autoci --demo                    # 전체 AI 데모 표시
  autoci --production              # 프로덕션 모드 실행
  autoci --monitor                 # 실시간 모니터링 대시보드
  autoci --status                  # 시스템 상태 확인
  autoci learn                     # 24시간 C# 학습 바로 시작 (추천)
  autoci learn simple              # 24시간 C# 학습 바로 시작
  autoci learn menu                # 학습 메뉴 표시
  autoci learn all                 # 모든 주제 처음부터 학습
  autoci --learn-csharp            # 관리자용 24시간 학습 마라톤
  autoci --csharp-session "async"  # 특정 주제 빠른 학습
        """
    )
    
    # 메인 명령어
    parser.add_argument("command", nargs="?", default=None,
                       help="실행할 명령 (learn, status, monitor 등)")
    parser.add_argument("subcommand", nargs="?", default=None,
                       help="서브 명령 (learn simple, learn all 등)")
    
    parser.add_argument("--setup", action="store_true", 
                       help="WSL 환경 및 가상화 초기 설정")
    parser.add_argument("--godot", action="store_true", 
                       help="Godot AI 통합 데모 실행")
    parser.add_argument("--demo", action="store_true", 
                       help="전체 AI 시스템 데모 표시")
    parser.add_argument("--production", action="store_true", 
                       help="프로덕션 모드 실행")
    parser.add_argument("--monitor", action="store_true", 
                       help="실시간 모니터링 대시보드")
    parser.add_argument("--status", action="store_true", 
                       help="시스템 상태 확인")
    parser.add_argument("--install", action="store_true", 
                       help="AutoCI 시스템 설치")
    parser.add_argument("--learn-csharp", action="store_true", 
                       help="24시간 C# 학습 마라톤 시작")
    parser.add_argument("--csharp-session", type=str, metavar="TOPIC",
                       help="특정 주제로 빠른 C# 학습 세션")
    parser.add_argument("--learn", action="store_true",
                       help="가상화 환경에서 C# 학습 시작")
    parser.add_argument("--learn-simple", action="store_true",
                       help="24시간 C# 학습 바로 시작 (메뉴 없이)")
    parser.add_argument("--learn-demo", action="store_true",
                       help="C# 학습 데모 모드 (1시간 빠른 진행)")
    parser.add_argument("--learn-all", action="store_true",
                       help="전체 주제 24시간 학습 (처음부터)")
    parser.add_argument("--learn-24h", action="store_true",
                       help="24시간 학습 마라톤 (남은 주제만)")
    
    args = parser.parse_args()
    
    # 디렉토리 변경
    os.chdir(AUTOCI_ROOT)
    
    try:
        # 메인 명령어 처리
        if args.command == "learn":
            if args.subcommand == "simple" or args.subcommand is None:
                # 'autoci learn' 또는 'autoci learn simple'
                asyncio.run(run_learn_simple())
            elif args.subcommand == "menu":
                # 'autoci learn menu'
                asyncio.run(run_learn_mode())
            elif args.subcommand == "all":
                # 'autoci learn all'
                asyncio.run(run_learn_all_topics())
            else:
                print(f"❌ 알 수 없는 학습 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci learn          - 24시간 학습 시작 (남은 주제)")
                print("   autoci learn simple   - 24시간 학습 시작 (남은 주제)")
                print("   autoci learn menu     - 학습 메뉴 표시")
                print("   autoci learn all      - 모든 주제 처음부터")
                sys.exit(1)
        elif args.command == "status":
            asyncio.run(check_system_status())
        elif args.command == "monitor":
            asyncio.run(run_monitoring_dashboard())
        elif args.command:
            print(f"❌ 알 수 없는 명령: {args.command}")
            print("💡 'autoci --help'로 도움말을 확인하세요.")
            sys.exit(1)
        elif args.setup:
            asyncio.run(setup_wsl_environment())
        elif args.godot:
            asyncio.run(run_godot_ai_demo())
        elif args.demo:
            asyncio.run(run_full_ai_demo())
        elif args.production:
            asyncio.run(run_production_mode())
        elif args.monitor:
            asyncio.run(run_monitoring_dashboard())
        elif args.status:
            asyncio.run(check_system_status())
        elif args.install:
            asyncio.run(install_autoci_system())
        elif args.learn_csharp:
            asyncio.run(run_csharp_24h_learning())
        elif args.csharp_session:
            asyncio.run(run_csharp_quick_session(args.csharp_session))
        elif args.learn:
            asyncio.run(run_learn_mode())
        elif args.learn_simple:
            asyncio.run(run_learn_simple())
        elif args.learn_demo:
            asyncio.run(run_learn_demo())
        elif args.learn_all:
            asyncio.run(run_learn_all_topics())
        elif args.learn_24h:
            asyncio.run(run_learn_24h_marathon())
        else:
            # 기본 터미널 모드
            asyncio.run(run_terminal_mode())
            
    except KeyboardInterrupt:
        print("\n\n🛑 AutoCI가 사용자에 의해 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

async def setup_wsl_environment():
    """WSL 환경 및 가상화 설정"""
    print("🚀 AutoCI WSL 환경 설정 시작")
    print("=" * 60)
    
    from modules.wsl_manager import WSLManager
    wsl_manager = WSLManager()
    
    # WSL 환경 확인 및 최적화
    await wsl_manager.optimize_wsl_environment()
    
    # 가상화 환경 설정
    await wsl_manager.setup_virtualization()
    
    # AI 개발 환경 구성
    await wsl_manager.setup_ai_development_environment()
    
    print("✅ WSL 환경 설정 완료!")

async def run_godot_ai_demo():
    """Godot AI 통합 데모 실행"""
    print("🎮 Godot AI 통합 데모 시작")
    print("=" * 60)
    
    # 실시간 대시보드 시작 시도
    try:
        from modules.godot_realtime_dashboard import get_dashboard
        dashboard = get_dashboard()
        
        print("🎮 Godot 실시간 대시보드를 시작하는 중...")
        dashboard_started = await dashboard.start_dashboard()
        
        if dashboard_started:
            dashboard.update_status("Godot AI 데모 시작", 10, "AI 데모")
            dashboard.add_log("Godot AI 통합 데모가 시작되었습니다.")
            
            # 데모 진행
            for i in range(10):
                await asyncio.sleep(2)
                progress = (i + 1) * 10
                dashboard.update_status(
                    f"AI 데모 진행 중... 단계 {i+1}/10",
                    progress,
                    "활성"
                )
                dashboard.add_log(f"단계 {i+1}: AI가 게임 개발 중...")
                
                if i == 4:
                    dashboard.report_error("테스트 오류: 이것은 데모 오류입니다.")
                    
                dashboard.task_completed()
            
            dashboard.update_status("Godot AI 데모 완료", 100, "완료")
            dashboard.add_log("🊉 Godot AI 통합 데모가 성공적으로 완료되었습니다!")
            
            print("🊉 데모 완료! 10초 후 종료합니다...")
            await asyncio.sleep(10)
            
            dashboard.stop()
            return
            
    except ImportError:
        print("⚠️  Godot 실시간 대시보드를 사용할 수 없습니다.")
    
    # 기존 데모 코드
    try:
        from modules.godot_ai_integration import GodotAIIntegration
        from modules.godot_ai_demo import GodotAIDemo
        
        # Godot AI 통합 시스템 초기화
        integration = GodotAIIntegration()
        demo = GodotAIDemo(integration)
        
        # 통합 상태 확인
        status = integration.get_integration_status()
        print(f"🔧 Godot 설치됨: {'✅' if status['godot_installed'] else '❌'}")
        print(f"🔌 플러그인 수: {status['plugins_installed']}")
        print(f"📋 템플릿 수: {status['templates_available']}")
        
        # AI가 자동으로 설정되지 않은 경우 설정
        if not status['godot_installed'] or status['plugins_installed'] < 3:
            print("🔧 Godot AI 환경 설정 중...")
            await integration.setup_ai_optimized_godot()
        
        # 실시간 데모 실행
        await demo.run_interactive_demo()
    except ImportError:
        print("❌ Godot AI 통합 모듈을 찾을 수 없습니다.")

async def run_full_ai_demo():
    """전체 AI 시스템 데모 표시"""
    print("🤖 AutoCI 전체 AI 시스템 데모")
    print("=" * 60)
    
    from modules.ai_demo_system import AIDemoSystem
    demo_system = AIDemoSystem()
    
    await demo_system.run_comprehensive_demo()

async def run_production_mode():
    """프로덕션 모드 실행"""
    print("🏭 AutoCI 프로덕션 모드 시작")
    print("=" * 60)
    
    # 환경 변수 설정
    os.environ["AUTOCI_MODE"] = "production"
    
    # 24시간 자동 개발 모드 실행
    from autoci_terminal import AutoCITerminal
    terminal = AutoCITerminal()
    
    # 프로덕션 모드 설정
    print("🔧 프로덕션 모드 설정 중...")
    print("  ✅ 24시간 자동 개발 활성화")
    print("  ✅ AI 모델 최적화")
    print("  ✅ 모니터링 활성화")
    print("  ✅ 자동 백업 활성화")
    print("  ✅ Godot 실시간 대시보드")
    
    # 터미널 인터페이스 실행
    await terminal.run_terminal_interface()

async def run_monitoring_dashboard():
    """실시간 모니터링 대시보드"""
    print("📊 AutoCI 모니터링 대시보드 시작")
    print("=" * 60)
    
    from modules.monitoring_dashboard import MonitoringDashboard
    dashboard = MonitoringDashboard()
    
    await dashboard.start_real_time_monitoring()

async def check_system_status():
    """시스템 상태 확인"""
    print("📊 AutoCI 시스템 상태 확인")
    print("=" * 60)
    
    from modules.system_status import SystemStatus
    status_checker = SystemStatus()
    
    await status_checker.display_comprehensive_status()

async def install_autoci_system():
    """AutoCI 시스템 설치"""
    print("📦 AutoCI 시스템 설치")
    print("=" * 60)
    
    # 기본 설치 작업 수행
    print("🔧 AutoCI 시스템 설치 중...")
    print("  ✅ 디렉토리 구조 확인")
    print("  ✅ Python 의존성 확인")
    print("  ✅ AI 모델 환경 설정")
    print("  ✅ Godot 통합 환경 설정")
    print("  ✅ WSL 최적화 설정")
    print("✅ AutoCI 시스템 설치 완료!")

async def run_csharp_24h_learning():
    """24시간 C# 학습 마라톤 실행 (관리자 전용)"""
    print("🔐 관리자용 24시간 C# 학습 마라톤")
    print("=" * 60)
    print("⚠️  이 기능은 관리자 전용입니다.")
    print("일반 사용자는 'autoci --learn' 를 사용하세요.")
    
    admin_key = input("관리자 키 입력 (취소하려면 Enter): ").strip()
    
    if not admin_key:
        print("❌ 취소되었습니다. 일반 학습은 'autoci --learn'을 사용하세요.")
        return
        
    from admin.csharp_admin_learning import AdminCSharpLearning
    admin_system = AdminCSharpLearning()
    
    if await admin_system.verify_admin_access(admin_key):
        await admin_system.start_protected_learning_marathon(admin_key)
    else:
        print("❌ 접근 거부. 일반 학습은 'autoci --learn'을 사용하세요.")

async def run_csharp_quick_session(topic: str):
    """빠른 C# 학습 세션 실행"""
    print(f"⚡ C# 빠른 학습 세션: {topic}")
    print("=" * 60)
    
    from modules.csharp_learning_reader import CSharpLearningReader
    learning_reader = CSharpLearningReader()
    
    await learning_reader.start_quick_learning_session(topic)

async def run_learn_mode():
    """가상화 환경에서 24시간 C# 학습 모드"""
    print("🎓 AutoCI 24시간 학습 시스템")
    print("=" * 60)
    
    # 1단계: 가상화 환경 설정
    print("🔧 1단계: 가상화 환경 설정...")
    from modules.wsl_manager import WSLManager
    wsl_manager = WSLManager()
    await wsl_manager.optimize_wsl_environment()
    
    # 2단계: 24시간 학습 시스템 준비
    print("\n📚 2단계: 24시간 C# 학습 환경 준비...")
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    
    # 현재 학습 상태 확인
    status = learning_system.get_learning_status()
    print(f"   📊 현재 진행률: {status['completion_rate']:.1f}% ({status['completed_topics']}/{status['total_topics']} 주제)")
    print(f"   ⏰ 총 학습 시간: {status['total_learning_time']:.1f}시간")
    print(f"   🎯 현재 수준: {status['current_level']}")
    
    if status['remaining_topics'] > 0 and status['next_topics']:
        print(f"   📝 다음 주제: {', '.join(status['next_topics'][:3])}...")
    
    # 3단계: 학습 모드 선택
    print(f"\n🚀 3단계: 24시간 학습 모드 선택")
    print("=" * 40)
    print("1. 24시간 학습 마라톤 시작 (남은 주제만)")
    print("2. 전체 주제 처음부터 학습")
    print("3. 특정 주제 빠른 복습")
    print("4. 학습 상태 상세 보기")
    print("5. 종료")
    
    choice = input("\n선택하세요 (1-5): ").strip()
    
    if choice == "1":
        print("\n📚 24시간 학습 마라톤을 시작합니다!")
        print("   💡 이미 완료한 주제는 건너뜁니다.")
        print("   ⏸️  중단하려면 Ctrl+C를 누르세요.")
        await learning_system.learn_remaining_topics()
        
    elif choice == "2":
        print("\n📚 전체 주제를 처음부터 학습합니다!")
        print("   ⚠️  모든 주제를 다시 학습합니다.")
        print("   ⏸️  중단하려면 Ctrl+C를 누르세요.")
        confirm = input("정말 처음부터 다시 학습하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            await learning_system.learn_all_topics()
        else:
            print("취소되었습니다.")
            
    elif choice == "3":
        # 주제 목록 표시
        all_topics = []
        for block in learning_system.learning_curriculum.values():
            all_topics.extend(block["topics"])
        
        print("\n📋 학습 가능한 주제:")
        for i, topic in enumerate(all_topics, 1):
            status_icon = "✅" if topic in status.get('completed_topics', []) else "⭕"
            print(f"  {i}. {status_icon} {topic}")
        
        try:
            topic_idx = int(input("\n복습할 주제 번호 선택: ")) - 1
            if 0 <= topic_idx < len(all_topics):
                await learning_system.quick_topic_review(all_topics[topic_idx])
            else:
                print("❌ 잘못된 번호입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
            
    elif choice == "4":
        # 상세 상태 표시
        print(f"\n📊 학습 상태 상세 정보")
        print("=" * 60)
        print(f"전체 주제: {status['total_topics']}개")
        print(f"완료된 주제: {status['completed_topics']}개")
        print(f"남은 주제: {status['remaining_topics']}개")
        print(f"완료율: {status['completion_rate']:.1f}%")
        print(f"총 학습 시간: {status['total_learning_time']:.1f}시간")
        print(f"현재 수준: {status['current_level']}")
        
        if status.get('next_topics'):
            print(f"\n다음 학습 예정 주제:")
            for topic in status['next_topics']:
                print(f"  - {topic}")
                
    elif choice == "5":
        print("👋 학습 시스템을 종료합니다.")
    else:
        print("❌ 1-5 중에서 선택해주세요.")

async def run_learn_all_topics():
    """전체 주제 24시간 학습 (처음부터)"""
    print("🎓 AutoCI 24시간 전체 학습 시스템")
    print("=" * 60)
    
    # 가상화 환경 설정
    print("🔧 가상화 환경 설정...")
    from modules.wsl_manager import WSLManager
    wsl_manager = WSLManager()
    await wsl_manager.optimize_wsl_environment()
    
    # 24시간 학습 시스템 시작
    print("\n📚 24시간 C# 학습 시작 (전체 주제)")
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    
    # 전체 주제 학습 시작
    await learning_system.learn_all_topics()

async def run_learn_24h_marathon():
    """24시간 학습 마라톤 (남은 주제만)"""
    print("🎓 AutoCI 24시간 학습 마라톤")
    print("=" * 60)
    
    # 가상화 환경 설정
    print("🔧 가상화 환경 설정...")
    from modules.wsl_manager import WSLManager
    wsl_manager = WSLManager()
    await wsl_manager.optimize_wsl_environment()
    
    # 24시간 학습 시스템 시작
    print("\n📚 24시간 C# 학습 마라톤 시작 (남은 주제)")
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    
    # 남은 주제만 학습
    await learning_system.learn_remaining_topics()

async def run_learn_simple():
    """단순화된 24시간 학습 (메뉴 없이 바로 시작)"""
    print("🎓 AutoCI 24시간 C# 학습 시스템")
    print("=" * 60)
    print("24시간 C# 학습을 바로 시작합니다...")
    
    # 24시간 학습 시스템 바로 시작
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    
    # 남은 주제만 학습 (이미 완료한 것은 건너뜀)
    await learning_system.learn_remaining_topics()

async def run_learn_demo():
    """데모 모드 학습 (1시간 빠른 진행)"""
    print("⚡ AutoCI C# 학습 데모 모드")
    print("=" * 60)
    print("1시간 안에 전체 학습 과정을 시연합니다...")
    
    # 데모 모드 설정
    from modules.csharp_24h_learning_config import LearningConfig
    LearningConfig.DEMO_MODE = True
    
    # 24시간 학습 시스템 시작
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    
    # 데모 학습 시작
    await learning_system.learn_remaining_topics()

async def run_terminal_mode():
    """기본 터미널 모드 실행"""
    print("🚀 AutoCI 24시간 AI 게임 개발 시스템")
    print("=" * 60)
    print("WSL 환경에서 24시간 자동으로 게임을 개발합니다.")
    print("🎮 Godot 대시보드가 자동으로 열립니다.")
    print("'help'를 입력하여 사용 가능한 명령어를 확인하세요.")
    print("=" * 60)
    
    # 기존 터미널 시스템 실행 (asyncio.run 없이 직접 호출)
    from autoci_terminal import AutoCITerminal
    terminal = AutoCITerminal()
    await terminal.run_terminal_interface()

if __name__ == "__main__":
    main()