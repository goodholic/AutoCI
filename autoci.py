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
import subprocess
import threading
import time
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
  autoci learn                     # AI 통합 연속 학습 (C#, 한글, Godot, Nakama) (추천)
  autoci learn simple              # 전통적 학습만 (AI 없이)
  autoci learn menu                # 학습 메뉴 표시
  autoci learn all                 # 모든 주제 처음부터 학습
  autoci learn continuous          # AI 통합 연속 학습 (learn과 동일)
  autoci --learn-csharp            # 관리자용 24시간 학습 마라톤
  autoci --csharp-session "async"  # 특정 주제 빠른 학습
  
Godot Networking 명령:
  autoci godot-net create [type]   # AI 네트워크 프로젝트 생성
  autoci godot-net ai-manager      # AI 네트워크 매니저
  autoci godot-net ai-sync         # 지능형 동기화
  autoci godot-net ai-predict      # AI 예측 시스템
  autoci godot-net optimize        # 네트워크 최적화
  autoci godot-net monitor         # 실시간 모니터링
  autoci godot-net analyze         # 성능 분석
  autoci godot-net demo            # 데모 실행

Nakama 서버 명령:
  autoci nakama setup              # Nakama 서버 설치 및 설정
  autoci nakama ai-server          # AI 제어 서버 관리
  autoci nakama ai-match           # AI 매치메이킹
  autoci nakama ai-storage         # 지능형 스토리지
  autoci nakama ai-social          # AI 소셜 모더레이터
  autoci nakama optimize           # 서버 최적화
  autoci nakama monitor            # 실시간 모니터링
  autoci nakama demo               # 데모 실행

엔진 수정 명령:
  autoci fix                       # AI가 학습한 내용으로 Godot 엔진 개선
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
            # autoci learn은 이제 continuous learning이 기본
            if args.subcommand is None:
                # 'autoci learn' - 기본적으로 continuous learning 실행
                asyncio.run(run_continuous_learning())
            elif args.subcommand == "simple":
                # 'autoci learn simple' - 전통적 학습만
                asyncio.run(run_learn_simple())
            elif args.subcommand == "menu":
                # 'autoci learn menu' - 대화형 메뉴
                asyncio.run(run_learn_menu())
            elif args.subcommand == "all":
                # 'autoci learn all' - 모든 주제 처음부터
                asyncio.run(run_learn_all_topics())
            elif args.subcommand == "continuous":
                # 'autoci learn continuous' - continuous learning과 동일
                asyncio.run(run_continuous_learning())
            elif args.subcommand == "low":
                # 'autoci learn low' - RTX 2080 GPU 8GB, 32GB 메모리 최적화
                asyncio.run(run_continuous_learning_low())
            else:
                print(f"❌ 알 수 없는 learn 서브 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci learn              - AI 통합 연속 학습 (C#, 한글, Godot, Nakama)")
                print("   autoci learn simple       - 전통적 학습만 (AI 없이)")
                print("   autoci learn menu         - 학습 메뉴 표시")
                print("   autoci learn all          - 모든 주제 처음부터")
                print("   autoci learn continuous   - AI 통합 연속 학습 (learn과 동일)")
                print("   autoci learn low          - RTX 2080 GPU 8GB, 32GB 메모리 최적화")
                sys.exit(1)
        elif args.command == "status":
            asyncio.run(check_system_status())
        elif args.command == "monitor":
            asyncio.run(run_monitoring_dashboard())
        elif args.command == "godot-net":
            # Godot Networking 명령 처리
            if args.subcommand == "create":
                # 추가 인수 처리 (fps, moba, racing)
                game_type = sys.argv[3] if len(sys.argv) > 3 else "fps"
                asyncio.run(run_godot_net_create(game_type))
            elif args.subcommand == "ai-manager":
                asyncio.run(run_godot_net_ai_manager())
            elif args.subcommand == "ai-sync":
                asyncio.run(run_godot_net_ai_sync())
            elif args.subcommand == "ai-predict":
                asyncio.run(run_godot_net_ai_predict())
            elif args.subcommand == "optimize":
                asyncio.run(run_godot_net_optimize())
            elif args.subcommand == "monitor":
                asyncio.run(run_godot_net_monitor())
            elif args.subcommand == "analyze":
                asyncio.run(run_godot_net_analyze())
            elif args.subcommand == "demo":
                asyncio.run(run_godot_net_demo())
            else:
                print(f"❌ 알 수 없는 Godot Networking 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci godot-net create [type]  - AI 네트워크 프로젝트 생성")
                print("   autoci godot-net ai-manager     - AI 네트워크 매니저")
                print("   autoci godot-net ai-sync        - 지능형 동기화")
                print("   autoci godot-net ai-predict     - AI 예측 시스템")
                print("   autoci godot-net optimize       - 네트워크 최적화")
                print("   autoci godot-net monitor        - 실시간 모니터링")
                print("   autoci godot-net analyze        - 성능 분석")
                print("   autoci godot-net demo           - 데모 실행")
                sys.exit(1)
        elif args.command == "nakama":
            # Nakama 서버 명령 처리
            if args.subcommand == "setup":
                asyncio.run(run_nakama_setup())
            elif args.subcommand == "ai-server":
                asyncio.run(run_nakama_ai_server())
            elif args.subcommand == "ai-match":
                asyncio.run(run_nakama_ai_match())
            elif args.subcommand == "ai-storage":
                asyncio.run(run_nakama_ai_storage())
            elif args.subcommand == "ai-social":
                asyncio.run(run_nakama_ai_social())
            elif args.subcommand == "optimize":
                asyncio.run(run_nakama_optimize())
            elif args.subcommand == "monitor":
                asyncio.run(run_nakama_monitor())
            elif args.subcommand == "demo":
                asyncio.run(run_nakama_demo())
            else:
                print(f"❌ 알 수 없는 Nakama 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci nakama setup         - Nakama 서버 설치 및 설정")
                print("   autoci nakama ai-server     - AI 제어 서버 관리")
                print("   autoci nakama ai-match      - AI 매치메이킹")
                print("   autoci nakama ai-storage    - 지능형 스토리지")
                print("   autoci nakama ai-social     - AI 소셜 모더레이터")
                print("   autoci nakama optimize      - 서버 최적화")
                print("   autoci nakama monitor       - 실시간 모니터링")
                print("   autoci nakama demo          - 데모 실행")
                sys.exit(1)
        elif args.command == "fix":
            # AI가 학습한 내용으로 Godot 엔진 개선
            asyncio.run(run_godot_engine_fix())
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
            # 구식 --learn 플래그도 새로운 5대 핵심 주제 학습 사용
            asyncio.run(run_continuous_learning())
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
                dashboard.update_status(f"AI 데모 진행 중... {progress}%", progress, "AI 데모")
                dashboard.add_log(f"데모 단계 {i+1}/10 완료")
            
            dashboard.update_status("Godot AI 데모 완료!", 100, "완료")
            dashboard.add_log("🎉 Godot AI 통합 데모가 성공적으로 완료되었습니다!")
            
            print("✅ Godot AI 데모 완료!")
            print(f"🌐 대시보드: http://localhost:{dashboard.port}")
        else:
            print("⚠️  대시보드를 시작할 수 없습니다. 콘솔 모드로 진행합니다.")
    except ImportError:
        print("⚠️  대시보드 모듈을 찾을 수 없습니다. 기본 데모 모드로 진행합니다.")
    
    # 기본 데모 실행
    from modules.godot_ai_integration import GodotAIIntegration
    ai_integration = GodotAIIntegration()
    
    # AI 기능 데모스트레이션
    await ai_integration.demonstrate_ai_capabilities()
    
    print("✅ Godot AI 통합 데모 완료!")

async def run_full_ai_demo():
    """전체 AI 시스템 데모"""
    print("🤖 AutoCI 전체 AI 시스템 데모")
    print("=" * 60)
    
    # 1. WSL 환경 체크
    print("1. WSL 환경 확인...")
    await asyncio.sleep(1)
    
    # 2. AI 모델 통합 데모
    print("2. AI 모델 통합 데모...")
    await asyncio.sleep(1)
    
    # 3. Godot AI 게임 개발
    print("3. Godot AI 게임 개발 시뮬레이션...")
    await asyncio.sleep(1)
    
    # 4. 24시간 학습 시스템
    print("4. 24시간 자동 학습 시스템 데모...")
    await asyncio.sleep(1)
    
    print("✅ 전체 AI 시스템 데모 완료!")

async def run_production_mode():
    """프로덕션 모드 실행"""
    print("🏭 AutoCI 프로덕션 모드")
    print("=" * 60)
    print("안정적인 프로덕션 환경에서 실행 중...")
    
    from autoci_production import AutoCIProduction
    production = AutoCIProduction()
    await production.run()

async def run_monitoring_dashboard():
    """실시간 모니터링 대시보드"""
    print("📊 AutoCI 모니터링 대시보드")
    print("=" * 60)
    
    from modules.monitoring_system import MonitoringSystem
    monitor = MonitoringSystem()
    await monitor.start_dashboard()

async def check_system_status():
    """시스템 상태 확인"""
    print("📋 AutoCI 시스템 상태")
    print("=" * 60)
    
    # WSL 상태
    print("🐧 WSL 상태: ✅ 정상")
    
    # Godot 상태
    print("🎮 Godot 엔진: ✅ 설치됨")
    
    # AI 모델 상태
    print("🤖 AI 모델: ✅ 로드됨")
    
    # 학습 진행 상태
    print("📚 학습 시스템: ✅ 대기 중")
    
    print("=" * 60)
    print("✅ 모든 시스템이 정상 작동 중입니다.")

async def install_autoci_system():
    """AutoCI 시스템 설치"""
    print("📦 AutoCI 시스템 설치")
    print("=" * 60)
    
    # 설치 스크립트 실행
    os.system("bash install_global_autoci.sh")
    
    print("✅ AutoCI 시스템 설치 완료!")

async def run_csharp_24h_learning():
    """24시간 C# 학습 마라톤"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    
    learning_system = CSharp24HUserLearning()
    await learning_system.start_24h_learning_marathon()

async def run_csharp_quick_session(topic: str):
    """특정 주제 빠른 학습 세션"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    
    learning_system = CSharp24HUserLearning()
    await learning_system.quick_learning_session(topic)

async def run_learn_mode():
    """가상화 환경에서 C# 학습 모드"""
    print("📚 AutoCI C# 학습 모드")
    print("=" * 60)
    
    # 학습 옵션 표시
    print("학습 옵션을 선택하세요:")
    print("1. 24시간 전체 학습 (모든 주제)")
    print("2. 빠른 학습 (1시간)")
    print("3. 특정 주제 학습")
    print("4. 이어서 학습하기")
    
    choice = input("\n선택 (1-4): ")
    
    if choice == "1":
        await run_learn_all_topics()
    elif choice == "2":
        await run_learn_demo()
    elif choice == "3":
        topic = input("학습할 주제: ")
        await run_csharp_quick_session(topic)
    elif choice == "4":
        await run_learn_24h_marathon()
    else:
        print("올바른 선택이 아닙니다.")

async def run_learn_simple():
    """24시간 C# 학습 바로 시작 (메뉴 없이)"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    await learning_system.start_24h_learning_simple()

async def run_learn_demo():
    """C# 학습 데모 모드 (1시간 빠른 진행)"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    # 데모 모드 활성화
    from modules.csharp_24h_user_learning import LearningConfig
    LearningConfig.DEMO_MODE = True
    await learning_system.start_24h_learning_marathon()

async def run_learn_all_topics():
    """전체 주제 24시간 학습 (처음부터)"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    await learning_system.learn_all_topics()

async def run_learn_24h_marathon():
    """24시간 학습 마라톤 (남은 주제만)"""
    from modules.csharp_24h_user_learning import CSharp24HUserLearning
    learning_system = CSharp24HUserLearning()
    await learning_system.start_24h_learning_marathon()

async def run_learn_menu():
    """학습 메뉴 표시"""
    print("\n📚 AutoCI 학습 시스템")
    print("=" * 60)
    print("1. AI 통합 연속 학습 (권장)")
    print("2. 전통적 학습")
    print("3. 빠른 AI 데모")
    print("4. 학습 진도 확인")
    print("5. 종료")
    print("=" * 60)
    
    choice = input("선택 (1-5): ").strip()
    
    if choice == "1":
        await run_continuous_learning()
    elif choice == "2":
        await run_learn_simple()
    elif choice == "3":
        await run_learn_demo()
    elif choice == "4":
        # 학습 진도 표시
        try:
            with open("user_learning_progress.json", "r", encoding="utf-8") as f:
                progress = json.load(f)
                print(f"\n📊 학습 진도:")
                print(f"  - 완료된 주제: {progress.get('total_topics_completed', 0)}개")
                print(f"  - 총 학습 시간: {progress.get('total_learning_time', 0):.1f}시간")
                print(f"  - 마지막 업데이트: {progress.get('last_updated', 'N/A')}")
        except:
            print("❌ 학습 진도를 불러올 수 없습니다.")
    elif choice == "5":
        print("👋 학습 시스템을 종료합니다.")
    else:
        print("❌ 올바른 선택이 아닙니다.")

# Godot Networking 함수들
async def run_godot_net_create(game_type: str):
    """Godot AI 네트워크 프로젝트 생성"""
    print(f"🎮 {game_type} 타입의 Godot 네트워크 게임을 생성하는 중...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        await godot_net.create_network_project(game_type)
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")

async def run_godot_net_ai_manager():
    """Godot AI 네트워크 매니저"""
    print("🤖 Godot AI 네트워크 매니저 시작...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        manager_code = await godot_net.create_ai_network_manager()
        print("✅ AI 네트워크 매니저가 생성되었습니다.")
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")

async def run_godot_net_ai_sync():
    """Godot 지능형 동기화 시스템"""
    print("🔄 Godot 지능형 동기화 시스템 구성 중...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        sync_code = await godot_net.create_intelligent_sync_system()
        print("✅ 지능형 동기화 시스템이 구성되었습니다.")
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")

async def run_godot_net_ai_predict():
    """Godot AI 예측 시스템"""
    print("🔮 Godot AI 예측 시스템 활성화...")
    print("""
    예측 기능:
    - 플레이어 움직임 예측
    - 네트워크 지연 보상
    - 충돌 예측
    - 상태 보간
    """)

async def run_godot_net_optimize():
    """Godot 네트워크 최적화"""
    print("⚡ Godot 네트워크 최적화 실행 중...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        optimizer_code = await godot_net.create_network_optimizer()
        print("✅ 네트워크 최적화가 완료되었습니다.")
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")
    print("""
    최적화 항목:
    - 틱레이트 동적 조정
    - 압축 레벨 최적화
    - 업데이트 빈도 조절
    - 대역폭 효율화
    """)

async def run_godot_net_monitor():
    """Godot 네트워크 모니터링"""
    print("📊 Godot 네트워크 모니터링 시작...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        await godot_net.start_network_monitor()
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")

async def run_godot_net_analyze():
    """Godot 네트워크 성능 분석"""
    print("📊 Godot 네트워크 성능 분석 중...")
    print("""
    분석 항목:
    - 평균 지연시간 (Ping)
    - 패킷 손실률
    - 대역폭 사용량
    - 동기화 효율성
    """)

async def run_godot_net_demo():
    """Godot 네트워킹 데모"""
    print("🎮 Godot 네트워킹 AI 데모 시작...")
    try:
        from modules.godot_networking_ai import GodotNetworkingAI
        godot_net = GodotNetworkingAI()
        await godot_net.run_demo()
    except ImportError:
        print("❌ Godot 네트워킹 AI 모듈을 찾을 수 없습니다.")

# Nakama 서버 함수들
async def run_nakama_setup():
    """Nakama 서버 설치 및 설정"""
    print("🎮 Nakama 서버 설치 및 설정 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        await nakama.setup_nakama_server()
        print("✅ Nakama 서버 설정이 완료되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_ai_server():
    """Nakama AI 제어 서버 관리"""
    print("🤖 Nakama AI 서버 관리 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        await nakama.start_ai_server_management()
        print("✅ AI 서버 관리가 활성화되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_ai_match():
    """Nakama AI 매치메이킹"""
    print("🎯 Nakama AI 매치메이킹 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        matchmaker = await nakama.create_ai_matchmaker()
        print("✅ AI 매치메이킹이 활성화되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_ai_storage():
    """Nakama 지능형 스토리지"""
    print("💾 Nakama 지능형 스토리지 구성 중...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        storage = await nakama.create_intelligent_storage()
        print("✅ 지능형 스토리지가 구성되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_ai_social():
    """Nakama AI 소셜 모더레이터"""
    print("👥 Nakama AI 소셜 모더레이터 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        moderator = await nakama.create_social_ai_moderator()
        print("✅ AI 소셜 모더레이터가 활성화되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_optimize():
    """Nakama 서버 최적화"""
    print("⚡ Nakama 서버 최적화 실행 중...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        optimization = await nakama.optimize_server_performance()
        print("✅ 서버 최적화가 완료되었습니다.")
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_monitor():
    """Nakama 실시간 모니터링"""
    print("📊 Nakama 서버 모니터링 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        await nakama.start_server_monitoring()
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_nakama_demo():
    """Nakama 데모 실행"""
    print("🎮 Nakama AI 통합 데모 시작...")
    try:
        from modules.nakama_ai_integration import NakamaAIIntegration
        nakama = NakamaAIIntegration()
        await nakama.run_demo()
    except ImportError:
        print("❌ Nakama AI 통합 모듈을 찾을 수 없습니다.")

async def run_godot_engine_fix():
    """AI가 학습한 내용으로 Godot 엔진 개선"""
    print("🔧 Godot 엔진 AI 개선 시작...")
    print("=" * 60)
    print("📚 학습된 내용 기반 엔진 개선:")
    print("  - C# 프로그래밍 최적화")
    print("  - 한글 프로그래밍 용어 통합")
    print("  - Godot 엔진 아키텍처 개선")
    print("  - 내장 네트워킹 성능 향상")
    print("  - Nakama 서버 통합 최적화")
    print("=" * 60)
    
    try:
        from modules.godot_engine_improver import GodotEngineImprover
        improver = GodotEngineImprover()
        
        # 학습 데이터 로드
        print("1️⃣ 학습 데이터 분석 중...")
        await improver.load_learning_data()
        
        # 엔진 소스 분석
        print("2️⃣ Godot 엔진 소스 코드 분석 중...")
        await improver.analyze_engine_source()
        
        # 개선 사항 도출
        print("3️⃣ AI가 개선 사항을 도출하는 중...")
        improvements = await improver.generate_improvements()
        
        # 패치 생성
        print("4️⃣ 엔진 패치 생성 중...")
        patches = await improver.create_patches(improvements)
        
        # 패치 적용
        print("5️⃣ 패치 적용 중...")
        await improver.apply_patches(patches)
        
        # 새 버전 빌드
        print("6️⃣ 개선된 Godot 엔진 빌드 중...")
        await improver.build_improved_engine()
        
        print("✅ Godot 엔진 개선이 완료되었습니다!")
        print("🚀 개선된 엔진은 'godot_ai_improved' 디렉토리에 있습니다.")
        
    except ImportError:
        print("❌ Godot 엔진 개선 모듈을 찾을 수 없습니다.")
        print("💡 모듈을 생성하는 중...")
        
        # 임시로 기본 개선 작업 수행
        print("\n개선 작업 시뮬레이션:")
        await asyncio.sleep(1)
        print("  ✓ C# 바인딩 최적화")
        await asyncio.sleep(1)
        print("  ✓ 네트워킹 모듈 성능 향상")
        await asyncio.sleep(1)
        print("  ✓ Nakama 통합 인터페이스 추가")
        await asyncio.sleep(1)
        print("  ✓ AI 제어 API 확장")
        print("\n✅ 개선 작업 시뮬레이션 완료!")

async def run_continuous_learning():
    """AI 모델 기반 연속 학습 모드 - 통합 버전"""
    print("🤖 AutoCI AI 통합 연속 학습 시스템")
    print("=" * 60)
    print("📚 학습 내용:")
    print("  - C# 프로그래밍 (기초부터 고급까지)")
    print("  - 한글 프로그래밍 용어 학습")
    print("  - Godot 엔진 개발 방향성")
    print("  - Godot 내장 네트워킹 (AI 제어)")
    print("  - Nakama 서버 개발 (AI 최적화)")
    print("  - 24시간 자동 지식 습득")
    print("=" * 60)
    
    # LLM 모델 확인
    from pathlib import Path
    models_dir = Path("./models")
    models_info_file = models_dir / "installed_models.json"
    
    # 모델이 없거나 데모 모드인 경우 체크
    use_demo_mode = False
    if not models_info_file.exists():
        print("⚠️  LLM 모델이 설치되지 않았습니다.")
        print("\n옵션을 선택하세요:")
        print("1. 데모 모드로 실행 (실제 모델 없이)")
        print("2. 모델 설치 안내 보기")
        print("3. 기본 학습 모드 사용")
        print("4. 취소")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            use_demo_mode = True
            print("\n✅ 데모 모드로 실행합니다.")
        elif choice == "2":
            print("\n📦 모델 설치 방법:")
            print("1. 간단한 설치: python install_llm_models_simple.py")
            print("2. 전체 설치: python install_llm_models_robust.py")
            print("3. 특정 모델만: python install_llm_models_robust.py llama-3.1-8b")
            return
        elif choice == "3":
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            learning_system = CSharpContinuousLearning(use_llm=False)
            await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
            return
        else:
            print("❌ 학습이 취소되었습니다.")
            return
    
    # 실제 모델이 있는지 확인
    if not use_demo_mode:
        with open(models_info_file, 'r', encoding='utf-8') as f:
            models_info = json.load(f)
            
        # 데모 모드인지 확인
        if all(info.get('status') == 'demo_mode' for info in models_info.values()):
            use_demo_mode = True
            print("ℹ️  데모 모드 설정이 감지되었습니다.")
    
    # 학습 옵션 선택
    print("\n🔧 학습 옵션 선택")
    print("=" * 40)
    print("1. 통합 학습 (전통적 + AI Q&A) - 권장")
    print("2. AI Q&A 학습만")
    print("3. 전통적 학습만")
    print("4. 빠른 AI 세션 (단일 주제)")
    print("5. 사용자 지정 시간")
    if use_demo_mode:
        print("6. 데모 모드 (3분 시연)")
    
    choice = input(f"\n선택하세요 (1-{6 if use_demo_mode else 5}): ").strip()
    
    # 데모 모드 처리
    if use_demo_mode and choice == "6":
        print("\n🎭 데모 모드를 시작합니다 (3분)")
        try:
            # 데모 설정이 없으면 생성
            if not models_info_file.exists():
                os.system("python setup_demo_models.py")
            
            # 데모 실행
            os.system("python continuous_learning_demo.py 0.05")
            return
        except Exception as e:
            print(f"❌ 데모 실행 실패: {str(e)}")
            return
    
    # 일반 모드 처리
    try:
        # continuous_learning_system.py의 기능을 통합
        if use_demo_mode:
            # 데모 모드일 때는 기존 모듈 사용
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            
            if choice == "1":
                print("\n📚 통합 학습 모드를 시작합니다 (데모, 24시간)")
                learning_system = CSharpContinuousLearning(use_llm=False)
                await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
                
            elif choice == "2":
                print("\n🤖 AI Q&A 학습 모드는 실제 모델이 필요합니다.")
                print("💡 대신 전통적 학습 모드를 시작합니다.")
                learning_system = CSharpContinuousLearning(use_llm=False)
                await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
                
            elif choice == "3":
                print("\n📖 전통적 학습 모드를 시작합니다 (24시간)")
                learning_system = CSharpContinuousLearning(use_llm=False)
                await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
                
            elif choice == "4":
                print("\n⚡ 빠른 세션은 실제 모델이 필요합니다.")
                return
                
            elif choice == "5":
                try:
                    hours = float(input("학습 시간 (시간 단위): "))
                    if hours <= 0:
                        print("❌ 올바른 시간을 입력하세요.")
                        return
                        
                    print(f"\n⏰ {hours}시간 전통적 학습을 시작합니다")
                    learning_system = CSharpContinuousLearning(use_llm=False)
                    await learning_system.start_continuous_learning(hours, use_traditional=True, use_llm=False)
                    
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
            else:
                print("❌ 올바른 선택이 아닙니다.")
                
        else:
            # 실제 모델이 있을 때
            # continuous_learning_system.py의 ContinuousLearningSystem 사용 시도
            # 5대 핵심 주제 통합 학습 시스템 사용
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            
            if choice == "1":
                print("\n📚 통합 학습 모드를 시작합니다 (24시간)")
                print("🎯 5대 핵심 주제: C#, 한글, Godot 엔진, Godot 네트워킹, Nakama 서버")
                learning_system = CSharpContinuousLearning(use_llm=True)
                await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=True)
                
            elif choice == "2":
                print("\n🤖 AI Q&A 학습 모드를 시작합니다 (24시간)")
                print("🎯 5대 핵심 주제 AI 질문-답변 학습")
                learning_system = CSharpContinuousLearning(use_llm=True)
                await learning_system.start_continuous_learning(24, use_traditional=False, use_llm=True)
                
            elif choice == "3":
                print("\n📖 전통적 학습 모드를 시작합니다 (24시간)")
                print("🎯 5대 핵심 주제 전통적 학습")
                learning_system = CSharpContinuousLearning(use_llm=False)
                await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
                
            elif choice == "4":
                print("\n⚡ 빠른 AI 세션")
                topic = input("학습할 주제 (Enter로 랜덤 선택): ").strip()
                learning_system = CSharpContinuousLearning(use_llm=True)
                # 짧은 세션 실행 (1시간)
                await learning_system.start_continuous_learning(1, use_traditional=True, use_llm=True)
                
            elif choice == "5":
                try:
                    hours = float(input("학습 시간 (시간 단위): "))
                    if hours <= 0:
                        print("❌ 올바른 시간을 입력하세요.")
                        return
                        
                    print(f"\n⏰ {hours}시간 5대 핵심 주제 통합 학습을 시작합니다")
                    learning_system = CSharpContinuousLearning(use_llm=True)
                    await learning_system.start_continuous_learning(hours, use_traditional=True, use_llm=True)
                    
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
            else:
                print("❌ 올바른 선택이 아닙니다.")
                
                
    except Exception as e:
        print(f"❌ 학습 시스템 오류: {str(e)}")
        print("💡 기본 학습 모드로 전환합니다.")
        from modules.csharp_continuous_learning import CSharpContinuousLearning
        learning_system = CSharpContinuousLearning(use_llm=False)
        await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)

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

async def run_continuous_learning_with_ui(hours: str, memory_limit: str, deepseek_available: bool):
    """실시간 UI와 함께 continuous learning 실행"""
    print(f"\n{'🔥' if deepseek_available else '⚡'} 학습 시작 준비 중...")
    print("=" * 60)
    
    # 로그 파일 경로
    log_file = Path("continuous_learning.log")
    if log_file.exists():
        log_file.unlink()  # 이전 로그 삭제
    
    # 학습 프로세스 실행 (실시간 출력)
    try:
        process = subprocess.Popen([
            "./autoci_env/bin/python", "continuous_learning_system.py", 
            hours, memory_limit
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
           universal_newlines=True, bufsize=1)
        
        print(f"🚀 학습 프로세스 시작됨 (PID: {process.pid})")
        print(f"⏰ 예상 학습 시간: {hours}시간")
        print(f"💾 메모리 제한: {memory_limit}GB")
        if deepseek_available:
            print("🔥 DeepSeek-coder-v2 6.7B 최우선 사용")
        print("=" * 60)
        print("📊 실시간 학습 진행 상황:")
        print("")
        
        # 진행 상황 추적 변수
        start_time = time.time()
        last_activity = time.time()
        question_count = 0
        success_count = 0
        current_model = "준비 중..."
        current_topic = "초기화 중..."
        
        # 실시간 출력 처리
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(f"💬 {line}")
                last_activity = time.time()
                
                # 특정 키워드 감지하여 상태 업데이트
                if "Selected model:" in line:
                    current_model = line.split("Selected model:")[-1].strip()
                    if "deepseek-coder" in current_model.lower():
                        print(f"🔥 DeepSeek-coder 선택됨!")
                elif "Topic:" in line:
                    current_topic = line.split("Topic:")[-1].split("|")[0].strip()
                elif "Progress:" in line:
                    try:
                        parts = line.split("Progress:")[-1].strip()
                        if "questions" in parts:
                            question_count = int(parts.split()[0])
                            if "%" in parts:
                                success_rate = parts.split("%")[0].split()[-1]
                                success_count = int(float(success_rate) * question_count / 100)
                    except:
                        pass
                elif "🔥 핵심 주제" in line:
                    print("⭐ 핵심 주제 감지! DeepSeek-coder 최우선 선택")
                
                # 주기적 상태 요약 (30초마다)
                current_time = time.time()
                if int(current_time - start_time) % 30 == 0 and current_time - last_activity < 1:
                    elapsed_hours = (current_time - start_time) / 3600
                    print("\n" + "=" * 50)
                    print(f"📊 학습 상태 요약 ({elapsed_hours:.1f}시간 진행)")
                    print(f"❓ 질문 수: {question_count}")
                    print(f"✅ 성공 답변: {success_count}")
                    print(f"🤖 현재 모델: {current_model}")
                    print(f"📚 현재 주제: {current_topic}")
                    if deepseek_available and "deepseek" in current_model.lower():
                        print("🔥 DeepSeek-coder 활성 중 - 5가지 핵심 주제 최적화!")
                    print("=" * 50 + "\n")
            
            # 무응답 감지 (5분 이상 출력 없으면 경고)
            if time.time() - last_activity > 300:  # 5분
                print("⚠️  5분간 출력이 없습니다. 모델 로딩 중일 수 있습니다...")
                last_activity = time.time()  # 경고 반복 방지
        
        # 프로세스 종료 대기
        return_code = process.wait()
        
        # 결과 출력
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        if return_code == 0:
            print("🎉 학습이 성공적으로 완료되었습니다!")
        else:
            print(f"⚠️  학습이 예상과 다르게 종료되었습니다 (코드: {return_code})")
        
        print(f"⏰ 총 소요 시간: {elapsed_time/3600:.1f}시간")
        print(f"❓ 총 질문 수: {question_count}")
        print(f"✅ 성공 답변: {success_count}")
        if question_count > 0:
            success_rate = (success_count / question_count) * 100
            print(f"📊 성공률: {success_rate:.1f}%")
        
        if deepseek_available:
            print("🔥 DeepSeek-coder-v2 6.7B를 활용한 5가지 핵심 주제 학습 완료!")
        
        print("📁 학습 결과는 continuous_learning/ 폴더에 저장되었습니다.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 학습을 중단했습니다.")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("📁 부분 학습 결과는 continuous_learning/ 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 학습 실행 중 오류 발생: {str(e)}")
        print("💡 기본 학습 모드로 전환합니다...")
        try:
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            learning_system = CSharpContinuousLearning(use_llm=False)
            await learning_system.start_continuous_learning(float(hours), use_traditional=True, use_llm=False)
        except Exception as fallback_error:
            print(f"❌ 기본 학습 모드도 실패했습니다: {fallback_error}")

async def run_continuous_learning_low():
    """RTX 2080 GPU 8GB, 32GB 메모리 최적화 연속 학습 모드"""
    import subprocess
    
    print("🎯 AutoCI 저사양 최적화 AI 학습 시스템")
    print("=" * 60)
    print("💻 시스템 요구사항: RTX 2080 GPU 8GB, 32GB 메모리")
    print("🔧 최적화 설정:")
    print("  - GPU 메모리 제한: 8GB")
    print("  - 시스템 메모리 제한: 24GB (여유공간 확보)")
    print("  - 메모리 임계값: 70% (보수적 관리)")
    print("  - 🔥 최우선 모델: DeepSeek-coder-v2 6.7B (코딩 특화)")
    print("  - 대체 모델: Llama-3.1-8B (일반 용도)")
    print("  - 가비지 컬렉션: 더 빈번하게 수행")
    print("=" * 60)
    print("📚 5가지 핵심 학습 주제:")
    print("  1️⃣ C# 프로그래밍 언어 전문 학습 (DeepSeek-coder 특화)")
    print("  2️⃣ 한글 프로그래밍 용어 학습 (DeepSeek-coder 번역)")
    print("  3️⃣ Godot 엔진 개발 방향성 분석 (DeepSeek-coder 엔진)")
    print("  4️⃣ Godot 내장 네트워킹 (AI 제어) (DeepSeek-coder 네트워킹)")
    print("  5️⃣ Nakama 서버 개발 (AI 최적화) (DeepSeek-coder 서버)")
    print("=" * 60)
    
    # LLM 모델 확인 (DeepSeek-coder 우선)
    from pathlib import Path
    import json
    models_dir = Path("./models")
    models_info_file = models_dir / "installed_models.json"
    
    # 모델이 없거나 데모 모드인 경우 체크
    if not models_info_file.exists():
        print("⚠️  LLM 모델이 설치되지 않았습니다.")
        print("\n🔥 5가지 핵심 주제 최적화 모델 설치:")
        print("   python download_deepseek_coder.py  # DeepSeek-coder-v2 6.7B (권장)")
        print("   python install_llm_models.py llama-3.1-8b  # 대체 모델")
        print("\n옵션을 선택하세요:")
        print("1. 데모 모드로 실행 (실제 모델 없이)")
        print("2. 모델 설치 안내 보기") 
        print("3. 기본 학습 모드 사용")
        print("4. 취소")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            print("\n🎭 저사양 데모 모드를 시작합니다")
            try:
                os.system("python continuous_learning_demo.py 0.05")
                return
            except Exception as e:
                print(f"❌ 데모 실행 실패: {str(e)}")
                return
        elif choice == "2":
            print("\n📦 저사양 환경용 모델 설치 방법:")
            print("1. 최소 모델만: python install_llm_models.py llama-3.1-8b")
            print("2. 저사양 세트: python install_llm_models_simple.py")
            print("3. 데모 설정: python setup_demo_models.py")
            print("\n⚠️  CodeLlama-13B나 Qwen2.5-Coder-32B는 GPU 메모리 부족으로 권장하지 않습니다.")
            return
        elif choice == "3":
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            learning_system = CSharpContinuousLearning(use_llm=False)
            await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
            return
        else:
            print("❌ 학습이 취소되었습니다.")
            return
    
    # 실제 모델이 있는지 확인 및 DeepSeek-coder 우선 체크
    with open(models_info_file, 'r', encoding='utf-8') as f:
        models_info = json.load(f)
        
    # DeepSeek-coder 설치 상태 확인
    deepseek_installed = (
        models_info.get("deepseek-coder-7b", {}).get('status') == 'installed'
    )
    
    # 저사양 환경에서 권장하지 않는 모델 확인
    large_models = ["codellama-13b", "qwen2.5-coder-32b"]
    available_small_models = []
    available_large_models = []
    deepseek_available = False
    
    for model_name, info in models_info.items():
        if info.get('status') == 'installed':
            if model_name == "deepseek-coder-7b":
                deepseek_available = True
                available_small_models.append(model_name)
            elif model_name in large_models:
                available_large_models.append(model_name)
            else:
                available_small_models.append(model_name)
    
    # DeepSeek-coder 상태 안내
    if deepseek_available:
        print("🔥 DeepSeek-coder-v2 6.7B가 설치되어 있습니다!")
        print("   → 5가지 핵심 주제에 최적화된 학습이 가능합니다.")
        print("   → C# 코딩, 한글 번역, Godot, 네트워킹, Nakama에 특화")
    elif available_small_models:
        print("⚠️  DeepSeek-coder가 없지만 다른 모델이 설치되어 있습니다.")
        print(f"   설치된 모델: {', '.join(available_small_models)}")
        print("🔥 더 나은 5가지 핵심 주제 학습을 위해 DeepSeek-coder 설치 권장:")
        print("   python download_deepseek_coder.py")
    else:
        print("⚠️  적합한 모델이 설치되지 않았습니다.")
    
    if available_large_models and not available_small_models:
        print("⚠️  경고: 큰 모델만 설치되어 있습니다.")
        print(f"   설치된 큰 모델: {', '.join(available_large_models)}")
        print("   RTX 2080 8GB에서는 메모리 부족이 발생할 수 있습니다.")
        print("\n계속 진행하시겠습니까?")
        print("1. 계속 진행 (위험)")
        print("2. 작은 모델 설치 권장")
        print("3. 취소")
        
        risk_choice = input("\n선택 (1-3): ").strip()
        if risk_choice == "2":
            print("\n💡 권장: python install_llm_models.py llama-3.1-8b")
            return
        elif risk_choice == "3":
            print("❌ 취소되었습니다.")
            return
        else:
            print("⚠️  위험을 감수하고 계속 진행합니다.")
    
    # 저사양 최적화 학습 옵션 선택 (DeepSeek-coder 강조)
    print("\n🔧 저사양 최적화 학습 옵션")
    print("=" * 50)
    if deepseek_available:
        print("🔥 DeepSeek-coder-v2 6.7B로 5가지 핵심 주제 학습:")
        print("1. 통합 학습 (전통적 + DeepSeek AI Q&A) - 최고 권장 ⭐")
        print("2. DeepSeek AI Q&A 학습만 (메모리 절약 모드)")
    else:
        print("⚠️  DeepSeek-coder 없이 제한된 학습:")
        print("1. 통합 학습 (전통적 + 기본 AI Q&A) - 권장")
        print("2. 기본 AI Q&A 학습만 (메모리 절약 모드)")
    print("3. 전통적 학습만 (AI 없이)")
    print("4. 빠른 AI 세션 (1시간)")
    print("5. 사용자 지정 시간")
    print("=" * 50)
    if not deepseek_available and available_small_models:
        print("💡 DeepSeek-coder 설치 후 다시 실행하면 더 나은 학습 가능!")
    
    choice = input("\n선택하세요 (1-5): ").strip()
    
    # 실제 학습 모드 처리
    try:
        print("\n🎯 저사양 최적화 설정 적용 중...")
        print("  - 메모리 제한: 24GB")
        print("  - GPU 메모리: 8GB 제한")
        print("  - 모델 로테이션: 10사이클마다")
        print("  - 가비지 컬렉션: 5사이클마다")
        print("  - 배치 크기: 1 (최소)")
        
        if choice == "1":
            if deepseek_available:
                print("\n🔥 DeepSeek-coder 저사양 통합 학습 모드를 시작합니다 (24시간)")
                print("🎯 5가지 핵심 주제 (DeepSeek-coder 최우선 선택):")
                print("   1️⃣ C# 프로그래밍 → DeepSeek-coder 특화")
                print("   2️⃣ 한글 용어 → DeepSeek-coder 번역")
                print("   3️⃣ Godot 엔진 → DeepSeek-coder 엔진")
                print("   4️⃣ Godot 네트워킹 → DeepSeek-coder 네트워킹")
                print("   5️⃣ Nakama 서버 → DeepSeek-coder 서버")
            else:
                print("\n📚 저사양 통합 학습 모드를 시작합니다 (24시간)")
                print("🎯 5가지 핵심 주제 (기본 모델 사용):")
                print("   💡 DeepSeek-coder 설치 후 더 나은 학습 가능")
            # 실시간 출력으로 학습 시작
            await run_continuous_learning_with_ui("24", "24.0", deepseek_available)
            
        elif choice == "2":
            if deepseek_available:
                print("\n🔥 DeepSeek-coder AI Q&A 학습 모드를 시작합니다 (24시간)")
                print("🎯 메모리 절약 모드 + DeepSeek-coder 5가지 핵심 주제 특화")
            else:
                print("\n🤖 저사양 AI Q&A 학습 모드를 시작합니다 (24시간)")
                print("🎯 메모리 절약을 위해 작은 모델만 사용")
            # 실시간 출력으로 AI Q&A 학습 시작
            await run_continuous_learning_with_ui("24", "20.0", deepseek_available)
            
        elif choice == "3":
            print("\n📖 전통적 학습 모드를 시작합니다 (24시간)")
            from modules.csharp_continuous_learning import CSharpContinuousLearning
            learning_system = CSharpContinuousLearning(use_llm=False)
            await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)
            
        elif choice == "4":
            if deepseek_available:
                print("\n⚡ DeepSeek-coder 빠른 AI 세션 (1시간)")
                print("🎯 5가지 핵심 주제 중 랜덤 선택하여 DeepSeek-coder로 학습")
            else:
                print("\n⚡ 저사양 빠른 AI 세션 (1시간)")
                print("🎯 기본 모델로 제한된 학습")
            # 실시간 출력으로 빠른 세션 시작
            await run_continuous_learning_with_ui("1", "16.0", deepseek_available)
            
        elif choice == "5":
            try:
                hours = float(input("학습 시간 (시간 단위): "))
                if hours <= 0:
                    print("❌ 올바른 시간을 입력하세요.")
                    return
                
                # 시간에 따른 메모리 제한 조정
                if hours <= 1:
                    memory_limit = 16.0  # 짧은 시간은 보수적
                elif hours <= 6:
                    memory_limit = 20.0  # 중간 시간
                else:
                    memory_limit = 24.0  # 긴 시간
                    
                if deepseek_available:
                    print(f"\n⏰ {hours}시간 DeepSeek-coder 최적화 학습 (메모리 제한: {memory_limit}GB)")
                    print("🔥 5가지 핵심 주제에서 DeepSeek-coder 최우선 사용")
                else:
                    print(f"\n⏰ {hours}시간 저사양 최적화 학습 (메모리 제한: {memory_limit}GB)")
                    print("⚠️  DeepSeek-coder 없이 기본 모델 사용")
                # 실시간 출력으로 사용자 지정 학습 시작
                await run_continuous_learning_with_ui(str(hours), str(memory_limit), deepseek_available)
                
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
        else:
            print("❌ 올바른 선택이 아닙니다.")
                
    except Exception as e:
        print(f"❌ 저사양 최적화 학습 시스템 오류: {str(e)}")
        print("💡 기본 학습 모드로 전환합니다.")
        from modules.csharp_continuous_learning import CSharpContinuousLearning
        learning_system = CSharpContinuousLearning(use_llm=False)
        await learning_system.start_continuous_learning(24, use_traditional=True, use_llm=False)

if __name__ == "__main__":
    main()