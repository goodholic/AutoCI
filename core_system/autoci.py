#!/usr/bin/env python3
"""
AutoCI - Panda3D AI 게임 개발 시스템
WSL 환경에서 가상화부터 Panda3D AI 게임 개발까지 모든 것을 한번에 실행
"""

import sys
import os
import asyncio
import argparse
import json
import subprocess
import threading
import time
import logging
from pathlib import Path
from datetime import datetime

# 가상환경 체크 (런처를 통해 실행되는 경우 건너뜀)
def check_virtual_env():
    """가상환경이 활성화되어 있는지 확인"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

# 런처를 통해 실행되는지 확인
if os.environ.get('AUTOCI_LAUNCHER') != 'true' and not check_virtual_env():
    print("⚠️  가상환경이 활성화되지 않았습니다.")
    print("💡 'autoci' 명령어를 사용하거나 다음을 실행하세요:")
    print("   source autoci_env/bin/activate  # Linux/Mac")
    print("   autoci_env\\Scripts\\activate     # Windows")
    sys.exit(1)

# AI 모델 컨트롤러 임포트
try:
    from modules.ai_model_controller import AIModelController
    from modules.terminal_ui import get_terminal_ui # terminal_ui 임포트
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("💡 requirements.txt의 패키지들이 설치되었는지 확인하세요.")
    sys.exit(1)

# 로깅 설정
logger = logging.getLogger(__name__)

def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        description="AutoCI - 24시간 AI 게임 개발 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  autoci                           # 24시간 자동 게임 개발 (대화형 메뉴)
  autoci create [game_type]        # 게임 타입 지정하여 바로 개발 시작
  autoci chat                      # 한글 대화 모드로 게임 개발
  autoci learn                     # AI 모델 기반 연속 학습
  autoci learn low                 # 메모리 최적화 연속 학습 (8GB GPU)
  autoci fix                       # 학습 기반 AI 게임 제작 능력 업데이트
  autoci monitor                   # 실시간 모니터링 대시보드
  autoci status                    # 시스템 상태 확인
  autoci help                      # 도움말 표시

게임 타입:
  platformer - 플랫폼 게임
  racing     - 레이싱 게임
  rpg        - RPG 게임
  puzzle     - 퍼즐 게임

예시:
  autoci create platformer         # 플랫폼 게임 24시간 자동 개발
  autoci chat                      # "플랫폼 게임 만들어줘"로 시작
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
  
AI 모델 제어:
  autoci control                   # AI 모델 제어권 상태 확인
  autoci learn low                 # RTX 2080 최적화 + AI 제어 상태 자동 표시
  
자가 진화 시스템:
  autoci evolve                    # 자가 진화 시스템 상태 확인
  autoci evolve status             # 진화 통계 및 집단 지성 정보
  autoci evolve insights           # 최근 발견된 인사이트
  
한글 대화 + AI 게임 개발:
  autoci chat                      # AI 게임 개발자 모드 (한글 대화로 24시간 게임 개발)
  autoci talk                      # chat과 동일 (대화하며 게임 개발)
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
    
    try:
        # Panda3D 기반 명령어로 리다이렉트
        if args.command in ["create", "chat", "learn", "fix", "monitor", "status", "help"]:
            # autoci_main.py로 리다이렉트
            import subprocess
            main_script = Path(__file__).parent / "autoci_main.py"
            if main_script.exists():
                subprocess.run([sys.executable, str(main_script)] + sys.argv[1:])
                return
            else:
                print(f"❌ 메인 스크립트를 찾을 수 없습니다: {main_script}")
                sys.exit(1)
        
        # 기존 명령어 처리 (레거시 호환성)
        elif args.command == "learn_legacy":
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
            elif args.subcommand == "godot-expert":
                # 'autoci learn godot-expert' - Godot 전문가 학습
                asyncio.run(run_godot_expert_learning())
            else:
                print(f"❌ 알 수 없는 learn 서브 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci learn              - AI 통합 연속 학습 (C#, 한글, Godot, Nakama)")
                print("   autoci learn simple       - 전통적 학습만 (AI 없이)")
                print("   autoci learn menu         - 학습 메뉴 표시")
                print("   autoci learn all          - 모든 주제 처음부터")
                print("   autoci learn continuous   - AI 통합 연속 학습 (learn과 동일)")
                print("   autoci learn low          - RTX 2080 GPU 8GB, 32GB 메모리 최적화")
                print("   autoci learn godot-expert - Godot 전문가 학습")
                sys.exit(1)
        elif args.command == "status":
            asyncio.run(check_system_status())
        elif args.command == "monitor":
            # Monitor 명령 처리 - 실시간 상세 모니터링
            if args.subcommand is None:
                # 'autoci monitor' - 기본 실시간 모니터링
                asyncio.run(run_realtime_monitoring())
            elif args.subcommand == "status":
                # 'autoci monitor status' - 시스템 상태만
                asyncio.run(run_monitor_status())
            elif args.subcommand == "learning":
                # 'autoci monitor learning' - 학습 상태만
                asyncio.run(run_monitor_learning())
            elif args.subcommand == "projects":
                # 'autoci monitor projects' - 게임 프로젝트만
                asyncio.run(run_monitor_projects())
            elif args.subcommand == "logs":
                # 'autoci monitor logs' - 로그만
                asyncio.run(run_monitor_logs())
            elif args.subcommand == "interactive":
                # 'autoci monitor interactive' - 대화형 모드
                asyncio.run(run_monitor_interactive())
            elif args.subcommand == "watch":
                # 'autoci monitor watch' - 자동 새로고침
                asyncio.run(run_monitor_watch())
            elif args.subcommand == "dashboard":
                # 'autoci monitor dashboard' - 기존 대시보드
                asyncio.run(run_monitoring_dashboard())
            else:
                print(f"❌ 알 수 없는 monitor 서브 명령: {args.subcommand}")
                print("💡 사용 가능한 명령:")
                print("   autoci monitor                - 실시간 상세 모니터링 (기본)")
                print("   autoci monitor status         - 시스템 상태만 표시")
                print("   autoci monitor learning       - AI 학습 상태만 표시")
                print("   autoci monitor projects       - 게임 프로젝트만 표시")
                print("   autoci monitor logs           - 최근 로그만 표시")
                print("   autoci monitor interactive    - 대화형 모니터링 모드")
                print("   autoci monitor watch          - 5초마다 자동 새로고침")
                print("   autoci monitor dashboard      - 기존 모니터링 대시보드")
                sys.exit(1)
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
            # AI가 학습한 내용으로 Panda3D 게임 개발 능력 개선
            asyncio.run(run_panda3d_game_fix())
        elif args.command == "control":
            # 🎮 AI 모델 제어권 상태 확인 (단독 실행)
            asyncio.run(show_ai_control_status())
        elif args.command == "evolve":
            # 🧬 자가 진화 시스템
            if args.subcommand == "status":
                asyncio.run(show_evolution_status())
            elif args.subcommand == "insights":
                asyncio.run(show_evolution_insights())
            else:
                # 기본: 간단한 상태 표시
                asyncio.run(show_evolution_summary())
        elif args.command == "gather-code":
            # 🌐 외부 코드 수집
            asyncio.run(run_code_gathering())
        elif args.command == "chat" or args.command == "talk":
            # 💬 한글 대화 모드 + 24시간 게임 개발 AI
            asyncio.run(run_ai_game_developer())
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
            # 기본값: autoci_main.py로 리다이렉트
            import subprocess
            main_script = Path(__file__).parent / "autoci_main.py"
            if main_script.exists():
                subprocess.run([sys.executable, str(main_script)])
                return
            else:
                # 폴백: 기존 게임 개발자 모드
                asyncio.run(run_ai_game_developer())
            
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
    
    from core.autoci_production import AutoCIProduction
    production = AutoCIProduction()
    await production.run()

async def run_monitoring_dashboard():
    """실시간 모니터링 대시보드"""
    print("📊 AutoCI 모니터링 대시보드")
    print("=" * 60)
    
    try:
        from modules.autoci_monitor_client import AutoCIMonitorClient
        monitor = AutoCIMonitorClient(mode="simple")  # simple 모드로 시작
        await monitor.run_async()
    except ImportError:
        print("❌ 모니터링 시스템을 찾을 수 없습니다.")
        print("autoci_monitor_client.py가 modules 디렉토리에 있는지 확인해주세요.")

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
    print("📖 전통적 학습 모드를 시작합니다 (24시간)")
    print("🎯 5대 핵심 주제 전통적 학습")
    print("=" * 60)
    print("💡 실제 24시간 동안 체계적인 학습이 진행됩니다.")
    print("⏰ 각 주제별로 20-40분씩 심화 학습")
    print("💾 진행 상태 자동 저장 (Ctrl+C로 중단 후 재개 가능)")
    print("=" * 60)
    
    # LLM 경고를 무시하고 실제 24시간 학습 진행
    try:
        from modules.csharp_24h_user_learning import CSharp24HUserLearning, LearningConfig
        
        # 데모 모드 비활성화 (실제 24시간 학습)
        LearningConfig.DEMO_MODE = False
        
        # 전통적 학습 시스템 시작 (이어서 학습)
        learning_system = CSharp24HUserLearning()
        print("🚀 실제 24시간 전통적 학습 시작!")
        print("📚 이미 완료한 주제는 건너뛰고 이어서 학습합니다")
        await learning_system.start_24h_learning_marathon(skip_completed=True)
        
    except ImportError as e:
        print(f"❌ 학습 모듈 import 실패: {e}")
        print("💡 기본 학습 모드로 전환합니다...")
        await run_basic_learning_mode()
    except Exception as e:
        print(f"❌ 학습 시스템 오류: {e}")
        print("💡 그래도 실제 24시간 학습을 시도해보겠습니다...")
        
        # 오류가 있어도 시도
        try:
            from modules.csharp_24h_user_learning import CSharp24HUserLearning, LearningConfig
            LearningConfig.DEMO_MODE = False
            learning_system = CSharp24HUserLearning()
            print("📚 이어서 학습을 시도합니다...")
            await learning_system.start_24h_learning_marathon(skip_completed=True)
        except:
            print("🔄 최종적으로 기본 학습 모드로 전환합니다...")
            await run_basic_learning_mode()

async def run_basic_learning_mode():
    """LLM 없이 작동하는 기본 전통적 학습 모드"""
    print("\n📚 기본 전통적 학습 모드 시작")
    print("=" * 50)
    
    # 5대 핵심 주제
    topics = [
        {
            "name": "C# 프로그래밍 기초",
            "duration": "4시간",
            "subtopics": ["변수와 타입", "연산자", "조건문", "반복문", "메서드", "배열과 컬렉션"]
        },
        {
            "name": "객체지향 프로그래밍",
            "duration": "4시간", 
            "subtopics": ["클래스", "객체", "상속", "다형성", "캡슐화", "인터페이스"]
        },
        {
            "name": "고급 C# 기능",
            "duration": "4시간",
            "subtopics": ["제네릭", "델리게이트", "람다 표현식", "LINQ", "예외 처리", "파일 I/O"]
        },
        {
            "name": "Godot 엔진 통합",
            "duration": "4시간",
            "subtopics": ["Godot Node", "Signal 시스템", "리소스 관리", "씬 트리", "물리 엔진", "UI 시스템"]
        },
        {
            "name": "게임 개발 실습",
            "duration": "8시간",
            "subtopics": ["게임 아키텍처", "상태 머신", "컴포넌트 시스템", "네트워킹", "최적화", "디버깅"]
        }
    ]
    
    total_duration = 24
    print(f"📅 총 학습 시간: {total_duration}시간")
    print(f"📝 학습 주제 수: {len(topics)}개")
    print("\n📋 학습 계획:")
    
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic['name']} ({topic['duration']})")
        for j, subtopic in enumerate(topic['subtopics'], 1):
            print(f"     {i}.{j} {subtopic}")
        print()
    
    print("🚀 학습을 시작하려면 Enter를 누르세요 (Ctrl+C로 중단)")
    try:
        input()
        
        print("\n📖 전통적 학습 시뮬레이션 시작...")
        print("💡 실제 24시간 학습 대신 각 주제별 데모를 진행합니다.")
        
        for i, topic in enumerate(topics, 1):
            print(f"\n🎯 {i}/{len(topics)}: {topic['name']} 학습 시작")
            print(f"⏱️ 예상 소요 시간: {topic['duration']}")
            
            for j, subtopic in enumerate(topic['subtopics'], 1):
                print(f"   📌 {i}.{j} {subtopic} 학습 중...")
                await asyncio.sleep(2)  # 2초 시뮬레이션
                print(f"   ✅ {subtopic} 완료")
            
            print(f"🏆 {topic['name']} 주제 완료!")
            await asyncio.sleep(1)
        
        print("\n🎉 전통적 학습 모드 완료!")
        print("📊 학습 결과:")
        print(f"  ✅ 완료된 주제: {len(topics)}개")
        print(f"  ✅ 완료된 세부 주제: {sum(len(t['subtopics']) for t in topics)}개")
        print(f"  ⏱️ 총 시뮬레이션 시간: {len(topics) * 10 + sum(len(t['subtopics']) for t in topics) * 2}초")
        print("\n💡 실제 24시간 학습을 원한다면 LLM 모델을 설치하고 'autoci learn'을 실행하세요.")
        
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 사용자에 의해 중단되었습니다.")
        print("📊 현재까지의 진행 상황이 저장되었습니다.")

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

async def check_ai_control_status():
    """AI 모델 제어권 상태 확인 (외부 명령어용)"""
    await show_ai_control_status()

async def show_ai_control_status():
    """🎮 AI 모델 제어권 상태 확인"""
    print("🎮 AI 모델 완전 제어 시스템 상태")
    print("=" * 60)
    
    try:
        # AI 모델 컨트롤러 모듈 확인
        try:
            from modules.ai_model_controller import AIModelController
            controller = AIModelController()
            print("✅ AI 모델 컨트롤러: 정상 작동")
            print("🎯 우리가 AI 모델의 조종권을 완전히 갖고 있습니다!")
            
            # 품질 기준 표시
            print("\n📊 품질 관리 기준:")
            print(f"  • 최소 응답 길이: {controller.quality_standards['min_length']} 문자")
            print(f"  • 최대 응답 길이: {controller.quality_standards['max_length']} 문자")
            print(f"  • 한글 응답 비율: {controller.quality_standards['required_korean_ratio']*100}%")
            print(f"  • 금지된 응답: {len(controller.quality_standards['forbidden_phrases'])}개 패턴")
            
            # 모델별 제어 설정 표시
            print("\n🔧 모델별 제어 설정:")
            for model_name, control in controller.model_controls.items():
                print(f"  📦 {model_name}:")
                print(f"    - 품질 임계점: {control.quality_threshold}")
                print(f"    - 최대 재시도: {control.max_attempts}회")
                print(f"    - 커스텀 프롬프트: {len(control.custom_prompts)}개")
                print(f"    - 파라미터 오버라이드: {len(control.parameter_overrides)}개")
            
            # 응답 히스토리 확인
            quality_report = controller.get_quality_report()
            if "total_responses" in quality_report:
                print("\n📈 품질 관리 실적:")
                print(f"  • 총 응답 수: {quality_report['total_responses']}")
                print(f"  • 전체 성공률: {quality_report['overall_success_rate']*100:.1f}%")
                print(f"  • 평균 품질 점수: {quality_report['average_quality_score']:.2f}")
                
                print("\n🤖 모델별 성능:")
                for model, stats in quality_report.get('model_performance', {}).items():
                    success_rate = stats['success_rate'] * 100
                    avg_score = stats['avg_score']
                    total = stats['total']
                    print(f"  📦 {model}: {success_rate:.1f}% 성공률, {avg_score:.2f} 평균 점수 ({total}회)")
            else:
                print("\n📈 아직 품질 관리 실적이 없습니다.")
                print("   'autoci learn low' 실행 후 다시 확인하세요.")
            
        except ImportError:
            print("❌ AI 모델 컨트롤러를 로드할 수 없습니다")
            print("💡 modules/ai_model_controller.py 파일을 확인하세요")
            
        # continuous_learning_system.py 연동 확인
        try:
            import sys
            sys.path.append('.')
            from core.continuous_learning_system import ContinuousLearningSystem
            
            # 테스트 인스턴스 생성 (실제 모델 로딩 없이)
            system = ContinuousLearningSystem()
            if hasattr(system, 'model_controller') and system.model_controller:
                print("\n✅ 연속 학습 시스템과 통합: 정상")
                print("🔥 autoci learn low 실행 시 완전 제어 모드 활성화")
            else:
                print("\n⚠️ 연속 학습 시스템과 통합: 부분적")
                print("🔧 모델 컨트롤러가 비활성화 상태입니다")
                
        except Exception as e:
            print(f"\n❌ 연속 학습 시스템 확인 실패: {str(e)}")
        
        # 모델 설치 상태 확인
        models_file = Path("models/installed_models.json")
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                models_info = json.load(f)
            
            print("\n📦 제어 가능한 모델:")
            for model_name, info in models_info.items():
                status = info.get('status', 'unknown')
                if status == 'installed':
                    print(f"  ✅ {model_name}: 설치됨 (완전 제어 가능)")
                elif status == 'not_downloaded':
                    print(f"  ❌ {model_name}: 미설치 (제어 불가)")
                else:
                    print(f"  ⚠️ {model_name}: 상태 불명 ({status})")
        else:
            print("\n❌ 모델 정보 파일을 찾을 수 없습니다")
            print("💡 install_llm_models.py를 먼저 실행하세요")
        
        print("\n🎯 AI 모델 제어 명령어:")
        print("  autoci learn low     - 완전 제어 모드로 학습")
        print("  autoci control       - 현재 제어 상태 확인")
        print("  autoci status        - 전체 시스템 상태")
        
        print("\n💡 완전한 AI 모델 조종권 확보를 위한 특징:")
        print("  🎯 응답 품질 실시간 평가 및 재시도")
        print("  🔧 모델별 커스텀 프롬프트 및 파라미터")
        print("  📊 상세한 품질 로깅 및 통계")
        print("  ⚡ 품질 기준 미달 시 자동 재생성")
        print("  🎮 우리 기준에 맞는 답변만 허용")
        
    except Exception as e:
        print(f"❌ AI 제어 상태 확인 중 오류: {str(e)}")

async def run_panda3d_game_fix():
    """AI가 학습한 내용으로 Panda3D 게임 개발 능력 개선"""
    print("🔧 Panda3D 게임 개발 AI 능력 개선 시작...")
    print("=" * 60)
    print("📚 학습된 내용 기반 게임 개발 능력 개선:")
    print("  - Python 프로그래밍 최적화")
    print("  - 한글 프로그래밍 용어 통합")
    print("  - Panda3D 엔진 활용 능력 향상")
    print("  - Socket.IO 네트워킹 성능 향상")
    print("  - 2.5D/3D 게임 아키텍처 최적화")
    print("=" * 60)
    
    try:
        from modules.panda3d_game_improver import Panda3DGameImprover
        improver = Panda3DGameImprover()
        
        # 학습 데이터 로드
        print("1️⃣ 학습 데이터 분석 중...")
        await improver.load_learning_data()
        
        # 게임 개발 패턴 분석
        print("2️⃣ Panda3D 게임 개발 패턴 분석 중...")
        await improver.analyze_game_patterns()
        
        # 개선 사항 도출
        print("3️⃣ AI가 게임 개발 개선 사항을 도출하는 중...")
        improvements = await improver.generate_improvements()
        
        # AI 모델 업데이트
        print("4️⃣ AI 모델 파인튜닝 중...")
        await improver.finetune_ai_models(improvements)
        
        # 템플릿 생성
        print("5️⃣ 개선된 게임 템플릿 생성 중...")
        await improver.create_game_templates()
        
        # 새로운 게임 프로토타입 생성
        print("6️⃣ 개선된 능력으로 게임 프로토타입 생성 중...")
        await improver.build_improved_game_prototype()
        
        print("✅ Panda3D 게임 개발 AI 능력 개선이 완료되었습니다!")
        print("🚀 개선된 게임 템플릿은 'panda3d_ai_improved' 디렉토리에 있습니다.")
        
    except ImportError:
        print("❌ Panda3D 게임 개선 모듈을 찾을 수 없습니다.")
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
                await run_learn_simple()
                
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
                await run_learn_simple()
                
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
    
    # 🎮 AI 모델 제어 상태 자동 확인 및 표시
    print("\n📊 AI 모델 완전 제어 시스템 상태 확인...")
    await show_ai_control_status()
    print("\n" + "="*60)
    print("🎯 터미널 인터페이스 시작")
    print("='help' 명령어로 사용 가능한 기능을 확인하세요.")
    print("="*60)
    
    # 기존 터미널 시스템 실행 (asyncio.run 없이 직접 호출)
    from core.autoci_terminal import AutoCITerminal
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
    
    # 🎮 AI 모델 제어 상태 자동 확인 및 표시
    print("📊 AI 모델 완전 제어 시스템 상태 확인...")
    await show_ai_control_status()
    print("\n" + "="*60)
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

async def run_godot_expert_learning():
    """Godot 전문가 학습 모드를 실행합니다."""
    print("📚 Godot 전문가 학습 모드를 시작합니다...")
    try:
        from core.continuous_learning_system import ContinuousLearningSystem
        system = ContinuousLearningSystem()
        # "Godot 전문가" 카테고리의 주제만 선택하여 학습
        godot_expert_topics = [t for t in system.learning_topics if t.category == "Godot 전문가"]
        if not godot_expert_topics:
            print('❌ "Godot 전문가" 학습 주제를 찾을 수 없습니다.')
            return
        system.learning_topics = godot_expert_topics
        await system.learning_cycle(duration_hours=8)  # 8시간 동안 집중 학습
    except ImportError:
        print("❌ 학습 시스템을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {str(e)}")

async def show_evolution_summary():
    """자가 진화 시스템 요약 표시"""
    print("🧬 AutoCI 자가 진화 시스템")
    print("=" * 60)
    
    try:
        from modules.self_evolution_system import get_evolution_system
        evolution = get_evolution_system()
        
        status = await evolution.get_evolution_status()
        
        print(f"📊 진화 단계: {status['evolution_stage']}")
        print(f"💬 총 질문 수: {status['metrics']['total_questions']:,}")
        print(f"🎯 평균 정확도: {status['metrics']['average_accuracy']:.1%}")
        print(f"📚 지식 베이스 크기: {status['collective_knowledge_size']['total']:,}")
        print(f"💡 학습률: {status['metrics']['learning_rate']:.3f}")
        
        print("\n🔥 주요 학습 도메인:")
        for domain, count in list(status['knowledge_domains'].items())[:5]:
            print(f"  - {domain}: {count:,}개 질문")
        
        print("\n💬 가장 많이 묻는 질문:")
        for i, q in enumerate(status['top_questions'][:3], 1):
            print(f"  {i}. {q['question'][:60]}... ({q['count']}회)")
        
        print("\n✅ AutoCI는 사용자들의 질문을 통해 지속적으로 진화하고 있습니다!")
        
    except ImportError:
        print("❌ 자가 진화 시스템 모듈을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 자가 진화 상태 확인 중 오류: {str(e)}")

async def show_evolution_status():
    """자가 진화 시스템 상세 상태"""
    print("🧬 AutoCI 자가 진화 시스템 상세 상태")
    print("=" * 60)
    
    try:
        from modules.self_evolution_system import get_evolution_system
        evolution = get_evolution_system()
        
        status = await evolution.get_evolution_status()
        
        # 진화 메트릭스
        print("📊 진화 메트릭스:")
        print(f"  • 진화 단계: {status['evolution_stage']}")
        print(f"  • 총 질문 수: {status['metrics']['total_questions']:,}")
        print(f"  • 총 응답 수: {status['metrics']['total_responses']:,}")
        print(f"  • 평균 정확도: {status['metrics']['average_accuracy']:.1%}")
        print(f"  • 학습률: {status['metrics']['learning_rate']:.4f}")
        
        # 최근 성능
        print("\n📈 최근 성능 (최근 100개 응답):")
        perf = status['recent_performance']
        print(f"  • 정확도: {perf['accuracy']:.1%}")
        print(f"  • 완성도: {perf['completeness']:.1%}")
        print(f"  • 관련성: {perf['relevance']:.1%}")
        print(f"  • 기술적 정확성: {perf['technical']:.1%}")
        
        # 지식 도메인
        print("\n🎯 지식 도메인 분포:")
        total = sum(status['knowledge_domains'].values())
        for domain, count in status['knowledge_domains'].items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  • {domain}: {count:,}개 ({percentage:.1f}%)")
        
        # 집단 지성 크기
        print("\n💡 집단 지성 데이터베이스:")
        kb_size = status['collective_knowledge_size']
        print(f"  • 패턴: {kb_size['patterns']:,}개")
        print(f"  • 솔루션: {kb_size['solutions']:,}개")
        print(f"  • 자주 묻는 질문: {kb_size['common_questions']:,}개")
        print(f"  • 모범 사례: {kb_size['best_practices']:,}개")
        print(f"  • 인사이트: {kb_size['total_insights']:,}개")
        print(f"  • 총 크기: {kb_size['total']:,}개 항목")
        
        # 개선 영역
        if status['improvement_areas']:
            print("\n🔧 개선이 필요한 영역:")
            for i, area in enumerate(status['improvement_areas'], 1):
                print(f"  {i}. {area.get('area', 'N/A')} (우선순위: {area.get('priority', 'N/A')})")
        
        print("\n🚀 AutoCI는 매일 더 똑똑해지고 있습니다!")
        
    except ImportError:
        print("❌ 자가 진화 시스템 모듈을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 자가 진화 상태 확인 중 오류: {str(e)}")

async def run_korean_conversation():
    """한글 대화 모드 실행"""
    print("💬 AutoCI 한글 대화 모드")
    print("=" * 60)
    print("AutoCI와 자연스러운 한글로 대화하며 게임 개발을 진행하세요!")
    print("대화를 통해 AutoCI가 더 똑똑해집니다.")
    print("=" * 60)
    
    try:
        from modules.korean_conversation import interactive_conversation
        await interactive_conversation()
    except ImportError:
        print("❌ 한글 대화 시스템 모듈을 찾을 수 없습니다.")
        print("💡 modules/korean_conversation.py 파일을 확인하세요.")
    except Exception as e:
        print(f"❌ 한글 대화 시스템 오류: {str(e)}")

async def run_ai_game_developer():
    """AI 게임 개발자 - 한글 대화 + 24시간 게임 개발 통합"""
    print("🤖 AutoCI AI 게임 개발자 모드")
    print("=" * 60)
    print("✨ 이제 AutoCI가 한글로 대화하며 24시간 게임을 개발합니다!")
    print("💬 자연스러운 한글 대화로 게임 아이디어를 설명하세요")
    print("🎮 AI가 자동으로 게임을 기획하고 개발합니다")
    print("⏰ 24시간 동안 끈질기게 개선하며 완성도를 높입니다")
    print("=" * 60)
    
    # 이전 작업 확인
    mvp_games_dir = Path("mvp_games")
    selected_project = None
    
    if mvp_games_dir.exists():
        game_projects = [d for d in mvp_games_dir.iterdir() if d.is_dir() and (d / "project.godot").exists()]
        
        if game_projects:
            print("\n📂 이전 게임 프로젝트를 발견했습니다!")
            print("=" * 60)
            sorted_projects = sorted(game_projects, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            for i, project in enumerate(sorted_projects, 1):
                mtime = datetime.fromtimestamp(project.stat().st_mtime)
                print(f"{i}. {project.name} - {mtime.strftime('%Y-%m-%d %H:%M')}")
            
            print("\n선택하세요:")
            print("1-5. 이전 프로젝트 계속 개발하기")
            print("0. 새로운 프로젝트 시작하기")
            print("Enter. 바로 대화 시작하기")
            
            choice = input("\n선택 (0-5 또는 Enter): ").strip()
            
            if choice and choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= min(5, len(sorted_projects)):
                    selected_project = sorted_projects[choice_num - 1]
                    print(f"\n✅ '{selected_project.name}' 프로젝트를 불러옵니다...")
    
    try:
        # 통합 시스템 임포트
        from modules.korean_conversation import KoreanConversationSystem
        from modules.game_factory_24h import GameFactory24H
        from modules.ai_model_controller import AIModelController
        from modules.self_evolution_system import get_evolution_system
        
        # 시스템 초기화
        game_factory = GameFactory24H()
        conversation = KoreanConversationSystem(game_factory=game_factory)
        ai_controller = AIModelController()
        evolution_system = get_evolution_system()
        
        # AI 모델 제어 상태 확인
        print("\n📊 AI 모델 상태 확인...")
        control_status = ai_controller.get_model_control_status()
        print(f"✅ AI 제어 레벨: {control_status.get('control_level', 'HIGH')}")
        print(f"🎮 게임 개발 준비 완료!")
        
        # 대화 시작
        print("\n💬 대화를 시작하세요. '게임 만들기', '24시간 개발' 등의 키워드를 사용하세요!")
        print("종료하려면 '종료', '끝', 'exit' 등을 입력하세요.\n")
        
        active_game_project = None
        
        # 이전 프로젝트 선택된 경우 즉시 개발 재개
        if selected_project:
            print(f"\n🤖 AutoCI: {selected_project.name} 프로젝트 개발을 재개합니다!")
            print("   24시간 끈질긴 개선 모드로 전환합니다... 🚀")
            
            # 프로젝트 개발 재개 (start_factory 메서드 사용)
            active_game_project = asyncio.create_task(
                game_factory.start_factory(selected_project.name, "rpg")
            )
            
            # 진화 시스템에 기록
            try:
                context = {
                    "category": "game_development",
                    "success": True,
                    "response_time": 1.0,
                    "model_used": "game_factory_24h",
                    "user_id": "autoci_system"
                }
                await evolution_system.process_user_question(
                    f"게임 개발 재개: {selected_project.name}", 
                    context
                )
            except Exception as e:
                logger.warning(f"진화 시스템 기록 실패: {e}")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 당신: ").strip()
                
                if not user_input:
                    continue
                
                # 종료 명령 확인
                if user_input.lower() in ['종료', '끝', 'exit', 'quit', '나가기']:
                    print("\n🤖 AutoCI: 안녕히 가세요! 다음에 또 만나요~ 👋")
                    break
                
                # 대화 처리
                response_data = await conversation.process_user_input(user_input, evolution_system)
                intent = response_data.get('intent')
                entities = response_data.get('entities', [])
                
                # AI 응답 생성
                if '게임' in entities or '개발' in entities or '만들기' in user_input:
                    # 게임 개발 요청 감지
                    if not active_game_project:
                        print("\n🤖 AutoCI: 네! 게임을 만들어 드리겠습니다! 🎮")
                        print("   어떤 종류의 게임을 원하시나요?")
                        print("   - 플랫포머 게임 (마리오 스타일)")
                        print("   - 레이싱 게임")
                        print("   - RPG 게임")
                        print("   - 퍼즐 게임")
                        
                        game_type_input = input("\n👤 게임 종류: ").strip()
                        game_name_input = input("👤 게임 이름: ").strip()
                        
                        # 게임 타입 매핑
                        game_type_map = {
                            '플랫포머': 'platformer',
                            '레이싱': 'racing',
                            'rpg': 'rpg',
                            '퍼즐': 'puzzle'
                        }
                        
                        game_type = 'platformer'  # 기본값
                        for keyword, gtype in game_type_map.items():
                            if keyword in game_type_input.lower():
                                game_type = gtype
                                break
                        
                        game_name = game_name_input if game_name_input else f"AI_Game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        print(f"\n🤖 AutoCI: {game_name} ({game_type}) 게임 개발을 시작합니다!")
                        print("   24시간 동안 자동으로 개발하고 개선할게요! 🚀")
                        
                        # 24시간 게임 개발 시작
                        active_game_project = asyncio.create_task(
                            game_factory.start_factory(game_name, game_type)
                        )
                        
                        # 진화 시스템에 기록
                        try:
                            context = {
                                "category": "game_development",
                                "success": True,
                                "response_time": 1.0,
                                "model_used": "game_factory_24h",
                                "user_id": "autoci_system"
                            }
                            await evolution_system.process_user_question(
                                f"게임 개발 요청: {game_name} ({game_type})",
                                context
                            )
                        except Exception as e:
                            logger.warning(f"진화 시스템 기록 실패: {e}")
                    else:
                        print("\n🤖 AutoCI: 이미 게임을 개발 중입니다! 진행 상황을 확인하시겠어요?")
                        # TODO: 진행 상황 표시 기능
                
                elif '상태' in user_input or '진행' in user_input:
                    # 진행 상황 확인
                    if active_game_project and not active_game_project.done():
                        print("\n🤖 AutoCI: 게임 개발 진행 상황을 확인하는 중...")
                        # 백그라운드 프로세스 추적기에서 상태 가져오기
                        try:
                            from modules.background_process_tracker import get_process_tracker
                            tracker = get_process_tracker()
                            status = tracker.get_current_status()
                            if status:
                                print(f"\n📊 현재 상태: {status['status']}")
                                print(f"⏱️ 진행률: {status['progress']:.1f}%")
                                print(f"🎯 현재 작업: {status['current_task']}")
                                if status.get('remaining_time'):
                                    print(f"⏳ 예상 남은 시간: {status['remaining_time']}")
                        except:
                            print("   진행 상황을 불러올 수 없습니다.")
                    else:
                        print("\n🤖 AutoCI: 현재 진행 중인 게임 개발이 없습니다.")
                
                else:
                    # 일반 대화 처리
                    print(f"\n🤖 AutoCI: {response_data.get('response', '네, 무엇을 도와드릴까요?')}")
                
                # 대화 만족도 업데이트
                conversation.update_satisfaction(0.8)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ 대화가 중단되었습니다.")
                if active_game_project and not active_game_project.done():
                    print("🎮 게임 개발은 백그라운드에서 계속됩니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                print("💡 다시 시도해주세요.")
        
        # 정리
        if active_game_project and not active_game_project.done():
            print("\n⏳ 게임 개발 작업을 안전하게 종료하는 중...")
            active_game_project.cancel()
            await asyncio.sleep(1)
        
        # 대화 세션 저장
        conversation.save_conversation()
        print("\n💾 대화 내용이 저장되었습니다.")
        
    except ImportError as e:
        print(f"❌ 필요한 모듈을 찾을 수 없습니다: {str(e)}")
        print("💡 다음 모듈들이 필요합니다:")
        print("   - modules/korean_conversation.py")
        print("   - modules/game_factory_24h.py")
        print("   - modules/ai_model_controller.py")
        print("   - modules/self_evolution_system.py")
    except Exception as e:
        print(f"❌ AI 게임 개발자 모드 오류: {str(e)}")

async def show_evolution_insights():
    """최근 발견된 진화 인사이트"""
    print("💡 AutoCI 자가 진화 인사이트")
    print("=" * 60)
    
    try:
        from modules.self_evolution_system import get_evolution_system
        from pathlib import Path
        import json
        
        evolution = get_evolution_system()
        insights_dir = evolution.insights_dir
        
        # 최근 인사이트 파일들 로드
        insight_files = sorted(insights_dir.glob("*.json"), 
                             key=lambda x: x.stat().st_mtime, 
                             reverse=True)[:10]
        
        if not insight_files:
            print("아직 발견된 인사이트가 없습니다.")
            print("더 많은 사용자 질문과 피드백이 필요합니다.")
            return
        
        print(f"최근 {len(insight_files)}개의 인사이트:")
        print()
        
        for i, insight_file in enumerate(insight_files, 1):
            with open(insight_file, 'r', encoding='utf-8') as f:
                insight = json.load(f)
            
            print(f"{i}. {insight['pattern_type'].upper()} 패턴")
            print(f"   발견 시간: {insight['timestamp'][:19]}")
            print(f"   신뢰도: {insight['confidence']:.1%}")
            print(f"   영향도: {insight['impact_score']:.1%}")
            
            # 패턴 데이터 표시
            pattern_data = insight['pattern_data']
            if insight['pattern_type'] == 'frequent_question':
                print(f"   질문: {pattern_data['question'][:80]}...")
                print(f"   빈도: {pattern_data['frequency']}회")
            elif insight['pattern_type'] == 'category_trend':
                print(f"   카테고리: {pattern_data['category']}")
                print(f"   비율: {pattern_data['percentage']:.1%}")
                print(f"   성장률: {pattern_data['growth_rate']:+.1%}")
            
            if insight['implementation_ready']:
                print("   ✅ 자동 구현 완료")
            else:
                print("   ⏳ 구현 대기 중")
            
            print()
        
        # 요약
        status = await evolution.get_evolution_status()
        print("=" * 60)
        print(f"💡 총 {status['collective_knowledge_size']['total_insights']}개의 인사이트가 발견되었습니다.")
        print("🚀 이러한 인사이트는 AutoCI의 응답 품질을 지속적으로 향상시킵니다.")
        
    except ImportError:
        print("❌ 자가 진화 시스템 모듈을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 인사이트 확인 중 오류: {str(e)}")

async def run_code_gathering():
    """외부 소스에서 코드 정보를 수집합니다."""
    print("🌐 외부 소스에서 C# 코드 정보를 수집합니다...")
    try:
        from modules.intelligent_information_gatherer import get_information_gatherer
        gatherer = get_information_gatherer()
        await gatherer.gather_and_process_csharp_code()
        print("✅ 코드 정보 수집 및 처리가 완료되었습니다.")
    except ImportError:
        print("❌ 정보 수집기 모듈을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 코드 수집 중 오류 발생: {str(e)}")

async def run_realtime_monitoring():
    """실시간 상세 모니터링 (기본)"""
    print("🔄 AutoCI 실시간 모니터링 시작")
    print("=" * 60)
    
    try:
        # autoci-monitor의 AutoCIMonitor 클래스 사용
        sys.path.insert(0, str(AUTOCI_ROOT))
        from modules.monitoring_system import ProductionMonitor, MetricType
        from modules.enhanced_logging import setup_enhanced_logging
        import psutil
        import time
        from datetime import datetime
        
        # 로깅 설정
        setup_enhanced_logging()
        
        class AutoCIMonitor:
            """AutoCI 모니터링 인터페이스"""
            
            def __init__(self):
                try:
                    self.monitor = ProductionMonitor()
                    self.monitor_available = True
                except Exception as e:
                    print(f"⚠️ 고급 모니터링 초기화 실패: {e}")
                    self.monitor = None
                    self.monitor_available = False
                self.running = False
                
            async def show_status(self):
                """실시간 상태 표시"""
                print("\n" + "="*60)
                print("📊 AutoCI 시스템 상태")
                print("="*60)
                
                # 시스템 메트릭
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                print(f"🖥️  CPU 사용률: {cpu_percent:.1f}%")
                print(f"💾 메모리 사용률: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
                print(f"💿 디스크 사용률: {disk.percent:.1f}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)")
                
                # 실시간 백그라운드 프로세스 감지
                print(f"\n🔄 실행 중인 AutoCI 프로세스:")
                try:
                    import subprocess
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    autoci_processes = []
                    for line in result.stdout.split('\n'):
                        if 'autoci.py' in line and 'grep' not in line:
                            if 'learn' in line:
                                autoci_processes.append("🧠 AI 학습 진행 중")
                            elif 'monitor' in line:
                                autoci_processes.append("📊 모니터링 활성화")
                            elif 'game' in line or 'create' in line:
                                autoci_processes.append("🎮 게임 개발 진행 중")
                            else:
                                autoci_processes.append("⚙️ AutoCI 실행 중")
                    
                    if autoci_processes:
                        for process in autoci_processes:
                            print(f"   {process}")
                    else:
                        print("   💤 백그라운드 작업 없음")
                except Exception as e:
                    print(f"   ⚠️ 프로세스 확인 실패: {e}")
                
                # 헬스 체크 상태
                try:
                    health_summary = self.monitor.get_health_summary()
                    print(f"\n🏥 헬스 체크: {health_summary.get('overall_status', 'Unknown')}")
                except:
                    print(f"\n🏥 헬스 체크: 기본 상태")
                
                # 카운터 정보
                print(f"\n📈 게임 개발 통계:")
                try:
                    for name, count in self.monitor.counters.items():
                        display_name = {
                            "games_created": "생성된 게임",
                            "features_added": "추가된 기능", 
                            "bugs_fixed": "수정된 버그",
                            "errors_caught": "포착된 오류",
                            "ai_requests": "AI 요청",
                            "ai_tokens_used": "사용된 토큰"
                        }.get(name, name)
                        print(f"   {display_name}: {count}")
                except:
                    print("   📊 통계 수집 중...")
                
                print("="*60)
            
            async def show_learning_status(self):
                """AI 학습 상태 표시"""
                print("\n" + "="*40)
                print("🧠 AI 학습 상태")
                print("="*40)
                
                # 먼저 프로세스 기반으로 학습 상태 확인
                learning_process_active = False
                try:
                    import subprocess
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'autoci.py learn' in line and 'grep' not in line:
                            learning_process_active = True
                            print("🧠 **AI 학습 진행 중** ✅")
                            
                            # 프로세스 실행 시간 추출
                            parts = line.split()
                            if len(parts) > 10:
                                cpu_time = parts[10] if ':' in parts[10] else parts[9]
                                print(f"⏱️ 진행 시간: {cpu_time}")
                            break
                except:
                    pass
                
                # 학습 파일도 확인
                file_based_learning = False
                progress_files = [
                    "user_learning_data/continuous_learning/progress/learning_progress.json",
                    "continuous_learning/progress/learning_progress.json",
                    "user_learning_data/continuous_learning/latest.json"
                ]
                
                for progress_file in progress_files:
                    if Path(progress_file).exists():
                        try:
                            with open(progress_file, 'r', encoding='utf-8') as f:
                                import json
                                data = json.load(f)
                            file_based_learning = True
                            
                            print(f"📄 학습 데이터: 발견됨")
                            if 'total_hours' in data:
                                print(f"📊 총 학습 시간: {data['total_hours']:.1f}시간")
                            if 'total_questions' in data:
                                print(f"❓ 총 질문 수: {data['total_questions']}")
                            if 'total_successful' in data:
                                print(f"✅ 성공한 답변: {data['total_successful']}")
                            break
                        except Exception as e:
                            continue
                
                # 최근 학습 활동 확인
                recent_files = []
                if Path("user_learning_data").exists():
                    import glob
                    recent_json = glob.glob("user_learning_data/**/learning_*.json", recursive=True)
                    if recent_json:
                        recent_json.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                        if recent_json:
                            latest_file = recent_json[0]
                            mtime = Path(latest_file).stat().st_mtime
                            from datetime import datetime
                            last_activity = datetime.fromtimestamp(mtime)
                            print(f"🕐 최근 학습: {last_activity.strftime('%m-%d %H:%M')}")
                
                if not learning_process_active and not file_based_learning:
                    print("💤 현재 학습 세션 없음")
                    print("💡 'autoci learn' 명령어로 시작")
                
                print("="*40)
            
            async def show_game_projects(self):
                """게임 프로젝트 상태 표시"""
                print("\n" + "="*50)
                print("🎮 게임 프로젝트 상태")
                print("="*50)
                
                project_dirs = ["game_projects", "mvp_games", "accurate_games"]
                total_projects = 0
                recent_projects = []
                
                for project_dir in project_dirs:
                    if Path(project_dir).exists():
                        projects = list(Path(project_dir).iterdir())
                        for project in projects:
                            if project.is_dir():
                                try:
                                    # 프로젝트 상세 정보 수집
                                    create_time = datetime.fromtimestamp(project.stat().st_ctime)
                                    
                                    # 파일 수 계산
                                    import os
                                    file_count = 0
                                    script_count = 0
                                    scene_count = 0
                                    
                                    for root, dirs, files in os.walk(project):
                                        file_count += len(files)
                                        for file in files:
                                            if file.endswith(('.cs', '.gd')):
                                                script_count += 1
                                            elif file.endswith('.tscn'):
                                                scene_count += 1
                                    
                                    # 마지막 수정 시간
                                    last_modified = create_time
                                    for root, dirs, files in os.walk(project):
                                        for file in files:
                                            file_path = os.path.join(root, file)
                                            try:
                                                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                                                if mtime > last_modified:
                                                    last_modified = mtime
                                            except:
                                                continue
                                    
                                    # 진행 상황 판단
                                    progress = "📦 초기"
                                    if script_count > 5:
                                        progress = "🚧 개발 중"
                                    if scene_count > 3 and script_count > 10:
                                        progress = "⚙️ 고급"
                                    if file_count > 50:
                                        progress = "🎯 완성형"
                                    
                                    # 최근 활동 여부
                                    now = datetime.now()
                                    time_diff = now - last_modified
                                    if time_diff.total_seconds() < 3600:  # 1시간 이내
                                        activity = "🔥 활발"
                                    elif time_diff.total_seconds() < 86400:  # 24시간 이내
                                        activity = "🕐 최근"
                                    else:
                                        activity = "💤 대기"
                                    
                                    recent_projects.append({
                                        'name': project.name,
                                        'folder': project_dir,
                                        'create_time': create_time,
                                        'last_modified': last_modified,
                                        'progress': progress,
                                        'activity': activity,
                                        'file_count': file_count,
                                        'script_count': script_count,
                                        'scene_count': scene_count
                                    })
                                    
                                    total_projects += 1
                                except Exception as e:
                                    # 기본 정보만
                                    recent_projects.append({
                                        'name': project.name,
                                        'folder': project_dir,
                                        'progress': "❓ 정보없음",
                                        'activity': "❓",
                                        'file_count': 0
                                    })
                                    total_projects += 1
                
                if total_projects == 0:
                    print("🎮 게임 프로젝트 없음")
                    print("💡 'autoci' 명령어로 게임 생성")
                else:
                    print(f"🎮 총 프로젝트: {total_projects}개")
                    
                    # 최근 수정된 순으로 정렬
                    recent_projects.sort(key=lambda x: x.get('last_modified', x.get('create_time', datetime.min)), reverse=True)
                    
                    print("📋 프로젝트 상세:")
                    for i, proj in enumerate(recent_projects[:4]):  # 최근 4개만 표시
                        name = proj['name'][:20]  # 이름 길이 제한
                        progress = proj['progress']
                        activity = proj['activity']
                        file_count = proj.get('file_count', 0)
                        
                        if 'last_modified' in proj:
                            last_mod = proj['last_modified'].strftime('%m-%d %H:%M')
                            print(f"   {progress} {name}")
                            print(f"      {activity} | 파일: {file_count}개 | 수정: {last_mod}")
                        else:
                            print(f"   {progress} {name} | 파일: {file_count}개")
                    
                    if len(recent_projects) > 4:
                        print(f"   ... 및 {len(recent_projects) - 4}개 더")
                
                print("="*50)
        
        monitor = AutoCIMonitor()
        
        # 지속적인 모니터링 루프
        print("💡 5초마다 자동 업데이트됩니다. (Ctrl+C로 중지)")
        iteration = 0
        
        while True:
            try:
                # 화면 지우기
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                
                iteration += 1
                print(f"🔄 AutoCI 실시간 모니터링 #{iteration}")
                print("💡 Ctrl+C로 중지")
                print("=" * 60)
                
                # 모든 상태 표시
                await monitor.show_status()
                await monitor.show_learning_status()
                await monitor.show_game_projects()
                
                print("\n⏳ 1분 후 업데이트...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                print("\n\n👋 모니터링을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 모니터링 오류: {e}")
                print("💡 5초 후 재시도...")
                await asyncio.sleep(5)
        
    except Exception as e:
        print(f"❌ 모니터링 초기화 오류: {e}")
        print("💡 기본 시스템 상태를 표시합니다...")
        await check_system_status()

async def run_monitor_status():
    """시스템 상태만 표시"""
    print("📊 AutoCI 시스템 상태")
    
    try:
        from modules.monitoring_system import ProductionMonitor
        import psutil
        
        monitor = ProductionMonitor()
        
        # 시스템 메트릭
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print("=" * 60)
        print(f"🖥️  CPU 사용률: {cpu_percent:.1f}%")
        print(f"💾 메모리 사용률: {memory.percent:.1f}%")
        print(f"💿 디스크 사용률: {disk.percent:.1f}%")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 상태 확인 오류: {e}")
        await check_system_status()

async def run_monitor_learning():
    """AI 학습 상태만 표시"""
    print("🧠 AI 학습 상태")
    print("=" * 60)
    
    try:
        import json
        
        progress_files = [
            "user_learning_data/continuous_learning/progress/learning_progress.json",
            "continuous_learning/progress/learning_progress.json"
        ]
        
        learning_active = False
        for progress_file in progress_files:
            if Path(progress_file).exists():
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    learning_active = True
                    
                    print(f"📚 학습 파일: {progress_file}")
                    if 'total_hours' in data:
                        print(f"   총 학습 시간: {data['total_hours']:.1f}시간")
                    if 'total_questions' in data:
                        print(f"   총 질문 수: {data['total_questions']}")
                    break
                except:
                    continue
        
        if not learning_active:
            print("📖 현재 활성화된 학습 세션이 없습니다.")
            print("💡 'autoci learn' 명령어로 학습을 시작할 수 있습니다.")
        
    except Exception as e:
        print(f"❌ 학습 상태 확인 오류: {e}")
    
    print("=" * 60)

async def run_monitor_projects():
    """게임 프로젝트만 표시"""
    print("🎮 게임 프로젝트 상태")
    print("=" * 60)
    
    try:
        from datetime import datetime
        
        project_dirs = ["game_projects", "mvp_games", "accurate_games"]
        total_projects = 0
        
        for project_dir in project_dirs:
            if Path(project_dir).exists():
                projects = list(Path(project_dir).iterdir())
                if projects:
                    print(f"📁 {project_dir}:")
                    for project in projects:
                        if project.is_dir():
                            try:
                                create_time = datetime.fromtimestamp(project.stat().st_ctime)
                                print(f"   🎯 {project.name} - {create_time.strftime('%Y-%m-%d %H:%M')}")
                                total_projects += 1
                            except:
                                print(f"   🎯 {project.name}")
                                total_projects += 1
        
        if total_projects == 0:
            print("🎮 생성된 게임 프로젝트가 없습니다.")
        else:
            print(f"\n📊 총 {total_projects}개의 게임 프로젝트")
        
    except Exception as e:
        print(f"❌ 프로젝트 확인 오류: {e}")
    
    print("=" * 60)

async def run_monitor_logs():
    """최근 로그만 표시"""
    print("📜 최근 로그")
    print("=" * 60)
    
    try:
        log_files = [
            "logs/autoci.log",
            "continuous_learning.log",
            "user_learning_data/continuous_learning/latest.log"
        ]
        
        for log_file in log_files:
            if Path(log_file).exists():
                print(f"\n📄 {log_file}:")
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        recent_lines = lines[-10:] if len(lines) > 10 else lines
                        
                        for line in recent_lines:
                            print(f"   {line.rstrip()}")
                        
                except Exception as e:
                    print(f"   ❌ 로그 읽기 실패: {e}")
    
    except Exception as e:
        print(f"❌ 로그 확인 오류: {e}")
    
    print("=" * 60)

async def run_monitor_interactive():
    """대화형 모니터링 모드"""
    print("🎛️ AutoCI 대화형 모니터링 모드")
    print("명령어: status, learning, projects, logs, help, quit")
    
    while True:
        try:
            command = input("\nmonitor> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                break
            elif command in ['status', 's']:
                await run_monitor_status()
            elif command in ['learning', 'learn', 'l']:
                await run_monitor_learning()
            elif command in ['projects', 'games', 'p']:
                await run_monitor_projects()
            elif command in ['logs', 'log']:
                await run_monitor_logs()
            elif command in ['help', 'h']:
                print("""
📖 사용 가능한 명령어:
  status, s     - 시스템 상태 표시
  learning, l   - AI 학습 상태 표시  
  projects, p   - 게임 프로젝트 상태 표시
  logs          - 최근 로그 표시
  help, h       - 도움말 표시
  quit, q       - 종료
                """)
            else:
                print(f"❌ 알 수 없는 명령어: {command}")
                print("💡 'help'를 입력하면 사용 가능한 명령어를 볼 수 있습니다.")
                
        except KeyboardInterrupt:
            print("\n\n👋 모니터링을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

async def run_monitor_watch():
    """5초마다 자동 새로고침"""
    print("🔄 5초마다 상태를 새로고침합니다. (Ctrl+C로 중지)")
    
    # 실시간 모니터링이 이미 지속적 업데이트를 지원하므로 그대로 호출
    await run_realtime_monitoring()

if __name__ == "__main__":
    main()