#!/usr/bin/env python3
"""
AutoCI 메인 명령어 처리
PyTorch + 변형된 Godot + C# 기반 24시간 자동 게임 개발 시스템
"""

import sys
import os
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Xlib 경고 억제
from core.xlib_suppressor import suppress_all_xlib_warnings
suppress_all_xlib_warnings()

# PyTorch 및 변형된 Godot 모듈 임포트
try:
    from modules.game_development_pipeline import GameDevelopmentPipeline
    from modules.pytorch_deep_learning_module import AutoCIPyTorchLearningSystem, PyTorchGameAI
    from modules.korean_conversation_interface import KoreanConversationInterface
    from modules.self_evolution_system import SelfEvolutionSystem
    from modules.realtime_monitoring_system import RealtimeMonitoringSystem
    from modules.ai_model_integration import get_ai_integration
    from modules.godot_automation_controller import GodotAutomationController
    from modules.socketio_realtime_system import SocketIORealtimeSystem
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("💡 requirements.txt의 패키지들이 설치되었는지 확인하세요.")
    sys.exit(1)


async def run_interactive_menu():
    """대화형 메뉴 실행 (autoci 명령어만 입력했을 때)"""
    # Since we're now using Godot instead of Panda3D, let's use the Korean conversation interface
    from modules.korean_conversation_interface import KoreanConversationInterface
    terminal = KoreanConversationInterface()
    await terminal.start_conversation()


async def run_create_game(game_type: str = None, project_name: str = None):
    """게임 타입을 지정하여 24시간 자동 개발 시작"""
    pipeline = GameDevelopmentPipeline()
    
    # 게임 타입이 없으면 대화형으로 선택
    if not game_type:
        print("\n🎮 AutoCI 게임 생성 마법사")
        print("=" * 50)
        print("\n어떤 종류의 게임을 만들고 싶으신가요?\n")
        
        game_types = {
            "1": ("platformer", "플랫폼 게임", "마리오 같은 점프 액션 게임"),
            "2": ("rpg", "RPG 게임", "스토리와 캐릭터 성장이 있는 롤플레잉 게임"),
            "3": ("puzzle", "퍼즐 게임", "테트리스나 매치3 같은 두뇌 게임"),
            "4": ("shooter", "슈팅 게임", "총알을 쏘며 적을 물리치는 게임"),
            "5": ("racing", "레이싱 게임", "자동차나 비행기로 경주하는 게임"),
            "6": ("strategy", "전략 게임", "자원을 관리하고 전략을 짜는 게임"),
            "7": ("adventure", "어드벤처 게임", "탐험과 모험이 있는 스토리 게임"),
            "8": ("simulation", "시뮬레이션 게임", "현실을 모방한 경영/생활 게임")
        }
        
        for key, (type_id, name, desc) in game_types.items():
            print(f"  {key}. {name:<15} - {desc}")
        
        while True:
            choice = input("\n선택하세요 (1-8): ").strip()
            if choice in game_types:
                game_type = game_types[choice][0]
                selected_name = game_types[choice][1]
                break
            else:
                print("❌ 잘못된 선택입니다. 1-8 중에서 선택해주세요.")
    
    # 프로젝트 이름이 미리 제공된 경우
    if project_name:
        game_name = project_name
        print(f"\n✨ 프로젝트 이름: {game_name}")
    else:
        # 게임 이름 입력받기
        print(f"\n✨ {selected_name if 'selected_name' in locals() else game_type} 게임을 만들겠습니다!")
        print("\n게임 이름을 정해주세요.")
        print("(Enter를 누르면 자동으로 이름이 생성됩니다)")
        
        game_name = input("\n게임 이름: ").strip()
        
        if not game_name:
            # 자동 이름 생성
            game_name = f"Auto{game_type.capitalize()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"📝 자동 생성된 이름: {game_name}")
    
    print(f"\n🎮 '{game_name}' 개발을 시작합니다!")
    print("📅 24시간 동안 AI가 자동으로 게임을 개발합니다.")
    print("⏸️  언제든지 Ctrl+C로 중단할 수 있습니다.\n")
    
    await pipeline.start_development(game_name, game_type)
    
    # 개발이 진행되는 동안 대기
    try:
        while pipeline.is_running:
            await asyncio.sleep(1)  # 1초마다 확인
    except KeyboardInterrupt:
        print("\n\n🛑 개발 중단 요청...")
        pipeline.stop()
        await asyncio.sleep(2)  # 정리 시간
    
    print("\n✅ 게임 개발이 완료되었습니다!")


async def run_chat_mode():
    """한글 대화 모드로 게임 개발"""
    interface = KoreanConversationInterface()
    await interface.start_conversation()


async def run_continuous_learning():
    """AI 모델 기반 연속 학습 (기본 24시간)"""
    print("🎓 AI 연속 학습을 시작합니다...")
    print("7가지 핵심 주제: C#, 한글 용어, 변형된 Godot, Socket.IO, AI 최적화, Godot 전문가, Godot 조작")
    
    from core_system.continuous_learning_system import ContinuousLearningSystem
    
    # 연속 학습 시스템 초기화
    learning_system = ContinuousLearningSystem()
    
    # 24시간 연속 학습 루프 실행
    await learning_system.continuous_learning_loop(duration_hours=24)


async def run_continuous_learning_low():
    """메모리 최적화 연속 학습 (8GB GPU)"""
    print("🎓 메모리 최적화 학습 모드")
    print("RTX 2080 8GB, 32GB RAM 환경에 최적화됨")
    
    # continuous_learning_system.py 실행
    import subprocess
    from pathlib import Path
    script_path = Path(project_root) / "core_system" / "continuous_learning_system.py"
    
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path), "24", "16.0"])
    else:
        # PyTorch 학습 시스템 사용 (메모리 최적화)
        import torch
        torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리 80%만 사용
        
        learning_system = AutoCIPyTorchLearningSystem(project_root)
        print("⚠️ 메모리 최적화 모드로 실행 중...")
        
        # 배치 크기를 줄여서 메모리 사용량 감소
        from modules.self_evolution_system import SelfEvolutionSystem
        evolution = SelfEvolutionSystem()
        experiences = await evolution.collect_experiences()
        
        if experiences:
            # 작은 배치로 나누어 학습
            batch_size = 8  # 메모리 최적화를 위한 작은 배치
            for i in range(0, len(experiences), batch_size):
                batch = experiences[i:i+batch_size]
                learning_system.train_on_experience(batch, epochs=5)
            print(f"✅ 메모리 최적화 학습 완료!")


async def run_fix():
    """AutoCI 지능형 가디언 시스템 - 24시간 지속적 감시/학습/조언 시스템"""
    print("🛡️ AutoCI 지능형 가디언 시스템 시작")
    print("=" * 70)
    print("🎯 목표: autoci learn과 autoci create의 바보같은 단순 학습을 방지하고")
    print("       부족한 부분을 자동으로 메꿔주는 가뭄의 단비 같은 핵심 시스템")
    print("=" * 70)
    
    # 지능형 가디언 시스템 초기화
    from modules.intelligent_guardian_system import get_guardian_system
    
    guardian = get_guardian_system()
    
    try:
        # 24시간 가디언 모드 시작
        print("\n🚀 24시간 지속적 감시 모드 활성화...")
        print("   - autoci learn 프로세스 실시간 감시")
        print("   - autoci create 결과물 분석 및 최적화")
        print("   - 반복적 학습 패턴 감지 및 차단")
        print("   - 지식 격차 자동 감지 및 보완")
        print("   - PyTorch 딥러닝 자동 실행")
        print("   - 24시간 지속적 정보 검색")
        print("   - 인간 맞춤형 조언 생성")
        
        print("\n⚡ 가디언 시스템 핵심 기능:")
        print("   🔍 감시: learn/create 프로세스 24시간 모니터링")
        print("   🚫 지양: 반복적/비효율적 학습 패턴 차단")
        print("   🔎 검색: 부족한 정보 자동 검색 및 보완")
        print("   🧠 딥러닝: PyTorch로 학습 데이터 자동 최적화")
        print("   💡 조언: 인간에게 다음 단계 맞춤형 조언")
        
        print("\n🎮 이제 단순한 반복 학습 대신 지능적 학습이 시작됩니다!")
        print("=" * 70)
        
        # 가디언 모드 실행 (무한 루프)
        await guardian.start_guardian_mode()
        
    except KeyboardInterrupt:
        print("\n\n🛑 사용자가 가디언 시스템 중단을 요청했습니다...")
        await guardian.stop_guardian_mode()
        
        print("\n✅ AutoCI 지능형 가디언 시스템이 안전하게 종료되었습니다.")
        print("💭 언제든지 'autoci fix'를 다시 실행하여 가디언을 재활성화할 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 가디언 시스템 오류: {e}")
        print("🔧 시스템을 재시작하거나 로그를 확인해주세요.")
        
        # 오류 상황에서도 안전하게 종료
        try:
            await guardian.stop_guardian_mode()
        except:
            pass


async def run_monitor():
    """실시간 모니터링 대시보드"""
    monitoring = RealtimeMonitoringSystem(port=5555)
    monitoring.start()
    
    print("\n📊 실시간 모니터링 대시보드")
    print(f"🌐 브라우저에서 열기: http://localhost:5555")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        # 계속 실행
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        monitoring.stop()


async def run_evolve_insights():
    """진화 인사이트 조회"""
    print("\n🧬 AI 진화 인사이트")
    print("=" * 50)
    
    evolution = SelfEvolutionSystem()
    insights = await evolution.get_insights()
    
    if not insights:
        print("아직 수집된 인사이트가 없습니다.")
        print("💡 'autoci learn'으로 학습을 진행하면 인사이트가 생성됩니다.")
    else:
        print(f"\n발견된 인사이트: {len(insights)}개\n")
        
        for i, insight in enumerate(insights[:10], 1):  # 상위 10개만
            print(f"{i}. {insight.get('title', '제목 없음')}")
            print(f"   {insight.get('description', '설명 없음')}")
            print(f"   신뢰도: {insight.get('confidence', 0):.1%}")
            print(f"   적용 횟수: {insight.get('usage_count', 0)}회")
            print()
    
    # PyTorch AI 통계
    pytorch_ai = PyTorchGameAI()
    print("\n🔥 PyTorch AI 통계:")
    print(f"   모델 크기: {sum(p.numel() for p in pytorch_ai.model.parameters()) / 1e6:.1f}M 파라미터")
    print(f"   디바이스: {pytorch_ai.device}")
    

async def run_status():
    """시스템 상태 확인"""
    print("📊 AutoCI 시스템 상태")
    print("=" * 50)
    
    # AI 모델 상태
    ai_model = get_ai_integration()
    print(f"\n🤖 AI 모델:")
    print(f"   로드된 모델: {ai_model.current_model or 'None'}")
    print(f"   사용 가능: {'Yes' if ai_model.is_model_loaded() else 'No'}")
    
    # 진화 시스템 상태
    evolution = SelfEvolutionSystem()
    report = evolution.get_evolution_report()
    print(f"\n🧬 진화 시스템:")
    print(f"   패턴 수: {report.get('total_patterns', 0)}")
    print(f"   평균 적합도: {report.get('average_fitness', 0):.2f}")
    
    # PyTorch 상태
    import torch
    print(f"\n🔥 PyTorch 상태:")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 프로젝트 상태
    game_projects = Path(project_root) / "game_projects"
    if game_projects.exists():
        projects = list(game_projects.iterdir())
        print(f"\n🎮 게임 프로젝트:")
        print(f"   총 프로젝트: {len(projects)}개")
        if projects:
            latest = max(projects, key=lambda p: p.stat().st_mtime)
            print(f"   최근 프로젝트: {latest.name}")
    
    print("\n✅ 시스템 정상 작동 중")


def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        description="AutoCI - PyTorch 기반 24시간 AI 게임 개발 시스템 (변형된 Godot + C# + Socket.IO)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 위치 인자 (명령어)
    parser.add_argument("command", nargs="?", help="실행할 명령어")
    parser.add_argument("args", nargs="*", help="추가 인자들")
    
    args = parser.parse_args()
    
    try:
        # 명령어 처리
        if args.command is None:
            # autoci만 입력 -> 대화형 메뉴
            asyncio.run(run_interactive_menu())
            
        elif args.command == "create":
            # autoci create [name] [game_type]
            project_name = None
            game_type = None
            
            if len(args.args) >= 1:
                # 첫 번째 인자 확인
                first_arg = args.args[0]
                valid_types = ["platformer", "racing", "rpg", "puzzle", "shooter", "strategy", "adventure", "simulation"]
                
                if first_arg in valid_types:
                    # 첫 번째 인자가 게임 타입인 경우
                    game_type = first_arg
                else:
                    # 첫 번째 인자가 프로젝트 이름인 경우
                    project_name = first_arg
                    
                    # 두 번째 인자가 있으면 게임 타입으로 확인
                    if len(args.args) >= 2:
                        if args.args[1] in valid_types:
                            game_type = args.args[1]
                        else:
                            print(f"❌ 지원하지 않는 게임 타입: {args.args[1]}")
                            print(f"💡 지원 타입: {', '.join(valid_types)}")
                            sys.exit(1)
            
            # 프로젝트 이름과 게임 타입 출력
            if project_name or game_type:
                print(f"\n🎮 AutoCI 게임 자동 생성")
                if project_name:
                    print(f"   프로젝트: {project_name}")
                if game_type:
                    print(f"   타입: {game_type}")
                print()
            
            # game_type이 None이면 대화형 모드로 실행
            asyncio.run(run_create_game(game_type, project_name))
            
        elif args.command == "chat":
            # autoci chat
            asyncio.run(run_chat_mode())
            
        elif args.command == "learn":
            # autoci learn 또는 autoci learn low
            if len(args.args) > 0 and args.args[0] == "low":
                asyncio.run(run_continuous_learning_low())
            else:
                asyncio.run(run_continuous_learning())
                
        elif args.command == "fix":
            # autoci fix
            asyncio.run(run_fix())
            
        elif args.command == "monitor":
            # autoci monitor
            asyncio.run(run_monitor())
            
        elif args.command == "status":
            # autoci status
            asyncio.run(run_status())
            
        elif args.command == "evolve":
            # autoci evolve insights
            if args.subcommand == "insights":
                asyncio.run(run_evolve_insights())
            else:
                print("💡 사용법: autoci evolve insights")
                
        elif args.command == "help":
            # autoci help
            parser.print_help()
            
        else:
            print(f"❌ 알 수 없는 명령어: {args.command}")
            print("💡 'autoci help'로 사용 가능한 명령어를 확인하세요.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 AutoCI를 종료합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()