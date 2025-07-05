#!/usr/bin/env python3
"""
AutoCI Panda3D Terminal Interface
Panda3D 기반 게임 개발을 위한 터미널 인터페이스
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

# WSL 환경 확인
def is_wsl():
    """WSL 환경인지 확인"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

# WSL 환경에서는 디스플레이 설정
if is_wsl():
    os.environ.setdefault('DISPLAY', ':0')
    # GUI 자동화 비활성화
    os.environ['PYAUTOGUI_FAILSAFE'] = 'False'
    print("🐧 WSL 환경 감지됨 - GUI 자동화가 제한됩니다")

# Panda3D 관련 모듈 import
try:
    from modules.panda3d_automation_controller import Panda3DAutomationController
    from modules.game_development_pipeline import GameDevelopmentPipeline
    from modules.panda3d_continuous_learning import Panda3DContinuousLearning
    from modules.korean_conversation_interface import KoreanConversationInterface
    from modules.panda3d_self_evolution_system import Panda3DSelfEvolutionSystem
    from modules.realtime_monitoring_system import RealtimeMonitoringSystem
    from modules.ai_model_integration import get_ai_integration
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("💡 필요한 패키지를 설치하세요:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


class Panda3DTerminal:
    """Panda3D 터미널 인터페이스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_directories()
        self.setup_logging()
        
        # 컴포넌트 초기화
        self.ai_model = get_ai_integration()
        self.panda3d_controller = Panda3DAutomationController(self.ai_model)
        self.game_pipeline = GameDevelopmentPipeline()
        self.learning_system = None
        self.conversation_interface = KoreanConversationInterface()
        self.evolution_system = Panda3DSelfEvolutionSystem()
        self.monitoring_system = RealtimeMonitoringSystem(port=5555)
        
        # 모니터링 시스템에 컴포넌트 등록
        self.monitoring_system.register_component("game_pipeline", self.game_pipeline)
        self.monitoring_system.register_component("ai_system", self.ai_model)
        
        self.logger.info("🚀 AutoCI Panda3D 터미널 초기화 완료")
    
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            "game_projects",
            "mvp_games",
            "continuous_learning",
            "evolution_data",
            "conversations",
            "user_feedback",
            "logs",
            "templates"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"autoci_panda3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def print_banner(self):
        """배너 출력"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    AutoCI Panda3D Terminal                    ║
║                 AI 기반 2.5D~3D 게임 개발 시스템              ║
╠═══════════════════════════════════════════════════════════════╣
║  🎮 24시간 자동 게임 개발 시스템                              ║
║  🤖 AI가 직접 Panda3D를 조작하여 게임 제작                    ║
║  💬 자연스러운 한국어 대화로 게임 개발                        ║
║  🧬 자가 진화 시스템으로 지속적 개선                          ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_menu(self):
        """메뉴 출력"""
        menu = """
📋 주요 명령어:
  1. 🎮 게임 개발 시작 (24시간 자동 개발)
  2. 💬 한글 대화 모드 (자연어로 게임 개발)
  3. 📚 AI 학습 모드 (5가지 핵심 주제)
  4. 📊 실시간 모니터링 대시보드
  5. 🧬 진화 시스템 상태
  6. ℹ️  도움말
  0. 🚪 종료

선택: """
        return menu
    
    async def run(self):
        """메인 실행 루프"""
        self.print_banner()
        
        # 모니터링 시스템 시작
        self.monitoring_system.start()
        
        while True:
            try:
                choice = input(self.print_menu()).strip()
                
                if choice == '1':
                    await self.start_game_development()
                elif choice == '2':
                    await self.start_conversation_mode()
                elif choice == '3':
                    await self.start_learning_mode()
                elif choice == '4':
                    self.open_monitoring_dashboard()
                elif choice == '5':
                    self.show_evolution_status()
                elif choice == '6':
                    self.show_help()
                elif choice == '0':
                    break
                else:
                    print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다...")
                break
            except Exception as e:
                self.logger.error(f"오류 발생: {e}", exc_info=True)
                print(f"❌ 오류가 발생했습니다: {e}")
        
        # 정리
        self.cleanup()
    
    async def start_game_development(self):
        """24시간 게임 개발 시작"""
        print("\n🎮 24시간 자동 게임 개발 모드")
        print("=" * 50)
        
        game_types = ["platformer", "racing", "rpg", "puzzle"]
        print("게임 타입을 선택하세요:")
        for i, game_type in enumerate(game_types, 1):
            print(f"  {i}. {game_type.capitalize()}")
        
        try:
            choice = int(input("선택 (1-4): ")) - 1
            if 0 <= choice < len(game_types):
                game_type = game_types[choice]
                game_name = input("게임 이름을 입력하세요: ").strip() or f"AutoGame_{int(time.time())}"
                
                print(f"\n🚀 {game_name} ({game_type}) 개발을 시작합니다...")
                print("24시간 동안 AI가 자동으로 게임을 개발합니다.")
                print("언제든지 Ctrl+C로 중단할 수 있습니다.\n")
                
                await self.game_pipeline.start_development(game_name, game_type)
            else:
                print("❌ 잘못된 선택입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n\n🛑 게임 개발이 중단되었습니다.")
            self.game_pipeline.stop()
    
    async def start_conversation_mode(self):
        """한글 대화 모드 시작"""
        print("\n💬 한글 대화 모드를 시작합니다...")
        print("자연스러운 한국어로 게임 개발을 진행할 수 있습니다.")
        print("=" * 50)
        
        await self.conversation_interface.start_conversation()
    
    async def start_learning_mode(self):
        """AI 학습 모드 시작"""
        print("\n📚 AI 학습 모드")
        print("=" * 50)
        print("1. 빠른 학습 (1시간)")
        print("2. 표준 학습 (6시간)")
        print("3. 집중 학습 (24시간)")
        
        try:
            choice = input("선택 (1-3): ").strip()
            duration_map = {'1': 1, '2': 6, '3': 24}
            
            if choice in duration_map:
                duration = duration_map[choice]
                print(f"\n🎓 {duration}시간 학습을 시작합니다...")
                
                self.learning_system = Panda3DContinuousLearning(
                    duration_hours=duration,
                    memory_limit_gb=16.0
                )
                
                await self.learning_system.start_learning()
            else:
                print("❌ 잘못된 선택입니다.")
        except KeyboardInterrupt:
            print("\n\n🛑 학습이 중단되었습니다.")
    
    def open_monitoring_dashboard(self):
        """모니터링 대시보드 열기"""
        import webbrowser
        dashboard_url = f"http://localhost:{self.monitoring_system.port}"
        
        print(f"\n📊 모니터링 대시보드: {dashboard_url}")
        print("웹 브라우저에서 대시보드가 열립니다...")
        
        try:
            webbrowser.open(dashboard_url)
        except:
            print("💡 브라우저에서 직접 열어주세요.")
    
    def show_evolution_status(self):
        """진화 시스템 상태 표시"""
        print("\n🧬 자가 진화 시스템 상태")
        print("=" * 50)
        
        report = self.evolution_system.get_evolution_report()
        
        print(f"총 패턴 수: {report['total_patterns']}")
        print(f"평균 적합도: {report['average_fitness']:.2f}")
        print(f"진화 사이클: {report['total_evolutions']}회")
        print(f"\n주제별 통계:")
        
        for topic, stats in report['topic_statistics'].items():
            print(f"  - {topic}:")
            print(f"    패턴: {stats['patterns']}개")
            print(f"    인사이트: {stats['insights']}개")
            print(f"    베스트 프랙티스: {stats['best_practices']}개")
    
    def show_help(self):
        """도움말 표시"""
        help_text = """
ℹ️  AutoCI Panda3D 도움말
========================

🎮 게임 개발:
  - AI가 24시간 동안 자동으로 완전한 게임을 개발합니다
  - 플랫폼, 레이싱, RPG, 퍼즐 게임 지원
  - 실시간으로 개발 과정을 모니터링할 수 있습니다

💬 한글 대화:
  - "플랫폼 게임 만들어줘" 같은 자연스러운 한국어로 명령
  - "점프 기능 추가해줘", "색상 바꿔줘" 등 실시간 수정
  - 게임 개발 중에도 대화로 기능 추가 가능

📚 AI 학습:
  - Python, 한글 용어, Panda3D, Socket.IO, AI 최적화 학습
  - 학습한 내용은 지식 베이스에 저장되어 재사용
  - 난이도별 진도 관리 시스템

🧬 자가 진화:
  - 사용자 상호작용에서 패턴을 학습
  - 유전 알고리즘으로 최적의 솔루션 진화
  - 집단지성 기반 지속적 개선

📊 모니터링:
  - 실시간 시스템 리소스 모니터링
  - 게임 개발 진행 상황 시각화
  - AI 모델 성능 추적

💡 팁:
  - 메모리가 부족하면 'autoci learn low' 사용
  - 개발 중 문제가 생기면 AI가 자동으로 해결
  - 모든 활동이 자동으로 학습 데이터로 활용됨
"""
        print(help_text)
    
    def cleanup(self):
        """종료 시 정리"""
        print("\n🧹 시스템 정리 중...")
        
        # 실행 중인 프로세스 중지
        if hasattr(self.game_pipeline, 'stop'):
            self.game_pipeline.stop()
        
        if hasattr(self.monitoring_system, 'stop'):
            self.monitoring_system.stop()
        
        print("✅ 정리 완료. 안녕히 가세요! 👋")


def main():
    """메인 진입점"""
    terminal = Panda3DTerminal()
    
    try:
        # 비동기 실행
        asyncio.run(terminal.run())
    except KeyboardInterrupt:
        print("\n\n👋 AutoCI를 종료합니다.")
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()