#!/usr/bin/env python3
"""
AutoCI Main System - Panda3D 24시간 자동 게임 개발 AI
AI가 직접 Panda3D를 조작하여 완전한 2.5D~3D 게임을 제작하는 시스템
"""

import os
import sys
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 필수 모듈 임포트
from modules.ai_model_integration import get_ai_integration
from modules.panda3d_automation_controller import Panda3DAutomationController
from modules.panda3d_self_evolution_system import Panda3DSelfEvolutionSystem
from modules.korean_conversation_interface import KoreanConversationInterface
from modules.game_development_pipeline import GameDevelopmentPipeline
from modules.realtime_monitoring_system import RealtimeMonitoringSystem
from modules.enterprise_ai_model_system import EnterpriseAIModelSystem
from modules.game_session_manager import GameSessionManager, GameSession

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs_current' / f'autoci_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class AutoCIPanda3DMain:
    """AutoCI 메인 시스템 - Panda3D 24시간 자동 게임 개발"""
    
    def __init__(self):
        """시스템 초기화"""
        self.is_running = False
        self.current_game = None
        self.components = {}
        self.current_session = None
        self.session_manager = GameSessionManager()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        logger.info("🚀 AutoCI Panda3D 시스템 초기화 완료")
    
    def _initialize_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            # AI 모델 시스템
            self.components['ai_model'] = get_ai_integration()
            self.components['enterprise_ai'] = EnterpriseAIModelSystem()
            
            # Panda3D 자동화
            self.components['panda3d_controller'] = Panda3DAutomationController(
                self.components['ai_model']
            )
            
            # 자가 진화 시스템
            self.components['evolution_system'] = Panda3DSelfEvolutionSystem()
            
            # 한국어 대화 인터페이스
            self.components['korean_interface'] = KoreanConversationInterface()
            
            # 게임 개발 파이프라인
            self.components['game_pipeline'] = GameDevelopmentPipeline()
            
            # 실시간 모니터링
            self.components['monitoring'] = RealtimeMonitoringSystem()
            
            # 컴포넌트 간 연결
            self._connect_components()
            
            # 세션 매니저 등록
            self.components['session_manager'] = self.session_manager
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def _connect_components(self):
        """컴포넌트 간 연결 설정"""
        # 모니터링 시스템에 컴포넌트 등록
        self.components['monitoring'].register_component(
            'game_pipeline', self.components['game_pipeline']
        )
        self.components['monitoring'].register_component(
            'ai_system', self.components['ai_model']
        )
        self.components['monitoring'].register_component(
            'learning_system', self.components['evolution_system']
        )
    
    async def start(self):
        """AutoCI 시스템 시작"""
        self.is_running = True
        logger.info("🎮 AutoCI Panda3D 시스템 시작!")
        
        # 모니터링 시작
        self.components['monitoring'].start()
        
        # AI 모델 초기화
        await self._initialize_ai_models()
        
        # 메인 루프 시작
        await self._main_loop()
    
    async def _initialize_ai_models(self):
        """AI 모델 초기화 및 로드"""
        logger.info("🧠 AI 모델 로드 중...")
        
        # AI 모델 초기화
        try:
            self.components['ai_model'].initialize_model()
            logger.info("✅ AI 모델 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ AI 모델 초기화 실패: {e}, 기본 모드로 실행")
    
    async def _main_loop(self):
        """메인 실행 루프"""
        while self.is_running:
            try:
                # 사용자 입력 대기
                user_input = await self._get_user_input()
                
                if user_input.lower() in ['exit', 'quit', '종료']:
                    break
                
                # 명령 처리
                await self._process_command(user_input)
                
            except KeyboardInterrupt:
                logger.info("사용자 중단")
                break
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(1)
        
        await self.stop()
    
    async def _get_user_input(self) -> str:
        """사용자 입력 받기 (비동기)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, "\n🤖 AutoCI> ")
    
    async def _process_command(self, command: str):
        """명령 처리"""
        command = command.strip().lower()
        
        if command.startswith('create'):
            # 게임 생성
            parts = command.split()
            if len(parts) >= 3 and parts[-1] == 'game':
                game_type = parts[1]
                await self._create_game(game_type)
            else:
                print("사용법: create [type] game (예: create platformer game)")
        
        elif command.startswith('add'):
            # 기능 추가
            if self.current_game:
                feature = command.replace('add feature', '').strip()
                await self._add_feature(feature)
            else:
                print("먼저 게임을 생성해주세요.")
        
        elif command == 'status':
            # 상태 확인
            self._show_status()
        
        elif command == 'help':
            # 도움말
            self._show_help()
        
        elif command == 'open_panda3d':
            # Panda3D 에디터 열기
            self._open_panda3d_editor()
        
        else:
            # 한국어 대화 처리
            response = await self.components['korean_interface'].process_input(command)
            print(f"\n{response}")
    
    async def _create_game(self, game_type: str):
        """게임 생성 시작"""
        game_name = f"{game_type}_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 세션 생성
        if self.current_session is None:
            self.current_session = self.session_manager.create_session(game_type, game_name)
        
        print(f"\n🎮 '{game_name}' 생성 시작...")
        print("📊 24시간 자동 개발이 시작됩니다!")
        
        # 게임 개발 파이프라인 시작
        success = await self.components['game_pipeline'].start_development(
            game_name, game_type
        )
        
        if success:
            self.current_game = game_name
            print(f"✅ 게임 개발이 시작되었습니다!")
            print(f"📂 프로젝트 위치: game_projects/{game_name}")
            
            # 세션 업데이트
            if self.current_session:
                self.session_manager.update_progress(self.current_session.session_id, {
                    'stage': 'development_started',
                    'current_task': 'initial_setup'
                })
        else:
            print("❌ 게임 생성 실패")
    
    async def create_game(self, game_type: str):
        """게임 생성 (외부 호출용)"""
        await self._create_game(game_type)
    
    async def resume_development(self, session: GameSession):
        """기존 게임 개발 재개"""
        self.current_session = session
        self.current_game = session.game_name
        
        print(f"\n🔄 '{session.game_name}' 개발 재개...")
        print(f"📊 현재 진행률: {session.progress.get('completion_percentage', 0)}%")
        
        # 게임 파이프라인 복원
        if hasattr(self.components['game_pipeline'], 'resume_development'):
            success = await self.components['game_pipeline'].resume_development(session)
        else:
            # 기본 개발 계속
            success = await self.components['game_pipeline'].start_development(
                session.game_name, session.game_type
            )
        
        if success:
            print(f"✅ 게임 개발이 재개되었습니다!")
            
            # 세션 상태 업데이트
            self.session_manager.resume_session(session.session_id)
        else:
            print("❌ 게임 개발 재개 실패")
    
    async def _add_feature(self, feature: str):
        """기능 추가"""
        print(f"\n➕ '{feature}' 기능 추가 중...")
        
        # 개발 중인 게임에 기능 추가
        if hasattr(self.components['game_pipeline'], 'add_feature'):
            success = await self.components['game_pipeline'].add_feature(feature)
            
            if success:
                print(f"✅ '{feature}' 기능이 추가되었습니다!")
                
                # 세션에 기능 추가
                if self.current_session:
                    self.session_manager.add_feature(self.current_session.session_id, feature)
            else:
                print(f"❌ '{feature}' 기능 추가 실패")
    
    def _show_status(self):
        """시스템 상태 표시"""
        print("\n📊 AutoCI 시스템 상태")
        print("=" * 50)
        
        # AI 모델 상태
        ai_status = "✅ 활성" if self.components['ai_model'].is_model_loaded() else "❌ 비활성"
        print(f"AI 모델: {ai_status}")
        
        # 현재 게임
        if self.current_game:
            print(f"현재 게임: {self.current_game}")
            
            # 개발 진행 상태
            if hasattr(self.components['game_pipeline'], 'get_status'):
                status = self.components['game_pipeline'].get_status()
                if status:
                    print(f"진행률: {status.get('progress', 0)}%")
                    print(f"현재 단계: {status.get('current_phase', 'N/A')}")
            
            # 세션 정보
            if self.current_session:
                print(f"\n💾 세션 정보:")
                print(f"   ID: {self.current_session.session_id}")
                print(f"   상태: {self.current_session.status}")
                print(f"   기능 수: {len(self.current_session.features)}")
        else:
            print("현재 게임: 없음")
        
        # 모니터링 URL
        print(f"\n🖥️ 모니터링 대시보드: http://localhost:5000")
    
    def _show_help(self):
        """도움말 표시"""
        print("\n📚 AutoCI 명령어 도움말")
        print("=" * 50)
        print("create [type] game    - 게임 생성 (platformer, racing, rpg, puzzle)")
        print("add feature [name]    - 기능 추가")
        print("modify [aspect]       - 게임 수정")
        print("open_panda3d         - Panda3D 에디터 열기")
        print("status               - 시스템 상태")
        print("help                 - 도움말")
        print("exit/quit/종료       - 종료")
        print("\n💬 한국어로 자유롭게 대화할 수 있습니다!")
    
    def _open_panda3d_editor(self):
        """Panda3D 에디터/뷰어 열기"""
        print("\n🎨 Panda3D 에디터를 여는 중...")
        
        # Panda3D 자동화 컨트롤러를 통해 에디터 실행
        if hasattr(self.components['panda3d_controller'], 'open_editor'):
            self.components['panda3d_controller'].open_editor()
            print("✅ Panda3D 에디터가 열렸습니다!")
        else:
            print("⚠️ Panda3D 에디터 기능이 아직 구현되지 않았습니다.")
    
    async def stop(self):
        """시스템 종료"""
        logger.info("🛑 AutoCI 시스템 종료 중...")
        self.is_running = False
        
        # 현재 세션 일시 정지
        if self.current_session and self.current_session.status == 'active':
            self.session_manager.pause_session(self.current_session.session_id)
            print(f"🟡 현재 세션이 일시 정지되었습니다: {self.current_session.session_id}")
        
        # 모든 컴포넌트 정리
        if 'monitoring' in self.components:
            self.components['monitoring'].stop()
        
        if 'game_pipeline' in self.components:
            self.components['game_pipeline'].stop()
        
        logger.info("✅ AutoCI 시스템 종료 완료")


async def main():
    """메인 실행 함수"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      🎮 AutoCI - AI 게임 개발 시스템 v5.0 🎮         ║
    ║                                                       ║
    ║   AI가 직접 Panda3D를 조작하여 게임을 만듭니다!      ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # AutoCI 시스템 생성 및 시작
    autoci = AutoCIPanda3DMain()
    await autoci.start()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())