"""
AutoCI Panda3D 통합 테스트
AI 자동 게임 개발 시스템의 통합 테스트
"""

import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.panda3d_ai_agent import Panda3DAIAgent, GameState, ActionSpace
from modules.ai_model_integration import AIModelIntegration
from modules.autoci_panda3d_integration import AutoCIPanda3DSystem


class TestPanda3DIntegration(unittest.TestCase):
    """Panda3D 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.system = AutoCIPanda3DSystem()
        self.test_project_name = "TestGame"
        self.test_game_type = "platformer"
    
    def tearDown(self):
        """테스트 정리"""
        # 테스트 프로젝트 정리
        project_path = Path(f"game_projects/{self.test_project_name}")
        if project_path.exists():
            import shutil
            shutil.rmtree(project_path)
    
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        self.assertIsNotNone(self.system.ai_integration)
        self.assertIsNotNone(self.system.socketio_system)
        self.assertEqual(len(self.system.agents), 0)
    
    def test_supported_game_types(self):
        """지원 게임 타입 테스트"""
        game_types = self.system.get_supported_game_types()
        self.assertIn("platformer", game_types)
        self.assertIn("racing", game_types)
        self.assertIn("rpg", game_types)
        self.assertGreaterEqual(len(game_types), 5)
    
    @patch('modules.panda3d_automation_controller.Panda3DAutomationController.start_panda3d_project')
    async def test_create_game_basic(self, mock_start):
        """기본 게임 생성 테스트"""
        mock_start.return_value = True
        
        # 짧은 시간으로 테스트
        result = await self.system.create_game(
            project_name=self.test_project_name,
            game_type=self.test_game_type,
            development_hours=0.001  # 매우 짧은 시간
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["project_name"], self.test_project_name)
        self.assertEqual(result["game_type"], self.test_game_type)
    
    def test_invalid_game_type(self):
        """잘못된 게임 타입 테스트"""
        with self.assertRaises(ValueError):
            asyncio.run(self.system.create_game(
                project_name="InvalidGame",
                game_type="invalid_type",
                development_hours=0.001
            ))


class TestPanda3DAIAgent(unittest.TestCase):
    """Panda3D AI 에이전트 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.agent = Panda3DAIAgent("TestAgent", "platformer")
    
    def test_agent_initialization(self):
        """에이전트 초기화 테스트"""
        self.assertEqual(self.agent.project_name, "TestAgent")
        self.assertEqual(self.agent.game_type, "platformer")
        self.assertIsInstance(self.agent.game_state, GameState)
        self.assertEqual(self.agent.game_state.current_phase, "initialization")
    
    def test_quality_score_calculation(self):
        """품질 점수 계산 테스트"""
        # 초기 점수
        self.agent._update_quality_score()
        initial_score = self.agent.game_state.quality_score
        self.assertEqual(initial_score, 0.0)
        
        # 기능 추가 후 점수
        self.agent.game_state.features.append("player")
        self.agent.game_state.features.append("movement")
        self.agent._update_quality_score()
        
        self.assertGreater(self.agent.game_state.quality_score, initial_score)
    
    def test_phase_progression(self):
        """개발 단계 진행 테스트"""
        # 초기 단계
        self.assertEqual(self.agent.game_state.current_phase, "initialization")
        
        # 필요 기능 추가
        self.agent.game_state.features.extend(["player", "level"])
        
        # 단계 완료 확인
        self.assertTrue(self.agent._check_phase_completion())
        
        # 다음 단계로 진행
        self.agent._advance_phase()
        self.assertEqual(self.agent.game_state.current_phase, "core_mechanics")
    
    @patch('modules.ai_model_integration.AIModelIntegration.generate_code')
    async def test_code_generation(self, mock_generate):
        """코드 생성 테스트"""
        # Mock 설정
        mock_generate.return_value = {
            "success": True,
            "code": "# Test generated code\nprint('Hello, Panda3D!')",
            "validation": {"syntax_valid": True}
        }
        
        # 코드 생성
        code = await self.agent._generate_code_for_action(ActionSpace.CREATE_PLAYER)
        
        self.assertIsNotNone(code)
        self.assertIn("print", code)
    
    def test_action_evaluation(self):
        """액션 평가 테스트"""
        # 성공적인 액션
        reward = self.agent._evaluate_action(ActionSpace.CREATE_PLAYER, success=True)
        self.assertGreater(reward, 0)
        
        # 실패한 액션
        penalty = self.agent._evaluate_action(ActionSpace.CREATE_PLAYER, success=False)
        self.assertLess(penalty, 0)
        
        # 중요 액션은 더 높은 보상
        normal_reward = self.agent._evaluate_action(ActionSpace.ADD_SOUND_EFFECT, success=True)
        critical_reward = self.agent._evaluate_action(ActionSpace.CREATE_PLAYER, success=True)
        self.assertGreater(critical_reward, normal_reward)


class TestGameStateManagement(unittest.TestCase):
    """게임 상태 관리 테스트"""
    
    def test_game_state_initialization(self):
        """게임 상태 초기화 테스트"""
        state = GameState(
            project_name="TestProject",
            game_type="rpg"
        )
        
        self.assertEqual(state.project_name, "TestProject")
        self.assertEqual(state.game_type, "rpg")
        self.assertEqual(state.quality_score, 0.0)
        self.assertEqual(state.completeness, 0.0)
        self.assertEqual(len(state.features), 0)
        self.assertEqual(state.current_phase, "initialization")
    
    def test_game_state_updates(self):
        """게임 상태 업데이트 테스트"""
        state = GameState("Test", "platformer")
        
        # 기능 추가
        state.features.append("player")
        state.features.append("movement")
        
        # 버그 추가
        state.bugs.append("collision_detection_error")
        
        # 메트릭 업데이트
        state.performance_metrics["fps"] = 45.5
        state.performance_metrics["memory_usage"] = 256.0
        
        self.assertEqual(len(state.features), 2)
        self.assertEqual(len(state.bugs), 1)
        self.assertEqual(state.performance_metrics["fps"], 45.5)


class TestIntegrationScenarios(unittest.TestCase):
    """통합 시나리오 테스트"""
    
    @patch('modules.panda3d_automation_controller.Panda3DAutomationController')
    @patch('modules.socketio_realtime_system.SocketIORealtimeSystem')
    async def test_complete_game_development_flow(self, mock_socketio, mock_panda):
        """완전한 게임 개발 플로우 테스트"""
        # Mock 설정
        mock_panda_instance = MagicMock()
        mock_panda.return_value = mock_panda_instance
        mock_panda_instance.start_panda3d_project.return_value = True
        mock_panda_instance.get_project_path.return_value = "game_projects/TestFlow"
        
        # 시스템 초기화
        system = AutoCIPanda3DSystem()
        
        # 게임 생성 (매우 짧은 시간)
        result = await system.create_game(
            project_name="TestFlowGame",
            game_type="platformer",
            development_hours=0.0001
        )
        
        # 결과 검증
        self.assertIn("TestFlowGame", system.agents)
        agent = system.agents["TestFlowGame"]
        self.assertIsNotNone(agent)
    
    def test_memory_and_learning(self):
        """메모리 및 학습 시스템 테스트"""
        agent = Panda3DAIAgent("MemoryTest", "puzzle")
        
        # 성공 패턴 추가
        agent.memory.successful_patterns.append({
            "action": "create_player",
            "state": "initialization",
            "code_snippet": "player = Player()",
            "timestamp": 123456
        })
        
        # 실패 시도 추가
        agent.memory.failed_attempts.append({
            "action": "add_physics",
            "error": "Physics engine not initialized",
            "timestamp": 123457
        })
        
        # 학습된 솔루션 추가
        agent.memory.learned_solutions["Physics engine not initialized"] = """
        # Initialize physics engine first
        self.physics_world = BulletWorld()
        """
        
        self.assertEqual(len(agent.memory.successful_patterns), 1)
        self.assertEqual(len(agent.memory.failed_attempts), 1)
        self.assertIn("Physics engine not initialized", agent.memory.learned_solutions)


def run_async_test(coro):
    """비동기 테스트 실행 헬퍼"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    # 테스트 실행
    unittest.main(verbosity=2)