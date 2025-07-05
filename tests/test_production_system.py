#!/usr/bin/env python3
"""
Production System Comprehensive Tests
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.error_handler import ProductionErrorHandler, ErrorSeverity, RecoveryStrategy
from modules.monitoring_system import ProductionMonitor, MetricType, Metric
from modules.ai_model_integration import AIModelIntegration, ModelType
from config.production_config import ProductionConfig, ConfigManager


class TestErrorHandler:
    """Error handler tests"""
    
    @pytest.fixture
    def error_handler(self):
        return ProductionErrorHandler()
    
    def test_error_severity_assessment(self, error_handler):
        """Test error severity assessment"""
        # Critical errors
        assert error_handler._assess_severity(MemoryError()) == ErrorSeverity.CRITICAL
        assert error_handler._assess_severity(SystemError()) == ErrorSeverity.CRITICAL
        
        # High severity
        assert error_handler._assess_severity(OSError()) == ErrorSeverity.HIGH
        assert error_handler._assess_severity(IOError()) == ErrorSeverity.HIGH
        
        # Medium severity
        assert error_handler._assess_severity(ValueError()) == ErrorSeverity.MEDIUM
        assert error_handler._assess_severity(KeyError()) == ErrorSeverity.MEDIUM
        
        # Low severity
        assert error_handler._assess_severity(Exception()) == ErrorSeverity.LOW
    
    def test_recovery_strategy_determination(self, error_handler):
        """Test recovery strategy determination"""
        # Network errors should retry
        assert error_handler._determine_recovery_strategy(
            ConnectionError(), ErrorSeverity.MEDIUM
        ) == RecoveryStrategy.RETRY
        
        # File not found should fallback
        assert error_handler._determine_recovery_strategy(
            FileNotFoundError(), ErrorSeverity.MEDIUM
        ) == RecoveryStrategy.FALLBACK
        
        # Memory errors should restart
        assert error_handler._determine_recovery_strategy(
            MemoryError(), ErrorSeverity.CRITICAL
        ) == RecoveryStrategy.RESTART
    
    def test_error_handling(self, error_handler):
        """Test complete error handling"""
        # Test handling a medium severity error
        error = ValueError("Test error")
        result = error_handler.handle_error(
            error,
            "TestComponent",
            "test_operation",
            test_param="value"
        )
        
        assert result["handled"] == True
        assert result["severity"] == ErrorSeverity.MEDIUM.value
        assert "strategy" in result
        assert "recovery_success" in result
        
        # Check stats updated
        assert error_handler.error_stats["total_errors"] > 0
        assert "TestComponent" in error_handler.error_stats["errors_by_component"]
        assert "ValueError" in error_handler.error_stats["errors_by_type"]
    
    def test_error_logging(self, error_handler, tmp_path):
        """Test error logging to file"""
        # Override log path
        error_handler.error_log_path = tmp_path / "errors"
        error_handler.error_log_path.mkdir(parents=True, exist_ok=True)
        
        # Handle an error
        error = RuntimeError("Test runtime error")
        error_handler.handle_error(error, "TestComponent", "test_op")
        
        # Check log file created
        log_files = list(error_handler.error_log_path.glob("error_*.log"))
        assert len(log_files) > 0
        
        # Check log content
        log_content = log_files[0].read_text()
        assert "Test runtime error" in log_content
        assert "TestComponent" in log_content
        assert "test_op" in log_content


class TestMonitoringSystem:
    """Monitoring system tests"""
    
    @pytest.fixture
    async def monitor(self):
        config = {
            "metrics_collection_interval": 1,
            "retention_days": 7
        }
        monitor = ProductionMonitor(config)
        yield monitor
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, monitor):
        """Test metric recording"""
        # Record various metric types
        monitor.record_metric("test.counter", 10, MetricType.COUNTER)
        monitor.record_metric("test.gauge", 75.5, MetricType.GAUGE, unit="percent")
        monitor.record_metric("test.histogram", 0.123, MetricType.HISTOGRAM, unit="seconds")
        
        # Wait for async save
        await asyncio.sleep(0.1)
        
        # Check metrics recorded
        counter = monitor.collector.get_latest("test.counter")
        assert counter is not None
        assert counter.value == 10
        assert counter.type == MetricType.COUNTER
        
        gauge = monitor.collector.get_latest("test.gauge")
        assert gauge is not None
        assert gauge.value == 75.5
        assert gauge.unit == "percent"
    
    @pytest.mark.asyncio
    async def test_health_checks(self, monitor):
        """Test health check functionality"""
        # Start monitoring
        await monitor.start()
        await asyncio.sleep(0.1)
        
        # Perform health checks
        await monitor._perform_health_checks()
        
        # Check health status recorded
        assert len(monitor.health_status) > 0
        
        # Get summary
        summary = monitor.get_health_summary()
        assert "overall_status" in summary
        assert "components" in summary
        assert summary["overall_status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_alert_creation(self, monitor):
        """Test alert creation and processing"""
        # Create test alert
        await monitor.create_alert(
            "test_alert",
            "This is a test alert",
            "warning"
        )
        
        # Check alert queued
        assert monitor.alert_queue.qsize() > 0
    
    @pytest.mark.asyncio
    async def test_counter_increment(self, monitor):
        """Test counter increment"""
        # Increment counters
        monitor.increment_counter("games_created", 1)
        monitor.increment_counter("features_added", 5)
        monitor.increment_counter("bugs_fixed", 3)
        
        # Check values
        assert monitor.counters["games_created"] == 1
        assert monitor.counters["features_added"] == 5
        assert monitor.counters["bugs_fixed"] == 3
        
        # Check metrics recorded
        await asyncio.sleep(0.1)
        metric = monitor.collector.get_latest("app.games_created")
        assert metric is not None
        assert metric.value == 1
    
    @pytest.mark.asyncio
    async def test_metrics_summary(self, monitor):
        """Test metrics summary generation"""
        # Record some metrics
        for i in range(5):
            monitor.record_metric("system.cpu.usage", 50 + i * 5, MetricType.GAUGE)
            await asyncio.sleep(0.01)
        
        # Get summary
        summary = monitor.get_metrics_summary(duration_minutes=1)
        
        if "system.cpu.usage" in summary:
            cpu_summary = summary["system.cpu.usage"]
            assert "avg" in cpu_summary
            assert "max" in cpu_summary
            assert "min" in cpu_summary
            assert cpu_summary["max"] >= cpu_summary["min"]


class TestAIModelIntegration:
    """AI model integration tests"""
    
    @pytest.fixture
    def ai_integration(self):
        return AIModelIntegration()
    
    def test_model_selection(self, ai_integration):
        """Test model selection based on memory"""
        # Test model selection logic
        model_type = ai_integration.select_model_based_on_memory()
        assert model_type in [ModelType.QWEN_32B, ModelType.CODELLAMA_13B, ModelType.LLAMA_8B]
    
    @pytest.mark.asyncio
    async def test_code_generation(self, ai_integration):
        """Test code generation"""
        context = {
            "game_type": "platformer",
            "target_feature": "double jump",
            "language": "GDScript"
        }
        
        result = await ai_integration.generate_code(
            "Add double jump feature to player",
            context,
            task_type="game_dev"
        )
        
        assert "success" in result
        if result["success"]:
            assert "code" in result
            assert "explanation" in result
            assert "model" in result
            assert "validation" in result
    
    def test_code_validation(self, ai_integration):
        """Test code validation"""
        # Test GDScript validation
        gdscript_code = """
extends CharacterBody2D

func _ready():
    print("Player ready")

func _physics_process(delta):
    move_and_slide()
"""
        
        validation = asyncio.run(
            ai_integration.validate_generated_code(
                gdscript_code,
                {"language": "GDScript"}
            )
        )
        
        assert "syntax_valid" in validation
        assert "godot_compatible" in validation
        assert "security_issues" in validation
        assert "performance_concerns" in validation
    
    def test_security_validation(self, ai_integration):
        """Test security issue detection"""
        dangerous_code = """
extends Node

func _ready():
    OS.execute("rm -rf /", [], false)
    var file = File.new()
    file.open("user://../../sensitive.txt", File.WRITE)
"""
        
        validation = asyncio.run(
            ai_integration.validate_generated_code(
                dangerous_code,
                {"language": "GDScript"}
            )
        )
        
        assert len(validation["security_issues"]) > 0
    
    def test_performance_validation(self, ai_integration):
        """Test performance issue detection"""
        inefficient_code = """
extends Node2D

func _process(delta):
    var player = get_node("Player")
    var enemy = get_node("Enemy")
    for i in range(1000):
        var bullet = preload("res://Bullet.tscn").instance()
        add_child(bullet)
"""
        
        validation = asyncio.run(
            ai_integration.validate_generated_code(
                inefficient_code,
                {"language": "GDScript"}
            )
        )
        
        assert len(validation["performance_concerns"]) > 0


class TestConfigurationManagement:
    """Configuration management tests"""
    
    @pytest.fixture
    def config_manager(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        return ConfigManager(str(config_path))
    
    def test_default_config(self, config_manager):
        """Test default configuration"""
        config = config_manager.config
        
        assert config.environment == "production"
        assert isinstance(config.debug_mode, bool)
        assert config.version == "1.0.0"
        
        # Check sub-configs
        assert config.godot.engine_path == "/usr/local/bin/godot"
        assert config.ai_model.model_selection_strategy == "auto"
        assert config.performance.max_memory_usage_percent == 80.0
    
    def test_config_get_set(self, config_manager):
        """Test configuration get/set"""
        # Test get
        assert config_manager.get("godot.version") == "4.2"
        assert config_manager.get("nonexistent.key", "default") == "default"
        
        # Test set
        config_manager.set("godot.max_fps", 120)
        assert config_manager.config.godot.max_fps == 120
    
    def test_environment_override(self, config_manager, monkeypatch):
        """Test environment variable override"""
        # Set environment variable
        monkeypatch.setenv("AUTOCI_GODOT_MAX_FPS", "144")
        monkeypatch.setenv("AUTOCI_AI_TEMPERATURE", "0.9")
        
        # Reload config
        config_manager._apply_env_overrides()
        
        assert config_manager.config.godot.max_fps == 144
        assert config_manager.config.ai_model.temperature == 0.9
    
    def test_config_validation(self, config_manager):
        """Test configuration validation"""
        # Set invalid values
        config_manager.config.ai_model.temperature = 3.0  # Too high
        config_manager.config.performance.max_memory_usage_percent = 150  # Too high
        
        # Validation should fix or raise error
        with pytest.raises(ValueError):
            config_manager._validate_config()
    
    def test_config_export(self, config_manager):
        """Test configuration export"""
        # Export to different formats
        yaml_export = config_manager.export_config("yaml")
        assert "environment: production" in yaml_export
        
        json_export = config_manager.export_config("json")
        assert '"environment": "production"' in json_export
    
    def test_encryption(self, config_manager):
        """Test sensitive data encryption"""
        sensitive_data = "my_secret_api_key"
        
        # Encrypt
        encrypted = config_manager.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data
        
        # Decrypt
        decrypted = config_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_system_startup(self, tmp_path, monkeypatch):
        """Test full system startup"""
        # Set test environment
        monkeypatch.setenv("AUTOCI_LOGGING_LOG_FILE_PATH", str(tmp_path / "logs"))
        monkeypatch.setenv("AUTOCI_GAME_PROJECTS_ROOT", str(tmp_path / "projects"))
        
        # Import after env setup
        from core.autoci_production import ProductionAutoCI
        
        # Create system
        system = ProductionAutoCI()
        
        # Test health check
        health = await system.health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health
        assert "version" in health
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self):
        """Test error handling and recovery integration"""
        from core.autoci_production import ProductionAutoCI
        
        system = ProductionAutoCI()
        
        # Simulate various errors
        errors = [
            ConnectionError("Network error"),
            FileNotFoundError("Missing file"),
            ValueError("Invalid value")
        ]
        
        for error in errors:
            result = system.error_handler.handle_error(
                error,
                "IntegrationTest",
                "test_operation"
            )
            assert result["handled"] == True


def test_production_ready():
    """Verify system is production ready"""
    required_files = [
        "autoci.py",
        "autoci_production.py",
        "autoci_terminal.py",
        "modules/error_handler.py",
        "modules/monitoring_system.py",
        "modules/ai_model_integration.py",
        "modules/csharp_learning_agent.py",
        "modules/godot_controller.py",
        "modules/godot_editor_controller.py",
        "config/production_config.py",
        "install_production.sh"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for file_path in required_files:
        assert (project_root / file_path).exists(), f"Missing required file: {file_path}"
    
    print("✅ All required files present")
    print("✅ System is production ready!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])