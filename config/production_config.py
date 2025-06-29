#!/usr/bin/env python3
"""
Production Configuration Management
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml
import toml

try:
    from cryptography.fernet import Fernet
except ImportError:
    # Fallback for when cryptography is not installed
    class Fernet:
        @staticmethod
        def generate_key():
            import secrets
            return secrets.token_bytes(32)
        
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            # Simple XOR encryption for fallback
            import base64
            result = bytes(a ^ b for a, b in zip(data, self.key * (len(data) // len(self.key) + 1)))
            return base64.b64encode(result)
        
        def decrypt(self, data):
            import base64
            data = base64.b64decode(data)
            result = bytes(a ^ b for a, b in zip(data, self.key * (len(data) // len(self.key) + 1)))
            return result

@dataclass
class GodotConfig:
    """Godot 설정"""
    engine_path: str = "/usr/local/bin/godot"
    version: str = "4.2"
    headless: bool = True
    render_thread_mode: int = 1  # Single-safe
    audio_driver: str = "Dummy"
    display_driver: str = "headless"
    max_fps: int = 60
    physics_fps: int = 60
    
@dataclass
class AIModelConfig:
    """AI 모델 설정"""
    model_selection_strategy: str = "auto"  # auto, manual, fixed
    fixed_model: Optional[str] = None
    model_cache_dir: str = "./models"
    max_context_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.95
    max_retries: int = 3
    timeout: int = 300  # seconds
    quantization_enabled: bool = True
    device_map: str = "auto"
    
@dataclass
class GameDevelopmentConfig:
    """게임 개발 설정"""
    projects_root: str = "./game_projects"
    max_concurrent_projects: int = 3
    
    # 시간 설정 (초 단위)
    game_creation_interval_min: int = 7200  # 2시간
    game_creation_interval_max: int = 14400  # 4시간
    feature_addition_interval: int = 1800  # 30분
    bug_check_interval: int = 900  # 15분
    optimization_interval: int = 3600  # 1시간
    
    # 기능 설정
    auto_git_commit: bool = True
    auto_backup: bool = True
    backup_retention_days: int = 7
    
    # 게임 타입별 가중치
    game_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "platformer": 0.3,
        "racing": 0.25,
        "puzzle": 0.25,
        "rpg": 0.2
    })

@dataclass
class PerformanceConfig:
    """성능 설정"""
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 75.0
    memory_cleanup_threshold: float = 70.0
    
    # 스레드/프로세스 설정
    max_worker_threads: int = 4
    async_io_threads: int = 2
    
    # 캐싱 설정
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600

@dataclass
class LoggingConfig:
    """로깅 설정"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path: str = "./logs"
    max_log_size_mb: int = 100
    backup_count: int = 10
    
    # 로그 카테고리별 레벨
    component_log_levels: Dict[str, str] = field(default_factory=lambda: {
        "AutoCI": "INFO",
        "AIModel": "INFO",
        "GodotController": "DEBUG",
        "ErrorHandler": "WARNING"
    })

@dataclass
class SecurityConfig:
    """보안 설정"""
    enable_encryption: bool = True
    api_key_encryption: bool = True
    secure_file_operations: bool = True
    allowed_file_extensions: list = field(default_factory=lambda: [
        ".gd", ".cs", ".tscn", ".tres", ".json", ".cfg", ".ini"
    ])
    max_file_size_mb: int = 50
    sanitize_user_input: bool = True

@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    health_check_interval: int = 300  # seconds
    alert_email: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # 알림 임계값
    error_rate_threshold: float = 0.1  # 10%
    memory_alert_threshold: float = 90.0  # 90%
    disk_space_alert_threshold: float = 85.0  # 85%

@dataclass
class ProductionConfig:
    """전체 프로덕션 설정"""
    environment: str = "production"
    debug_mode: bool = False
    version: str = "1.0.0"
    
    # 하위 설정
    godot: GodotConfig = field(default_factory=GodotConfig)
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    game_development: GameDevelopmentConfig = field(default_factory=GameDevelopmentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config/settings.yaml")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("ConfigManager")
        self._encryption_key = None
        
        # 설정 로드
        self.config = self._load_config()
        
        # 환경 변수 오버라이드
        self._apply_env_overrides()
        
        # 설정 검증
        self._validate_config()
        
        # 설정 저장
        self._save_config()
    
    def _load_config(self) -> ProductionConfig:
        """설정 파일 로드"""
        config_data = {}
        
        # 여러 형식 지원
        if self.config_path.suffix == ".yaml" or self.config_path.suffix == ".yml":
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
        
        elif self.config_path.suffix == ".json":
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
        
        elif self.config_path.suffix == ".toml":
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = toml.load(f)
        
        # 기본 설정으로 시작
        config = ProductionConfig()
        
        # 로드된 데이터로 업데이트
        self._update_config(config, config_data)
        
        return config
    
    def _update_config(self, config: ProductionConfig, data: Dict[str, Any]):
        """설정 객체 업데이트"""
        if not data:
            return
            
        # 최상위 필드
        for field in ["environment", "debug_mode", "version"]:
            if field in data:
                setattr(config, field, data[field])
        
        # 하위 설정
        if "godot" in data:
            for key, value in data["godot"].items():
                if hasattr(config.godot, key):
                    setattr(config.godot, key, value)
        
        if "ai_model" in data:
            for key, value in data["ai_model"].items():
                if hasattr(config.ai_model, key):
                    setattr(config.ai_model, key, value)
        
        if "game_development" in data:
            for key, value in data["game_development"].items():
                if hasattr(config.game_development, key):
                    setattr(config.game_development, key, value)
        
        if "performance" in data:
            for key, value in data["performance"].items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        if "security" in data:
            for key, value in data["security"].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        if "monitoring" in data:
            for key, value in data["monitoring"].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
    
    def _apply_env_overrides(self):
        """환경 변수로 설정 오버라이드"""
        # AUTOCI_ 접두사를 가진 환경 변수 찾기
        for key, value in os.environ.items():
            if key.startswith("AUTOCI_"):
                self._apply_env_override(key[7:], value)  # AUTOCI_ 제거
    
    def _apply_env_override(self, key: str, value: str):
        """단일 환경 변수 적용"""
        # 키를 파싱 (예: GODOT_ENGINE_PATH -> godot.engine_path)
        parts = key.lower().split('_')
        
        if len(parts) < 2:
            return
        
        section = parts[0]
        field = '_'.join(parts[1:])
        
        # 섹션별 처리
        config_section = None
        if section == "godot":
            config_section = self.config.godot
        elif section == "ai":
            config_section = self.config.ai_model
        elif section == "game":
            config_section = self.config.game_development
        elif section == "performance":
            config_section = self.config.performance
        elif section == "logging":
            config_section = self.config.logging
        elif section == "security":
            config_section = self.config.security
        elif section == "monitoring":
            config_section = self.config.monitoring
        
        if config_section and hasattr(config_section, field):
            # 타입 변환
            current_value = getattr(config_section, field)
            if isinstance(current_value, bool):
                setattr(config_section, field, value.lower() in ['true', '1', 'yes'])
            elif isinstance(current_value, int):
                setattr(config_section, field, int(value))
            elif isinstance(current_value, float):
                setattr(config_section, field, float(value))
            else:
                setattr(config_section, field, value)
            
            self.logger.info(f"Config override: {section}.{field} = {value}")
    
    def _validate_config(self):
        """설정 검증"""
        errors = []
        
        # 경로 검증
        if not Path(self.config.godot.engine_path).exists():
            self.logger.warning(f"Godot engine not found at {self.config.godot.engine_path}")
        
        # 디렉토리 생성
        for path in [
            self.config.ai_model.model_cache_dir,
            self.config.game_development.projects_root,
            self.config.logging.log_file_path
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # 값 범위 검증
        if not 0 <= self.config.ai_model.temperature <= 2:
            errors.append("AI temperature must be between 0 and 2")
        
        if not 0 < self.config.performance.max_memory_usage_percent <= 100:
            errors.append("Max memory usage must be between 0 and 100")
        
        # 게임 타입 가중치 합 검증
        weight_sum = sum(self.config.game_development.game_type_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # 정규화
            for game_type in self.config.game_development.game_type_weights:
                self.config.game_development.game_type_weights[game_type] /= weight_sum
        
        if errors:
            raise ValueError(f"Config validation failed: {'; '.join(errors)}")
    
    def _save_config(self):
        """설정 파일 저장"""
        config_dict = self._config_to_dict(self.config)
        
        if self.config_path.suffix in [".yaml", ".yml"]:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        elif self.config_path.suffix == ".json":
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        elif self.config_path.suffix == ".toml":
            with open(self.config_path, 'w', encoding='utf-8') as f:
                toml.dump(config_dict, f)
        
        self.logger.info(f"Configuration saved to {self.config_path}")
    
    def _config_to_dict(self, config: ProductionConfig) -> Dict[str, Any]:
        """설정 객체를 딕셔너리로 변환"""
        return {
            "environment": config.environment,
            "debug_mode": config.debug_mode,
            "version": config.version,
            "godot": asdict(config.godot),
            "ai_model": asdict(config.ai_model),
            "game_development": asdict(config.game_development),
            "performance": asdict(config.performance),
            "logging": asdict(config.logging),
            "security": asdict(config.security),
            "monitoring": asdict(config.monitoring)
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        parts = key.split('.')
        if len(parts) < 2:
            return
        
        # 마지막 부분은 필드명
        field = parts[-1]
        
        # 섹션 찾기
        section = self.config
        for part in parts[:-1]:
            if hasattr(section, part):
                section = getattr(section, part)
            else:
                return
        
        if hasattr(section, field):
            setattr(section, field, value)
            self._save_config()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """민감한 데이터 암호화"""
        if not self._encryption_key:
            key_file = Path("config/.encryption_key")
            if key_file.exists():
                self._encryption_key = key_file.read_bytes()
            else:
                self._encryption_key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_bytes(self._encryption_key)
                key_file.chmod(0o600)  # 소유자만 읽기 가능
        
        f = Fernet(self._encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """민감한 데이터 복호화"""
        if not self._encryption_key:
            key_file = Path("config/.encryption_key")
            if key_file.exists():
                self._encryption_key = key_file.read_bytes()
            else:
                raise ValueError("Encryption key not found")
        
        f = Fernet(self._encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    def export_config(self, format: str = "yaml") -> str:
        """설정 내보내기"""
        config_dict = self._config_to_dict(self.config)
        
        if format == "yaml":
            return yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        elif format == "json":
            return json.dumps(config_dict, indent=2, ensure_ascii=False)
        elif format == "toml":
            return toml.dumps(config_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")


# 싱글톤 인스턴스
_config_manager = None

def get_config() -> ProductionConfig:
    """설정 가져오기"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """설정 관리자 가져오기"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager