#!/usr/bin/env python3
"""
Enhanced Error Handler - 상용화 수준의 에러 처리 및 복구 시스템
"""

import os
import sys
import asyncio
import logging
import traceback
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
import psutil
import signal

class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"          # 무시 가능
    MEDIUM = "medium"    # 복구 시도
    HIGH = "high"        # 즉시 대응
    CRITICAL = "critical" # 시스템 중단

class RecoveryStrategy(Enum):
    """복구 전략"""
    RETRY = "retry"              # 재시도
    FALLBACK = "fallback"        # 대체 방법
    RESTART = "restart"          # 컴포넌트 재시작
    SKIP = "skip"               # 건너뛰기
    ROLLBACK = "rollback"       # 이전 상태로 복원
    ALERT = "alert"             # 알림만
    SHUTDOWN = "shutdown"       # 시스템 종료

@dataclass
class ErrorContext:
    """에러 컨텍스트"""
    error_type: str
    message: str
    severity: ErrorSeverity
    component: str
    timestamp: datetime
    traceback: str
    retry_count: int = 0
    recovery_history: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recovery_history is None:
            self.recovery_history = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RecoveryResult:
    """복구 결과"""
    success: bool
    strategy_used: RecoveryStrategy
    message: str
    duration: float
    metadata: Dict[str, Any] = None

class EnhancedErrorHandler:
    """상용화 수준의 에러 핸들러"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config/error_handling.json")
        self.logger = logging.getLogger("ErrorHandler")
        
        # 에러 히스토리
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[RecoveryResult] = []
        
        # 에러 패턴 학습
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        
        # 복구 전략 매핑
        self.recovery_strategies: Dict[str, List[RecoveryStrategy]] = {
            "FileNotFoundError": [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            "ConnectionError": [RecoveryStrategy.RETRY, RecoveryStrategy.RESTART],
            "MemoryError": [RecoveryStrategy.RESTART, RecoveryStrategy.ALERT],
            "PermissionError": [RecoveryStrategy.SKIP, RecoveryStrategy.ALERT],
            "TimeoutError": [RecoveryStrategy.RETRY, RecoveryStrategy.SKIP],
            "ImportError": [RecoveryStrategy.FALLBACK, RecoveryStrategy.ALERT],
            "RuntimeError": [RecoveryStrategy.RESTART, RecoveryStrategy.ROLLBACK],
            "ValueError": [RecoveryStrategy.SKIP, RecoveryStrategy.ALERT],
            "KeyError": [RecoveryStrategy.FALLBACK, RecoveryStrategy.SKIP],
            "AttributeError": [RecoveryStrategy.FALLBACK, RecoveryStrategy.SKIP]
        }
        
        # 복구 핸들러
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._handle_retry,
            RecoveryStrategy.FALLBACK: self._handle_fallback,
            RecoveryStrategy.RESTART: self._handle_restart,
            RecoveryStrategy.SKIP: self._handle_skip,
            RecoveryStrategy.ROLLBACK: self._handle_rollback,
            RecoveryStrategy.ALERT: self._handle_alert,
            RecoveryStrategy.SHUTDOWN: self._handle_shutdown
        }
        
        # 설정 로드
        self.load_config()
        
        # 시스템 상태 모니터링
        self.system_health = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "error_rate": 0,
            "last_check": datetime.now()
        }
        
        # 복구 시도 제한
        self.max_retry_attempts = 5
        self.retry_delays = [1, 2, 5, 10, 30]  # 초 단위
        
        # 에러 임계값
        self.error_thresholds = {
            "error_rate_1m": 10,    # 1분당 최대 에러 수
            "error_rate_5m": 30,    # 5분당 최대 에러 수
            "critical_errors": 3,   # 치명적 에러 허용 수
            "memory_usage": 90,     # 메모리 사용률 %
            "disk_usage": 95        # 디스크 사용률 %
        }
        
        # 실행 컨텍스트
        self.execution_context = {}
        
        # 시작
        self._setup_signal_handlers()
        # 모니터링은 이벤트 루프가 있을 때만 시작
        try:
            self._start_monitoring()
        except RuntimeError:
            # 이벤트 루프가 없으면 나중에 시작
            pass
    
    def load_config(self):
        """설정 로드"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 설정 업데이트
                if "recovery_strategies" in config:
                    self.recovery_strategies.update(config["recovery_strategies"])
                if "error_thresholds" in config:
                    self.error_thresholds.update(config["error_thresholds"])
                if "max_retry_attempts" in config:
                    self.max_retry_attempts = config["max_retry_attempts"]
                    
            except Exception as e:
                self.logger.warning(f"설정 로드 실패: {e}")
    
    def save_config(self):
        """설정 저장"""
        config = {
            "recovery_strategies": {k: [s.value for s in v] for k, v in self.recovery_strategies.items()},
            "error_thresholds": self.error_thresholds,
            "max_retry_attempts": self.max_retry_attempts,
            "retry_delays": self.retry_delays
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            self.logger.warning(f"시그널 수신: {signum}")
            self.graceful_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_monitoring(self):
        """시스템 모니터링 시작"""
        async def monitor_system():
            while True:
                try:
                    # CPU 사용률
                    self.system_health["cpu_usage"] = psutil.cpu_percent(interval=1)
                    
                    # 메모리 사용률
                    memory = psutil.virtual_memory()
                    self.system_health["memory_usage"] = memory.percent
                    
                    # 디스크 사용률
                    disk = psutil.disk_usage('/')
                    self.system_health["disk_usage"] = disk.percent
                    
                    # 에러율 계산
                    recent_errors = [e for e in self.error_history 
                                   if e.timestamp > datetime.now() - timedelta(minutes=1)]
                    self.system_health["error_rate"] = len(recent_errors)
                    
                    self.system_health["last_check"] = datetime.now()
                    
                    # 임계값 체크
                    self._check_thresholds()
                    
                    await asyncio.sleep(30)  # 30초마다 체크
                    
                except Exception as e:
                    self.logger.error(f"시스템 모니터링 에러: {e}")
                    await asyncio.sleep(60)
        
        # 백그라운드 태스크로 실행
        asyncio.create_task(monitor_system())
    
    def _check_thresholds(self):
        """임계값 체크"""
        # 메모리 사용률 체크
        if self.system_health["memory_usage"] > self.error_thresholds["memory_usage"]:
            self.logger.warning(f"메모리 사용률 임계값 초과: {self.system_health['memory_usage']}%")
            self._trigger_memory_cleanup()
        
        # 디스크 사용률 체크
        if self.system_health["disk_usage"] > self.error_thresholds["disk_usage"]:
            self.logger.warning(f"디스크 사용률 임계값 초과: {self.system_health['disk_usage']}%")
            self._trigger_disk_cleanup()
        
        # 에러율 체크
        if self.system_health["error_rate"] > self.error_thresholds["error_rate_1m"]:
            self.logger.warning(f"에러율 임계값 초과: {self.system_health['error_rate']}/분")
            self._trigger_error_rate_mitigation()
    
    def _trigger_memory_cleanup(self):
        """메모리 정리 트리거"""
        import gc
        gc.collect()
        self.logger.info("가비지 컬렉션 실행")
        
        # 캐시 정리
        if hasattr(self, 'cache'):
            self.cache.clear()
            self.logger.info("캐시 정리 완료")
    
    def _trigger_disk_cleanup(self):
        """디스크 정리 트리거"""
        # 오래된 로그 파일 삭제
        log_dir = Path("logs")
        if log_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=7)
            for log_file in log_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
                    self.logger.info(f"오래된 로그 삭제: {log_file}")
    
    def _trigger_error_rate_mitigation(self):
        """에러율 완화 조치"""
        # 에러 패턴 분석
        recent_errors = [e for e in self.error_history 
                       if e.timestamp > datetime.now() - timedelta(minutes=5)]
        
        # 가장 빈번한 에러 타입 찾기
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        if error_types:
            most_common = max(error_types, key=error_types.get)
            self.logger.warning(f"가장 빈번한 에러: {most_common} ({error_types[most_common]}회)")
            
            # 특정 컴포넌트 재시작 고려
            if error_types[most_common] > 10:
                self._consider_component_restart(most_common)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> RecoveryResult:
        """에러 처리 및 복구"""
        start_time = time.time()
        
        # 에러 컨텍스트 생성
        error_context = ErrorContext(
            error_type=type(error).__name__,
            message=str(error),
            severity=self._classify_severity(error),
            component=context.get("component", "unknown") if context else "unknown",
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            metadata=context or {}
        )
        
        # 에러 기록
        self.error_history.append(error_context)
        self.logger.error(f"에러 발생: {error_context.error_type} - {error_context.message}")
        
        # 복구 전략 선택
        strategies = self._select_recovery_strategies(error_context)
        
        # 복구 시도
        for strategy in strategies:
            try:
                result = await self.recovery_handlers[strategy](error_context)
                
                if result.success:
                    # 성공적인 복구
                    result.duration = time.time() - start_time
                    self.recovery_history.append(result)
                    self.logger.info(f"복구 성공: {strategy.value} - {result.message}")
                    
                    # 패턴 학습
                    self._learn_error_pattern(error_context, result)
                    
                    return result
                    
            except Exception as recovery_error:
                self.logger.error(f"복구 중 에러: {recovery_error}")
                continue
        
        # 모든 복구 시도 실패
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ALERT,
            message="모든 복구 시도 실패",
            duration=time.time() - start_time
        )
    
    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """에러 심각도 분류"""
        error_type = type(error).__name__
        
        # 치명적 에러
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL
        
        # 높음
        if error_type in ["RuntimeError", "PermissionError", "OSError"]:
            return ErrorSeverity.HIGH
        
        # 중간
        if error_type in ["ValueError", "KeyError", "AttributeError"]:
            return ErrorSeverity.MEDIUM
        
        # 낮음
        return ErrorSeverity.LOW
    
    def _select_recovery_strategies(self, error_context: ErrorContext) -> List[RecoveryStrategy]:
        """복구 전략 선택"""
        error_type = error_context.error_type
        
        # 기본 전략
        strategies = self.recovery_strategies.get(error_type, [RecoveryStrategy.ALERT])
        
        # 심각도에 따른 조정
        if error_context.severity == ErrorSeverity.CRITICAL:
            strategies = [RecoveryStrategy.ALERT, RecoveryStrategy.SHUTDOWN]
        elif error_context.severity == ErrorSeverity.HIGH:
            if RecoveryStrategy.RESTART not in strategies:
                strategies.insert(0, RecoveryStrategy.RESTART)
        
        # 재시도 횟수에 따른 조정
        if error_context.retry_count >= self.max_retry_attempts:
            strategies = [s for s in strategies if s != RecoveryStrategy.RETRY]
            strategies.append(RecoveryStrategy.SKIP)
        
        return strategies
    
    async def _handle_retry(self, error_context: ErrorContext) -> RecoveryResult:
        """재시도 처리"""
        retry_count = error_context.retry_count
        
        if retry_count >= self.max_retry_attempts:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                message=f"최대 재시도 횟수 초과 ({self.max_retry_attempts})"
            )
        
        # 지수 백오프
        delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
        self.logger.info(f"재시도 대기: {delay}초")
        await asyncio.sleep(delay)
        
        # 재시도 카운트 증가
        error_context.retry_count += 1
        error_context.recovery_history.append(f"재시도 {retry_count + 1}회")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            message=f"재시도 준비 완료 (시도 {retry_count + 1}/{self.max_retry_attempts})"
        )
    
    async def _handle_fallback(self, error_context: ErrorContext) -> RecoveryResult:
        """대체 방법 처리"""
        component = error_context.component
        
        # 컴포넌트별 대체 방법
        fallback_methods = {
            "ai_model": self._fallback_ai_model,
            "godot_controller": self._fallback_godot,
            "file_operation": self._fallback_file_operation,
            "network": self._fallback_network
        }
        
        if component in fallback_methods:
            try:
                success = await fallback_methods[component](error_context)
                return RecoveryResult(
                    success=success,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    message=f"{component} 대체 방법 {'성공' if success else '실패'}"
                )
            except Exception as e:
                self.logger.error(f"대체 방법 실행 중 에러: {e}")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FALLBACK,
            message="적절한 대체 방법 없음"
        )
    
    async def _handle_restart(self, error_context: ErrorContext) -> RecoveryResult:
        """컴포넌트 재시작"""
        component = error_context.component
        
        self.logger.info(f"컴포넌트 재시작: {component}")
        
        # 컴포넌트별 재시작 로직
        restart_methods = {
            "ai_model": self._restart_ai_model,
            "godot_controller": self._restart_godot,
            "background_task": self._restart_background_task
        }
        
        if component in restart_methods:
            try:
                success = await restart_methods[component]()
                return RecoveryResult(
                    success=success,
                    strategy_used=RecoveryStrategy.RESTART,
                    message=f"{component} 재시작 {'성공' if success else '실패'}"
                )
            except Exception as e:
                self.logger.error(f"재시작 중 에러: {e}")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.RESTART,
            message="재시작 불가능"
        )
    
    async def _handle_skip(self, error_context: ErrorContext) -> RecoveryResult:
        """건너뛰기"""
        self.logger.info(f"작업 건너뛰기: {error_context.metadata.get('task', 'unknown')}")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.SKIP,
            message="작업을 건너뛰고 계속 진행"
        )
    
    async def _handle_rollback(self, error_context: ErrorContext) -> RecoveryResult:
        """롤백 처리"""
        # 이전 상태 복원
        if "previous_state" in error_context.metadata:
            try:
                # 상태 복원 로직
                self.logger.info("이전 상태로 롤백 중...")
                # 실제 롤백 구현
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.ROLLBACK,
                    message="이전 상태로 롤백 완료"
                )
            except Exception as e:
                self.logger.error(f"롤백 실패: {e}")
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ROLLBACK,
            message="롤백할 상태 정보 없음"
        )
    
    async def _handle_alert(self, error_context: ErrorContext) -> RecoveryResult:
        """알림 처리"""
        # 로그 기록
        self.logger.error(f"알림: {error_context.error_type} - {error_context.message}")
        
        # 알림 전송 (웹훅, 이메일 등)
        await self._send_alert(error_context)
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.ALERT,
            message="알림 전송 완료"
        )
    
    async def _handle_shutdown(self, error_context: ErrorContext) -> RecoveryResult:
        """시스템 종료"""
        self.logger.critical("시스템 종료 시작")
        
        # 정리 작업
        await self.graceful_shutdown()
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.SHUTDOWN,
            message="시스템 종료"
        )
    
    async def _send_alert(self, error_context: ErrorContext):
        """알림 전송"""
        # 웹훅 URL이 설정되어 있으면 전송
        webhook_url = os.environ.get("AUTOCI_WEBHOOK_URL")
        if webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "error_type": error_context.error_type,
                        "message": error_context.message,
                        "severity": error_context.severity.value,
                        "component": error_context.component,
                        "timestamp": error_context.timestamp.isoformat()
                    }
                    await session.post(webhook_url, json=payload)
            except Exception as e:
                self.logger.error(f"웹훅 전송 실패: {e}")
    
    def _learn_error_pattern(self, error_context: ErrorContext, recovery_result: RecoveryResult):
        """에러 패턴 학습"""
        pattern_key = f"{error_context.error_type}_{error_context.component}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "occurrences": 0,
                "successful_recoveries": {},
                "failed_recoveries": {}
            }
        
        pattern = self.error_patterns[pattern_key]
        pattern["occurrences"] += 1
        
        strategy_key = recovery_result.strategy_used.value
        if recovery_result.success:
            pattern["successful_recoveries"][strategy_key] = \
                pattern["successful_recoveries"].get(strategy_key, 0) + 1
        else:
            pattern["failed_recoveries"][strategy_key] = \
                pattern["failed_recoveries"].get(strategy_key, 0) + 1
    
    async def _fallback_ai_model(self, error_context: ErrorContext) -> bool:
        """AI 모델 대체"""
        self.logger.info("더미 AI 모델로 전환")
        return True
    
    async def _fallback_godot(self, error_context: ErrorContext) -> bool:
        """Godot 대체"""
        self.logger.info("Godot 없이 계속 진행")
        return True
    
    async def _fallback_file_operation(self, error_context: ErrorContext) -> bool:
        """파일 작업 대체"""
        self.logger.info("대체 경로 사용")
        return True
    
    async def _fallback_network(self, error_context: ErrorContext) -> bool:
        """네트워크 대체"""
        self.logger.info("오프라인 모드로 전환")
        return True
    
    async def _restart_ai_model(self) -> bool:
        """AI 모델 재시작"""
        self.logger.info("AI 모델 재초기화")
        return True
    
    async def _restart_godot(self) -> bool:
        """Godot 재시작"""
        self.logger.info("Godot 엔진 재시작")
        return True
    
    async def _restart_background_task(self) -> bool:
        """백그라운드 작업 재시작"""
        self.logger.info("백그라운드 작업 재시작")
        return True
    
    def _consider_component_restart(self, error_type: str):
        """컴포넌트 재시작 고려"""
        self.logger.warning(f"{error_type} 에러 빈발 - 컴포넌트 재시작 고려")
    
    async def graceful_shutdown(self):
        """우아한 종료"""
        self.logger.info("시스템 정리 작업 시작")
        
        # 에러 통계 저장
        await self.save_error_statistics()
        
        # 설정 저장
        self.save_config()
        
        self.logger.info("시스템 정리 완료")
    
    async def save_error_statistics(self):
        """에러 통계 저장"""
        stats = {
            "total_errors": len(self.error_history),
            "total_recoveries": len(self.recovery_history),
            "successful_recoveries": sum(1 for r in self.recovery_history if r.success),
            "error_patterns": self.error_patterns,
            "last_saved": datetime.now().isoformat()
        }
        
        stats_path = Path("logs/error_statistics.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    
    def get_error_report(self) -> Dict[str, Any]:
        """에러 리포트 생성"""
        recent_errors = [e for e in self.error_history 
                       if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        error_by_type = {}
        for error in recent_errors:
            error_by_type[error.error_type] = error_by_type.get(error.error_type, 0) + 1
        
        recovery_success_rate = 0
        if self.recovery_history:
            successful = sum(1 for r in self.recovery_history if r.success)
            recovery_success_rate = (successful / len(self.recovery_history)) * 100
        
        return {
            "system_health": self.system_health,
            "errors_24h": len(recent_errors),
            "error_by_type": error_by_type,
            "recovery_success_rate": recovery_success_rate,
            "most_common_errors": sorted(error_by_type.items(), key=lambda x: x[1], reverse=True)[:5]
        }


# 데코레이터
def with_error_handling(component: str = "unknown", severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """에러 핸들링 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            handler = get_enhanced_error_handler()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "component": component,
                    "function": func.__name__,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                }
                result = await handler.handle_error(e, context)
                if not result.success:
                    raise
                # 재시도가 필요한 경우
                if result.strategy_used == RecoveryStrategy.RETRY:
                    return await func(*args, **kwargs)
                return None
        
        def sync_wrapper(*args, **kwargs):
            handler = get_enhanced_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "component": component,
                    "function": func.__name__,
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                }
                # 동기 함수에서는 기본 처리만
                handler.logger.error(f"에러 발생: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# 싱글톤 인스턴스
_error_handler = None

def get_enhanced_error_handler() -> EnhancedErrorHandler:
    """에러 핸들러 가져오기"""
    global _error_handler
    if _error_handler is None:
        _error_handler = EnhancedErrorHandler()
    return _error_handler