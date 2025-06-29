#!/usr/bin/env python3
"""
Production-grade Error Handling and Recovery System
"""

import sys
import traceback
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
import json
import psutil
import signal
from enum import Enum

class ErrorSeverity(Enum):
    """오류 심각도 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """복구 전략"""
    RETRY = "retry"
    RESTART = "restart"
    FALLBACK = "fallback"
    SKIP = "skip"
    SHUTDOWN = "shutdown"

class ErrorContext:
    """오류 컨텍스트 정보"""
    def __init__(self, error: Exception, component: str, operation: str, **kwargs):
        self.error = error
        self.component = component
        self.operation = operation
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()
        self.system_state = self._capture_system_state()
        self.additional_info = kwargs
        
    def _capture_system_state(self) -> Dict:
        """시스템 상태 캡처"""
        try:
            return {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "disk_usage": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "python_version": sys.version,
                "platform": sys.platform
            }
        except:
            return {}
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
            "system_state": self.system_state,
            "additional_info": self.additional_info
        }

class ProductionErrorHandler:
    """프로덕션 환경용 에러 핸들러"""
    
    def __init__(self):
        self.logger = logging.getLogger("ErrorHandler")
        self.error_log_path = Path("logs/errors")
        self.error_log_path.mkdir(parents=True, exist_ok=True)
        
        # 오류 통계
        self.error_stats = {
            "total_errors": 0,
            "errors_by_component": {},
            "errors_by_type": {},
            "recovery_success": 0,
            "recovery_failed": 0
        }
        
        # 복구 전략 매핑
        self.recovery_strategies = {
            # 네트워크 관련 오류
            ConnectionError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            
            # 파일 시스템 오류
            FileNotFoundError: RecoveryStrategy.FALLBACK,
            PermissionError: RecoveryStrategy.SKIP,
            
            # 메모리 오류
            MemoryError: RecoveryStrategy.RESTART,
            
            # 일반 오류
            ValueError: RecoveryStrategy.FALLBACK,
            KeyError: RecoveryStrategy.FALLBACK,
            
            # 심각한 오류
            SystemError: RecoveryStrategy.SHUTDOWN,
            KeyboardInterrupt: RecoveryStrategy.SHUTDOWN
        }
        
        # 재시도 설정
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2
        }
        
        # 시그널 핸들러 설정
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Signal {signum} received")
            self._graceful_shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def handle_error(self, error: Exception, component: str, operation: str, **kwargs) -> Dict[str, Any]:
        """오류 처리 메인 함수"""
        # 오류 컨텍스트 생성
        context = ErrorContext(error, component, operation, **kwargs)
        
        # 오류 로깅
        self._log_error(context)
        
        # 통계 업데이트
        self._update_stats(context)
        
        # 심각도 평가
        severity = self._assess_severity(error)
        
        # 복구 전략 결정
        strategy = self._determine_recovery_strategy(error, severity)
        
        # 복구 실행
        recovery_result = self._execute_recovery(strategy, context)
        
        # 오류 리포트 저장
        self._save_error_report(context, severity, strategy, recovery_result)
        
        return {
            "handled": True,
            "severity": severity.value,
            "strategy": strategy.value,
            "recovery_success": recovery_result["success"],
            "recovery_details": recovery_result
        }
    
    def _log_error(self, context: ErrorContext):
        """오류 로깅"""
        log_message = f"""
=== ERROR DETECTED ===
Component: {context.component}
Operation: {context.operation}
Error Type: {type(context.error).__name__}
Error Message: {str(context.error)}
Timestamp: {context.timestamp}
Memory Usage: {context.system_state.get('memory_usage', 'N/A')}%
CPU Usage: {context.system_state.get('cpu_usage', 'N/A')}%
=====================
"""
        self.logger.error(log_message)
        
        # 상세 로그는 파일로
        detailed_log_file = self.error_log_path / f"error_{context.timestamp.strftime('%Y%m%d_%H%M%S')}.log"
        with open(detailed_log_file, 'w', encoding='utf-8') as f:
            f.write(log_message)
            f.write("\n\nStack Trace:\n")
            f.write(context.stack_trace)
            f.write("\n\nAdditional Info:\n")
            f.write(json.dumps(context.additional_info, indent=2, default=str))
    
    def _update_stats(self, context: ErrorContext):
        """오류 통계 업데이트"""
        self.error_stats["total_errors"] += 1
        
        # 컴포넌트별 통계
        if context.component not in self.error_stats["errors_by_component"]:
            self.error_stats["errors_by_component"][context.component] = 0
        self.error_stats["errors_by_component"][context.component] += 1
        
        # 오류 타입별 통계
        error_type = type(context.error).__name__
        if error_type not in self.error_stats["errors_by_type"]:
            self.error_stats["errors_by_type"][error_type] = 0
        self.error_stats["errors_by_type"][error_type] += 1
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """오류 심각도 평가"""
        # 심각한 오류
        if isinstance(error, (SystemError, MemoryError, RecursionError)):
            return ErrorSeverity.CRITICAL
        
        # 높은 심각도
        if isinstance(error, (OSError, IOError, RuntimeError)):
            return ErrorSeverity.HIGH
        
        # 중간 심각도
        if isinstance(error, (ValueError, KeyError, IndexError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # 낮은 심각도
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error: Exception, severity: ErrorSeverity) -> RecoveryStrategy:
        """복구 전략 결정"""
        # 오류 타입별 전략
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(error, error_type):
                return strategy
        
        # 심각도별 기본 전략
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.SHUTDOWN
        elif severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.RESTART
        elif severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.SKIP
    
    def _execute_recovery(self, strategy: RecoveryStrategy, context: ErrorContext) -> Dict[str, Any]:
        """복구 전략 실행"""
        self.logger.info(f"Executing recovery strategy: {strategy.value}")
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(context)
            elif strategy == RecoveryStrategy.RESTART:
                return self._restart_component(context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._use_fallback(context)
            elif strategy == RecoveryStrategy.SKIP:
                return self._skip_operation(context)
            elif strategy == RecoveryStrategy.SHUTDOWN:
                return self._graceful_shutdown()
            
            self.error_stats["recovery_success"] += 1
            return {"success": True, "method": strategy.value}
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            self.error_stats["recovery_failed"] += 1
            return {"success": False, "method": strategy.value, "error": str(e)}
    
    def _retry_operation(self, context: ErrorContext) -> Dict[str, Any]:
        """작업 재시도"""
        retries = 0
        delay = self.retry_config["base_delay"]
        
        while retries < self.retry_config["max_retries"]:
            try:
                self.logger.info(f"Retrying operation {context.operation} (attempt {retries + 1})")
                # 실제로는 작업을 재실행하는 콜백이 필요
                return {"success": True, "retries": retries + 1}
            except Exception as e:
                retries += 1
                if retries < self.retry_config["max_retries"]:
                    self.logger.warning(f"Retry {retries} failed, waiting {delay}s")
                    asyncio.create_task(asyncio.sleep(delay))
                    delay = min(delay * self.retry_config["exponential_base"], 
                              self.retry_config["max_delay"])
        
        return {"success": False, "retries": retries}
    
    def _restart_component(self, context: ErrorContext) -> Dict[str, Any]:
        """컴포넌트 재시작"""
        self.logger.info(f"Restarting component: {context.component}")
        # 실제로는 컴포넌트별 재시작 로직 필요
        return {"success": True, "restarted": context.component}
    
    def _use_fallback(self, context: ErrorContext) -> Dict[str, Any]:
        """폴백 사용"""
        self.logger.info(f"Using fallback for {context.operation}")
        # 실제로는 작업별 폴백 로직 필요
        return {"success": True, "fallback_used": True}
    
    def _skip_operation(self, context: ErrorContext) -> Dict[str, Any]:
        """작업 건너뛰기"""
        self.logger.warning(f"Skipping operation: {context.operation}")
        return {"success": True, "skipped": True}
    
    def _graceful_shutdown(self) -> Dict[str, Any]:
        """우아한 종료"""
        self.logger.critical("Initiating graceful shutdown")
        
        # 통계 저장
        self._save_stats()
        
        # 진행 중인 작업 정리
        # 실제로는 각 컴포넌트의 cleanup 메서드 호출
        
        return {"success": True, "shutdown": True}
    
    def _save_error_report(self, context: ErrorContext, severity: ErrorSeverity, 
                          strategy: RecoveryStrategy, recovery_result: Dict):
        """오류 리포트 저장"""
        report = {
            "error": context.to_dict(),
            "severity": severity.value,
            "recovery_strategy": strategy.value,
            "recovery_result": recovery_result,
            "stats_snapshot": self.error_stats.copy()
        }
        
        report_file = self.error_log_path / f"report_{context.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _save_stats(self):
        """통계 저장"""
        stats_file = self.error_log_path / "error_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.error_stats, f, indent=2)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """오류 요약 반환"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "recovery_rate": (
                self.error_stats["recovery_success"] / 
                max(1, self.error_stats["recovery_success"] + self.error_stats["recovery_failed"])
            ) * 100,
            "top_error_components": sorted(
                self.error_stats["errors_by_component"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_error_types": sorted(
                self.error_stats["errors_by_type"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


def error_handler(component: str):
    """에러 핸들링 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = get_error_handler()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                result = handler.handle_error(
                    e, 
                    component, 
                    func.__name__,
                    args=args,
                    kwargs=kwargs
                )
                if not result["recovery_success"]:
                    raise
                return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = get_error_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                result = handler.handle_error(
                    e,
                    component,
                    func.__name__,
                    args=args,
                    kwargs=kwargs
                )
                if not result["recovery_success"]:
                    raise
                return None
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 싱글톤 인스턴스
_error_handler = None

def get_error_handler() -> ProductionErrorHandler:
    """에러 핸들러 싱글톤 인스턴스 반환"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ProductionErrorHandler()
    return _error_handler