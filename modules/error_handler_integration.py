#!/usr/bin/env python3
"""
에러 핸들러와 Godot 대시보드 통합
모든 오류를 실시간으로 대시보드에 표시
"""

import logging
import traceback
from functools import wraps
from typing import Optional, Callable, Any

class DashboardErrorHandler:
    """대시보드 연동 에러 핸들러"""
    
    def __init__(self, dashboard=None):
        self.dashboard = dashboard
        self.logger = logging.getLogger("DashboardErrorHandler")
        self.error_count = 0
        
    def set_dashboard(self, dashboard):
        """대시보드 설정"""
        self.dashboard = dashboard
        
    def report_error(self, error: Exception, context: str = ""):
        """오류를 대시보드에 보고"""
        self.error_count += 1
        
        error_msg = f"{type(error).__name__}: {str(error)}"
        if context:
            error_msg = f"[{context}] {error_msg}"
            
        self.logger.error(error_msg)
        
        # 대시보드에 오류 전송
        if self.dashboard:
            self.dashboard.report_error(error_msg)
            self.dashboard.add_log(f"❌ 오류 발생: {error_msg}")
            
        # 스택 트레이스도 로그에 추가
        tb = traceback.format_exc()
        if self.dashboard and tb.strip() != "NoneType: None":
            for line in tb.split('\n'):
                if line.strip():
                    self.dashboard.add_log(f"  {line}")
                    
    def dashboard_error_handler(self, context: str = ""):
        """데코레이터: 오류를 대시보드에 자동 보고"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.report_error(e, context or func.__name__)
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.report_error(e, context or func.__name__)
                    raise
                    
            # 비동기 함수인지 확인
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator

# 전역 에러 핸들러 인스턴스
_error_handler = DashboardErrorHandler()

def get_error_handler() -> DashboardErrorHandler:
    """에러 핸들러 싱글톤 가져오기"""
    return _error_handler

def dashboard_error_handler(context: str = ""):
    """데코레이터 바로가기"""
    return _error_handler.dashboard_error_handler(context)

def report_error(error: Exception, context: str = ""):
    """오류 보고 바로가기"""
    _error_handler.report_error(error, context)