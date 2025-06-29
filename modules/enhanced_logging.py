#!/usr/bin/env python3
"""
Enhanced Logging System - 상용화 수준의 로깅 시스템
"""

import os
import sys
import logging
import logging.handlers
import json
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading
import queue
import traceback

class JsonFormatter(logging.Formatter):
    """JSON 포맷터"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.extra_fields = kwargs.get('extra_fields', {})
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
            'process_name': record.processName
        }
        
        # 추가 필드
        log_data.update(self.extra_fields)
        
        # 예외 정보
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # 추가 속성
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """압축 지원 로테이팅 파일 핸들러"""
    
    def doRollover(self):
        """로테이션 시 압축"""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # 현재 로그 파일 백업
        dfn = self.rotation_filename(self.baseFilename + ".1")
        if os.path.exists(dfn):
            os.remove(dfn)
        
        # 기존 백업 파일들 이동
        for i in range(self.backupCount - 1, 0, -1):
            sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
            dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
            if os.path.exists(sfn):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(sfn, dfn)
        
        # 현재 파일을 .1로 이동
        dfn = self.rotation_filename(self.baseFilename + ".1")
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
            
            # 압축
            with open(dfn, 'rb') as f_in:
                with gzip.open(dfn + '.gz', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(dfn)
        
        # 새 로그 파일 생성
        self.stream = self._open()

class AsyncLogHandler(logging.Handler):
    """비동기 로그 핸들러"""
    
    def __init__(self, handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """워커 스레드"""
        while True:
            try:
                record = self.queue.get()
                if record is None:  # 종료 시그널
                    break
                self.handler.emit(record)
            except Exception:
                pass
    
    def emit(self, record: logging.LogRecord):
        """로그 레코드 큐에 추가"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # 큐가 가득 찬 경우 가장 오래된 것 제거
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(record)
            except:
                pass
    
    def close(self):
        """핸들러 종료"""
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.handler.close()

class LogFilter:
    """로그 필터"""
    
    def __init__(self, exclude_patterns: List[str] = None, include_patterns: List[str] = None):
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or []
    
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        
        # 포함 패턴 체크
        if self.include_patterns:
            if not any(pattern in message for pattern in self.include_patterns):
                return False
        
        # 제외 패턴 체크
        if self.exclude_patterns:
            if any(pattern in message for pattern in self.exclude_patterns):
                return False
        
        return True

class EnhancedLogger:
    """향상된 로거 설정"""
    
    def __init__(self, name: str = "AutoCI", config_path: Optional[str] = None):
        self.name = name
        self.config_path = Path(config_path or "config/logging.json")
        self.config = self._load_config()
        
        # 로그 디렉토리 생성
        self.log_dir = Path(self.config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self._setup_logger()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        default_config = {
            "log_dir": "logs",
            "log_level": "INFO",
            "max_bytes": 10 * 1024 * 1024,  # 10MB
            "backup_count": 10,
            "enable_json": True,
            "enable_compression": True,
            "enable_async": True,
            "enable_syslog": False,
            "syslog_host": "localhost",
            "syslog_port": 514,
            "components": {
                "AutoCI": "INFO",
                "GodotController": "DEBUG",
                "AIModel": "INFO",
                "ErrorHandler": "WARNING",
                "Monitor": "INFO"
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_config.update(config)
            except Exception as e:
                print(f"로깅 설정 로드 실패: {e}")
        
        return default_config
    
    def _setup_logger(self):
        """로거 설정"""
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 파일 핸들러 설정
        self._setup_file_handlers(root_logger)
        
        # 콘솔 핸들러 설정
        self._setup_console_handler(root_logger)
        
        # Syslog 핸들러 설정
        if self.config.get("enable_syslog"):
            self._setup_syslog_handler(root_logger)
        
        # 컴포넌트별 로그 레벨 설정
        for component, level in self.config.get("components", {}).items():
            component_logger = logging.getLogger(component)
            component_logger.setLevel(getattr(logging, level))
    
    def _setup_file_handlers(self, logger: logging.Logger):
        """파일 핸들러 설정"""
        # 일반 로그 파일
        log_file = self.log_dir / f"{self.name.lower()}.log"
        
        if self.config.get("enable_compression"):
            file_handler = CompressedRotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 10)
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 10)
            )
        
        # 포맷 설정
        if self.config.get("enable_json"):
            file_handler.setFormatter(JsonFormatter(
                extra_fields={"service": self.name}
            ))
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        file_handler.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # 비동기 처리
        if self.config.get("enable_async"):
            file_handler = AsyncLogHandler(file_handler)
        
        logger.addHandler(file_handler)
        
        # 에러 전용 로그 파일
        error_file = self.log_dir / f"{self.name.lower()}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
            backupCount=self.config.get("backup_count", 5)
        )
        
        if self.config.get("enable_json"):
            error_handler.setFormatter(JsonFormatter(
                extra_fields={"service": self.name, "type": "error"}
            ))
        else:
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'
            ))
        
        error_handler.setLevel(logging.ERROR)
        
        if self.config.get("enable_async"):
            error_handler = AsyncLogHandler(error_handler)
        
        logger.addHandler(error_handler)
    
    def _setup_console_handler(self, logger: logging.Logger):
        """콘솔 핸들러 설정"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        # 컬러 포맷터
        try:
            import colorlog
            console_handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'
                }
            ))
        except ImportError:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        console_handler.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        logger.addHandler(console_handler)
    
    def _setup_syslog_handler(self, logger: logging.Logger):
        """Syslog 핸들러 설정"""
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address=(
                    self.config.get("syslog_host", "localhost"),
                    self.config.get("syslog_port", 514)
                )
            )
            
            syslog_handler.setFormatter(logging.Formatter(
                f'{self.name}: %(levelname)s %(message)s'
            ))
            
            syslog_handler.setLevel(logging.WARNING)
            logger.addHandler(syslog_handler)
        except Exception as e:
            print(f"Syslog 핸들러 설정 실패: {e}")
    
    def cleanup_old_logs(self, days: int = 30):
        """오래된 로그 파일 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.is_file():
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    try:
                        log_file.unlink()
                        logging.info(f"오래된 로그 파일 삭제: {log_file}")
                    except Exception as e:
                        logging.error(f"로그 파일 삭제 실패: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """로그 통계"""
        stats = {
            "total_size": 0,
            "file_count": 0,
            "oldest_log": None,
            "newest_log": None,
            "by_level": {}
        }
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.is_file():
                stats["total_size"] += log_file.stat().st_size
                stats["file_count"] += 1
                
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if stats["oldest_log"] is None or file_time < stats["oldest_log"]:
                    stats["oldest_log"] = file_time
                if stats["newest_log"] is None or file_time > stats["newest_log"]:
                    stats["newest_log"] = file_time
        
        # 크기를 MB로 변환
        stats["total_size_mb"] = stats["total_size"] / (1024 * 1024)
        
        return stats

class LogContextManager:
    """로그 컨텍스트 관리자"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{self.operation} 시작", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"{self.operation} 실패 (소요시간: {duration:.2f}초)",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={**self.context, "duration": duration}
            )
        else:
            self.logger.info(
                f"{self.operation} 완료 (소요시간: {duration:.2f}초)",
                extra={**self.context, "duration": duration}
            )

def setup_enhanced_logging(name: str = "AutoCI") -> EnhancedLogger:
    """향상된 로깅 설정"""
    return EnhancedLogger(name)

def get_logger(name: str) -> logging.Logger:
    """로거 가져오기"""
    return logging.getLogger(name)

# 전역 로깅 설정
_enhanced_logger = None

def init_logging():
    """로깅 초기화"""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = setup_enhanced_logging()
    return _enhanced_logger