#!/usr/bin/env python3
"""
Production Monitoring and Metrics System
"""

import os
import sys
import time
import json
import logging
import asyncio
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import aiohttp
import sqlite3
from enum import Enum
try:
    import torch
except ImportError:
    torch = None  # Handle missing torch gracefully

class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    """메트릭 데이터"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = None
    description: str = None

@dataclass
class HealthStatus:
    """헬스 체크 상태"""
    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any] = None
    error: Optional[str] = None

class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        self.metrics = defaultdict(deque)
        self.max_history = 10000  # 메트릭당 최대 보관 개수
        self._lock = threading.Lock()
        
    def record(self, metric: Metric):
        """메트릭 기록"""
        with self._lock:
            self.metrics[metric.name].append(metric)
            
            # 오래된 메트릭 제거
            if len(self.metrics[metric.name]) > self.max_history:
                self.metrics[metric.name].popleft()
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """메트릭 조회"""
        with self._lock:
            metrics = list(self.metrics.get(name, []))
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_latest(self, name: str) -> Optional[Metric]:
        """최신 메트릭 조회"""
        with self._lock:
            metrics = self.metrics.get(name)
            return metrics[-1] if metrics else None
    
    def clear(self, name: Optional[str] = None):
        """메트릭 삭제"""
        with self._lock:
            if name:
                self.metrics.pop(name, None)
            else:
                self.metrics.clear()

class ProductionMonitor:
    """프로덕션 모니터링 시스템"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger("Monitor")
        
        # 메트릭 수집기
        self.collector = MetricsCollector()
        
        # 헬스 체크 상태
        self.health_status = {}
        
        # 알림 큐
        self.alert_queue = asyncio.Queue()
        
        # 데이터베이스 초기화
        self.db_path = Path("monitoring/metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 모니터링 태스크
        self.monitoring_task = None
        self.running = False
        
        # 성능 카운터
        self.counters = {
            "games_created": 0,
            "features_added": 0,
            "bugs_fixed": 0,
            "errors_caught": 0,
            "ai_requests": 0,
            "ai_tokens_used": 0
        }
        
        # 임계값 설정
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 0.1,
            "response_time": 5.0  # seconds
        }
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 메트릭 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                labels TEXT,
                unit TEXT,
                description TEXT
            )
        """)
        
        # 메트릭 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
            ON metrics (name, timestamp)
        """)
        
        # 이벤트 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                details TEXT
            )
        """)
        
        # 이벤트 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events (timestamp)
        """)
        
        # 알림 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME
            )
        """)
        
        # 알림 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
            ON alerts (timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    async def start(self):
        """모니터링 시작"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Production monitoring started")
        
        # 모니터링 태스크 시작
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # 알림 처리 태스크
        asyncio.create_task(self._alert_processor())
        
        # 데이터 정리 태스크
        asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """모니터링 중지"""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Production monitoring stopped")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        interval = self.config.get("metrics_collection_interval", 60)
        
        while self.running:
            try:
                # 시스템 메트릭 수집
                await self._collect_system_metrics()
                
                # 애플리케이션 메트릭 수집
                await self._collect_app_metrics()
                
                # 헬스 체크
                await self._perform_health_checks()
                
                # 임계값 체크
                await self._check_thresholds()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        timestamp = datetime.now()
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric("system.cpu.usage", cpu_percent, MetricType.GAUGE, 
                         unit="percent", labels={"host": os.uname().nodename})
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        self.record_metric("system.memory.usage", memory.percent, MetricType.GAUGE,
                         unit="percent")
        self.record_metric("system.memory.available", memory.available / (1024**3), 
                         MetricType.GAUGE, unit="GB")
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        self.record_metric("system.disk.usage", disk.percent, MetricType.GAUGE,
                         unit="percent")
        self.record_metric("system.disk.free", disk.free / (1024**3), 
                         MetricType.GAUGE, unit="GB")
        
        # 네트워크 I/O
        net_io = psutil.net_io_counters()
        self.record_metric("system.network.bytes_sent", net_io.bytes_sent, 
                         MetricType.COUNTER, unit="bytes")
        self.record_metric("system.network.bytes_recv", net_io.bytes_recv,
                         MetricType.COUNTER, unit="bytes")
        
        # 프로세스 정보
        process = psutil.Process()
        self.record_metric("process.cpu.usage", process.cpu_percent(), 
                         MetricType.GAUGE, unit="percent")
        self.record_metric("process.memory.rss", process.memory_info().rss / (1024**2),
                         MetricType.GAUGE, unit="MB")
        self.record_metric("process.threads", process.num_threads(),
                         MetricType.GAUGE)
    
    async def _collect_app_metrics(self):
        """애플리케이션 메트릭 수집"""
        # 카운터 메트릭
        for name, value in self.counters.items():
            self.record_metric(f"app.{name}", value, MetricType.COUNTER)
        
        # 큐 크기
        self.record_metric("app.alert_queue.size", self.alert_queue.qsize(),
                         MetricType.GAUGE)
        
        # 에러율 계산
        total_operations = sum([
            self.counters["games_created"],
            self.counters["features_added"],
            self.counters["bugs_fixed"]
        ])
        
        if total_operations > 0:
            error_rate = self.counters["errors_caught"] / total_operations
            self.record_metric("app.error_rate", error_rate, MetricType.GAUGE)
    
    async def _perform_health_checks(self):
        """헬스 체크 수행"""
        components = [
            ("godot_controller", self._check_godot_health),
            ("ai_integration", self._check_ai_health),
            ("database", self._check_database_health),
            ("disk_space", self._check_disk_space_health)
        ]
        
        for component, check_func in components:
            try:
                status = await check_func()
                self.health_status[component] = status
                
                # 헬스 메트릭 기록
                health_value = 1.0 if status.status == "healthy" else 0.0
                self.record_metric(f"health.{component}", health_value, 
                                 MetricType.GAUGE)
                
            except Exception as e:
                self.health_status[component] = HealthStatus(
                    component=component,
                    status="unhealthy",
                    last_check=datetime.now(),
                    error=str(e)
                )
    
    async def _check_godot_health(self) -> HealthStatus:
        """Godot 헬스 체크"""
        try:
            # Godot 프로세스 확인
            godot_running = any("godot" in p.name().lower() 
                              for p in psutil.process_iter(['name']))
            
            return HealthStatus(
                component="godot_controller",
                status="healthy" if godot_running else "degraded",
                last_check=datetime.now(),
                details={"process_running": godot_running}
            )
        except Exception as e:
            return HealthStatus(
                component="godot_controller",
                status="unhealthy",
                last_check=datetime.now(),
                error=str(e)
            )
    
    async def _check_ai_health(self) -> HealthStatus:
        """AI 모델 헬스 체크"""
        try:
            # AI 모델 메모리 사용량 확인
            if torch and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_percent = (torch.cuda.memory_allocated() / 
                                    torch.cuda.max_memory_allocated() * 100)
                
                status = "healthy"
                if gpu_memory_percent > 90:
                    status = "degraded"
                
                return HealthStatus(
                    component="ai_integration",
                    status=status,
                    last_check=datetime.now(),
                    details={
                        "gpu_memory_gb": gpu_memory,
                        "gpu_memory_percent": gpu_memory_percent
                    }
                )
            else:
                return HealthStatus(
                    component="ai_integration",
                    status="healthy",
                    last_check=datetime.now(),
                    details={"device": "cpu"}
                )
        except Exception as e:
            return HealthStatus(
                component="ai_integration",
                status="unhealthy",
                last_check=datetime.now(),
                error=str(e)
            )
    
    async def _check_database_health(self) -> HealthStatus:
        """데이터베이스 헬스 체크"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()
            
            # 간단한 쿼리로 연결 확인
            cursor.execute("SELECT COUNT(*) FROM metrics")
            metric_count = cursor.fetchone()[0]
            
            conn.close()
            
            return HealthStatus(
                component="database",
                status="healthy",
                last_check=datetime.now(),
                details={"metric_count": metric_count}
            )
        except Exception as e:
            return HealthStatus(
                component="database",
                status="unhealthy",
                last_check=datetime.now(),
                error=str(e)
            )
    
    async def _check_disk_space_health(self) -> HealthStatus:
        """디스크 공간 헬스 체크"""
        try:
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            if disk.percent > 90:
                status = "unhealthy"
            elif disk.percent > 80:
                status = "degraded"
            
            return HealthStatus(
                component="disk_space",
                status=status,
                last_check=datetime.now(),
                details={
                    "used_percent": disk.percent,
                    "free_gb": disk.free / (1024**3)
                }
            )
        except Exception as e:
            return HealthStatus(
                component="disk_space",
                status="unhealthy",
                last_check=datetime.now(),
                error=str(e)
            )
    
    async def _check_thresholds(self):
        """임계값 체크 및 알림"""
        # CPU 임계값
        cpu_metric = self.collector.get_latest("system.cpu.usage")
        if cpu_metric and cpu_metric.value > self.thresholds["cpu_percent"]:
            await self.create_alert(
                "high_cpu_usage",
                f"CPU usage is {cpu_metric.value:.1f}%",
                "warning"
            )
        
        # 메모리 임계값
        memory_metric = self.collector.get_latest("system.memory.usage")
        if memory_metric and memory_metric.value > self.thresholds["memory_percent"]:
            await self.create_alert(
                "high_memory_usage",
                f"Memory usage is {memory_metric.value:.1f}%",
                "warning"
            )
        
        # 디스크 임계값
        disk_metric = self.collector.get_latest("system.disk.usage")
        if disk_metric and disk_metric.value > self.thresholds["disk_percent"]:
            await self.create_alert(
                "low_disk_space",
                f"Disk usage is {disk_metric.value:.1f}%",
                "critical"
            )
        
        # 에러율 임계값
        error_rate_metric = self.collector.get_latest("app.error_rate")
        if error_rate_metric and error_rate_metric.value > self.thresholds["error_rate"]:
            await self.create_alert(
                "high_error_rate",
                f"Error rate is {error_rate_metric.value:.1%}",
                "warning"
            )
    
    async def _alert_processor(self):
        """알림 처리"""
        while self.running:
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                await self._send_alert(alert)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """알림 발송"""
        # 로그 기록
        self.logger.warning(f"ALERT: {alert['type']} - {alert['message']}")
        
        # 데이터베이스 저장
        self._save_alert(alert)
        
        # 웹훅 발송
        webhook_url = self.config.get("webhook_url")
        if webhook_url:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=alert)
            except Exception as e:
                self.logger.error(f"Webhook send failed: {e}")
    
    async def _cleanup_loop(self):
        """오래된 데이터 정리"""
        while self.running:
            try:
                # 24시간마다 정리
                await asyncio.sleep(86400)
                
                retention_days = self.config.get("retention_days", 7)
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                # 데이터베이스 정리
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
                cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_date,))
                cursor.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", 
                             (cutoff_date,))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Cleaned up data older than {retention_days} days")
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     unit: Optional[str] = None, labels: Optional[Dict] = None,
                     description: Optional[str] = None):
        """메트릭 기록"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            labels=labels or {},
            description=description
        )
        
        # 메모리 수집
        self.collector.record(metric)
        
        # 데이터베이스 저장 (비동기)
        asyncio.create_task(self._save_metric_async(metric))
    
    async def _save_metric_async(self, metric: Metric):
        """메트릭 비동기 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO metrics (name, type, value, timestamp, labels, unit, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.type.value,
                metric.value,
                metric.timestamp,
                json.dumps(metric.labels) if metric.labels else None,
                metric.unit,
                metric.description
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save metric: {e}")
    
    def log_event(self, event_type: str, component: str, message: str,
                  severity: str = "info", details: Optional[Dict] = None):
        """이벤트 로그"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO events (event_type, component, message, severity, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_type,
            component,
            message,
            severity,
            datetime.now(),
            json.dumps(details) if details else None
        ))
        
        conn.commit()
        conn.close()
    
    async def create_alert(self, alert_type: str, message: str, severity: str = "info"):
        """알림 생성"""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.alert_queue.put(alert)
    
    def _save_alert(self, alert: Dict[str, Any]):
        """알림 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (alert_type, message, severity, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            alert["type"],
            alert["message"],
            alert["severity"],
            datetime.fromisoformat(alert["timestamp"])
        ))
        
        conn.commit()
        conn.close()
    
    def increment_counter(self, name: str, value: int = 1):
        """카운터 증가"""
        if name in self.counters:
            self.counters[name] += value
            self.record_metric(f"app.{name}", self.counters[name], MetricType.COUNTER)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """헬스 상태 요약"""
        overall_health = "healthy"
        unhealthy_components = []
        
        for component, status in self.health_status.items():
            if status.status == "unhealthy":
                overall_health = "unhealthy"
                unhealthy_components.append(component)
            elif status.status == "degraded" and overall_health == "healthy":
                overall_health = "degraded"
        
        return {
            "overall_status": overall_health,
            "components": {k: v.status for k, v in self.health_status.items()},
            "unhealthy_components": unhealthy_components,
            "last_check": max(s.last_check for s in self.health_status.values()) 
                         if self.health_status else None
        }
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """메트릭 요약"""
        since = datetime.now() - timedelta(minutes=duration_minutes)
        
        summary = {}
        
        # 주요 메트릭 평균값
        for metric_name in ["system.cpu.usage", "system.memory.usage", "system.disk.usage"]:
            metrics = self.collector.get_metrics(metric_name, since)
            if metrics:
                values = [m.value for m in metrics]
                summary[metric_name] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "current": metrics[-1].value
                }
        
        return summary


# 싱글톤 인스턴스
_monitor = None

def get_monitor() -> ProductionMonitor:
    """모니터 싱글톤 인스턴스"""
    global _monitor
    if _monitor is None:
        from config.production_config import get_config
        config = get_config()
        _monitor = ProductionMonitor(asdict(config.monitoring))
    return _monitor