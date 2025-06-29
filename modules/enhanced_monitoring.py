#!/usr/bin/env python3
"""
Enhanced Monitoring System - 상용화 수준의 모니터링 및 메트릭스
"""

import os
import asyncio
import logging
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import platform
import socket

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

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
    labels: Dict[str, str]
    timestamp: datetime
    unit: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Alert:
    """알림 데이터"""
    name: str
    severity: str  # info, warning, error, critical
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

class EnhancedMonitor:
    """상용화 수준의 모니터링 시스템"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config/monitoring.json")
        self.logger = logging.getLogger("Monitor")
        
        # 데이터베이스 설정
        self.db_path = Path("data/metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 메트릭스 저장소
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        
        # Prometheus 메트릭스 (가능한 경우)
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        # 모니터링 설정
        self.config = self._load_config()
        
        # 수집 간격
        self.collection_intervals = {
            "system": 30,      # 시스템 메트릭
            "application": 60,  # 애플리케이션 메트릭
            "business": 300    # 비즈니스 메트릭
        }
        
        # 알림 규칙
        self.alert_rules = {
            "cpu_high": {
                "metric": "system.cpu.usage",
                "threshold": 80,
                "duration": 300,  # 5분
                "severity": "warning"
            },
            "memory_high": {
                "metric": "system.memory.usage",
                "threshold": 85,
                "duration": 300,
                "severity": "warning"
            },
            "error_rate_high": {
                "metric": "app.error.rate",
                "threshold": 10,
                "duration": 60,
                "severity": "error"
            },
            "disk_full": {
                "metric": "system.disk.usage",
                "threshold": 90,
                "duration": 60,
                "severity": "critical"
            }
        }
        
        # 모니터링 시작 (이벤트 루프가 있을 때만)
        try:
            self._start_monitoring()
        except RuntimeError:
            # 이벤트 루프가 없으면 나중에 시작
            pass
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 메트릭스 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                unit TEXT,
                description TEXT
            )
        ''')
        
        # 알림 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                threshold REAL,
                current_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                metadata TEXT
            )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
        
        conn.commit()
        conn.close()
    
    def _init_prometheus_metrics(self):
        """Prometheus 메트릭스 초기화"""
        # 카운터
        self.prom_counters = {
            "games_created": Counter('autoci_games_created_total', 'Total games created'),
            "features_added": Counter('autoci_features_added_total', 'Total features added'),
            "bugs_fixed": Counter('autoci_bugs_fixed_total', 'Total bugs fixed'),
            "errors": Counter('autoci_errors_total', 'Total errors', ['component', 'error_type'])
        }
        
        # 게이지
        self.prom_gauges = {
            "cpu_usage": Gauge('autoci_cpu_usage_percent', 'CPU usage percentage'),
            "memory_usage": Gauge('autoci_memory_usage_percent', 'Memory usage percentage'),
            "disk_usage": Gauge('autoci_disk_usage_percent', 'Disk usage percentage'),
            "active_projects": Gauge('autoci_active_projects', 'Number of active projects')
        }
        
        # 히스토그램
        self.prom_histograms = {
            "game_creation_time": Histogram('autoci_game_creation_seconds', 
                                          'Game creation time in seconds',
                                          buckets=(60, 300, 600, 1800, 3600, 7200)),
            "feature_implementation_time": Histogram('autoci_feature_implementation_seconds',
                                                   'Feature implementation time in seconds')
        }
        
        # 서머리
        self.prom_summaries = {
            "api_latency": Summary('autoci_api_latency_seconds', 'API latency in seconds')
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        default_config = {
            "retention_days": 30,
            "alert_webhook": None,
            "export_interval": 3600,
            "prometheus_port": 8000
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_config.update(config)
            except Exception as e:
                self.logger.warning(f"설정 로드 실패: {e}")
        
        return default_config
    
    def _start_monitoring(self):
        """모니터링 시작"""
        async def monitor_loop():
            tasks = [
                self._collect_system_metrics(),
                self._collect_application_metrics(),
                self._collect_business_metrics(),
                self._check_alerts(),
                self._cleanup_old_data(),
                self._export_metrics()
            ]
            await asyncio.gather(*tasks)
        
        asyncio.create_task(monitor_loop())
        
        # Prometheus 서버 시작 (가능한 경우)
        if PROMETHEUS_AVAILABLE and self.config.get("prometheus_port"):
            try:
                prometheus_client.start_http_server(self.config["prometheus_port"])
                self.logger.info(f"Prometheus 서버 시작: http://localhost:{self.config['prometheus_port']}/metrics")
            except Exception as e:
                self.logger.warning(f"Prometheus 서버 시작 실패: {e}")
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        while True:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric("system.cpu.usage", cpu_percent, 
                                       MetricType.GAUGE, unit="%")
                
                # CPU 코어별 사용률
                cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
                for i, percent in enumerate(cpu_per_core):
                    await self.record_metric(f"system.cpu.core{i}.usage", percent,
                                           MetricType.GAUGE, unit="%")
                
                # 메모리
                memory = psutil.virtual_memory()
                await self.record_metric("system.memory.usage", memory.percent,
                                       MetricType.GAUGE, unit="%")
                await self.record_metric("system.memory.available", memory.available / (1024**3),
                                       MetricType.GAUGE, unit="GB")
                await self.record_metric("system.memory.used", memory.used / (1024**3),
                                       MetricType.GAUGE, unit="GB")
                
                # 스왑
                swap = psutil.swap_memory()
                await self.record_metric("system.swap.usage", swap.percent,
                                       MetricType.GAUGE, unit="%")
                
                # 디스크
                disk = psutil.disk_usage('/')
                await self.record_metric("system.disk.usage", disk.percent,
                                       MetricType.GAUGE, unit="%")
                await self.record_metric("system.disk.free", disk.free / (1024**3),
                                       MetricType.GAUGE, unit="GB")
                
                # 네트워크
                net_io = psutil.net_io_counters()
                await self.record_metric("system.network.bytes_sent", net_io.bytes_sent,
                                       MetricType.COUNTER, unit="bytes")
                await self.record_metric("system.network.bytes_recv", net_io.bytes_recv,
                                       MetricType.COUNTER, unit="bytes")
                
                # 프로세스
                process_count = len(psutil.pids())
                await self.record_metric("system.process.count", process_count,
                                       MetricType.GAUGE)
                
                # 시스템 정보
                boot_time = datetime.fromtimestamp(psutil.boot_time())
                uptime = (datetime.now() - boot_time).total_seconds()
                await self.record_metric("system.uptime", uptime,
                                       MetricType.GAUGE, unit="seconds")
                
                await asyncio.sleep(self.collection_intervals["system"])
                
            except Exception as e:
                self.logger.error(f"시스템 메트릭 수집 에러: {e}")
                await asyncio.sleep(60)
    
    async def _collect_application_metrics(self):
        """애플리케이션 메트릭 수집"""
        while True:
            try:
                # AutoCI 프로세스 정보
                current_process = psutil.Process()
                
                # 프로세스 CPU 사용률
                cpu_percent = current_process.cpu_percent(interval=1)
                await self.record_metric("app.process.cpu", cpu_percent,
                                       MetricType.GAUGE, unit="%")
                
                # 프로세스 메모리
                memory_info = current_process.memory_info()
                await self.record_metric("app.process.memory", memory_info.rss / (1024**2),
                                       MetricType.GAUGE, unit="MB")
                
                # 열린 파일 수
                try:
                    open_files = len(current_process.open_files())
                    await self.record_metric("app.process.open_files", open_files,
                                           MetricType.GAUGE)
                except:
                    pass
                
                # 스레드 수
                thread_count = current_process.num_threads()
                await self.record_metric("app.process.threads", thread_count,
                                       MetricType.GAUGE)
                
                # 게임 프로젝트 수
                game_projects_dir = Path("game_projects")
                if game_projects_dir.exists():
                    project_count = len(list(game_projects_dir.iterdir()))
                    await self.record_metric("app.projects.count", project_count,
                                           MetricType.GAUGE)
                
                await asyncio.sleep(self.collection_intervals["application"])
                
            except Exception as e:
                self.logger.error(f"애플리케이션 메트릭 수집 에러: {e}")
                await asyncio.sleep(60)
    
    async def _collect_business_metrics(self):
        """비즈니스 메트릭 수집"""
        while True:
            try:
                # 게임 개발 통계
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 24시간 동안 생성된 게임 수
                cursor.execute('''
                    SELECT COUNT(*) FROM metrics
                    WHERE name = 'business.games.created'
                    AND timestamp > datetime('now', '-24 hours')
                ''')
                games_24h = cursor.fetchone()[0]
                await self.record_metric("business.games.created_24h", games_24h,
                                       MetricType.GAUGE)
                
                # 평균 게임 생성 시간
                cursor.execute('''
                    SELECT AVG(value) FROM metrics
                    WHERE name = 'business.game.creation_time'
                    AND timestamp > datetime('now', '-7 days')
                ''')
                avg_creation_time = cursor.fetchone()[0] or 0
                await self.record_metric("business.game.avg_creation_time", avg_creation_time,
                                       MetricType.GAUGE, unit="seconds")
                
                conn.close()
                
                await asyncio.sleep(self.collection_intervals["business"])
                
            except Exception as e:
                self.logger.error(f"비즈니스 메트릭 수집 에러: {e}")
                await asyncio.sleep(300)
    
    async def record_metric(self, name: str, value: float, 
                          metric_type: MetricType = MetricType.GAUGE,
                          labels: Dict[str, str] = None,
                          unit: str = None,
                          description: str = None):
        """메트릭 기록"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {},
            timestamp=datetime.now(),
            unit=unit,
            description=description
        )
        
        # 메모리에 저장
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)
        
        # 최근 데이터만 유지 (메모리 절약)
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # 데이터베이스에 저장
        await self._save_metric_to_db(metric)
        
        # Prometheus 업데이트
        if PROMETHEUS_AVAILABLE:
            self._update_prometheus_metric(metric)
    
    async def _save_metric_to_db(self, metric: Metric):
        """메트릭을 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (name, type, value, labels, timestamp, unit, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.name,
                metric.type.value,
                metric.value,
                json.dumps(metric.labels),
                metric.timestamp,
                metric.unit,
                metric.description
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"메트릭 저장 에러: {e}")
    
    def _update_prometheus_metric(self, metric: Metric):
        """Prometheus 메트릭 업데이트"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # 메트릭 이름에서 Prometheus 메트릭 찾기
        metric_parts = metric.name.split('.')
        
        if metric.type == MetricType.COUNTER:
            if metric_parts[0] == "app" and metric_parts[1] in ["games", "features", "bugs"]:
                counter_name = f"{metric_parts[1]}_{'created' if metric_parts[1] == 'games' else 'fixed'}"
                if counter_name in self.prom_counters:
                    self.prom_counters[counter_name].inc(metric.value)
        
        elif metric.type == MetricType.GAUGE:
            if metric.name == "system.cpu.usage" and "cpu_usage" in self.prom_gauges:
                self.prom_gauges["cpu_usage"].set(metric.value)
            elif metric.name == "system.memory.usage" and "memory_usage" in self.prom_gauges:
                self.prom_gauges["memory_usage"].set(metric.value)
            elif metric.name == "system.disk.usage" and "disk_usage" in self.prom_gauges:
                self.prom_gauges["disk_usage"].set(metric.value)
    
    async def _check_alerts(self):
        """알림 체크"""
        while True:
            try:
                for rule_name, rule in self.alert_rules.items():
                    metric_name = rule["metric"]
                    threshold = rule["threshold"]
                    duration = rule["duration"]
                    severity = rule["severity"]
                    
                    # 최근 메트릭 확인
                    if metric_name in self.metrics:
                        recent_metrics = [m for m in self.metrics[metric_name]
                                        if m.timestamp > datetime.now() - timedelta(seconds=duration)]
                        
                        if recent_metrics:
                            # 평균값 계산
                            avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
                            
                            # 임계값 체크
                            if avg_value > threshold:
                                # 기존 알림 확인
                                existing_alert = next((a for a in self.alerts 
                                                     if a.name == rule_name and not a.resolved), None)
                                
                                if not existing_alert:
                                    # 새 알림 생성
                                    alert = Alert(
                                        name=rule_name,
                                        severity=severity,
                                        message=f"{metric_name} exceeded threshold: {avg_value:.2f} > {threshold}",
                                        threshold=threshold,
                                        current_value=avg_value,
                                        timestamp=datetime.now()
                                    )
                                    
                                    await self._trigger_alert(alert)
                            else:
                                # 임계값 이하 - 알림 해제
                                for alert in self.alerts:
                                    if alert.name == rule_name and not alert.resolved:
                                        alert.resolved = True
                                        await self._resolve_alert(alert)
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                self.logger.error(f"알림 체크 에러: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_alert(self, alert: Alert):
        """알림 발생"""
        self.alerts.append(alert)
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        # 데이터베이스에 저장
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (name, severity, message, threshold, current_value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.name,
                alert.severity,
                alert.message,
                alert.threshold,
                alert.current_value,
                alert.timestamp,
                json.dumps(alert.metadata or {})
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"알림 저장 에러: {e}")
        
        # 웹훅 전송
        if self.config.get("alert_webhook"):
            await self._send_webhook(alert)
    
    async def _resolve_alert(self, alert: Alert):
        """알림 해제"""
        self.logger.info(f"Alert resolved: {alert.name}")
        
        # 데이터베이스 업데이트
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts SET resolved = TRUE
                WHERE name = ? AND resolved = FALSE
            ''', (alert.name,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"알림 해제 에러: {e}")
    
    async def _send_webhook(self, alert: Alert):
        """웹훅 전송"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "alert": asdict(alert),
                    "system": platform.node(),
                    "timestamp": datetime.now().isoformat()
                }
                await session.post(self.config["alert_webhook"], json=payload)
        except Exception as e:
            self.logger.error(f"웹훅 전송 실패: {e}")
    
    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        while True:
            try:
                retention_days = self.config.get("retention_days", 30)
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 오래된 메트릭 삭제
                cursor.execute('''
                    DELETE FROM metrics WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # 오래된 알림 삭제
                cursor.execute('''
                    DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE
                ''', (cutoff_date,))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"오래된 데이터 정리 완료 (기준: {retention_days}일)")
                
                await asyncio.sleep(86400)  # 하루에 한 번
                
            except Exception as e:
                self.logger.error(f"데이터 정리 에러: {e}")
                await asyncio.sleep(3600)
    
    async def _export_metrics(self):
        """메트릭 내보내기"""
        while True:
            try:
                export_interval = self.config.get("export_interval", 3600)
                
                # JSON 형식으로 내보내기
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system": {
                        "hostname": socket.gethostname(),
                        "platform": platform.platform(),
                        "python_version": platform.python_version()
                    },
                    "metrics": {},
                    "alerts": []
                }
                
                # 최근 메트릭 수집
                for name, metrics in self.metrics.items():
                    if metrics:
                        latest = metrics[-1]
                        export_data["metrics"][name] = {
                            "value": latest.value,
                            "timestamp": latest.timestamp.isoformat(),
                            "unit": latest.unit
                        }
                
                # 활성 알림 추가
                for alert in self.alerts:
                    if not alert.resolved:
                        export_data["alerts"].append({
                            "name": alert.name,
                            "severity": alert.severity,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat()
                        })
                
                # 파일로 저장
                export_path = Path("data/metrics_export.json")
                export_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
                
                await asyncio.sleep(export_interval)
                
            except Exception as e:
                self.logger.error(f"메트릭 내보내기 에러: {e}")
                await asyncio.sleep(600)
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """메트릭 요약 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as count,
                AVG(value) as avg,
                MIN(value) as min,
                MAX(value) as max,
                SUM(value) as sum
            FROM metrics
            WHERE name = ? AND timestamp > ?
        ''', (metric_name, cutoff_time))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "metric": metric_name,
            "period_hours": hours,
            "count": result[0],
            "average": result[1],
            "minimum": result[2],
            "maximum": result[3],
            "total": result[4]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "metrics_collected": sum(len(m) for m in self.metrics.values()),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "database_size": os.path.getsize(self.db_path) / (1024**2),  # MB
            "uptime": self._get_uptime(),
            "last_metrics": {
                name: metrics[-1].value if metrics else None
                for name, metrics in self.metrics.items()
                if name.startswith("system.")
            }
        }
    
    def _get_uptime(self) -> float:
        """시스템 가동 시간"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return uptime_seconds
        except:
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # 데이터베이스 체크
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM metrics")
            cursor.fetchone()
            conn.close()
            health["components"]["database"] = "healthy"
        except Exception as e:
            health["components"]["database"] = f"unhealthy: {e}"
            health["status"] = "unhealthy"
        
        # 메트릭 수집 체크
        recent_metric = any(
            any(m.timestamp > datetime.now() - timedelta(minutes=5) for m in metrics)
            for metrics in self.metrics.values()
        )
        health["components"]["metric_collection"] = "healthy" if recent_metric else "unhealthy"
        
        # 시스템 리소스 체크
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 90 or memory_usage > 90:
            health["status"] = "degraded"
            health["components"]["resources"] = f"high usage - CPU: {cpu_usage}%, Memory: {memory_usage}%"
        else:
            health["components"]["resources"] = "healthy"
        
        return health


# 싱글톤 인스턴스
_monitor = None

def get_enhanced_monitor() -> EnhancedMonitor:
    """모니터 가져오기"""
    global _monitor
    if _monitor is None:
        _monitor = EnhancedMonitor()
    return _monitor


# 메트릭 데코레이터
def with_metrics(metric_name: str, metric_type: MetricType = MetricType.HISTOGRAM):
    """메트릭 수집 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_enhanced_monitor()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                await monitor.record_metric(
                    f"{metric_name}.duration",
                    duration,
                    metric_type,
                    unit="seconds"
                )
                
                await monitor.record_metric(
                    f"{metric_name}.success",
                    1,
                    MetricType.COUNTER
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                await monitor.record_metric(
                    f"{metric_name}.duration",
                    duration,
                    metric_type,
                    unit="seconds"
                )
                
                await monitor.record_metric(
                    f"{metric_name}.error",
                    1,
                    MetricType.COUNTER,
                    labels={"error_type": type(e).__name__}
                )
                
                raise
        
        def sync_wrapper(*args, **kwargs):
            # 동기 함수는 기본 처리만
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator