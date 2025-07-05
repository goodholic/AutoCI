"""
실시간 상태 모니터링 시스템
AutoCI의 모든 활동을 실시간으로 모니터링하고 시각화
"""

import os
import json
import time
import asyncio
import threading
import psutil
import GPUtil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify
app = Flask(__name__)
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_percent: float
    gpu_memory_percent: float
    gpu_temperature: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class GameDevelopmentMetrics:
    """게임 개발 메트릭"""
    timestamp: str
    project_name: str
    current_phase: str
    progress_percent: float
    quality_score: int
    iteration_count: int
    error_count: int
    features_completed: int
    features_pending: int
    elapsed_time: str
    remaining_time: str


@dataclass
class AIModelMetrics:
    """AI 모델 메트릭"""
    timestamp: str
    active_models: List[str]
    total_requests: int
    average_response_time: float
    cache_hit_rate: float
    memory_usage_gb: float
    queue_length: int


class RealtimeMonitoringSystem:
    """실시간 모니터링 시스템"""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'autoci-monitoring-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 메트릭 저장소 (최근 1시간 데이터)
        self.metrics_history = {
            "system": deque(maxlen=3600),  # 1초마다 1시간
            "game_dev": deque(maxlen=360),  # 10초마다 1시간
            "ai_model": deque(maxlen=360),  # 10초마다 1시간
            "learning": deque(maxlen=60),   # 1분마다 1시간
            "evolution": deque(maxlen=60)   # 1분마다 1시간
        }
        
        # 실시간 알림
        self.alerts = deque(maxlen=100)
        
        # 모니터링 스레드
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 연결된 컴포넌트
        self.connected_components = {}
        
        # 라우트 설정
        self._setup_routes()
        
        # SocketIO 이벤트 설정
        self._setup_socketio_events()
        
    def _setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            return render_template('monitoring_dashboard.html')
        
        @self.app.route('/api/metrics/system')
        def get_system_metrics():
            return jsonify(list(self.metrics_history["system"]))
        
        @self.app.route('/api/metrics/gamedev')
        def get_gamedev_metrics():
            return jsonify(list(self.metrics_history["game_dev"]))
        
        @self.app.route('/api/metrics/ai')
        def get_ai_metrics():
            return jsonify(list(self.metrics_history["ai_model"]))
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify(list(self.alerts))
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify(self._get_current_status())
    
    def _setup_socketio_events(self):
        """SocketIO 이벤트 설정"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("클라이언트 연결됨")
            emit('connected', {'data': 'Connected to AutoCI Monitoring'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("클라이언트 연결 해제됨")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            self._broadcast_current_metrics()
    
    def start(self):
        """모니터링 시스템 시작"""
        self.monitoring_active = True
        
        # 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # 대시보드 HTML 생성
        self._create_dashboard_html()
        
        # Flask 서버 시작 (별도 스레드)
        server_thread = threading.Thread(
            target=lambda: self.socketio.run(self.app, host='0.0.0.0', port=self.port)
        )
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"🖥️ 모니터링 대시보드: http://localhost:{self.port}")
    
    def stop(self):
        """모니터링 시스템 중지"""
        self.monitoring_active = False
        logger.info("모니터링 시스템 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        last_gamedev_update = time.time()
        last_ai_update = time.time()
        last_learning_update = time.time()
        
        while self.monitoring_active:
            current_time = time.time()
            
            # 시스템 메트릭 (매초)
            system_metrics = self._collect_system_metrics()
            self.metrics_history["system"].append(asdict(system_metrics))
            
            # 게임 개발 메트릭 (10초마다)
            if current_time - last_gamedev_update >= 10:
                gamedev_metrics = self._collect_gamedev_metrics()
                if gamedev_metrics:
                    self.metrics_history["game_dev"].append(asdict(gamedev_metrics))
                last_gamedev_update = current_time
            
            # AI 모델 메트릭 (10초마다)
            if current_time - last_ai_update >= 10:
                ai_metrics = self._collect_ai_metrics()
                if ai_metrics:
                    self.metrics_history["ai_model"].append(asdict(ai_metrics))
                last_ai_update = current_time
            
            # 학습 메트릭 (1분마다)
            if current_time - last_learning_update >= 60:
                learning_metrics = self._collect_learning_metrics()
                if learning_metrics:
                    self.metrics_history["learning"].append(learning_metrics)
                last_learning_update = current_time
            
            # 실시간 브로드캐스트
            self._broadcast_current_metrics()
            
            # 알림 체크
            self._check_alerts(system_metrics)
            
            time.sleep(1)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        # CPU 및 메모리
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU (NVIDIA)
        gpu_percent = 0
        gpu_memory_percent = 0
        gpu_temperature = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_temperature = gpu.temperature
        except:
            pass
        
        # 디스크
        disk = psutil.disk_usage('/')
        
        # 네트워크
        net_io = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature,
            disk_usage_percent=disk.percent,
            network_sent_mb=net_io.bytes_sent / (1024**2),
            network_recv_mb=net_io.bytes_recv / (1024**2)
        )
    
    def _collect_gamedev_metrics(self) -> Optional[GameDevelopmentMetrics]:
        """게임 개발 메트릭 수집"""
        # 게임 개발 파이프라인에서 메트릭 가져오기
        if "game_pipeline" in self.connected_components:
            pipeline = self.connected_components["game_pipeline"]
            if hasattr(pipeline, 'current_project') and pipeline.current_project:
                project = pipeline.current_project
                return GameDevelopmentMetrics(
                    timestamp=datetime.now().isoformat(),
                    project_name=project.name,
                    current_phase=project.current_phase.value,
                    progress_percent=project.progress_percentage,
                    quality_score=project.quality_metrics.total_score,
                    iteration_count=project.iteration_count,
                    error_count=project.error_count,
                    features_completed=len(project.completed_features),
                    features_pending=len(project.pending_features),
                    elapsed_time=str(project.elapsed_time),
                    remaining_time=str(project.remaining_time)
                )
        return None
    
    def _collect_ai_metrics(self) -> Optional[AIModelMetrics]:
        """AI 모델 메트릭 수집"""
        # AI 모델 시스템에서 메트릭 가져오기
        if "ai_system" in self.connected_components:
            ai_system = self.connected_components["ai_system"]
            if hasattr(ai_system, 'get_metrics'):
                metrics = ai_system.get_metrics()
                return AIModelMetrics(
                    timestamp=datetime.now().isoformat(),
                    active_models=metrics.get("active_models", []),
                    total_requests=metrics.get("total_requests", 0),
                    average_response_time=metrics.get("avg_response_time", 0),
                    cache_hit_rate=metrics.get("cache_hit_rate", 0),
                    memory_usage_gb=metrics.get("memory_usage_gb", 0),
                    queue_length=metrics.get("queue_length", 0)
                )
        return None
    
    def _collect_learning_metrics(self) -> Optional[Dict[str, Any]]:
        """학습 메트릭 수집"""
        if "learning_system" in self.connected_components:
            learning = self.connected_components["learning_system"]
            if hasattr(learning, 'stats'):
                return {
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": learning.stats.get("total_questions", 0),
                    "quality_answers": learning.stats.get("quality_answers", 0),
                    "topics_covered": learning.stats.get("topics_covered", {}),
                    "learning_progress": learning.stats.get("learning_progress", {})
                }
        return None
    
    def _broadcast_current_metrics(self):
        """현재 메트릭 브로드캐스트"""
        try:
            # 최신 메트릭
            latest_metrics = {
                "system": list(self.metrics_history["system"])[-10:] if self.metrics_history["system"] else [],
                "game_dev": list(self.metrics_history["game_dev"])[-1:] if self.metrics_history["game_dev"] else [],
                "ai_model": list(self.metrics_history["ai_model"])[-1:] if self.metrics_history["ai_model"] else [],
                "alerts": list(self.alerts)[-5:]
            }
            
            self.socketio.emit('metrics_update', latest_metrics)
        except Exception as e:
            logger.error(f"메트릭 브로드캐스트 오류: {e}")
    
    def _check_alerts(self, metrics: SystemMetrics):
        """알림 체크"""
        alerts = []
        
        # CPU 과부하
        if metrics.cpu_percent > 90:
            alerts.append({
                "type": "warning",
                "message": f"CPU 사용률 높음: {metrics.cpu_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # 메모리 부족
        if metrics.memory_percent > 85:
            alerts.append({
                "type": "warning",
                "message": f"메모리 사용률 높음: {metrics.memory_percent:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # GPU 온도
        if metrics.gpu_temperature > 80:
            alerts.append({
                "type": "danger",
                "message": f"GPU 온도 높음: {metrics.gpu_temperature}°C",
                "timestamp": datetime.now().isoformat()
            })
        
        # 디스크 공간
        if metrics.disk_usage_percent > 90:
            alerts.append({
                "type": "danger",
                "message": f"디스크 공간 부족: {metrics.disk_usage_percent:.1f}% 사용",
                "timestamp": datetime.now().isoformat()
            })
        
        # 알림 추가 및 브로드캐스트
        for alert in alerts:
            self.alerts.append(alert)
            self.socketio.emit('new_alert', alert)
    
    def _get_current_status(self) -> Dict[str, Any]:
        """현재 상태 요약"""
        # 최신 시스템 메트릭
        latest_system = list(self.metrics_history["system"])[-1] if self.metrics_history["system"] else {}
        
        # 게임 개발 상태
        game_status = "No active project"
        if self.metrics_history["game_dev"]:
            latest_game = list(self.metrics_history["game_dev"])[-1]
            game_status = f"{latest_game.get('project_name')} - {latest_game.get('current_phase')}"
        
        # AI 모델 상태
        ai_status = "No active models"
        if self.metrics_history["ai_model"]:
            latest_ai = list(self.metrics_history["ai_model"])[-1]
            active_models = latest_ai.get('active_models', [])
            ai_status = f"{len(active_models)} models active"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": self._calculate_system_health(latest_system),
            "game_development": game_status,
            "ai_models": ai_status,
            "total_alerts": len(self.alerts),
            "uptime": self._calculate_uptime()
        }
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> str:
        """시스템 건강도 계산"""
        if not metrics:
            return "Unknown"
        
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        
        if cpu < 70 and memory < 70:
            return "Healthy"
        elif cpu < 85 and memory < 85:
            return "Warning"
        else:
            return "Critical"
    
    def _calculate_uptime(self) -> str:
        """가동 시간 계산"""
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{days}d {hours}h {minutes}m"
    
    def register_component(self, name: str, component: Any):
        """컴포넌트 등록"""
        self.connected_components[name] = component
        logger.info(f"컴포넌트 등록됨: {name}")
    
    def _create_dashboard_html(self):
        """대시보드 HTML 생성"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoCI 실시간 모니터링</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <style>
        .metric-card { 
            margin-bottom: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .alert-box {
            max-height: 200px;
            overflow-y: auto;
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
        .status-badge {
            font-size: 0.9em;
            padding: 5px 10px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🖥️ AutoCI 실시간 모니터링</span>
            <span class="text-white" id="connection-status">⚪ 연결 중...</span>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- 상태 요약 -->
        <div class="row">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">시스템 상태</h6>
                        <h3 id="system-health" class="text-success">Healthy</h3>
                        <small class="text-muted">CPU: <span id="cpu-usage">0</span>%</small><br>
                        <small class="text-muted">메모리: <span id="memory-usage">0</span>%</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">게임 개발</h6>
                        <p id="game-status" class="mb-1">대기 중</p>
                        <div class="progress">
                            <div id="game-progress" class="progress-bar" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">AI 모델</h6>
                        <h3 id="ai-models-count">0</h3>
                        <small class="text-muted">활성 모델</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body">
                        <h6 class="card-title">알림</h6>
                        <h3 id="alert-count" class="text-warning">0</h3>
                        <small class="text-muted">활성 알림</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- 차트 -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">시스템 리소스</div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="system-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">게임 개발 진행</div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="gamedev-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 알림 -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">실시간 알림</div>
                    <div class="card-body alert-box" id="alerts-container">
                        <p class="text-muted">알림이 없습니다.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Socket.IO 연결
        const socket = io();
        
        // 차트 초기화
        const systemChart = new Chart(document.getElementById('system-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }, {
                    label: '메모리 %',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        const gamedevChart = new Chart(document.getElementById('gamedev-chart'), {
            type: 'bar',
            data: {
                labels: ['품질', '진행률', '완료 기능', '남은 기능'],
                datasets: [{
                    label: '게임 개발 메트릭',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(255, 159, 64, 0.2)'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });

        // Socket 이벤트 핸들러
        socket.on('connect', function() {
            document.getElementById('connection-status').innerHTML = '🟢 연결됨';
        });

        socket.on('disconnect', function() {
            document.getElementById('connection-status').innerHTML = '🔴 연결 끊김';
        });

        socket.on('metrics_update', function(data) {
            updateMetrics(data);
        });

        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });

        // 메트릭 업데이트
        function updateMetrics(data) {
            // 시스템 메트릭
            if (data.system && data.system.length > 0) {
                const latest = data.system[data.system.length - 1];
                document.getElementById('cpu-usage').textContent = latest.cpu_percent.toFixed(1);
                document.getElementById('memory-usage').textContent = latest.memory_percent.toFixed(1);
                
                // 차트 업데이트
                systemChart.data.labels = data.system.map((_, i) => i);
                systemChart.data.datasets[0].data = data.system.map(m => m.cpu_percent);
                systemChart.data.datasets[1].data = data.system.map(m => m.memory_percent);
                systemChart.update();
            }

            // 게임 개발 메트릭
            if (data.game_dev && data.game_dev.length > 0) {
                const gameData = data.game_dev[0];
                document.getElementById('game-status').textContent = 
                    gameData.project_name + ' - ' + gameData.current_phase;
                document.getElementById('game-progress').style.width = 
                    gameData.progress_percent + '%';
                
                // 차트 업데이트
                gamedevChart.data.datasets[0].data = [
                    gameData.quality_score,
                    gameData.progress_percent,
                    gameData.features_completed,
                    gameData.features_pending
                ];
                gamedevChart.update();
            }

            // AI 모델 메트릭
            if (data.ai_model && data.ai_model.length > 0) {
                const aiData = data.ai_model[0];
                document.getElementById('ai-models-count').textContent = 
                    aiData.active_models.length;
            }

            // 알림 수
            if (data.alerts) {
                document.getElementById('alert-count').textContent = data.alerts.length;
            }
        }

        // 알림 추가
        function addAlert(alert) {
            const container = document.getElementById('alerts-container');
            const alertClass = alert.type === 'danger' ? 'alert-danger' : 
                              alert.type === 'warning' ? 'alert-warning' : 'alert-info';
            
            const alertHtml = `
                <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                    <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
                    <strong>${alert.message}</strong>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            
            container.innerHTML = alertHtml + container.innerHTML;
            
            // 최대 10개 알림 유지
            const alerts = container.querySelectorAll('.alert');
            if (alerts.length > 10) {
                alerts[alerts.length - 1].remove();
            }
        }

        // 주기적 업데이트 요청
        setInterval(function() {
            socket.emit('request_update');
        }, 1000);
    </script>
</body>
</html>
"""
        
        # templates 디렉토리 생성
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        # HTML 파일 저장
        with open(templates_dir / "monitoring_dashboard.html", 'w', encoding='utf-8') as f:
            f.write(html_content)


@app.route("/")
def index():
    return "AutoCI 실시간 모니터링 시스템이 정상적으로 실행 중입니다!"

# 테스트 및 예제
if __name__ == "__main__":
    monitoring = RealtimeMonitoringSystem(port=5555)
    monitoring.start()
    
    try:
        # 모니터링 계속 실행
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitoring.stop()