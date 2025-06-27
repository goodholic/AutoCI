#!/usr/bin/env python3
"""
AI 학습 환경 1분 모니터링 시스템
- 1분마다 AI 모델의 학습환경을 모니터링
- 실시간 학습 진행률, 메모리 사용량, 에러 감지
- 웹 대시보드 및 알림 시스템
"""

import os
import sys
import time
import json
import logging
import threading
import psutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
from collections import deque
import signal

# 웹 서버용
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ai_learning_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AILearningMonitor:
    """AI 학습 환경 모니터링 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.monitoring_interval = 60  # 1분
        self.is_running = False
        self.monitor_thread = None
        
        # 모니터링 데이터 저장
        self.metrics_history = deque(maxlen=1440)  # 24시간 데이터 보관
        self.alerts = deque(maxlen=100)
        
        # 모니터링 대상 프로세스
        self.target_processes = [
            'python', 'python3', 'uvicorn', 'dotnet',
            'node', 'npm', 'torch', 'tensorflow'
        ]
        
        # 임계값 설정
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0,
            'learning_rate_drop': 0.001,  # 학습률 급감 감지
            'error_rate': 0.1  # 10% 이상 에러율
        }
        
        # 데이터베이스 초기화
        self.init_database()
        
        # 시그널 핸들러
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def init_database(self):
        """모니터링 데이터베이스 초기화"""
        db_path = self.base_path / "monitoring_data.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 모니터링 메트릭 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                disk_percent REAL,
                gpu_percent REAL,
                gpu_memory_mb REAL,
                active_processes INTEGER,
                learning_status TEXT,
                error_count INTEGER,
                data TEXT
            )
        ''')
        
        # 알림 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                severity TEXT,
                category TEXT,
                message TEXT,
                data TEXT
            )
        ''')
        
        # 학습 진행률 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                epoch INTEGER,
                total_epochs INTEGER,
                loss REAL,
                accuracy REAL,
                learning_rate REAL,
                eta_seconds INTEGER
            )
        ''')
        
        self.conn.commit()
        
    def signal_handler(self, sig, frame):
        """시그널 핸들러"""
        logger.info("종료 신호 받음. 모니터링을 중지합니다.")
        self.stop()
        sys.exit(0)
        
    def start(self):
        """모니터링 시작"""
        if self.is_running:
            logger.warning("모니터링이 이미 실행 중입니다.")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 웹 서버 시작
        self.start_web_server()
        
        logger.info("AI 학습 환경 모니터링이 시작되었습니다. (1분 간격)")
        
    def stop(self):
        """모니터링 중지"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if hasattr(self, 'conn'):
            self.conn.close()
            
        logger.info("모니터링이 중지되었습니다.")
        
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 메트릭 수집
                metrics = self.collect_metrics()
                
                # 데이터베이스 저장
                self.save_metrics(metrics)
                
                # 임계값 체크 및 알림
                self.check_thresholds(metrics)
                
                # 학습 상태 체크
                self.check_learning_status()
                
                # 히스토리에 추가
                self.metrics_history.append(metrics)
                
                # 콘솔 출력 (요약)
                self.print_summary(metrics)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                
            # 1분 대기
            time.sleep(self.monitoring_interval)
            
    def collect_metrics(self) -> Dict:
        """시스템 메트릭 수집"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory(),
            'disk': psutil.disk_usage('/'),
            'processes': self.get_ai_processes(),
            'gpu': self.get_gpu_metrics(),
            'network': self.get_network_stats(),
            'learning': self.get_learning_metrics()
        }
        
        return metrics
        
    def get_ai_processes(self) -> List[Dict]:
        """AI 관련 프로세스 정보 수집"""
        ai_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                name = pinfo['name'].lower()
                
                # AI 관련 프로세스 필터링
                if any(target in name for target in self.target_processes):
                    # 명령줄 인자 확인
                    cmdline = ' '.join(proc.cmdline())
                    
                    # AI 관련 키워드 체크
                    ai_keywords = ['train', 'learning', 'model', 'torch', 'tensorflow', 
                                  'autoci', 'enhanced_server', 'dual_phase', 'rag']
                    
                    if any(keyword in cmdline.lower() for keyword in ai_keywords):
                        ai_processes.append({
                            'pid': pinfo['pid'],
                            'name': pinfo['name'],
                            'cpu_percent': pinfo['cpu_percent'],
                            'memory_percent': pinfo['memory_percent'],
                            'cmdline': cmdline[:100]  # 처음 100자
                        })
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return ai_processes
        
    def get_gpu_metrics(self) -> Dict:
        """GPU 메트릭 수집"""
        gpu_metrics = {
            'available': False,
            'gpu_percent': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 0,
            'temperature': 0
        }
        
        try:
            # nvidia-smi 사용
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 4:
                    gpu_metrics.update({
                        'available': True,
                        'gpu_percent': float(values[0]),
                        'memory_used_mb': float(values[1]),
                        'memory_total_mb': float(values[2]),
                        'temperature': float(values[3])
                    })
                    
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            # GPU가 없거나 nvidia-smi가 없는 경우
            pass
            
        return gpu_metrics
        
    def get_network_stats(self) -> Dict:
        """네트워크 통계"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
    def get_learning_metrics(self) -> Dict:
        """학습 관련 메트릭 수집"""
        learning_metrics = {
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'learning_rate': 0.0,
            'eta_minutes': 0
        }
        
        # 학습 로그 파일 확인
        log_files = [
            'csharp_expert_learning.log',
            'fine_tuning.log',
            'hybrid_rag_training.log',
            'autoci_learning.log'
        ]
        
        for log_file in log_files:
            log_path = self.base_path / log_file
            if log_path.exists():
                try:
                    # 최근 로그 라인 읽기
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-100:]  # 마지막 100줄
                        
                    for line in reversed(lines):
                        # 학습 진행률 파싱
                        if 'epoch' in line.lower() and ('loss' in line.lower() or 'accuracy' in line.lower()):
                            learning_metrics['is_training'] = True
                            
                            # 간단한 파싱 (실제로는 더 정교하게)
                            import re
                            
                            # Epoch 파싱
                            epoch_match = re.search(r'epoch[:\s]+(\d+)/(\d+)', line, re.IGNORECASE)
                            if epoch_match:
                                learning_metrics['current_epoch'] = int(epoch_match.group(1))
                                learning_metrics['total_epochs'] = int(epoch_match.group(2))
                                
                            # Loss 파싱
                            loss_match = re.search(r'loss[:\s]+([\d.]+)', line, re.IGNORECASE)
                            if loss_match:
                                learning_metrics['loss'] = float(loss_match.group(1))
                                
                            # Accuracy 파싱
                            acc_match = re.search(r'acc(?:uracy)?[:\s]+([\d.]+)', line, re.IGNORECASE)
                            if acc_match:
                                learning_metrics['accuracy'] = float(acc_match.group(1))
                                
                            break
                            
                except Exception as e:
                    logger.debug(f"로그 파일 읽기 오류: {e}")
                    
        return learning_metrics
        
    def save_metrics(self, metrics: Dict):
        """메트릭을 데이터베이스에 저장"""
        cursor = self.conn.cursor()
        
        # 주요 메트릭 저장
        cursor.execute('''
            INSERT INTO metrics (
                cpu_percent, memory_percent, memory_used_gb,
                disk_percent, gpu_percent, gpu_memory_mb,
                active_processes, learning_status, error_count, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['cpu_percent'],
            metrics['memory'].percent,
            metrics['memory'].used / (1024**3),
            metrics['disk'].percent,
            metrics['gpu']['gpu_percent'],
            metrics['gpu']['memory_used_mb'],
            len(metrics['processes']),
            'training' if metrics['learning']['is_training'] else 'idle',
            0,  # TODO: 에러 카운트 구현
            json.dumps(metrics)
        ))
        
        # 학습 진행률 저장
        if metrics['learning']['is_training']:
            cursor.execute('''
                INSERT INTO learning_progress (
                    model_name, epoch, total_epochs,
                    loss, accuracy, learning_rate, eta_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'AutoCI Model',
                metrics['learning']['current_epoch'],
                metrics['learning']['total_epochs'],
                metrics['learning']['loss'],
                metrics['learning']['accuracy'],
                metrics['learning']['learning_rate'],
                metrics['learning']['eta_minutes'] * 60
            ))
        
        self.conn.commit()
        
    def check_thresholds(self, metrics: Dict):
        """임계값 체크 및 알림 생성"""
        alerts = []
        
        # CPU 사용률 체크
        if metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'severity': 'warning',
                'category': 'resource',
                'message': f"CPU 사용률이 높습니다: {metrics['cpu_percent']:.1f}%"
            })
            
        # 메모리 사용률 체크
        if metrics['memory'].percent > self.thresholds['memory_percent']:
            alerts.append({
                'severity': 'warning',
                'category': 'resource',
                'message': f"메모리 사용률이 높습니다: {metrics['memory'].percent:.1f}%"
            })
            
        # 디스크 사용률 체크
        if metrics['disk'].percent > self.thresholds['disk_percent']:
            alerts.append({
                'severity': 'critical',
                'category': 'resource',
                'message': f"디스크 공간이 부족합니다: {metrics['disk'].percent:.1f}%"
            })
            
        # GPU 메모리 체크
        if metrics['gpu']['available']:
            gpu_memory_percent = (metrics['gpu']['memory_used_mb'] / 
                                metrics['gpu']['memory_total_mb'] * 100)
            if gpu_memory_percent > self.thresholds['gpu_memory_percent']:
                alerts.append({
                    'severity': 'warning',
                    'category': 'gpu',
                    'message': f"GPU 메모리 사용률이 높습니다: {gpu_memory_percent:.1f}%"
                })
                
        # 학습 상태 체크
        if metrics['learning']['is_training']:
            # 학습률이 너무 낮은 경우
            if (metrics['learning']['learning_rate'] > 0 and 
                metrics['learning']['learning_rate'] < self.thresholds['learning_rate_drop']):
                alerts.append({
                    'severity': 'info',
                    'category': 'learning',
                    'message': f"학습률이 매우 낮습니다: {metrics['learning']['learning_rate']:.6f}"
                })
                
        # 알림 저장 및 처리
        for alert in alerts:
            self.save_alert(alert)
            self.process_alert(alert)
            
    def save_alert(self, alert: Dict):
        """알림 저장"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (severity, category, message, data)
            VALUES (?, ?, ?, ?)
        ''', (
            alert['severity'],
            alert['category'],
            alert['message'],
            json.dumps(alert)
        ))
        self.conn.commit()
        
        # 메모리에도 저장
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            **alert
        })
        
    def process_alert(self, alert: Dict):
        """알림 처리 (로깅, 알림 등)"""
        # 심각도에 따른 로깅
        if alert['severity'] == 'critical':
            logger.critical(f"[{alert['category']}] {alert['message']}")
        elif alert['severity'] == 'warning':
            logger.warning(f"[{alert['category']}] {alert['message']}")
        else:
            logger.info(f"[{alert['category']}] {alert['message']}")
            
    def check_learning_status(self):
        """학습 상태 상세 체크"""
        # 학습 프로세스 확인
        learning_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.cmdline()).lower()
                if any(keyword in cmdline for keyword in 
                      ['train', 'fine_tune', 'learning', 'enhanced_server']):
                    learning_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # 학습 프로세스가 없는데 이전에는 있었던 경우
        if not learning_processes and hasattr(self, '_last_learning_check'):
            if self._last_learning_check:
                self.save_alert({
                    'severity': 'info',
                    'category': 'learning',
                    'message': '학습 프로세스가 종료되었습니다.'
                })
                
        self._last_learning_check = bool(learning_processes)
        
    def print_summary(self, metrics: Dict):
        """콘솔에 요약 정보 출력"""
        print(f"\n{'='*60}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AI 학습 환경 모니터링")
        print(f"{'='*60}")
        
        # 시스템 리소스
        print(f"📊 시스템 리소스:")
        print(f"   CPU: {metrics['cpu_percent']:.1f}% | "
              f"메모리: {metrics['memory'].percent:.1f}% "
              f"({metrics['memory'].used / (1024**3):.1f}GB / "
              f"{metrics['memory'].total / (1024**3):.1f}GB)")
        
        # GPU 정보
        if metrics['gpu']['available']:
            print(f"   GPU: {metrics['gpu']['gpu_percent']:.1f}% | "
                  f"VRAM: {metrics['gpu']['memory_used_mb']:.0f}MB / "
                  f"{metrics['gpu']['memory_total_mb']:.0f}MB | "
                  f"온도: {metrics['gpu']['temperature']:.0f}°C")
        
        # AI 프로세스
        if metrics['processes']:
            print(f"\n🤖 활성 AI 프로세스: {len(metrics['processes'])}개")
            for proc in metrics['processes'][:3]:  # 상위 3개만
                print(f"   - {proc['name']} (PID: {proc['pid']}) - "
                      f"CPU: {proc['cpu_percent']:.1f}% | "
                      f"메모리: {proc['memory_percent']:.1f}%")
        
        # 학습 상태
        if metrics['learning']['is_training']:
            print(f"\n📚 학습 진행 중:")
            print(f"   Epoch: {metrics['learning']['current_epoch']}/{metrics['learning']['total_epochs']} | "
                  f"Loss: {metrics['learning']['loss']:.4f} | "
                  f"정확도: {metrics['learning']['accuracy']:.2%}")
        else:
            print(f"\n💤 학습 대기 중")
        
        # 최근 알림
        if self.alerts:
            recent_alerts = list(self.alerts)[-3:]  # 최근 3개
            if recent_alerts:
                print(f"\n⚠️  최근 알림:")
                for alert in recent_alerts:
                    icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(alert['severity'], '⚪')
                    print(f"   {icon} {alert['message']}")
        
        print(f"{'='*60}")
        
    def start_web_server(self):
        """웹 대시보드 서버 시작"""
        server_thread = threading.Thread(
            target=self._run_web_server,
            daemon=True
        )
        server_thread.start()
        logger.info("웹 대시보드: http://localhost:8888")
        
    def _run_web_server(self):
        """웹 서버 실행"""
        class MonitorHandler(BaseHTTPRequestHandler):
            monitor = self
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(self.monitor.get_dashboard_html().encode())
                    
                elif self.path == '/api/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    # 최근 메트릭 반환
                    recent_metrics = list(self.monitor.metrics_history)[-60:]  # 최근 1시간
                    self.wfile.write(json.dumps(recent_metrics).encode())
                    
                elif self.path == '/api/alerts':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    alerts = list(self.monitor.alerts)
                    self.wfile.write(json.dumps(alerts).encode())
                    
                else:
                    self.send_error(404)
                    
            def log_message(self, format, *args):
                pass  # 로그 억제
        
        MonitorHandler.monitor = self
        server = HTTPServer(('localhost', 8888), MonitorHandler)
        server.serve_forever()
        
    def get_dashboard_html(self) -> str:
        """대시보드 HTML 생성"""
        # 최신 메트릭
        latest = self.metrics_history[-1] if self.metrics_history else None
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AutoCI AI 학습 모니터링</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="60">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-detail {{
            font-size: 14px;
            color: #999;
        }}
        .alert-box {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .alert-critical {{
            background: #f8d7da;
            border-color: #f5c6cb;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            position: relative;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            background: #4caf50;
            height: 100%;
            transition: width 0.3s ease;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-active {{
            background: #4caf50;
            animation: pulse 2s infinite;
        }}
        .status-idle {{
            background: #999;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .timestamp {{
            text-align: right;
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AutoCI AI 학습 환경 모니터링</h1>
        <div class="timestamp">
            마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            <br>자동 새로고침: 60초
        </div>
"""
        
        if latest:
            # 메트릭 카드들
            html += '<div class="metrics-grid">'
            
            # CPU 사용률
            html += f'''
            <div class="metric-card">
                <div class="metric-title">CPU 사용률</div>
                <div class="metric-value">{latest["cpu_percent"]:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {latest["cpu_percent"]}%; 
                         background: {"#ff5252" if latest["cpu_percent"] > 80 else "#4caf50"};"></div>
                </div>
            </div>
            '''
            
            # 메모리 사용률
            memory_percent = latest["memory"].percent
            memory_used_gb = latest["memory"].used / (1024**3)
            memory_total_gb = latest["memory"].total / (1024**3)
            
            html += f'''
            <div class="metric-card">
                <div class="metric-title">메모리 사용률</div>
                <div class="metric-value">{memory_percent:.1f}%</div>
                <div class="metric-detail">{memory_used_gb:.1f} / {memory_total_gb:.1f} GB</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {memory_percent}%;
                         background: {"#ff5252" if memory_percent > 85 else "#4caf50"};"></div>
                </div>
            </div>
            '''
            
            # GPU 상태
            if latest["gpu"]["available"]:
                gpu_memory_percent = (latest["gpu"]["memory_used_mb"] / 
                                    latest["gpu"]["memory_total_mb"] * 100)
                html += f'''
                <div class="metric-card">
                    <div class="metric-title">GPU 상태</div>
                    <div class="metric-value">{latest["gpu"]["gpu_percent"]:.0f}%</div>
                    <div class="metric-detail">
                        VRAM: {latest["gpu"]["memory_used_mb"]:.0f} / {latest["gpu"]["memory_total_mb"]:.0f} MB
                        | 온도: {latest["gpu"]["temperature"]:.0f}°C
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {gpu_memory_percent}%;
                             background: {"#ff5252" if gpu_memory_percent > 90 else "#4caf50"};"></div>
                    </div>
                </div>
                '''
            
            # 학습 상태
            learning = latest["learning"]
            status_class = "status-active" if learning["is_training"] else "status-idle"
            status_text = "학습 중" if learning["is_training"] else "대기 중"
            
            html += f'''
            <div class="metric-card">
                <div class="metric-title">학습 상태</div>
                <div style="display: flex; align-items: center;">
                    <span class="status-indicator {status_class}"></span>
                    <div class="metric-value" style="font-size: 24px;">{status_text}</div>
                </div>
            '''
            
            if learning["is_training"]:
                progress = (learning["current_epoch"] / learning["total_epochs"] * 100) if learning["total_epochs"] > 0 else 0
                html += f'''
                <div class="metric-detail">
                    Epoch: {learning["current_epoch"]} / {learning["total_epochs"]}
                    | Loss: {learning["loss"]:.4f}
                    | 정확도: {learning["accuracy"]:.2%}
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%;"></div>
                </div>
                '''
                
            html += '</div>'
            
            # AI 프로세스
            html += f'''
            <div class="metric-card">
                <div class="metric-title">활성 AI 프로세스</div>
                <div class="metric-value">{len(latest["processes"])}</div>
                <div class="metric-detail">
            '''
            
            for proc in latest["processes"][:3]:
                html += f'{proc["name"]} (CPU: {proc["cpu_percent"]:.1f}%)<br>'
                
            html += '</div></div>'
            
            html += '</div>'  # metrics-grid 끝
            
            # 최근 알림
            if self.alerts:
                html += '<h2>⚠️ 최근 알림</h2>'
                for alert in list(self.alerts)[-10:]:  # 최근 10개
                    alert_class = 'alert-critical' if alert['severity'] == 'critical' else 'alert-box'
                    html += f'''
                    <div class="{alert_class}">
                        <strong>[{alert["severity"].upper()}]</strong> {alert["message"]}
                        <span style="float: right; color: #999;">{alert.get("timestamp", "")}</span>
                    </div>
                    '''
        else:
            html += '<p>데이터 수집 중... 잠시 기다려주세요.</p>'
            
        html += '''
    </div>
    <script>
        // 60초마다 자동 새로고침
        setTimeout(() => location.reload(), 60000);
    </script>
</body>
</html>
'''
        return html
        
    def get_summary_report(self) -> str:
        """요약 리포트 생성"""
        cursor = self.conn.cursor()
        
        # 최근 1시간 평균
        cursor.execute('''
            SELECT 
                AVG(cpu_percent) as avg_cpu,
                AVG(memory_percent) as avg_memory,
                AVG(gpu_percent) as avg_gpu,
                COUNT(DISTINCT learning_status) as status_changes
            FROM metrics
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        
        hour_stats = cursor.fetchone()
        
        # 최근 24시간 평균
        cursor.execute('''
            SELECT 
                AVG(cpu_percent) as avg_cpu,
                AVG(memory_percent) as avg_memory,
                MAX(cpu_percent) as max_cpu,
                MAX(memory_percent) as max_memory
            FROM metrics
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        
        day_stats = cursor.fetchone()
        
        # 알림 통계
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM alerts
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY severity
        ''')
        
        alert_stats = cursor.fetchall()
        
        report = f"""
# AI 학습 환경 모니터링 리포트
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 최근 1시간 요약
- 평균 CPU 사용률: {hour_stats[0]:.1f}%
- 평균 메모리 사용률: {hour_stats[1]:.1f}%
- 평균 GPU 사용률: {hour_stats[2]:.1f}%
- 상태 변경 횟수: {hour_stats[3]}회

## 📈 최근 24시간 통계
- 평균 CPU: {day_stats[0]:.1f}% (최대: {day_stats[2]:.1f}%)
- 평균 메모리: {day_stats[1]:.1f}% (최대: {day_stats[3]:.1f}%)

## ⚠️ 알림 통계 (24시간)
"""
        
        for severity, count in alert_stats:
            report += f"- {severity.upper()}: {count}건\n"
            
        return report


def main():
    """메인 함수"""
    monitor = AILearningMonitor()
    
    print("🤖 AutoCI AI 학습 환경 모니터링 시스템")
    print("=" * 60)
    print("📊 1분마다 AI 모델의 학습환경을 모니터링합니다.")
    print("🌐 웹 대시보드: http://localhost:8888")
    print("📝 로그 파일: ai_learning_monitor.log")
    print("⏹️  종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    try:
        monitor.start()
        
        # 메인 스레드는 계속 실행
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n\n모니터링을 종료합니다...")
        monitor.stop()
        
        # 최종 리포트 출력
        print("\n" + monitor.get_summary_report())
        
    except Exception as e:
        logger.error(f"모니터링 중 오류 발생: {e}")
        monitor.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()