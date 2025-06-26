#!/usr/bin/env python3
"""
강력한 이중 단계 시스템
실시간 모니터링과 함께 RAG + 파인튜닝 동시 실행
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import subprocess
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import aiohttp
import aiofiles
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import sqlite3
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_dual_phase.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """시스템 상태"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]
    disk_usage: float
    network_io: Dict[str, float]
    rag_queries_per_minute: int
    training_loss: Optional[float]
    training_accuracy: Optional[float]
    processed_chunks: int
    active_connections: int


class RobustDualPhase:
    """강력한 이중 단계 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config_path = self.base_path / "robust_dual_phase_config.json"
        self.state_path = self.base_path / "robust_dual_phase_state.json"
        
        # 디렉토리 구조
        self.dirs = {
            'logs': self.base_path / 'dual_phase_logs',
            'checkpoints': self.base_path / 'models' / 'checkpoints',
            'metrics': self.base_path / 'dual_phase_metrics',
            'cache': self.base_path / 'dual_phase_cache'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 설정 로드
        self.config = self.load_config()
        
        # 시스템 상태
        self.status = SystemStatus.INITIALIZING
        self.start_time = None
        self.rag_process = None
        self.training_process = None
        self.monitor_process = None
        
        # 메트릭 큐
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.event_queue = queue.Queue(maxsize=1000)
        
        # 웹소켓 연결
        self.websocket_clients = set()
        
        # FastAPI 앱
        self.app = self.create_api_app()
        
        # 데이터베이스 초기화
        self.init_database()
        
    def load_config(self) -> Dict:
        """설정 로드"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            default_config = {
                # RAG 설정
                'rag': {
                    'enabled': True,
                    'port': 8000,
                    'model_name': 'CodeLlama-7b-Instruct-hf',
                    'max_context': 4096,
                    'temperature': 0.7,
                    'cache_size': 1000,
                    'vector_index_path': 'expert_learning_data/vector_index'
                },
                # 파인튜닝 설정
                'training': {
                    'enabled': True,
                    'batch_size': 4,
                    'learning_rate': 2e-5,
                    'num_epochs': 3,
                    'warmup_steps': 500,
                    'gradient_accumulation_steps': 4,
                    'save_steps': 1000,
                    'eval_steps': 500,
                    'max_seq_length': 2048,
                    'use_gpu': torch.cuda.is_available(),
                    'mixed_precision': True
                },
                # 모니터링 설정
                'monitoring': {
                    'enabled': True,
                    'web_ui_port': 8080,
                    'metrics_interval': 10,  # 초
                    'alert_thresholds': {
                        'cpu_percent': 90,
                        'memory_percent': 85,
                        'gpu_percent': 95,
                        'disk_usage': 90
                    },
                    'enable_prometheus': True,
                    'enable_grafana': False
                },
                # 시스템 설정
                'system': {
                    'auto_restart': True,
                    'max_retries': 3,
                    'health_check_interval': 30,
                    'log_level': 'INFO',
                    'enable_profiling': False,
                    'backup_interval': 3600  # 1시간
                }
            }
            self.save_config(default_config)
            return default_config
            
    def save_config(self, config: Dict):
        """설정 저장"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def init_database(self):
        """데이터베이스 초기화"""
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 메트릭 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            cpu_percent REAL,
            memory_percent REAL,
            gpu_percent REAL,
            disk_usage REAL,
            network_in REAL,
            network_out REAL,
            rag_qpm INTEGER,
            training_loss REAL,
            training_accuracy REAL,
            processed_chunks INTEGER,
            active_connections INTEGER
        )
        ''')
        
        # 이벤트 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            event_type TEXT,
            severity TEXT,
            component TEXT,
            message TEXT,
            details TEXT
        )
        ''')
        
        # RAG 쿼리 로그
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            query TEXT,
            response_time REAL,
            tokens_used INTEGER,
            cache_hit BOOLEAN,
            quality_score REAL
        )
        ''')
        
        # 학습 체크포인트
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            accuracy REAL,
            learning_rate REAL,
            checkpoint_path TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def create_api_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        app = FastAPI(title="AutoCI Dual Phase System")
        
        @app.get("/")
        async def root():
            """웹 UI"""
            return HTMLResponse(self.get_web_ui_html())
            
        @app.get("/api/status")
        async def get_status():
            """시스템 상태"""
            return JSONResponse(self.get_system_status())
            
        @app.get("/api/metrics")
        async def get_metrics():
            """최신 메트릭"""
            return JSONResponse(self.get_latest_metrics())
            
        @app.post("/api/control/{action}")
        async def control_system(action: str):
            """시스템 제어"""
            if action == "pause":
                self.pause_system()
            elif action == "resume":
                self.resume_system()
            elif action == "restart":
                await self.restart_system()
            else:
                return JSONResponse({"error": "Invalid action"}, status_code=400)
                
            return JSONResponse({"status": "ok", "action": action})
            
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """실시간 업데이트"""
            await websocket.accept()
            self.websocket_clients.add(websocket)
            
            try:
                while True:
                    # 클라이언트 메시지 대기
                    data = await websocket.receive_text()
                    
                    # 브로드캐스트
                    await self.broadcast_to_clients(data)
                    
            except Exception as e:
                logger.debug(f"WebSocket 연결 종료: {e}")
            finally:
                self.websocket_clients.remove(websocket)
                
        return app
        
    async def start_system(self):
        """전체 시스템 시작"""
        logger.info("🚀 강력한 이중 단계 시스템 시작")
        
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        
        # 상태 저장
        self.save_state({
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'pid': os.getpid(),
            'config': self.config
        })
        
        try:
            # 사전 체크
            await self.pre_start_checks()
            
            # 병렬 태스크 시작
            tasks = []
            
            # 1. RAG 시스템 (즉시)
            if self.config['rag']['enabled']:
                tasks.append(self.start_rag_system())
                
            # 2. 파인튜닝 (백그라운드)
            if self.config['training']['enabled']:
                tasks.append(self.start_training_system())
                
            # 3. 모니터링
            if self.config['monitoring']['enabled']:
                tasks.append(self.start_monitoring_system())
                
            # 4. 웹 UI
            tasks.append(self.start_web_ui())
            
            # 5. 메인 루프
            tasks.append(self.main_loop())
            
            self.status = SystemStatus.RUNNING
            
            # 모든 태스크 실행
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("🛑 종료 신호 받음")
            await self.shutdown()
        except Exception as e:
            logger.error(f"시스템 오류: {e}")
            self.status = SystemStatus.ERROR
            await self.handle_system_error(e)
            
    async def pre_start_checks(self):
        """시작 전 체크"""
        logger.info("🔍 시스템 사전 체크...")
        
        # 데이터 존재 확인
        data_path = self.base_path / "expert_learning_data"
        if not data_path.exists() or not any(data_path.iterdir()):
            logger.warning("📥 학습 데이터가 없습니다. 수집을 시작합니다...")
            await self.collect_initial_data()
            
        # 벡터 인덱스 확인
        vector_index = Path(self.config['rag']['vector_index_path'])
        if not vector_index.exists():
            logger.warning("🔍 벡터 인덱스가 없습니다. 생성합니다...")
            await self.create_vector_index()
            
        # 모델 확인
        model_path = self.base_path / self.config['rag']['model_name']
        if not model_path.exists():
            logger.warning("🤖 모델이 없습니다. 다운로드가 필요합니다.")
            # 모델 다운로드는 별도 처리
            
        # 리소스 확인
        self.check_system_resources()
        
    async def collect_initial_data(self):
        """초기 데이터 수집"""
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.base_path / "deep_csharp_collector.py"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 데이터 수집 완료")
            else:
                logger.error(f"데이터 수집 실패: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            
    async def create_vector_index(self):
        """벡터 인덱스 생성"""
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.base_path / "vector_indexer.py"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 벡터 인덱스 생성 완료")
            else:
                logger.error(f"인덱스 생성 실패: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"인덱스 생성 오류: {e}")
            
    def check_system_resources(self):
        """시스템 리소스 확인"""
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # 디스크
        disk = psutil.disk_usage(str(self.base_path))
        
        logger.info(f"💻 시스템 리소스:")
        logger.info(f"  - CPU: {cpu_count}코어, 사용률 {cpu_percent}%")
        logger.info(f"  - 메모리: {available_gb:.1f}GB 사용 가능")
        logger.info(f"  - 디스크: {disk.free / (1024**3):.1f}GB 여유")
        
        # GPU (있는 경우)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"  - GPU: {gpu_name}, {gpu_memory:.1f}GB")
            
        # 경고
        if cpu_percent > 80:
            logger.warning("⚠️ CPU 사용률이 높습니다")
        if available_gb < 4:
            logger.warning("⚠️ 메모리가 부족할 수 있습니다")
        if disk.percent > 90:
            logger.warning("⚠️ 디스크 공간이 부족합니다")
            
    async def start_rag_system(self):
        """RAG 시스템 시작"""
        logger.info("📚 RAG 시스템 시작...")
        
        try:
            # enhanced_rag_server.py 생성
            await self.create_enhanced_rag_server()
            
            # 서버 프로세스 시작
            self.rag_process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.base_path / "enhanced_rag_server.py"),
                "--port", str(self.config['rag']['port']),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 이벤트 기록
            self.log_event("rag_started", "info", "RAG", "RAG 시스템 시작됨")
            
            # 프로세스 모니터링
            while self.status == SystemStatus.RUNNING:
                if self.rag_process.returncode is not None:
                    # 프로세스 종료됨
                    logger.error("RAG 프로세스가 종료되었습니다")
                    
                    if self.config['system']['auto_restart']:
                        logger.info("RAG 재시작 중...")
                        await asyncio.sleep(5)
                        await self.start_rag_system()
                    break
                    
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"RAG 시작 오류: {e}")
            self.log_event("rag_error", "error", "RAG", str(e))
            
    async def create_enhanced_rag_server(self):
        """향상된 RAG 서버 생성"""
        server_code = '''#!/usr/bin/env python3
"""
향상된 RAG 서버
벡터 검색 기반 고품질 응답 생성
"""

import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
sys.path.append(str(Path(__file__).parent))

from vector_indexer import VectorIndexer

app = FastAPI(title="Enhanced RAG Server")
indexer = VectorIndexer()

class Query(BaseModel):
    query: str
    k: int = 5
    filters: dict = {}

@app.post("/query")
async def process_query(query: Query):
    try:
        # 벡터 검색
        results = indexer.search(query.query, query.k, query.filters)
        
        # 응답 생성
        response = {
            "query": query.query,
            "results": []
        }
        
        for chunk, similarity in results:
            response["results"].append({
                "content": chunk.content[:500],
                "similarity": similarity,
                "category": chunk.category,
                "type": chunk.chunk_type,
                "quality": chunk.quality_score
            })
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
'''
        
        server_path = self.base_path / "enhanced_rag_server.py"
        with open(server_path, 'w', encoding='utf-8') as f:
            f.write(server_code)
            
    async def start_training_system(self):
        """파인튜닝 시스템 시작"""
        logger.info("🎯 파인튜닝 시스템 시작...")
        
        # RAG가 시작될 때까지 대기
        await asyncio.sleep(10)
        
        try:
            # 학습 스크립트 실행
            cmd = [
                sys.executable,
                str(self.base_path / "hybrid_rag_training_system.py"),
                "--config", json.dumps(self.config['training'])
            ]
            
            self.training_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 이벤트 기록
            self.log_event("training_started", "info", "Training", "파인튜닝 시작됨")
            
            # 학습 진행 모니터링
            async for line in self.training_process.stdout:
                line_str = line.decode().strip()
                
                # 학습 메트릭 파싱
                if "loss:" in line_str:
                    self.parse_training_metrics(line_str)
                    
                # 체크포인트 저장
                if "checkpoint saved" in line_str.lower():
                    self.log_event("checkpoint_saved", "info", "Training", line_str)
                    
            # 학습 완료
            if self.training_process.returncode == 0:
                logger.info("✅ 파인튜닝 완료")
                self.log_event("training_completed", "info", "Training", "파인튜닝 성공적으로 완료")
                
                # 새 모델로 RAG 재시작
                if self.config['system']['auto_restart']:
                    await self.restart_rag_with_new_model()
            else:
                logger.error("❌ 파인튜닝 실패")
                self.log_event("training_failed", "error", "Training", "파인튜닝 실패")
                
        except Exception as e:
            logger.error(f"학습 시작 오류: {e}")
            self.log_event("training_error", "error", "Training", str(e))
            
    def parse_training_metrics(self, line: str):
        """학습 메트릭 파싱"""
        try:
            # 간단한 파싱 (실제로는 더 복잡)
            import re
            
            loss_match = re.search(r'loss:\s*([\d.]+)', line)
            acc_match = re.search(r'accuracy:\s*([\d.]+)', line)
            
            metrics = {}
            if loss_match:
                metrics['training_loss'] = float(loss_match.group(1))
            if acc_match:
                metrics['training_accuracy'] = float(acc_match.group(1))
                
            if metrics:
                self.update_metrics(metrics)
                
        except Exception as e:
            logger.debug(f"메트릭 파싱 오류: {e}")
            
    async def restart_rag_with_new_model(self):
        """새 모델로 RAG 재시작"""
        logger.info("🔄 새 모델로 RAG 재시작...")
        
        # 기존 RAG 종료
        if self.rag_process:
            self.rag_process.terminate()
            await self.rag_process.wait()
            
        # 새 모델 경로 설정
        # (실제 구현 필요)
        
        # RAG 재시작
        await asyncio.sleep(5)
        await self.start_rag_system()
        
    async def start_monitoring_system(self):
        """모니터링 시스템 시작"""
        logger.info("📊 모니터링 시스템 시작...")
        
        while self.status == SystemStatus.RUNNING:
            try:
                # 시스템 메트릭 수집
                metrics = await self.collect_system_metrics()
                
                # 메트릭 저장
                self.save_metrics(metrics)
                
                # 큐에 추가
                if not self.metrics_queue.full():
                    self.metrics_queue.put(asdict(metrics))
                    
                # 웹소켓으로 브로드캐스트
                await self.broadcast_metrics(metrics)
                
                # 임계값 체크
                self.check_alert_thresholds(metrics)
                
                # 대기
                await asyncio.sleep(self.config['monitoring']['metrics_interval'])
                
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                await asyncio.sleep(60)
                
    async def collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        # CPU, 메모리
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU (있는 경우)
        gpu_percent = None
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = util.gpu
            except:
                pass
                
        # 디스크
        disk = psutil.disk_usage(str(self.base_path))
        
        # 네트워크
        net_io = psutil.net_io_counters()
        
        # RAG 쿼리 수 (최근 1분)
        rag_qpm = self.get_rag_queries_per_minute()
        
        # 활성 연결
        active_connections = len(self.websocket_clients)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_percent=gpu_percent,
            disk_usage=disk.percent,
            network_io={
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            },
            rag_queries_per_minute=rag_qpm,
            training_loss=None,  # 별도 업데이트
            training_accuracy=None,
            processed_chunks=0,  # 별도 계산
            active_connections=active_connections
        )
        
    def get_rag_queries_per_minute(self) -> int:
        """분당 RAG 쿼리 수"""
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 최근 1분간 쿼리 수
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        cursor.execute('''
        SELECT COUNT(*) FROM rag_queries 
        WHERE timestamp > ?
        ''', (one_minute_ago,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
        
    def save_metrics(self, metrics: SystemMetrics):
        """메트릭 저장"""
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO system_metrics
        (timestamp, cpu_percent, memory_percent, gpu_percent, disk_usage,
         network_in, network_out, rag_qpm, training_loss, training_accuracy,
         processed_chunks, active_connections)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp,
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.gpu_percent,
            metrics.disk_usage,
            metrics.network_io.get('bytes_recv', 0),
            metrics.network_io.get('bytes_sent', 0),
            metrics.rag_queries_per_minute,
            metrics.training_loss,
            metrics.training_accuracy,
            metrics.processed_chunks,
            metrics.active_connections
        ))
        
        conn.commit()
        conn.close()
        
    async def broadcast_metrics(self, metrics: SystemMetrics):
        """웹소켓으로 메트릭 브로드캐스트"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': 'metrics',
            'data': {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu': metrics.cpu_percent,
                'memory': metrics.memory_percent,
                'gpu': metrics.gpu_percent,
                'disk': metrics.disk_usage,
                'rag_qpm': metrics.rag_queries_per_minute
            }
        }
        
        # 연결이 끊긴 클라이언트 제거
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except:
                disconnected.add(client)
                
        self.websocket_clients -= disconnected
        
    def check_alert_thresholds(self, metrics: SystemMetrics):
        """임계값 체크 및 알림"""
        thresholds = self.config['monitoring']['alert_thresholds']
        
        # CPU
        if metrics.cpu_percent > thresholds['cpu_percent']:
            self.log_event(
                "high_cpu", "warning", "System",
                f"CPU 사용률 높음: {metrics.cpu_percent}%"
            )
            
        # 메모리
        if metrics.memory_percent > thresholds['memory_percent']:
            self.log_event(
                "high_memory", "warning", "System",
                f"메모리 사용률 높음: {metrics.memory_percent}%"
            )
            
        # GPU
        if metrics.gpu_percent and metrics.gpu_percent > thresholds['gpu_percent']:
            self.log_event(
                "high_gpu", "warning", "System",
                f"GPU 사용률 높음: {metrics.gpu_percent}%"
            )
            
        # 디스크
        if metrics.disk_usage > thresholds['disk_usage']:
            self.log_event(
                "high_disk", "warning", "System",
                f"디스크 사용률 높음: {metrics.disk_usage}%"
            )
            
    async def start_web_ui(self):
        """웹 UI 시작"""
        logger.info(f"🌐 웹 UI 시작 (포트: {self.config['monitoring']['web_ui_port']})")
        
        # uvicorn 서버 설정
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config['monitoring']['web_ui_port'],
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    def get_web_ui_html(self) -> str:
        """웹 UI HTML"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>AutoCI Dual Phase System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .status { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        .status.running { background: #2ecc71; color: white; }
        .status.stopped { background: #e74c3c; color: white; }
        .chart { height: 300px; background: white; border-radius: 5px; padding: 20px; margin: 20px 0; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        button.primary { background: #3498db; color: white; }
        button.danger { background: #e74c3c; color: white; }
        .log { background: #2c3e50; color: #2ecc71; padding: 15px; border-radius: 5px; font-family: monospace; height: 200px; overflow-y: auto; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AutoCI Dual Phase System</h1>
            <p>실시간 모니터링 대시보드</p>
            <span id="status" class="status">연결 중...</span>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value" id="cpu">-</div>
                <div class="metric-label">CPU 사용률</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="memory">-</div>
                <div class="metric-label">메모리 사용률</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="gpu">-</div>
                <div class="metric-label">GPU 사용률</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="rag_qpm">-</div>
                <div class="metric-label">RAG 쿼리/분</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="primary" onclick="controlSystem('pause')">⏸ 일시정지</button>
            <button class="primary" onclick="controlSystem('resume')">▶ 재개</button>
            <button class="danger" onclick="controlSystem('restart')">🔄 재시작</button>
        </div>
        
        <div class="chart">
            <canvas id="metricsChart"></canvas>
        </div>
        
        <div class="log" id="log">
            시스템 로그가 여기에 표시됩니다...
        </div>
    </div>
    
    <script>
        // WebSocket 연결
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            document.getElementById('status').textContent = '연결됨';
            document.getElementById('status').className = 'status running';
        };
        
        ws.onclose = () => {
            document.getElementById('status').textContent = '연결 끊김';
            document.getElementById('status').className = 'status stopped';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'metrics') {
                updateMetrics(data.data);
            } else if (data.type === 'log') {
                addLog(data.message);
            }
        };
        
        // 메트릭 업데이트
        function updateMetrics(metrics) {
            document.getElementById('cpu').textContent = metrics.cpu.toFixed(1) + '%';
            document.getElementById('memory').textContent = metrics.memory.toFixed(1) + '%';
            document.getElementById('gpu').textContent = metrics.gpu ? metrics.gpu.toFixed(1) + '%' : 'N/A';
            document.getElementById('rag_qpm').textContent = metrics.rag_qpm;
            
            // 차트 업데이트
            updateChart(metrics);
        }
        
        // 로그 추가
        function addLog(message) {
            const log = document.getElementById('log');
            log.innerHTML += message + '<br>';
            log.scrollTop = log.scrollHeight;
        }
        
        // 시스템 제어
        async function controlSystem(action) {
            try {
                const response = await fetch(`/api/control/${action}`, { method: 'POST' });
                const result = await response.json();
                addLog(`시스템 ${action} 명령 실행됨`);
            } catch (error) {
                addLog(`오류: ${error.message}`);
            }
        }
        
        // 차트 초기화
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#3498db',
                    tension: 0.1
                }, {
                    label: '메모리 %',
                    data: [],
                    borderColor: '#2ecc71',
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
        
        // 차트 업데이트
        function updateChart(metrics) {
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(metrics.cpu);
            chart.data.datasets[1].data.push(metrics.memory);
            
            // 최대 20개 데이터만 유지
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update();
        }
        
        // 초기 상태 로드
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                addLog(`시스템 상태: ${data.status}`);
            });
    </script>
</body>
</html>'''
        
    async def main_loop(self):
        """메인 루프"""
        while self.status == SystemStatus.RUNNING:
            try:
                # 헬스 체크
                await self.health_check()
                
                # 백업 (설정된 간격마다)
                if self.should_backup():
                    await self.backup_system()
                    
                # 이벤트 처리
                self.process_events()
                
                # 대기
                await asyncio.sleep(self.config['system']['health_check_interval'])
                
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                await asyncio.sleep(60)
                
    async def health_check(self):
        """시스템 헬스 체크"""
        checks = {
            'rag': self.check_rag_health(),
            'training': self.check_training_health(),
            'database': self.check_database_health(),
            'disk_space': self.check_disk_space()
        }
        
        for component, is_healthy in checks.items():
            if not is_healthy:
                self.log_event(
                    f"{component}_unhealthy", "warning", component,
                    f"{component} 헬스 체크 실패"
                )
                
    def check_rag_health(self) -> bool:
        """RAG 시스템 헬스 체크"""
        if not self.rag_process:
            return False
            
        return self.rag_process.returncode is None
        
    def check_training_health(self) -> bool:
        """학습 시스템 헬스 체크"""
        if not self.config['training']['enabled']:
            return True
            
        if not self.training_process:
            return False
            
        return self.training_process.returncode is None
        
    def check_database_health(self) -> bool:
        """데이터베이스 헬스 체크"""
        try:
            db_path = self.dirs['metrics'] / 'metrics.db'
            conn = sqlite3.connect(db_path)
            conn.execute('SELECT 1')
            conn.close()
            return True
        except:
            return False
            
    def check_disk_space(self) -> bool:
        """디스크 공간 체크"""
        disk = psutil.disk_usage(str(self.base_path))
        return disk.percent < 95
        
    def should_backup(self) -> bool:
        """백업 필요 여부"""
        backup_file = self.dirs['cache'] / 'last_backup.txt'
        
        if not backup_file.exists():
            return True
            
        with open(backup_file, 'r') as f:
            last_backup = datetime.fromisoformat(f.read().strip())
            
        return (datetime.now() - last_backup).seconds > self.config['system']['backup_interval']
        
    async def backup_system(self):
        """시스템 백업"""
        logger.info("💾 시스템 백업 중...")
        
        try:
            # 중요 파일 백업
            backup_dir = self.base_path / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 설정 파일
            import shutil
            shutil.copy(self.config_path, backup_dir / 'config.json')
            
            # 데이터베이스
            for db_file in self.dirs['metrics'].glob('*.db'):
                shutil.copy(db_file, backup_dir / db_file.name)
                
            # 체크포인트 (최신 것만)
            if self.dirs['checkpoints'].exists():
                latest_checkpoint = max(self.dirs['checkpoints'].iterdir(), 
                                      key=lambda x: x.stat().st_mtime, default=None)
                if latest_checkpoint:
                    shutil.copytree(latest_checkpoint, backup_dir / 'checkpoint')
                    
            # 백업 시간 기록
            backup_file = self.dirs['cache'] / 'last_backup.txt'
            with open(backup_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
            self.log_event("backup_completed", "info", "System", f"백업 완료: {backup_dir}")
            
        except Exception as e:
            logger.error(f"백업 실패: {e}")
            self.log_event("backup_failed", "error", "System", str(e))
            
    def process_events(self):
        """이벤트 처리"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                # 이벤트 처리 로직
                logger.debug(f"이벤트 처리: {event}")
            except queue.Empty:
                break
                
    def log_event(self, event_type: str, severity: str, component: str, message: str, details: Dict = None):
        """이벤트 로깅"""
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO system_events
        (timestamp, event_type, severity, component, message, details)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            event_type,
            severity,
            component,
            message,
            json.dumps(details) if details else None
        ))
        
        conn.commit()
        conn.close()
        
        # 이벤트 큐에 추가
        if not self.event_queue.full():
            self.event_queue.put({
                'type': event_type,
                'severity': severity,
                'component': component,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            
    def update_metrics(self, updates: Dict):
        """메트릭 업데이트"""
        # 현재 메트릭 가져오기
        current = self.get_latest_metrics()
        
        # 업데이트 적용
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
                
        # 저장
        self.save_metrics(current)
        
    def get_latest_metrics(self) -> Dict:
        """최신 메트릭 가져오기"""
        if not self.metrics_queue.empty():
            # 큐에서 최신 것 가져오기
            metrics = None
            while not self.metrics_queue.empty():
                try:
                    metrics = self.metrics_queue.get_nowait()
                except queue.Empty:
                    break
                    
            if metrics:
                return metrics
                
        # 데이터베이스에서 가져오기
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM system_metrics 
        ORDER BY timestamp DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # row를 딕셔너리로 변환
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
            
        return {}
        
    def get_system_status(self) -> Dict:
        """시스템 상태 반환"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'runtime': str(runtime),
            'components': {
                'rag': 'running' if self.check_rag_health() else 'stopped',
                'training': 'running' if self.check_training_health() else 'stopped',
                'monitoring': 'running',
                'web_ui': 'running'
            },
            'config': self.config
        }
        
    async def broadcast_to_clients(self, message: str):
        """모든 클라이언트에 메시지 브로드캐스트"""
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
                
        self.websocket_clients -= disconnected
        
    def pause_system(self):
        """시스템 일시정지"""
        logger.info("⏸ 시스템 일시정지")
        self.status = SystemStatus.PAUSED
        self.log_event("system_paused", "info", "System", "시스템이 일시정지되었습니다")
        
    def resume_system(self):
        """시스템 재개"""
        logger.info("▶ 시스템 재개")
        self.status = SystemStatus.RUNNING
        self.log_event("system_resumed", "info", "System", "시스템이 재개되었습니다")
        
    async def restart_system(self):
        """시스템 재시작"""
        logger.info("🔄 시스템 재시작")
        
        # 현재 프로세스 종료
        await self.shutdown()
        
        # 잠시 대기
        await asyncio.sleep(5)
        
        # 재시작
        await self.start_system()
        
    async def handle_system_error(self, error: Exception):
        """시스템 오류 처리"""
        logger.error(f"🚨 시스템 오류: {error}")
        
        self.log_event(
            "system_error", "critical", "System",
            f"치명적 오류: {str(error)}",
            {'traceback': str(error.__traceback__)}
        )
        
        # 자동 재시작 시도
        if self.config['system']['auto_restart']:
            retry_count = 0
            max_retries = self.config['system']['max_retries']
            
            while retry_count < max_retries:
                logger.info(f"재시작 시도 {retry_count + 1}/{max_retries}")
                
                await asyncio.sleep(30)  # 30초 대기
                
                try:
                    await self.start_system()
                    break
                except Exception as e:
                    logger.error(f"재시작 실패: {e}")
                    retry_count += 1
                    
            if retry_count >= max_retries:
                logger.critical("재시작 실패. 시스템 종료.")
                sys.exit(1)
                
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 시스템 종료 중...")
        
        self.status = SystemStatus.STOPPING
        
        # 프로세스 종료
        processes = [
            ('RAG', self.rag_process),
            ('Training', self.training_process),
            ('Monitoring', self.monitor_process)
        ]
        
        for name, process in processes:
            if process:
                logger.info(f"{name} 프로세스 종료 중...")
                process.terminate()
                
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning(f"{name} 프로세스 강제 종료")
                    process.kill()
                    
        # 최종 리포트 생성
        await self.generate_final_report()
        
        # 상태 저장
        self.save_state({
            'status': SystemStatus.STOPPED.value,
            'stop_time': datetime.now().isoformat(),
            'total_runtime': str(datetime.now() - self.start_time) if self.start_time else None
        })
        
        self.status = SystemStatus.STOPPED
        logger.info("✅ 시스템 종료 완료")
        
    async def generate_final_report(self):
        """최종 리포트 생성"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        # 통계 수집
        db_path = self.dirs['metrics'] / 'metrics.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # RAG 통계
        cursor.execute('SELECT COUNT(*) FROM rag_queries')
        total_queries = cursor.fetchone()[0]
        
        # 이벤트 통계
        cursor.execute('SELECT event_type, COUNT(*) FROM system_events GROUP BY event_type')
        event_stats = dict(cursor.fetchall())
        
        # 학습 체크포인트
        cursor.execute('SELECT COUNT(*) FROM training_checkpoints')
        checkpoint_count = cursor.fetchone()[0]
        
        conn.close()
        
        report = f"""# Robust Dual Phase System 최종 리포트

## 🚀 시스템 개요
- **시작 시간**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}
- **종료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **총 실행 시간**: {runtime}

## 📊 성과 통계

### RAG 시스템
- **총 쿼리 수**: {total_queries:,}
- **평균 응답 시간**: N/A
- **캐시 히트율**: N/A

### 파인튜닝
- **체크포인트 수**: {checkpoint_count}
- **최종 손실**: N/A
- **최종 정확도**: N/A

### 시스템 이벤트
"""
        
        for event_type, count in sorted(event_stats.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{event_type}**: {count}회\n"
            
        report += f"""
## 💻 리소스 사용
- **평균 CPU**: N/A
- **평균 메모리**: N/A
- **최대 GPU**: N/A

## 🔍 주요 이슈
- 시스템 오류: {event_stats.get('system_error', 0)}회
- 경고: {event_stats.get('warning', 0)}회

## 💡 개선 사항
- RAG 응답 품질 향상 필요
- 메모리 사용량 최적화 필요
- 학습 속도 개선 가능

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_path = self.base_path / 'autoci_reports' / f'final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📝 최종 리포트 생성: {report_path}")
        
    def save_state(self, state: Dict):
        """시스템 상태 저장"""
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Dual Phase System")
    parser.add_argument("command", choices=["start", "stop", "status"],
                       help="실행할 명령")
    
    args = parser.parse_args()
    
    system = RobustDualPhase()
    
    if args.command == "start":
        asyncio.run(system.start_system())
        
    elif args.command == "stop":
        # PID로 종료 신호 전송
        state_path = Path(__file__).parent / "robust_dual_phase_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            if 'pid' in state:
                try:
                    os.kill(state['pid'], signal.SIGINT)
                    print("종료 신호 전송됨")
                except ProcessLookupError:
                    print("프로세스를 찾을 수 없습니다")
        else:
            print("실행 중인 시스템이 없습니다")
            
    elif args.command == "status":
        state_path = Path(__file__).parent / "robust_dual_phase_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            print(json.dumps(state, indent=2, ensure_ascii=False))
        else:
            print("상태 정보가 없습니다")


if __name__ == "__main__":
    main()