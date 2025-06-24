#!/usr/bin/env python3
"""
C# 전문가 학습 시스템 Web API
실시간 학습 진행 상황 모니터링 및 제어
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel
import subprocess
import psutil

# FastAPI 앱 생성
app = FastAPI(
    title="C# Expert Learning API",
    description="24시간 C# 전문가 학습 시스템 모니터링 및 제어 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 상태 관리
class LearningSystemState:
    def __init__(self):
        self.is_running = False
        self.start_time = None
        self.crawler_process = None
        self.server_process = None
        self.current_phase = "대기 중"
        self.last_update = datetime.now()
        self.collected_data_count = 0
        self.improvements_count = 0
        self.training_cycles = 0
        self.current_model_score = 0.0
        
learning_state = LearningSystemState()

# Pydantic 모델
class LearningConfig(BaseModel):
    github_stars_threshold: int = 1000
    stackoverflow_score_threshold: int = 50
    code_quality_threshold: float = 0.7
    max_files_per_repo: int = 100
    crawl_interval_hours: int = 4
    auto_improvement: bool = True

class ImprovementRequest(BaseModel):
    code: str
    language: str = "csharp"
    context: Optional[str] = None

class TrainingStatus(BaseModel):
    is_running: bool
    current_phase: str
    uptime_hours: float
    collected_data_count: int
    improvements_count: int
    training_cycles: int
    current_model_score: float
    memory_usage_gb: float
    cpu_usage_percent: float

# API 엔드포인트
@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "C# Expert Learning System API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "start": "/api/start",
            "stop": "/api/stop",
            "stats": "/api/stats",
            "logs": "/api/logs",
            "config": "/api/config",
            "improve": "/api/improve"
        }
    }

@app.get("/api/status", response_model=TrainingStatus)
async def get_status():
    """현재 학습 시스템 상태"""
    uptime = 0
    if learning_state.is_running and learning_state.start_time:
        uptime = (datetime.now() - learning_state.start_time).total_seconds() / 3600
    
    # 시스템 리소스 사용량
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return TrainingStatus(
        is_running=learning_state.is_running,
        current_phase=learning_state.current_phase,
        uptime_hours=round(uptime, 2),
        collected_data_count=learning_state.collected_data_count,
        improvements_count=learning_state.improvements_count,
        training_cycles=learning_state.training_cycles,
        current_model_score=learning_state.current_model_score,
        memory_usage_gb=round(memory.used / (1024**3), 2),
        cpu_usage_percent=cpu_percent
    )

@app.post("/api/start")
async def start_learning(background_tasks: BackgroundTasks):
    """학습 시스템 시작"""
    if learning_state.is_running:
        raise HTTPException(status_code=400, detail="학습 시스템이 이미 실행 중입니다")
    
    # 백그라운드에서 학습 시작
    background_tasks.add_task(start_learning_system)
    
    return {
        "status": "started",
        "message": "C# 전문가 학습 시스템을 시작합니다",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/stop")
async def stop_learning():
    """학습 시스템 중지"""
    if not learning_state.is_running:
        raise HTTPException(status_code=400, detail="학습 시스템이 실행 중이지 않습니다")
    
    # 프로세스 종료
    if learning_state.crawler_process:
        learning_state.crawler_process.terminate()
    if learning_state.server_process:
        learning_state.server_process.terminate()
    
    learning_state.is_running = False
    learning_state.current_phase = "중지됨"
    
    return {
        "status": "stopped",
        "message": "학습 시스템을 중지했습니다",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_statistics():
    """학습 통계 조회"""
    stats_file = Path("learning_stats.json")
    
    if not stats_file.exists():
        return {
            "total_data_collected": 0,
            "total_training_hours": 0,
            "model_improvements": [],
            "last_update": None
        }
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    # 최근 7일간의 통계 추가
    recent_stats = calculate_recent_stats(stats)
    stats["recent_7_days"] = recent_stats
    
    return stats

@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """최근 로그 조회"""
    log_file = Path("csharp_expert_learning.log")
    
    if not log_file.exists():
        return {"logs": [], "message": "로그 파일이 없습니다"}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    # 로그 파싱
    parsed_logs = []
    for line in recent_lines:
        try:
            parts = line.strip().split(' - ', 3)
            if len(parts) >= 4:
                parsed_logs.append({
                    "timestamp": parts[0],
                    "logger": parts[1],
                    "level": parts[2],
                    "message": parts[3]
                })
            else:
                parsed_logs.append({"raw": line.strip()})
        except:
            parsed_logs.append({"raw": line.strip()})
    
    return {
        "logs": parsed_logs,
        "total_lines": len(all_lines),
        "returned_lines": len(parsed_logs)
    }

@app.get("/api/config")
async def get_config():
    """현재 설정 조회"""
    config_file = Path("expert_learning_config.json")
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="설정 파일이 없습니다")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

@app.put("/api/config")
async def update_config(config: LearningConfig):
    """설정 업데이트"""
    config_file = Path("expert_learning_config.json")
    
    # 기존 설정 로드
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            current_config = json.load(f)
    else:
        current_config = {}
    
    # 크롤러 설정 업데이트
    current_config["crawler_config"] = {
        "github_stars_threshold": config.github_stars_threshold,
        "stackoverflow_score_threshold": config.stackoverflow_score_threshold,
        "code_quality_threshold": config.code_quality_threshold,
        "max_files_per_repo": config.max_files_per_repo,
        "crawl_interval_hours": config.crawl_interval_hours
    }
    
    current_config["improvement_config"]["auto_fix"] = config.auto_improvement
    
    # 저장
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)
    
    return {
        "status": "updated",
        "message": "설정이 업데이트되었습니다",
        "config": current_config
    }

@app.post("/api/improve")
async def improve_code(request: ImprovementRequest):
    """코드 개선 요청"""
    if not learning_state.is_running:
        raise HTTPException(
            status_code=503, 
            detail="학습 시스템이 실행 중이지 않습니다. 먼저 시스템을 시작하세요."
        )
    
    # AI 서버에 개선 요청
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/improve",
                json={
                    "code": request.code,
                    "language": request.language,
                    "context": request.context
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # 개선 횟수 증가
                    learning_state.improvements_count += 1
                    
                    return {
                        "improved_code": result.get("improved_code"),
                        "suggestions": result.get("suggestions", []),
                        "quality_score": result.get("quality_score", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail="AI 서버에서 코드 개선에 실패했습니다"
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI 서버에 연결할 수 없습니다: {str(e)}"
        )

@app.get("/api/training-data")
async def get_training_data():
    """수집된 학습 데이터 정보"""
    data_dir = Path("expert_training_data")
    
    if not data_dir.exists():
        return {"sources": {}, "total": 0}
    
    sources = {}
    total_count = 0
    
    # 각 소스별 데이터 카운트
    for source_file in data_dir.glob("*.json"):
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
                sources[source_file.stem] = count
                total_count += count
        except:
            sources[source_file.stem] = 0
    
    return {
        "sources": sources,
        "total": total_count,
        "last_update": datetime.now().isoformat()
    }

@app.get("/api/improvements")
async def get_improvements():
    """최근 코드 개선 내역"""
    improvements_dir = Path("expert_training_data/improvements")
    
    if not improvements_dir.exists():
        return {"improvements": [], "total": 0}
    
    improvements = []
    
    # 최근 10개 개선 파일
    for imp_file in sorted(improvements_dir.glob("*_improvements.md"), 
                          key=lambda x: x.stat().st_mtime, 
                          reverse=True)[:10]:
        try:
            with open(imp_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 간단한 파싱
            improvements.append({
                "file": imp_file.stem.replace("_improvements", ""),
                "timestamp": datetime.fromtimestamp(imp_file.stat().st_mtime).isoformat(),
                "preview": content[:200] + "..." if len(content) > 200 else content
            })
        except:
            pass
    
    return {
        "improvements": improvements,
        "total": len(list(improvements_dir.glob("*_improvements.md")))
    }

@app.get("/api/model-info")
async def get_model_info():
    """모델 정보 조회"""
    model_path = Path("CodeLlama-7b-Instruct-hf")
    
    if not model_path.exists():
        return {
            "status": "not_found",
            "message": "모델이 설치되지 않았습니다"
        }
    
    # 모델 크기 계산
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    
    return {
        "status": "installed",
        "model_name": "CodeLlama-7b-Instruct-hf",
        "size_gb": round(total_size / (1024**3), 2),
        "path": str(model_path),
        "files_count": len(list(model_path.rglob("*")))
    }

# 정적 파일 서빙 (대시보드)
app.mount("/dashboard", StaticFiles(directory=".", html=True), name="dashboard")

# 헬퍼 함수들
def calculate_recent_stats(stats: Dict) -> Dict:
    """최근 7일간의 통계 계산"""
    recent = {
        "data_collected": 0,
        "improvements": 0,
        "training_hours": 0,
        "avg_quality_score": 0
    }
    
    # 실제 구현은 타임스탬프 기반으로 계산
    # 여기서는 예시 데이터
    recent["data_collected"] = stats.get("total_data_collected", 0) // 7
    recent["training_hours"] = stats.get("total_training_hours", 0) / 7
    
    return recent

async def start_learning_system():
    """학습 시스템 시작 (백그라운드)"""
    try:
        learning_state.is_running = True
        learning_state.start_time = datetime.now()
        learning_state.current_phase = "시작 중..."
        
        # 크롤러 프로세스 시작
        learning_state.crawler_process = subprocess.Popen(
            ["python", "csharp_expert_crawler.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # AI 서버 프로세스 시작 (있는 경우)
        enhanced_server = Path("MyAIWebApp/Models/enhanced_server.py")
        if enhanced_server.exists():
            learning_state.server_process = subprocess.Popen(
                ["python", "-m", "uvicorn", "enhanced_server:app", 
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd="MyAIWebApp/Models",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        learning_state.current_phase = "실행 중"
        
        # 주기적으로 상태 업데이트
        while learning_state.is_running:
            await update_learning_state()
            await asyncio.sleep(30)  # 30초마다 업데이트
            
    except Exception as e:
        logger.error(f"학습 시스템 시작 오류: {e}")
        learning_state.is_running = False
        learning_state.current_phase = f"오류: {str(e)}"

async def update_learning_state():
    """학습 상태 업데이트"""
    # 통계 파일에서 최신 정보 읽기
    stats_file = Path("learning_stats.json")
    if stats_file.exists():
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                
            learning_state.collected_data_count = stats.get("total_data_collected", 0)
            learning_state.training_cycles = len(stats.get("model_improvements", []))
            
            if stats.get("model_improvements"):
                latest = stats["model_improvements"][-1]
                learning_state.current_model_score = latest.get("score_change", 0)
        except:
            pass
    
    # 현재 진행 단계 추정
    now = datetime.now()
    hours_since_start = (now - learning_state.start_time).total_seconds() / 3600
    
    if hours_since_start < 4:
        learning_state.current_phase = "데이터 수집 중..."
    elif hours_since_start < 5:
        learning_state.current_phase = "데이터 전처리 중..."
    elif hours_since_start < 11:
        learning_state.current_phase = "모델 학습 중..."
    elif hours_since_start < 12:
        learning_state.current_phase = "모델 평가 중..."
    else:
        learning_state.current_phase = "코드 개선 서비스 실행 중..."

# 실행
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 C# Expert Learning API 시작...")
    uvicorn.run(app, host="0.0.0.0", port=8080)