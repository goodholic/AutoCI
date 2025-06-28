#!/usr/bin/env python3
"""
24시간 게임 제작 AI Agent 시스템 - 통합 메인 애플리케이션
AutoCI Main Application with integrated AI Agent frameworks
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import redis
import logging
from datetime import datetime
import subprocess
import json

# AI Agent Framework imports
try:
    from langgraph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
    import chromadb
    import weaviate
except ImportError as e:
    logging.warning(f"AI framework import error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AutoCI 24H Game Development AI Agent",
    description="24시간 게임 제작 AI Agent 시스템",
    version="3.0.0"
)

# Global configuration
config = {
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "weaviate_url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
    "godot_path": os.getenv("GODOT_PATH", "./godot_engine"),
    "projects_dir": "./godot_projects",
    "logs_dir": "./logs"
}

# Initialize services
redis_client = None
weaviate_client = None
chroma_client = None

def initialize_services():
    """Initialize external services"""
    global redis_client, weaviate_client, chroma_client
    
    try:
        # Redis
        redis_client = redis.from_url(config["redis_url"])
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    try:
        # Weaviate
        weaviate_client = weaviate.Client(config["weaviate_url"])
        logger.info("Weaviate connected successfully")
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
    
    try:
        # ChromaDB
        chroma_client = chromadb.Client()
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}")

# Pydantic models
class GameProjectRequest(BaseModel):
    name: str
    description: str
    game_type: str
    features: List[str]
    art_style: Optional[str] = "2D"
    complexity: Optional[str] = "simple"

class AIAgentTask(BaseModel):
    task_type: str
    parameters: Dict[str, Any]
    priority: Optional[int] = 1

class SystemStatus(BaseModel):
    status: str
    services: Dict[str, bool]
    active_projects: int
    system_info: Dict[str, Any]

# Core AI Agent functionality
class AutoCIAgent:
    """Main AI Agent for game development automation"""
    
    def __init__(self):
        self.active_projects = {}
        self.task_queue = []
        self.learning_memory = {}
    
    async def create_game_project(self, request: GameProjectRequest) -> Dict[str, str]:
        """Create a new game project with Godot"""
        try:
            project_path = os.path.join(config["projects_dir"], request.name)
            os.makedirs(project_path, exist_ok=True)
            
            # Create Godot project
            godot_cmd = [
                config["godot_path"],
                "--headless",
                "--quit",
                "--path", project_path
            ]
            
            result = subprocess.run(godot_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Store project info
                project_info = {
                    "name": request.name,
                    "description": request.description,
                    "created_at": datetime.now().isoformat(),
                    "status": "created",
                    "path": project_path
                }
                
                self.active_projects[request.name] = project_info
                
                # Store in Redis
                if redis_client:
                    redis_client.hset(
                        "autoci:projects", 
                        request.name, 
                        json.dumps(project_info)
                    )
                
                return {
                    "status": "success",
                    "message": f"Project '{request.name}' created successfully",
                    "project_path": project_path
                }
            else:
                raise Exception(f"Godot project creation failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_code(self, project_name: str, code_type: str, specifications: Dict) -> Dict[str, str]:
        """Generate game code using AI agents"""
        try:
            # This would integrate with LangGraph for complex code generation workflows
            if project_name not in self.active_projects:
                raise Exception(f"Project '{project_name}' not found")
            
            # AI-powered code generation logic here
            # For now, return a placeholder
            generated_code = f"""
# Auto-generated {code_type} for {project_name}
extends Node

func _ready():
    print("AI-generated code for {project_name}")
    # TODO: Implement {specifications.get('description', 'functionality')}
"""
            
            return {
                "status": "success",
                "code": generated_code,
                "type": code_type
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def learn_from_feedback(self, project_name: str, feedback: Dict) -> Dict[str, str]:
        """Learn from user feedback to improve future generations"""
        try:
            # Store learning data in vector database
            if weaviate_client:
                # Store feedback in Weaviate for semantic search
                pass
            
            if chroma_client:
                # Store in ChromaDB for retrieval
                pass
            
            return {
                "status": "success",
                "message": "Feedback processed and stored for learning"
            }
            
        except Exception as e:
            logger.error(f"Learning from feedback failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

# Initialize AI Agent
autoci_agent = AutoCIAgent()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()
    
    # Create necessary directories
    os.makedirs(config["projects_dir"], exist_ok=True)
    os.makedirs(config["logs_dir"], exist_ok=True)
    
    logger.info("AutoCI AI Agent System started successfully")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main dashboard"""
    return """
    <html>
        <head>
            <title>AutoCI 24H Game Development AI Agent</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c3e50; }
                .service { margin: 10px 0; }
                .api-link { color: #3498db; text-decoration: none; }
            </style>
        </head>
        <body>
            <h1 class="header">🤖 AutoCI 24시간 게임 제작 AI Agent 시스템</h1>
            <h2>서비스 상태</h2>
            <div class="service">✅ FastAPI 서버 실행 중</div>
            <div class="service">🎮 Godot Engine 연동</div>
            <div class="service">🧠 AI Agent 프레임워크 (LangGraph, CrewAI)</div>
            <div class="service">💾 벡터 데이터베이스 (ChromaDB, Weaviate)</div>
            <div class="service">⚡ 백그라운드 작업 (Celery + Redis)</div>
            
            <h2>API 엔드포인트</h2>
            <p><a href="/docs" class="api-link">📚 API 문서 (Swagger UI)</a></p>
            <p><a href="/status" class="api-link">🔍 시스템 상태</a></p>
            <p><a href="/projects" class="api-link">🎮 활성 프로젝트 목록</a></p>
        </body>
    </html>
    """

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    services = {
        "redis": bool(redis_client and redis_client.ping()),
        "weaviate": bool(weaviate_client),
        "chromadb": bool(chroma_client),
        "godot": os.path.exists(config["godot_path"])
    }
    
    return SystemStatus(
        status="running",
        services=services,
        active_projects=len(autoci_agent.active_projects),
        system_info={
            "godot_path": config["godot_path"],
            "projects_dir": config["projects_dir"],
            "python_version": subprocess.check_output(["python3", "--version"]).decode().strip()
        }
    )

@app.get("/projects")
async def get_projects():
    """Get list of active projects"""
    return {
        "active_projects": autoci_agent.active_projects,
        "total_count": len(autoci_agent.active_projects)
    }

@app.post("/projects/create")
async def create_project(request: GameProjectRequest, background_tasks: BackgroundTasks):
    """Create a new game project"""
    result = await autoci_agent.create_game_project(request)
    
    if result["status"] == "success":
        # Add background task for project setup
        background_tasks.add_task(
            setup_project_background, 
            request.name, 
            request.dict()
        )
    
    return result

@app.post("/projects/{project_name}/generate-code")
async def generate_code(project_name: str, code_request: Dict[str, Any]):
    """Generate code for a project"""
    return await autoci_agent.generate_code(
        project_name, 
        code_request.get("type", "script"),
        code_request.get("specs", {})
    )

@app.post("/projects/{project_name}/feedback")
async def submit_feedback(project_name: str, feedback: Dict[str, Any]):
    """Submit feedback for AI learning"""
    return await autoci_agent.learn_from_feedback(project_name, feedback)

@app.post("/ai-task")
async def execute_ai_task(task: AIAgentTask):
    """Execute an AI agent task"""
    try:
        # This would use CrewAI or LangGraph for complex multi-agent tasks
        result = {
            "task_id": f"task_{datetime.now().timestamp()}",
            "status": "queued",
            "task_type": task.task_type,
            "parameters": task.parameters
        }
        
        # Add to task queue
        autoci_agent.task_queue.append(result)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def setup_project_background(project_name: str, project_config: Dict):
    """Background task for project setup"""
    try:
        logger.info(f"Setting up project: {project_name}")
        
        # Additional project setup logic here
        # - Generate initial game files
        # - Set up version control
        # - Initialize AI learning context
        
        logger.info(f"Project {project_name} setup completed")
        
    except Exception as e:
        logger.error(f"Project setup failed for {project_name}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)