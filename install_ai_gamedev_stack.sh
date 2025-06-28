#!/bin/bash

# 24시간 게임 제작 AI Agent 시스템 설치 스크립트
# WSL Ubuntu 환경 기준

set -e

echo "=== 24시간 게임 제작 AI Agent 시스템 설치 시작 ==="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Python 버전 확인
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d"." -f1,2)
        log_info "Python $PYTHON_VERSION 발견"
    else
        log_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
}

# 시스템 업데이트
update_system() {
    log_info "시스템 패키지 업데이트 중..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y \
        build-essential \
        curl \
        wget \
        git \
        python3-pip \
        python3-venv \
        docker.io \
        docker-compose \
        redis-server \
        postgresql \
        postgresql-contrib \
        nodejs \
        npm
}

# 가상환경 생성
create_venv() {
    log_info "Python 가상환경 생성 중..."
    python3 -m venv venv_ai_gamedev
    source venv_ai_gamedev/bin/activate
    pip install --upgrade pip
}

# 1. AI Agent 프레임워크 설치
install_ai_agents() {
    log_info "AI Agent 프레임워크 설치 중..."
    
    # AutoGPT 설치
    log_info "AutoGPT 설치 중..."
    git clone https://github.com/Significant-Gravitas/AutoGPT.git
    cd AutoGPT
    pip install -r requirements.txt
    cd ..
    
    # LangGraph 설치
    log_info "LangGraph 설치 중..."
    pip install langgraph langchain langchain-community
    
    # CrewAI 설치
    log_info "CrewAI 설치 중..."
    pip install crewai crewai-tools
    
    # OpenDevin 설치
    log_info "OpenDevin 설치 중..."
    git clone https://github.com/OpenDevin/OpenDevin.git
    cd OpenDevin
    pip install -r requirements.txt
    cd ..
}

# 2. Llama 7B 및 관련 도구 설치
install_llama_tools() {
    log_info "Llama 7B 도구 설치 중..."
    
    # Ollama 설치 (WSL 지원)
    log_info "Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Ollama 서비스 시작
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    # Llama 7B 모델 다운로드
    log_info "Llama 7B 모델 다운로드 중..."
    ollama pull llama3.2:7b
    
    # LlamaFile 설치
    log_info "LlamaFile 설치 중..."
    mkdir -p llamafile
    cd llamafile
    wget https://github.com/Mozilla-Ocho/llamafile/releases/latest/download/llamafile
    chmod +x llamafile
    cd ..
    
    # Text Generation WebUI 설치
    log_info "Text Generation WebUI 설치 중..."
    git clone https://github.com/oobabooga/text-generation-webui.git
    cd text-generation-webui
    pip install -r requirements.txt
    cd ..
    
    # LangChain 설치
    log_info "LangChain 설치 중..."
    pip install langchain langchain-openai langchain-community
}

# 3. Godot 자동화 도구 설치
install_godot_tools() {
    log_info "Godot 자동화 도구 설치 중..."
    
    # Godot 헤드리스 버전 다운로드
    log_info "Godot Headless 설치 중..."
    wget https://downloads.tuxfamily.org/godotengine/4.2.1/Godot_v4.2.1-stable_linux.x86_64.zip
    unzip Godot_v4.2.1-stable_linux.x86_64.zip
    sudo mv Godot_v4.2.1-stable_linux.x86_64 /usr/local/bin/godot
    
    # Godot Export Templates 다운로드
    log_info "Godot Export Templates 다운로드 중..."
    wget https://downloads.tuxfamily.org/godotengine/4.2.1/Godot_v4.2.1-stable_export_templates.tpz
    mkdir -p ~/.local/share/godot/export_templates/4.2.1.stable
    unzip Godot_v4.2.1-stable_export_templates.tpz -d ~/.local/share/godot/export_templates/4.2.1.stable
    
    # GDScript LSP 설정
    log_info "GDScript LSP 설정 중..."
    pip install gdtoolkit
    
    # Godot CI/CD 템플릿 다운로드
    log_info "Godot CI/CD 템플릿 다운로드 중..."
    mkdir -p godot-ci-templates
    cd godot-ci-templates
    wget https://raw.githubusercontent.com/abarichello/godot-ci/master/.gitlab-ci.yml
    wget https://raw.githubusercontent.com/abarichello/godot-ci/master/.github/workflows/godot-ci.yml
    cd ..
}

# 4. 시스템 통합 도구 설치
install_integration_tools() {
    log_info "시스템 통합 도구 설치 중..."
    
    # FastAPI 설치
    log_info "FastAPI 설치 중..."
    pip install fastapi uvicorn[standard] pydantic
    
    # Celery 설치
    log_info "Celery 설치 중..."
    pip install celery[redis] flower
    
    # Redis 설정
    log_info "Redis 설정 중..."
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
    
    # Docker Compose 프로젝트 생성
    log_info "Docker Compose 설정 생성 중..."
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_gamedev
      POSTGRES_USER: ai_gamedev
      POSTGRES_PASSWORD: ai_gamedev_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  redis_data:
  postgres_data:
  ollama_data:
  chroma_data:
  weaviate_data:
EOF
}

# 5. 벡터 데이터베이스 설치
install_vector_databases() {
    log_info "벡터 데이터베이스 설치 중..."
    
    # ChromaDB 설치
    log_info "ChromaDB 설치 중..."
    pip install chromadb
    
    # Weaviate 클라이언트 설치
    log_info "Weaviate 클라이언트 설치 중..."
    pip install weaviate-client
    
    # Pinecone 대안 설치
    log_info "추가 벡터 DB 도구 설치 중..."
    pip install qdrant-client faiss-cpu
}

# 6. 통합 API 서버 생성
create_api_server() {
    log_info "통합 API 서버 생성 중..."
    
    mkdir -p api_server
    cat > api_server/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from celery import Celery
import redis
import json

app = FastAPI(title="24시간 게임 제작 AI Agent API")

# Celery 설정
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379',
    backend='redis://localhost:6379'
)

# Redis 연결
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class GameProjectRequest(BaseModel):
    project_name: str
    game_type: str
    description: str
    platform: List[str]
    ai_agents: List[str]

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    result: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "24시간 게임 제작 AI Agent 시스템"}

@app.post("/projects/create")
async def create_project(project: GameProjectRequest):
    # 프로젝트 생성 로직
    task = celery_app.send_task('create_game_project', args=[project.dict()])
    return {"task_id": task.id, "status": "프로젝트 생성 시작"}

@app.get("/projects/{project_id}/status")
async def get_project_status(project_id: str):
    # 프로젝트 상태 확인
    status = redis_client.get(f"project:{project_id}:status")
    if status:
        return json.loads(status)
    raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

@app.post("/ai/generate-code")
async def generate_code(prompt: str, language: str = "gdscript"):
    # AI 코드 생성
    task = celery_app.send_task('generate_code', args=[prompt, language])
    return {"task_id": task.id, "status": "코드 생성 중"}

@app.post("/godot/build")
async def build_godot_project(project_path: str, platform: str):
    # Godot 프로젝트 빌드
    task = celery_app.send_task('build_godot', args=[project_path, platform])
    return {"task_id": task.id, "status": "빌드 시작"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

    # Celery 작업 파일 생성
    cat > api_server/tasks.py << 'EOF'
from celery import Celery
import subprocess
import os
from langchain.llms import Ollama
from crewai import Agent, Task, Crew

app = Celery('tasks', broker='redis://localhost:6379', backend='redis://localhost:6379')

# Ollama LLM 초기화
llm = Ollama(model="llama3.2:7b")

@app.task
def create_game_project(project_data):
    # 게임 프로젝트 생성 로직
    project_name = project_data['project_name']
    
    # CrewAI 에이전트 생성
    game_designer = Agent(
        role='게임 디자이너',
        goal='혁신적인 게임 컨셉 설계',
        backstory='경험 많은 게임 디자이너',
        llm=llm
    )
    
    programmer = Agent(
        role='프로그래머',
        goal='효율적인 게임 코드 작성',
        backstory='Godot 전문 개발자',
        llm=llm
    )
    
    # 작업 생성 및 실행
    design_task = Task(
        description=f"{project_data['description']}에 기반한 게임 디자인 문서 작성",
        agent=game_designer
    )
    
    crew = Crew(
        agents=[game_designer, programmer],
        tasks=[design_task]
    )
    
    result = crew.kickoff()
    return {"status": "완료", "result": result}

@app.task
def generate_code(prompt, language):
    # AI를 사용한 코드 생성
    response = llm.invoke(f"다음 요구사항에 맞는 {language} 코드를 생성해주세요: {prompt}")
    return {"code": response}

@app.task
def build_godot(project_path, platform):
    # Godot 프로젝트 빌드
    cmd = f"godot --headless --export {platform} {project_path}/build/{platform}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"status": "완료", "output": result.stdout, "error": result.stderr}
EOF
}

# 7. 시작 스크립트 생성
create_start_script() {
    log_info "시작 스크립트 생성 중..."
    
    cat > start_ai_gamedev.sh << 'EOF'
#!/bin/bash

echo "24시간 게임 제작 AI Agent 시스템 시작..."

# Redis 시작
sudo systemctl start redis-server

# Ollama 시작
sudo systemctl start ollama

# Docker 컨테이너 시작
docker-compose up -d

# Celery 워커 시작
cd api_server
celery -A tasks worker --loglevel=info &

# FastAPI 서버 시작
python main.py &

echo "모든 서비스가 시작되었습니다!"
echo "API 서버: http://localhost:8000"
echo "API 문서: http://localhost:8000/docs"
EOF

    chmod +x start_ai_gamedev.sh
}

# 8. 환경 설정 파일 생성
create_config_files() {
    log_info "환경 설정 파일 생성 중..."
    
    cat > .env << 'EOF'
# AI 모델 설정
OLLAMA_MODEL=llama3.2:7b
OPENAI_API_KEY=your_openai_api_key_here

# 데이터베이스 설정
POSTGRES_DB=ai_gamedev
POSTGRES_USER=ai_gamedev
POSTGRES_PASSWORD=ai_gamedev_pass

# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379

# Godot 설정
GODOT_PATH=/usr/local/bin/godot
GODOT_VERSION=4.2.1

# 벡터 DB 설정
CHROMA_HOST=localhost
CHROMA_PORT=8000
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
EOF
}

# 메인 설치 프로세스
main() {
    check_python
    update_system
    create_venv
    
    install_ai_agents
    install_llama_tools
    install_godot_tools
    install_integration_tools
    install_vector_databases
    
    create_api_server
    create_start_script
    create_config_files
    
    log_info "=== 설치 완료 ==="
    log_info "시작하려면: ./start_ai_gamedev.sh"
    log_info "API 문서는 http://localhost:8000/docs 에서 확인하세요"
}

# 스크립트 실행
main