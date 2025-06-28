#!/bin/bash

# 24시간 게임 제작 AI Agent 시스템 시작 스크립트
echo "🤖 Starting AutoCI 24H Game Development AI Agent System..."

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export REDIS_URL="redis://localhost:6379/0"
export WEAVIATE_URL="http://localhost:8080"
export GODOT_PATH="$(pwd)/godot_engine"

# 가상환경 활성화
if [ -d "autoci_env" ]; then
    echo "🐍 Activating virtual environment..."
    source autoci_env/bin/activate
else
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# 로그 디렉토리 생성
mkdir -p logs

# Docker Compose로 외부 서비스 시작
echo "🐳 Starting external services with Docker Compose..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d redis weaviate t2v-transformers
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    docker compose up -d redis weaviate t2v-transformers
else
    echo "⚠️  Docker not available. Starting without containerized services."
fi

# 서비스 대기
echo "⏳ Waiting for services to be ready..."
sleep 10

# Redis 연결 테스트
echo "🔍 Testing Redis connection..."
python3 -c "
import redis
try:
    r = redis.from_url('redis://localhost:6379/0')
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"

# Weaviate 연결 테스트
echo "🔍 Testing Weaviate connection..."
python3 -c "
import requests
try:
    response = requests.get('http://localhost:8080/v1/meta')
    if response.status_code == 200:
        print('✅ Weaviate connection successful')
    else:
        print(f'❌ Weaviate connection failed: {response.status_code}')
except Exception as e:
    print(f'❌ Weaviate connection failed: {e}')
"

# Godot 실행 권한 확인
if [ -f "$GODOT_PATH" ]; then
    chmod +x "$GODOT_PATH"
    echo "✅ Godot engine ready at $GODOT_PATH"
else
    echo "⚠️  Godot engine not found at $GODOT_PATH"
fi

# 프로젝트 디렉토리 생성
mkdir -p godot_projects

# Celery Worker 백그라운드 시작
echo "🚀 Starting Celery worker..."
celery -A autoci_main worker --loglevel=info --detach --pidfile=logs/celery_worker.pid --logfile=logs/celery_worker.log

# Celery Beat 스케줄러 시작
echo "⏰ Starting Celery beat scheduler..."
celery -A autoci_main beat --loglevel=info --detach --pidfile=logs/celery_beat.pid --logfile=logs/celery_beat.log

# FastAPI 서버 시작
echo "🌐 Starting FastAPI server..."
echo "🔗 Server will be available at: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🎮 Dashboard: http://localhost:8000"

# 서버 시작
uvicorn autoci_main:app --host 0.0.0.0 --port 8000 --reload --log-level info

echo "🛑 Shutting down services..."

# Celery 프로세스 종료
if [ -f "logs/celery_worker.pid" ]; then
    kill $(cat logs/celery_worker.pid) 2>/dev/null || true
    rm -f logs/celery_worker.pid
fi

if [ -f "logs/celery_beat.pid" ]; then
    kill $(cat logs/celery_beat.pid) 2>/dev/null || true
    rm -f logs/celery_beat.pid
fi

echo "✅ AutoCI AI Agent System stopped."