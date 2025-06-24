#!/usr/bin/env python3
"""
C# 전문가 학습 시스템 시작 스크립트
24시간 자동 학습 및 코드 개선 서비스 실행
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expert_learning_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExpertLearningStartup:
    """전문가 학습 시스템 시작 관리자"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.models_dir = self.base_dir / "MyAIWebApp" / "Models"
        self.venv_path = self.models_dir / "llm_venv"
        self.requirements = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "aiohttp>=3.8.0",
            "beautifulsoup4>=4.12.0",
            "requests>=2.31.0",
            "watchdog>=3.0.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
            "protobuf>=4.23.0",
            "datasets>=2.14.0",
            "peft>=0.4.0",  # Parameter-Efficient Fine-Tuning
            "bitsandbytes>=0.40.0",  # 8-bit quantization
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "tqdm>=4.65.0",
            "colorama>=0.4.6",
            "python-dotenv>=1.0.0"
        ]
        
    def check_system_requirements(self):
        """시스템 요구사항 확인"""
        logger.info("🔍 시스템 요구사항 확인 중...")
        
        # Python 버전 확인
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("❌ Python 3.8 이상이 필요합니다!")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor} 확인")
        
        # Git 확인
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            logger.info("✅ Git 설치 확인")
        except:
            logger.error("❌ Git이 설치되지 않았습니다!")
            return False
        
        # 메모리 확인
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            if total_gb < 16:
                logger.warning(f"⚠️  메모리가 {total_gb:.1f}GB입니다. 16GB 이상 권장!")
            else:
                logger.info(f"✅ 메모리 {total_gb:.1f}GB 확인")
        except:
            logger.warning("⚠️  psutil이 없어 메모리를 확인할 수 없습니다")
        
        return True
    
    def setup_environment(self):
        """환경 설정"""
        logger.info("🛠️  개발 환경 설정 중...")
        
        # 필요한 디렉토리 생성
        directories = [
            self.base_dir / "expert_training_data",
            self.base_dir / "expert_training_data" / "github_projects",
            self.base_dir / "expert_training_data" / "stackoverflow",
            self.base_dir / "expert_training_data" / "microsoft_docs",
            self.base_dir / "expert_training_data" / "improvements",
            self.base_dir / "expert_training_data" / "models",
            self.base_dir / "expert_training_data" / "checkpoints",
            self.base_dir / "expert_training_data" / "logs"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 디렉토리 생성: {dir_path}")
        
        # 설정 파일 생성
        config = {
            "github_token": os.getenv("GITHUB_TOKEN", ""),
            "stackoverflow_key": os.getenv("STACKOVERFLOW_KEY", ""),
            "model_name": "codellama/CodeLlama-7b-Instruct-hf",
            "learning_config": {
                "batch_size": 4,
                "learning_rate": 1e-5,
                "epochs": 3,
                "max_length": 2048,
                "warmup_steps": 500,
                "save_steps": 1000,
                "evaluation_steps": 500,
                "gradient_checkpointing": True,
                "fp16": True,  # Mixed precision training
                "load_in_8bit": True  # 8-bit quantization
            },
            "crawler_config": {
                "github_stars_threshold": 1000,
                "stackoverflow_score_threshold": 50,
                "code_quality_threshold": 0.7,
                "max_files_per_repo": 100,
                "crawl_interval_hours": 4
            },
            "improvement_config": {
                "scan_interval_minutes": 5,
                "max_files_per_scan": 10,
                "min_quality_for_improvement": 0.7,
                "auto_fix": False  # 자동 수정 비활성화 (안전을 위해)
            }
        }
        
        config_path = self.base_dir / "expert_learning_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"⚙️  설정 파일 생성: {config_path}")
        
        # .env 파일 생성 (없는 경우)
        env_path = self.base_dir / ".env"
        if not env_path.exists():
            with open(env_path, 'w') as f:
                f.write("# GitHub API Token (선택사항 - Rate limit 증가)\n")
                f.write("# https://github.com/settings/tokens 에서 생성\n")
                f.write("GITHUB_TOKEN=\n\n")
                f.write("# Stack Exchange API Key (선택사항)\n")
                f.write("# https://stackapps.com/apps/oauth/register 에서 생성\n")
                f.write("STACKOVERFLOW_KEY=\n")
            logger.info("📝 .env 파일 생성 - API 키를 추가해주세요 (선택사항)")
    
    def install_dependencies(self):
        """의존성 설치"""
        logger.info("📦 필요한 패키지 설치 중...")
        
        # requirements.txt 생성
        req_path = self.base_dir / "requirements_expert.txt"
        with open(req_path, 'w') as f:
            for req in self.requirements:
                f.write(f"{req}\n")
        
        # pip 업그레이드
        logger.info("🔄 pip 업그레이드...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 패키지 설치
        logger.info("📥 패키지 설치 중... (시간이 걸릴 수 있습니다)")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"❌ 패키지 설치 실패: {result.stderr}")
            return False
        
        logger.info("✅ 모든 패키지 설치 완료!")
        return True
    
    def download_model(self):
        """Code Llama 모델 다운로드"""
        logger.info("🤖 Code Llama 7B-Instruct 모델 확인 중...")
        
        model_path = self.base_dir / "CodeLlama-7b-Instruct-hf"
        
        if model_path.exists() and any(model_path.iterdir()):
            logger.info("✅ 모델이 이미 존재합니다!")
            return True
        
        logger.info("📥 모델 다운로드 시작... (약 13GB)")
        
        # 모델 다운로드 스크립트
        download_script = """
import os
from huggingface_hub import snapshot_download
import torch

model_id = "codellama/CodeLlama-7b-Instruct-hf"
local_dir = "./CodeLlama-7b-Instruct-hf"

print(f"모델 다운로드 중: {model_id}")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print("✅ 모델 다운로드 완료!")

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  GPU를 사용할 수 없습니다. CPU로 실행됩니다 (느림)")
"""
        
        # 임시 스크립트 파일 생성
        temp_script = self.base_dir / "temp_download_model.py"
        with open(temp_script, 'w') as f:
            f.write(download_script)
        
        # 스크립트 실행
        result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True)
        
        # 임시 파일 삭제
        temp_script.unlink()
        
        if result.returncode != 0:
            logger.error(f"❌ 모델 다운로드 실패: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def create_startup_scripts(self):
        """시작 스크립트 생성"""
        logger.info("📝 시작 스크립트 생성 중...")
        
        # Windows 배치 파일
        bat_content = r"""@echo off
echo 🚀 C# Expert Learning System 시작...
echo.

REM Python 가상환경 활성화 (있는 경우)
if exist "MyAIWebApp\Models\llm_venv\Scripts\activate.bat" (
    call MyAIWebApp\Models\llm_venv\Scripts\activate.bat
)

REM 전문가 학습 시스템 시작
echo 🧠 24시간 전문가 학습 시스템 시작...
start "Expert Learning" python csharp_expert_crawler.py

REM 5초 대기
timeout /t 5 /nobreak > nul

REM 향상된 서버 시작 (있는 경우)
if exist "MyAIWebApp\\Models\\enhanced_server.py" (
    echo 🌐 AI 서버 시작...
    cd MyAIWebApp\\Models
    start "AI Server" python -m uvicorn enhanced_server:app --host 0.0.0.0 --port 8000 --reload
    cd ..\..
)

echo.
echo ✅ 모든 서비스가 시작되었습니다!
echo.
echo 📊 학습 진행 상황 확인: learning_stats.json
echo 📝 로그 확인: csharp_expert_learning.log
echo.
echo 종료하려면 Ctrl+C를 누르세요...
pause
"""
        
        with open(self.base_dir / "start_expert_learning.bat", 'w') as f:
            f.write(bat_content)
        
        # Linux/Mac 쉘 스크립트
        sh_content = """#!/bin/bash
echo "🚀 C# Expert Learning System 시작..."
echo

# Python 가상환경 활성화 (있는 경우)
if [ -f "MyAIWebApp/Models/llm_venv/bin/activate" ]; then
    source MyAIWebApp/Models/llm_venv/bin/activate
fi

# 전문가 학습 시스템 시작
echo "🧠 24시간 전문가 학습 시스템 시작..."
python csharp_expert_crawler.py &
LEARNING_PID=$!

# 5초 대기
sleep 5

# 향상된 서버 시작 (있는 경우)
if [ -f "MyAIWebApp/Models/enhanced_server.py" ]; then
    echo "🌐 AI 서버 시작..."
    cd MyAIWebApp/Models
    uvicorn enhanced_server:app --host 0.0.0.0 --port 8000 --reload &
    SERVER_PID=$!
    cd ../..
fi

echo
echo "✅ 모든 서비스가 시작되었습니다!"
echo
echo "📊 학습 진행 상황 확인: learning_stats.json"
echo "📝 로그 확인: csharp_expert_learning.log"
echo
echo "종료하려면 Ctrl+C를 누르세요..."

# 종료 시그널 대기
trap "kill $LEARNING_PID $SERVER_PID 2>/dev/null; exit" INT TERM
wait
"""
        
        sh_path = self.base_dir / "start_expert_learning.sh"
        with open(sh_path, 'w') as f:
            f.write(sh_content)
        
        # 실행 권한 부여 (Linux/Mac)
        if os.name != 'nt':
            os.chmod(sh_path, 0o755)
        
        logger.info("✅ 시작 스크립트 생성 완료!")
    
    def create_monitoring_dashboard(self):
        """모니터링 대시보드 생성"""
        logger.info("📊 모니터링 대시보드 생성 중...")
        
        dashboard_html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>C# Expert Learning Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 16px;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #007bff;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .log-container {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 10px;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            max-height: 400px;
            overflow-y: auto;
        }
        .refresh-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #218838;
        }
        .progress-bar {
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #007bff, #0056b3);
            height: 100%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 C# Expert Learning Dashboard</h1>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 새로고침</button>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>📚 수집된 데이터</h3>
                <div class="stat-value" id="total-data">0</div>
            </div>
            <div class="stat-card">
                <h3>⏱️ 총 학습 시간</h3>
                <div class="stat-value" id="training-hours">0h</div>
            </div>
            <div class="stat-card">
                <h3>📈 모델 성능</h3>
                <div class="stat-value" id="model-score">0.00</div>
            </div>
            <div class="stat-card">
                <h3>🔧 코드 개선</h3>
                <div class="stat-value" id="improvements">0</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>학습 진행률</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress" style="width: 0%"></div>
            </div>
            <p id="progress-text">대기 중...</p>
        </div>
        
        <div class="chart-container">
            <h3>최근 활동</h3>
            <div id="recent-activities">
                <p>활동 내역을 불러오는 중...</p>
            </div>
        </div>
        
        <div class="log-container" id="log-output">
            로그를 불러오는 중...
        </div>
    </div>
    
    <script>
        async function refreshData() {
            try {
                // 통계 파일 읽기
                const statsResponse = await fetch('learning_stats.json');
                if (statsResponse.ok) {
                    const stats = await statsResponse.json();
                    
                    document.getElementById('total-data').textContent = 
                        (stats.total_data_collected || 0).toLocaleString();
                    
                    document.getElementById('training-hours').textContent = 
                        (stats.total_training_hours || 0).toFixed(1) + 'h';
                    
                    const latestScore = stats.model_improvements && stats.model_improvements.length > 0
                        ? stats.model_improvements[stats.model_improvements.length - 1].score_change
                        : 0;
                    document.getElementById('model-score').textContent = latestScore.toFixed(2);
                    
                    document.getElementById('improvements').textContent = 
                        (stats.code_improvements_count || 0).toLocaleString();
                }
                
                // 로그 파일 읽기 (최근 50줄)
                const logResponse = await fetch('csharp_expert_learning.log');
                if (logResponse.ok) {
                    const logText = await logResponse.text();
                    const lines = logText.split('\\n').slice(-50).reverse();
                    document.getElementById('log-output').innerHTML = lines.join('<br>');
                }
                
                // 진행률 업데이트
                const now = new Date();
                const hours = now.getHours();
                const progress = (hours / 24) * 100;
                document.getElementById('progress').style.width = progress + '%';
                document.getElementById('progress-text').textContent = 
                    `오늘 진행률: ${progress.toFixed(1)}% (${hours}/24시간)`;
                
            } catch (error) {
                console.error('데이터 로드 오류:', error);
            }
        }
        
        // 초기 로드 및 자동 새로고침
        refreshData();
        setInterval(refreshData, 30000); // 30초마다 새로고침
    </script>
</body>
</html>"""
        
        with open(self.base_dir / "expert_learning_dashboard.html", 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info("✅ 모니터링 대시보드 생성 완료!")
    
    async def start_learning_system(self):
        """학습 시스템 시작"""
        logger.info("🚀 C# 전문가 학습 시스템을 시작합니다...")
        
        # 시스템 요구사항 확인
        if not self.check_system_requirements():
            logger.error("❌ 시스템 요구사항을 충족하지 못했습니다.")
            return
        
        # 환경 설정
        self.setup_environment()
        
        # 의존성 설치
        if not self.install_dependencies():
            logger.error("❌ 의존성 설치에 실패했습니다.")
            return
        
        # 모델 다운로드
        if not self.download_model():
            logger.error("❌ 모델 다운로드에 실패했습니다.")
            return
        
        # 시작 스크립트 생성
        self.create_startup_scripts()
        
        # 모니터링 대시보드 생성
        self.create_monitoring_dashboard()
        
        logger.info("\n" + "="*60)
        logger.info("✅ C# 전문가 학습 시스템 준비 완료!")
        logger.info("="*60)
        logger.info("\n🎯 시작 방법:")
        logger.info("  Windows: start_expert_learning.bat 실행")
        logger.info("  Linux/Mac: ./start_expert_learning.sh 실행")
        logger.info("\n📊 모니터링:")
        logger.info("  브라우저에서 expert_learning_dashboard.html 열기")
        logger.info("\n📝 설정:")
        logger.info("  expert_learning_config.json 편집")
        logger.info("  .env 파일에 API 키 추가 (선택사항)")
        logger.info("\n💡 팁:")
        logger.info("  - GitHub API 토큰을 추가하면 더 많은 데이터 수집 가능")
        logger.info("  - GPU가 있으면 학습 속도가 크게 향상됩니다")
        logger.info("  - 24시간 동안 지속적으로 학습하며 발전합니다")
        logger.info("="*60)

def main():
    """메인 함수"""
    startup = ExpertLearningStartup()
    asyncio.run(startup.start_learning_system())

if __name__ == "__main__":
    main()