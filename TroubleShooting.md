# AutoCI 트러블슈팅 가이드

## 🚀 실행 방법

### 빠른 시작 (권장)
```bash
# AutoCI 루트 디렉토리에서 실행
python start_all.py
```

### 수동 실행 (4개 터미널 필요)

#### 터미널 1: Python AI 서버 (포트 8000)
```bash
# 가상환경 활성화
source llm_venv_wsl/bin/activate  # WSL/Linux
# 또는
llm_venv\Scripts\activate  # Windows

# 서버 실행
python simple_server.py
# 또는 enhanced_server.py 사용
```

#### 터미널 2: 자동 학습 시스템
```bash
python auto_train_collector.py
```

#### 터미널 3: C# Backend (포트 5049)
```bash
cd MyAIWebApp/Backend
dotnet run
```

#### 터미널 4: Frontend (포트 7100/5100)
```bash
cd MyAIWebApp/Frontend
dotnet run
```

## 📋 실행 전 체크리스트

### 1. 필수 소프트웨어 설치 확인
```bash
# Python 버전 확인 (3.8 이상)
python --version

# .NET SDK 확인 (8.0 이상)
dotnet --version

# Node.js 확인 (선택사항)
node --version
```

### 2. Python 환경 설정
```bash
# 가상환경 생성 (처음 한 번만)
python -m venv llm_venv

# 패키지 설치
pip install -r MyAIWebApp/Models/requirements.txt
```

### 3. Code Llama 모델 다운로드
```bash
# 모델 다운로드 (약 13GB)
python download_model.py

# 다운로드 확인
ls -la CodeLlama-7b-Instruct-hf/
```

## 🔧 자주 발생하는 문제와 해결법

### Frontend가 http://localhost:5100에서 실행되지 않는 경우

#### 1. 포트 충돌 확인
```bash
# Windows
netstat -ano | findstr :5100
netstat -ano | findstr :7100
netstat -ano | findstr :5049
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :5100
lsof -i :7100
lsof -i :5049
lsof -i :8000
```

#### 2. launchSettings.json 확인
Frontend의 포트 설정을 확인하세요:
```json
// MyAIWebApp/Frontend/Properties/launchSettings.json
{
  "profiles": {
    "http": {
      "applicationUrl": "http://localhost:5100"
    },
    "https": {
      "applicationUrl": "https://localhost:7100;http://localhost:5100"
    }
  }
}
```

### Python 관련 문제

#### "transformers 라이브러리가 설치되지 않았습니다"
```bash
# 가상환경 활성화 확인
which python  # Linux/Mac
where python  # Windows

# 패키지 재설치
pip install transformers torch accelerate sentencepiece protobuf
```

#### "No module named 'fastapi'"
```bash
pip install fastapi uvicorn[standard] watchdog
```

#### CUDA/GPU 오류
```bash
# CPU 모드로 전환
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 버전 확인
nvidia-smi

# CUDA 11.8용 PyTorch 설치
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 모델 관련 문제

#### "모델 경로를 찾을 수 없습니다"
```bash
# 모델 존재 확인
ls CodeLlama-7b-Instruct-hf/

# 모델 재다운로드
python download_model.py
```

#### 메모리 부족
- 최소 16GB RAM 필요
- 8-bit 양자화 사용:
```python
# simple_server.py 수정
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,  # 메모리 절약
    device_map="auto"
)
```

### .NET/C# 관련 문제

#### "dotnet: command not found"
```bash
# .NET 설치 확인
dotnet --version

# 설치 필요시
# https://dotnet.microsoft.com/download
```

#### HTTP/HTTPS 리다이렉션 문제
```csharp
// Backend/Program.cs에서 HTTPS 리다이렉션 비활성화
// app.UseHttpsRedirection(); // 주석 처리
```

### CORS 문제

#### Frontend-Backend 연결 실패
```csharp
// Backend/Program.cs 확인
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowBlazorClient",
        builder => builder
            .WithOrigins("https://localhost:7100", "http://localhost:5100")
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials());
});

// 미들웨어 순서 확인
app.UseCors("AllowBlazorClient");  // UseRouting() 다음에 위치
```

### WSL 특화 문제

#### localhost 접근 문제
```bash
# WSL IP 확인
hostname -I

# Windows에서 WSL 서비스 접근
# localhost 대신 WSL IP 사용
```

#### 파일 권한
```bash
chmod +x start_all.py
chmod +x download_model.py
chmod +x start.sh
```

## 🛠️ 디버깅 방법

### 1. 서비스별 상태 확인
- Python AI Server: http://localhost:8000/status
- Backend Swagger: http://localhost:5049/swagger
- Frontend: http://localhost:7100

### 2. 상세 로그 활성화
```bash
# Python 서버
uvicorn simple_server:app --log-level debug

# .NET 애플리케이션
dotnet run --verbosity detailed
```

### 3. 네트워크 테스트
```bash
# API 엔드포인트 테스트
curl http://localhost:8000/generate -X POST -H "Content-Type: application/json" -d '{"prompt":"Hello"}'

# Backend 상태 확인
curl http://localhost:5049/api/ai/status
```

## 💡 성능 최적화 팁

### 1. GPU 사용
```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 2. 모델 캐싱
```python
# 환경 변수 설정
os.environ['TRANSFORMERS_CACHE'] = './model_cache'
```

### 3. 배치 처리
```python
# 여러 요청을 한 번에 처리
batch_size = 4  # GPU 메모리에 따라 조정
```

## 🆘 긴급 복구

### 모든 프로세스 종료
```bash
# Windows
taskkill /F /IM python.exe
taskkill /F /IM dotnet.exe

# Linux/Mac
pkill -f python
pkill -f dotnet
```

### 완전 재설치
```bash
# 가상환경 삭제
rm -rf llm_venv llm_venv_wsl

# 캐시 삭제
rm -rf ~/.cache/huggingface
rm -rf model_cache

# 재설치
python -m venv llm_venv
source llm_venv/bin/activate
pip install -r MyAIWebApp/Models/requirements.txt
```

## 📞 추가 지원

문제 해결이 어려운 경우:
1. 전체 에러 로그 수집
2. 시스템 정보: OS, Python 버전, .NET 버전
3. 실행한 명령어 순서
4. GitHub Issues에 상세 내용 제출