# AutoCI 실행 가이드 및 문제 해결

## 🚀 실행 방법

### 1. 빠른 시작 (권장)

```bash
# AutoCI 루트 디렉토리에서 실행
python start_all.py
```

이 명령어 하나로 모든 서비스가 자동으로 시작됩니다.

### 2. 수동 실행 (개별 서비스)

#### 2.1 환경 준비
```bash
# Python 가상환경 생성 (처음 한 번만)
python -m venv llm_venv

# 가상환경 활성화
# Windows
llm_venv\Scripts\activate

# Linux/Mac/WSL
source llm_venv/bin/activate

# 패키지 설치
cd MyAIWebApp/Models
pip install -r requirements.txt
```

#### 2.2 모델 다운로드 (처음 한 번만)
```bash
# AutoCI 루트에서 실행
python download_model.py
```

#### 2.3 서비스 개별 실행

**터미널 1: Python AI 서버**
```bash
# 가상환경 활성화 후
python simple_server.py
# 또는
uvicorn simple_server:app --host 0.0.0.0 --port 8000 --reload
```

**터미널 2: 자동 학습 시스템**
```bash
python auto_train_collector.py
```

**터미널 3: C# Backend**
```bash
cd MyAIWebApp/Backend
dotnet run
```

**터미널 4: Frontend**
```bash
cd MyAIWebApp/Frontend
dotnet run
```

## 🔧 일반적인 문제 해결

### 1. Python 관련 문제

#### "transformers 라이브러리가 설치되지 않았습니다" 오류
```bash
pip install transformers torch accelerate sentencepiece
```

#### "No module named 'fastapi'" 오류
```bash
pip install fastapi uvicorn[standard]
```

#### CUDA/GPU 관련 오류
```bash
# CPU만 사용하도록 설정
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 또는 CUDA 버전 확인 후 재설치
nvidia-smi  # CUDA 버전 확인
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8용
```

### 2. 모델 관련 문제

#### "모델 경로를 찾을 수 없습니다" 오류
```bash
# 모델 다운로드
python download_model.py

# 다운로드 확인
python download_model.py --check-only
```

#### 메모리 부족 오류
- 최소 16GB RAM 필요
- 다른 프로그램 종료 후 재시도
- 8-bit 양자화 사용:
```python
# enhanced_server.py에서 load_in_8bit=True 설정 확인
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True  # 메모리 절약
)
```

### 3. .NET/C# 관련 문제

#### "dotnet: command not found" 오류
```bash
# .NET SDK 설치 확인
dotnet --version

# 설치되지 않은 경우
# https://dotnet.microsoft.com/download 에서 .NET 8.0 SDK 다운로드
```

#### 포트 사용 중 오류
```bash
# Windows
netstat -ano | findstr :5049
netstat -ano | findstr :7100

# Linux/Mac
lsof -i :5049
lsof -i :7100

# 프로세스 종료 후 재실행
```

### 4. 네트워크/연결 문제

#### Frontend가 Backend에 연결되지 않음
1. `MyAIWebApp/Frontend/Program.cs` 확인:
```csharp
builder.Services.AddScoped(sp => new HttpClient 
{ 
    BaseAddress = new Uri("http://localhost:5049/")  // Backend 주소 확인
});
```

2. CORS 설정 확인 (`MyAIWebApp/Backend/Program.cs`):
```csharp
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowBlazorClient",
        builder => builder
            .WithOrigins("https://localhost:7100", "http://localhost:5100")
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials());
});
```

#### Python 서버 연결 실패
1. Python 서버 실행 확인:
```bash
curl http://localhost:8000/status
```

2. `MyAIWebApp/Backend/Services/LlamaService.cs`에서 URL 확인:
```csharp
private readonly string _pythonApiUrl = "http://localhost:8000";
```

### 5. WSL 특화 문제

#### WSL에서 localhost 접근 문제
```bash
# WSL IP 확인
hostname -I

# Windows hosts 파일에 추가 (관리자 권한)
# C:\Windows\System32\drivers\etc\hosts
# WSL_IP wsl.local
```

#### 파일 권한 문제
```bash
chmod +x start_all.py
chmod +x download_model.py
```

## 📝 체크리스트

### 시작 전 확인사항
- [ ] Python 3.8 이상 설치
- [ ] .NET 8.0 SDK 설치
- [ ] 16GB 이상 RAM
- [ ] 20GB 이상 여유 공간

### 첫 실행 시
1. [ ] 가상환경 생성 및 활성화
2. [ ] requirements.txt 설치
3. [ ] Code Llama 모델 다운로드
4. [ ] start_all.py 실행

### 실행 확인
- [ ] http://localhost:8000/docs - Python AI 서버
- [ ] http://localhost:5049/swagger - Backend API
- [ ] http://localhost:7100 - Frontend UI

## 🆘 긴급 해결책

### 모든 서비스 강제 종료
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
# 가상환경 삭제 및 재생성
rm -rf llm_venv
python -m venv llm_venv
source llm_venv/bin/activate  # 또는 llm_venv\Scripts\activate (Windows)

# 패키지 재설치
cd MyAIWebApp/Models
pip install -r requirements.txt

# 모델 재다운로드
cd ../..
python download_model.py
```

### 로그 확인
```bash
# Python 서버 로그
uvicorn simple_server:app --log-level debug

# .NET 로그
dotnet run --verbosity detailed
```

## 💡 팁

1. **개발 중**: `--reload` 옵션 사용으로 코드 변경 시 자동 재시작
2. **성능 향상**: GPU 사용 가능한 경우 CUDA 설치
3. **메모리 절약**: 8-bit 양자화 사용
4. **디버깅**: 각 서비스를 개별 터미널에서 실행하여 로그 확인

## 📞 추가 도움말

문제가 지속되는 경우:
1. 에러 메시지 전체를 복사
2. 시스템 정보 확인: `python --version`, `dotnet --version`
3. GitHub Issues에 문제 제출: https://github.com/[your-repo]/issues