# 🤖 AutoCI - 24시간 AI 코딩 공장

<div align="center">
  <img src="https://img.shields.io/badge/Code%20Llama-7B--Instruct-blue?style=for-the-badge" alt="Code Llama">
  <img src="https://img.shields.io/badge/C%23-Expert%20Level-purple?style=for-the-badge" alt="C# Expert">
  <img src="https://img.shields.io/badge/24%2F7-Learning-green?style=for-the-badge" alt="24/7 Learning">
</div>

## 📋 프로젝트 개요

AutoCI는 Code Llama 7B-Instruct 모델을 기반으로 C# 전문가 수준의 AI 코딩 어시스턴트를 구축하는 프로젝트입니다. 24시간 자동 학습을 통해 지속적으로 발전하며, 실시간으로 코드를 개선할 수 있는 개인 코딩 공장입니다.

### 🚀 주요 개선사항
1. **WSL 터미널 통합**: `autoci` 명령으로 어디서나 즉시 실행
2. **고급 C# 데이터 수집**: GitHub Stars 10,000+, Stack Overflow Score 50+ 프로젝트 자동 수집
3. **벡터 기반 인덱싱**: FAISS + Sentence Transformers로 고속 검색
4. **학습 결과 자동 저장**: `autoci_reports/` 폴더에 MD 파일로 정리
5. **Dual Phase System**: RAG (즉시 활용) + Fine-tuning (백그라운드 학습) 동시 실행


1분마다 AI 모델의 학습환경을 모니터링 해야합니다.


## 💻 24시간 자동 코드 수정 시스템

### 🎯 핵심 기능

#### 1. 터미널 명령어로 즉시 실행
```bash
# WSL 터미널에서
autoci enhance start /path/to/project  # 프로젝트 분석 및 개선
autoci create "Unity PlayerController with jump"  # 코드 생성
autoci improve /path/to/file.cs  # 특정 파일 개선
autoci fix "null reference exception in GameManager"  # 버그 수정
```

#### 2. 고급 데이터 인덱싱
```bash
# 전문가 데이터 수집 (GitHub, Stack Overflow, MS Docs)
autoci enhance collect

# 벡터 기반 인덱싱 (FAISS + Sentence Transformers)
autoci data index

# 인덱싱 상태 확인
autoci status
```

#### 3. Dual Phase System (동시 실행)
```bash
# 1단계: 즉시 사용 가능한 RAG 시스템
autoci rag start  # http://localhost:8000/rag

# 2단계: 백그라운드 파인튜닝 (578개 전문가 데이터)
autoci dual start  # 모니터링: http://localhost:8080
```

#### 4. 학습 결과 저장
```bash
# 자동 생성되는 리포트
autoci_reports/
├── final_report_20250625_150000.md     # 전체 분석 결과
├── improvement_summary_20250625.md      # 개선 요약
└── learning_metrics_20250625.json      # 학습 메트릭

# 리포트 확인
autoci enhance report
autoci dual report
```

### 🎯 핵심 목표
- **C# 박사/전문가 수준의 AI 모델 구축**
- **24시간 자동 학습 및 개선 시스템**
- **실시간 코드 생성, 검색, 개선 서비스**

### 🚀 3대 핵심 기능

1. **🧠 AI 코드 생성** (Code Llama 7B-Instruct)
   - C# 전문 코드 생성
   - 디자인 패턴 적용
   - 베스트 프랙티스 준수

2. **🔍 지능형 코드 검색** (ML.NET)
   - 의미 기반 검색
   - 코드 유사도 분석
   - 실시간 인덱싱

3. **💬 프로젝트 Q&A** (RAG 시스템)
   - README 기반 질의응답
   - 프로젝트 문서 이해
   - 컨텍스트 기반 답변

## 🛠️ 시스템 요구사항

### 최소 사양
- **RAM**: 16GB (Code Llama 7B 실행에 필요)
- **GPU**: CUDA 지원 GPU 권장 (선택사항, CPU만으로도 실행 가능)
- **Storage**: 50GB (모델 파일 약 13GB + 학습 데이터)
- **Python**: 3.8 이상
- **.NET**: 8.0 SDK 이상
- **OS**: Windows 10/11, Linux, macOS, WSL2

### 권장 사양
- **RAM**: 32GB 이상
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 100GB+ SSD
- **Internet**: 안정적인 연결 (데이터 수집)

### 필수 소프트웨어
- **Git**: 프로젝트 클론용
- **Visual Studio Code** 또는 Visual Studio 2022 (선택사항)
- **Node.js**: Frontend 개발용 (선택사항)

## ⚡ 빠른 시작 (5분 안에 시작하기)

### 1단계: 전체 시스템 자동 설치 및 실행
```bash
# 1. 프로젝트 클론
git clone https://github.com/yourusername/AutoCI.git
cd AutoCI

# 2. 전문가 학습 시스템 설치 및 시작
python start_expert_learning.py

# 3. 한 번에 모든 서비스 시작
python start_all.py
```

### 2단계: 웹 인터페이스 접속
- **AI 코드 생성**: http://localhost:7100/codegen
- **스마트 검색**: http://localhost:7100/codesearch
- **프로젝트 Q&A**: http://localhost:7100/rag
- **학습 대시보드**: http://localhost:8080/dashboard

## 🚀 WSL 환경 설정 (Windows 사용자)

### WSL2 설치 및 설정
```bash
# PowerShell 관리자 권한으로 실행
wsl --install
wsl --set-default-version 2

# Ubuntu 설치 후
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git build-essential -y
```

### AutoCI 명령어 설정
```bash
# AutoCI 디렉토리에서
chmod +x setup_autoci.sh
./setup_autoci.sh
source ~/.bashrc

# 이제 어디서나 'autoci' 명령 사용 가능!
autoci  # 대화형 모드 시작
```

### 🖥️ WSL 터미널에서 즉시 시작
```bash
# WSL 터미널에서
autoci  # 자동으로 가상환경 활성화 + 시스템 초기화 + 대화형 모드

# 또는 특정 명령 실행
autoci enhance start /path/to/project  # 24시간 자동 코드 수정
autoci create "Unity Player Controller"  # 코드 생성
autoci improve GameManager.cs  # 코드 개선
```

### 💡 자동 초기화 기능
`autoci` 명령 실행 시 자동으로:
- ✅ 가상환경 활성화 (llm_venv_wsl)
- ✅ 전문가 데이터 확인 및 수집
- ✅ 벡터 인덱싱 자동 실행
- ✅ Dual Phase 시스템 백그라운드 시작
- ✅ 대화형 모드로 진입

### 💡 자동 초기화 기능
`autoci` 명령 실행 시 자동으로:
- ✅ 가상환경 활성화 (llm_venv_wsl)
- ✅ 전문가 데이터 확인 및 수집
- ✅ 벡터 인덱싱 자동 실행
- ✅ Dual Phase 시스템 백그라운드 시작
- ✅ 대화형 모드로 진입

### Windows에서 WSL 서비스 접속
1. WSL IP 확인:
   ```bash
   hostname -I
   ```
2. Windows 브라우저에서 `http://[WSL_IP]:7100` 접속

### Windows 방화벽 설정 (관리자 PowerShell)
```powershell
# WSL 포트 허용
New-NetFirewallRule -DisplayName "AutoCI" -Direction Inbound -LocalPort 7100,8000,8080 -Protocol TCP -Action Allow
```

## 📚 상세 구축 가이드

### 1. Code Llama 7B-Instruct 설정

#### 1.1 환경 준비
```bash
# Python 가상환경 생성
python -m venv llm_venv
source llm_venv/bin/activate  # Linux/Mac
llm_venv\Scripts\activate     # Windows

# 필요 패키지 설치 (자동으로 설치됨)
pip install -r requirements_expert.txt
```

#### 1.2 모델 자동 다운로드
```bash
# start_expert_learning.py가 자동으로 처리
# 수동 다운로드가 필요한 경우:
python download_model.py
```

#### 1.3 AI 서버 실행
```bash
# 자동 실행 (start_all.py 사용 시)
# 수동 실행:
cd MyAIWebApp/Models
uvicorn enhanced_server:app --host 0.0.0.0 --port 8000
```

### 2. 24시간 C# 전문가 학습 시스템

#### 2.1 자동 학습 아키텍처
```
┌─────────────────────────────────────────────────┐
│           24시간 학습 사이클                      │
├─────────────────────────────────────────────────┤
│  4시간: GitHub/StackOverflow 데이터 수집         │
│  1시간: 데이터 전처리 및 품질 검증               │
│  6시간: Code Llama 모델 파인튜닝                │
│  1시간: 모델 평가 및 배포                       │
│ 12시간: 실시간 코드 개선 서비스                 │
└─────────────────────────────────────────────────┘
```

#### 2.2 학습 데이터 소스
- **GitHub**: Stars 10,000+ C# 프로젝트
- **Stack Overflow**: Score 50+ C# Q&A
- **Microsoft Docs**: 공식 C# 문서
- **사용자 프로젝트**: 로컬 C# 코드

#### 2.3 코드 품질 평가 기준
| 평가 항목 | 가중치 | 설명 |
|-----------|--------|------|
| XML 문서 주석 | 20% | /// 주석 포함 여부 |
| 디자인 패턴 | 15% | SOLID, GoF 패턴 사용 |
| 현대적 C# 기능 | 15% | async/await, LINQ, 패턴 매칭 |
| 에러 처리 | 10% | try-catch, 예외 처리 |
| 코드 구조 | 10% | 적절한 길이, 모듈화 |
| 테스트 코드 | 5% | 단위 테스트 포함 |

### 3. ML.NET 기반 지능형 코드 검색

C# Backend의 `SearchService.cs`에서 ML.NET을 사용한 지능형 코드 검색 구현:
- TF-IDF 기반 텍스트 임베딩
- 코사인 유사도를 이용한 검색
- 실시간 코드 인덱싱

### 4. RAG 기반 프로젝트 Q&A

C# Backend의 `RAGService.cs`에서 README 기반 질의응답 시스템 구현:
- README.md 파일 자동 파싱
- ML.NET을 이용한 문서 검색
- 컨텍스트 기반 답변 생성

### 5. 🚀 통합 실행

#### 5.1 한 번에 모든 서비스 시작
```bash
# 모든 서비스 자동 시작
python start_all.py
```

#### 5.2 개별 서비스 실행
```bash
# 24시간 전문가 학습 시스템
python csharp_expert_crawler.py

# AI 모델 서버
cd MyAIWebApp/Models
uvicorn enhanced_server:app --host 0.0.0.0 --port 8000

# C# Backend
cd MyAIWebApp/Backend
dotnet run

# Blazor Frontend
cd MyAIWebApp/Frontend
dotnet run

# 모니터링 API
python expert_learning_api.py
```

## 💡 주요 사용 시나리오

### 1. AI 코드 생성
```csharp
// 입력: "Repository 패턴을 사용한 사용자 관리 서비스 생성"
// 출력: 완전한 C# 코드 with 베스트 프랙티스
public interface IUserRepository { ... }
public class UserRepository : IUserRepository { ... }
public class UserService { ... }
```

### 2. 스마트 코드 검색
- "비동기 파일 업로드 구현" → 관련 코드 즉시 검색
- "Entity Framework 성능 최적화" → 최적화 패턴 제시

### 3. 프로젝트 Q&A
- "이 프로젝트의 인증 방식은?" → README 기반 정확한 답변
- "API 엔드포인트 목록?" → 프로젝트 구조 분석 후 답변

### 4. 24시간 코드 개선
- 자동으로 코드 품질 분석
- 개선 제안 파일 생성 (*_improvements.md)
- 실시간 리팩토링 추천

## ⚠️ 주의사항 및 최적화 팁

### 필수 확인사항
- ✅ RAM 16GB 이상 (32GB 권장)
- ✅ Python 3.8 이상
- ✅ 안정적인 인터넷 연결
- ✅ 50GB 이상 여유 공간

### 성능 최적화
1. **GPU 사용** (10배 빠른 학습)
   ```bash
   # CUDA 설치 확인
   nvidia-smi
   ```

2. **API 키 설정** (더 많은 데이터 수집)
   ```bash
   # .env 파일 편집
   GITHUB_TOKEN=your_github_token
   STACKOVERFLOW_KEY=your_stack_key
   ```

3. **메모리 최적화**
   - 8-bit 양자화 사용 (메모리 50% 절약)
   - Gradient checkpointing 활성화

## 📊 모니터링 및 관리

### 웹 대시보드
```bash
# 브라우저에서 열기
http://localhost:8080/dashboard/expert_learning_dashboard.html
```

### 실시간 모니터링
- 📈 학습 진행률
- 📊 수집된 데이터 통계
- 🎯 모델 성능 지표
- 🔧 코드 개선 횟수

### API 엔드포인트
| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/status` | GET | 현재 시스템 상태 |
| `/api/start` | POST | 학습 시작 |
| `/api/stop` | POST | 학습 중지 |
| `/api/stats` | GET | 통계 조회 |
| `/api/improve` | POST | 코드 개선 요청 |
| `/api/logs` | GET | 로그 조회 |

## 🔧 종합 문제 해결 가이드

### 📋 실행 전 체크리스트
```bash
# 필수 소프트웨어 확인
python --version  # 3.8 이상
dotnet --version  # 8.0 이상
nvidia-smi       # GPU 사용 시

# 모델 파일 확인
ls -la CodeLlama-7b-Instruct-hf/
```

### 🐍 Python 관련 문제

#### "transformers 라이브러리가 설치되지 않았습니다" 오류
```bash
# 가상환경 활성화 확인
which python  # Linux/Mac/WSL
where python  # Windows

# 필수 패키지 재설치
pip install transformers torch accelerate sentencepiece protobuf
```

#### "No module named 'fastapi'" 오류
```bash
pip install fastapi uvicorn[standard] watchdog
```

#### CUDA/GPU 오류 해결
```bash
# CPU 모드로 전환 (GPU 없을 때)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 버전 확인 후 재설치
nvidia-smi  # CUDA 버전 확인
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8용
```

#### 메모리 부족 오류
```python
# enhanced_server.py에서 8-bit 양자화 활성화
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,  # 메모리 50% 절약
    device_map="auto"
)
```

### 💻 .NET/C# 관련 문제

#### "dotnet: command not found" 오류
```bash
# .NET SDK 설치 확인
dotnet --version

# 설치 필요시: https://dotnet.microsoft.com/download
```

#### 포트 사용 중 오류
```bash
# Windows
netstat -ano | findstr :5049
netstat -ano | findstr :7100

# Linux/Mac/WSL
lsof -i :5049
lsof -i :7100

# 프로세스 종료 후 재실행
```

#### Frontend가 http://localhost:5100에서 실행되는 경우
```json
// MyAIWebApp/Frontend/Properties/launchSettings.json 확인
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

### 🌐 네트워크/연결 문제

#### Frontend-Backend 연결 실패
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
```bash
# 서버 상태 확인
curl http://localhost:8000/status

# LlamaService.cs URL 확인
private readonly string _pythonApiUrl = "http://localhost:8000";
```

### 🐧 WSL 특화 문제

#### localhost 접근 문제
```bash
# WSL IP 확인
hostname -I

# Windows hosts 파일에 추가 (관리자 권한)
# C:\Windows\System32\drivers\etc\hosts
# [WSL_IP] wsl.local
```

#### 파일 권한 문제
```bash
chmod +x start_all.py
chmod +x download_model.py
chmod +x wsl_setup.sh
```

#### WSL 메모리 제한 설정
```bash
# Windows 사용자 홈에 .wslconfig 생성
# C:\Users\[사용자명]\.wslconfig
[wsl2]
memory=16GB
swap=8GB
```

### 🔄 모델 관련 문제

#### 모델 다운로드 문제
```bash
# 수동으로 모델 다운로드
python download_model.py

# 모델 존재 확인
python download_model.py --check-only

# Hugging Face CLI로 직접 다운로드
pip install huggingface-hub
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir ./CodeLlama-7b-Instruct-hf
```

#### 모델 경로를 찾을 수 없습니다
```bash
# 모델 존재 확인
ls CodeLlama-7b-Instruct-hf/

# 모델 재다운로드
rm -rf CodeLlama-7b-Instruct-hf
python download_model.py
```

### 🆘 긴급 복구

#### 모든 프로세스 강제 종료
```bash
# Windows
taskkill /F /IM python.exe
taskkill /F /IM dotnet.exe

# Linux/Mac/WSL
pkill -f python
pkill -f dotnet
```

#### 완전 재설치
```bash
# 가상환경 삭제 및 재생성
rm -rf llm_venv llm_venv_wsl
python -m venv llm_venv
source llm_venv/bin/activate  # 또는 llm_venv\Scripts\activate (Windows)

# 캐시 삭제
rm -rf ~/.cache/huggingface
rm -rf model_cache

# 패키지 재설치
cd MyAIWebApp/Models
pip install -r requirements.txt

# 모델 재다운로드
cd ../..
python download_model.py
```

### 📊 디버깅 방법

#### 상세 로그 활성화
```bash
# Python 서버
uvicorn enhanced_server:app --log-level debug

# .NET 애플리케이션
dotnet run --verbosity detailed
```

#### 서비스 상태 확인
- Python AI Server: http://localhost:8000/docs
- Backend Swagger: http://localhost:5049/swagger
- Frontend: http://localhost:7100
- 모니터링 API: http://localhost:8080/api/status

#### 네트워크 테스트
```bash
# API 엔드포인트 테스트
curl http://localhost:8000/generate -X POST -H "Content-Type: application/json" -d '{"prompt":"Hello"}'

# Backend 상태 확인
curl http://localhost:5049/api/ai/status
```

## 🧠 고급 기능: 24시간 전문가 학습 시스템

### 자동 진화하는 AI
Code Llama가 24시간 동안 지속적으로 학습하여 C# 전문가로 성장:

```mermaid
graph LR
    A[데이터 수집] --> B[품질 평가]
    B --> C[모델 학습]
    C --> D[성능 평가]
    D --> E[서비스 배포]
    E --> A
```

### 학습 데이터 품질 기준
```python
# 최소 품질 점수: 0.8/1.0
quality_criteria = {
    "has_xml_docs": 0.20,      # XML 문서화
    "uses_patterns": 0.15,     # 디자인 패턴
    "modern_csharp": 0.15,     # 최신 C# 기능
    "follows_solid": 0.15,     # SOLID 원칙
    "error_handling": 0.10,    # 예외 처리
    "appropriate_length": 0.10  # 적절한 크기
}
```

### 수집되는 전문 지식
1. **GitHub 프로젝트** (Stars 10,000+)
   - dotnet/roslyn
   - dotnet/aspnetcore
   - Unity-Technologies/UnityCsReference
   - JamesNK/Newtonsoft.Json

2. **Stack Overflow** (Score 50+)
   - 베스트 프랙티스 Q&A
   - 성능 최적화 팁
   - 디버깅 솔루션

3. **Microsoft 공식 문서**
   - C# 언어 레퍼런스
   - .NET API 문서
   - 디자인 가이드라인

### 실시간 코드 개선 예시
```csharp
// 😰 개선 전 (품질 점수: 0.4)
public class UserManager {
    public List<User> users = new List<User>();
    public void AddUser(string name) {
        users.Add(new User { Name = name });
    }
}

// 😊 AI 개선 후 (품질 점수: 0.9)
/// <summary>
/// 사용자 관리를 위한 서비스
/// </summary>
public interface IUserManager {
    Task<User> AddUserAsync(string name, CancellationToken cancellationToken = default);
}

public class UserManager : IUserManager {
    private readonly IUserRepository _repository;
    private readonly ILogger<UserManager> _logger;
    
    public UserManager(IUserRepository repository, ILogger<UserManager> logger) {
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }
    
    public async Task<User> AddUserAsync(string name, CancellationToken cancellationToken = default) {
        if (string.IsNullOrWhiteSpace(name)) {
            throw new ArgumentException("사용자 이름은 필수입니다.", nameof(name));
        }
        
        try {
            var user = new User { Name = name, CreatedAt = DateTime.UtcNow };
            await _repository.AddAsync(user, cancellationToken);
            _logger.LogInformation("새 사용자 추가: {UserName}", name);
            return user;
        }
        catch (Exception ex) {
            _logger.LogError(ex, "사용자 추가 실패: {UserName}", name);
            throw;
        }
    }
}
```

## 🎯 프로젝트 구조

```
AutoCI/
├── 📁 MyAIWebApp/
│   ├── 📁 Backend/         # ASP.NET Core API
│   │   ├── Services/       # AI, 검색, RAG 서비스
│   │   └── Controllers/    # API 컨트롤러
│   ├── 📁 Frontend/        # Blazor WebAssembly
│   │   ├── Pages/          # UI 페이지
│   │   └── wwwroot/        # 정적 파일
│   └── 📁 Models/          # Python AI 모델
│       ├── enhanced_server.py
│       └── fine_tune.py
├── 📁 expert_training_data/  # 학습 데이터
├── 📄 csharp_expert_crawler.py  # 24시간 학습 엔진
├── 📄 start_expert_learning.py  # 설치 스크립트
├── 📄 expert_learning_api.py    # 모니터링 API
└── 📄 start_all.py              # 통합 실행
```

## 🔒 보안 및 프로덕션 고려사항

### API 키 관리
```bash
# .env 파일 생성
cat > .env << EOF
GITHUB_TOKEN=your_github_token
STACKOVERFLOW_KEY=your_stack_key
HUGGINGFACE_TOKEN=your_hf_token
EOF

# 환경 변수 사용
export $(cat .env | xargs)
```

### CORS 설정
```csharp
// 프로덕션 환경에서는 특정 도메인만 허용
builder.Services.AddCors(options =>
{
    options.AddPolicy("Production",
        builder => builder
            .WithOrigins("https://yourdomain.com")
            .AllowAnyMethod()
            .AllowAnyHeader());
});
```

### 입력 검증
- 사용자 입력 길이 제한 (최대 2000자)
- SQL 인젝션 방지
- 파일 경로 검증
- Rate limiting 적용

## 💡 성능 최적화 팁

### 1. GPU 가속 (10배 빠른 처리)
```bash
# CUDA 설치 확인
nvidia-smi

# GPU 메모리 확인
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. 모델 캐싱
```python
# 환경 변수 설정
export TRANSFORMERS_CACHE=./model_cache
export HF_HOME=./huggingface_cache
```

### 3. 배치 처리 최적화
```python
# enhanced_server.py에서 배치 크기 조정
batch_size = 4  # GPU 메모리에 따라 조정
```

### 4. 메모리 최적화
- 8-bit 양자화 사용 (메모리 50% 절약)
- Gradient checkpointing 활성화
- 불필요한 프로세스 종료

### 5. 백그라운드 실행 (tmux 사용)
```bash
# tmux 설치 및 실행
sudo apt install tmux
tmux new -s autoci

# autoci 실행
autoci

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t autoci
```

## 📁 프로젝트 구조 상세

```
AutoCI/
├── 📁 MyAIWebApp/
│   ├── 📁 Backend/         # ASP.NET Core API
│   │   ├── Services/       # AI, Search, RAG 서비스
│   │   │   ├── AIService.cs
│   │   │   ├── LlamaService.cs
│   │   │   ├── SearchService.cs
│   │   │   └── RAGService.cs
│   │   ├── Controllers/    # API 컨트롤러
│   │   └── Properties/     # 설정 파일
│   ├── 📁 Frontend/        # Blazor WebAssembly
│   │   ├── Pages/          # UI 페이지
│   │   │   ├── CodeGenerator.razor
│   │   │   ├── CodeSearch.razor
│   │   │   └── RAG.razor
│   │   └── wwwroot/        # 정적 파일
│   └── 📁 Models/          # Python AI 모델
│       ├── enhanced_server.py
│       ├── simple_server.py
│       └── requirements.txt
├── 📁 expert_training_data/  # 학습 데이터
├── 📁 learning_results/      # 학습 결과 보고서
├── 📄 csharp_expert_crawler.py  # 24시간 학습 엔진
├── 📄 start_expert_learning.py  # 설치 스크립트
├── 📄 expert_learning_api.py    # 모니터링 API
├── 📄 auto_train_collector.py   # 자동 학습 수집기
├── 📄 advanced_indexer.py       # 고급 데이터 인덱서
├── 📄 dual_phase_system.py      # RAG + 파인튜닝 시스템
├── 📄 autoci_terminal.py        # CLI 인터페이스
├── 📄 wsl_start_all.py          # WSL 전용 실행기
└── 📄 start_all.py              # 통합 실행
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

MIT License - 자유롭게 사용하세요!

## 🙏 감사의 말

- Meta AI의 Code Llama 팀
- Microsoft의 ML.NET 팀
- 오픈소스 커뮤니티

## ✅ 구현 완료 상태

### 최종 검증 완료 (2025-06-24)
- **검증 도구**: complete_verification.py
- **총 검증 항목**: 79개
- **성공률**: 100%

### 구현된 주요 기능
- ✅ Code Llama 7B-Instruct 기반 C# 코드 생성
- ✅ ML.NET 기반 지능형 코드 검색
- ✅ RAG 시스템 기반 README Q&A
- ✅ 24시간 자동 학습 시스템
- ✅ 실시간 코드 품질 개선
- ✅ WSL 환경 완벽 지원
- ✅ 웹 기반 모니터링 대시보드
- ✅ CLI 터미널 인터페이스
- ✅ 듀얼 페이즈 학습 시스템 (RAG + 파인튜닝)

### 포트 및 엔드포인트
- ✅ AI Server: 8000
- ✅ Monitoring API: 8080
- ✅ Backend: 5049
- ✅ Frontend: 7100
- ✅ 모든 API 엔드포인트 작동 확인

---

<div align="center">
  <h3>🚀 24시간 동안 진화하는 당신만의 AI 코딩 공장을 만들어보세요!</h3>
  <p>문의사항이나 버그 리포트는 Issues에 남겨주세요.</p>
</div>