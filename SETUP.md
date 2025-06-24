# AutoCI 설치 및 사용 가이드

## 프로젝트 개요

AutoCI는 AI 기반 지능형 코드 개발 도구로, 다음 세 가지 주요 기능을 제공합니다:

1. **Code Llama 7B-Instruct 기반 코드 생성** - C# 코드 자동 생성
2. **ML.NET 기반 지능형 코드 검색** - 효율적인 코드 탐색 및 검색
3. **RAG 시스템** - README 기반 질의응답 시스템

## 시스템 요구사항

### 하드웨어
- **RAM**: 최소 16GB (Code Llama 7B 실행에 필요)
- **GPU**: CUDA 지원 GPU 권장 (선택사항, CPU만으로도 실행 가능)
- **저장공간**: 최소 20GB (모델 파일 포함)

### 소프트웨어
- **.NET 8.0** 이상
- **Python 3.8** 이상
- **Visual Studio Code** 또는 Visual Studio 2022
- **Git**

## 설치 가이드

### 1. 프로젝트 클론

```bash
git clone [repository-url]
cd AutoCI
```

### 2. Code Llama 모델 다운로드

Code Llama 7B-Instruct 모델이 이미 `CodeLlama-7b-Instruct-hf` 폴더에 다운로드되어 있습니다.
만약 없다면 Hugging Face에서 다운로드하세요:

```bash
# Hugging Face CLI 설치
pip install huggingface-hub

# 모델 다운로드
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir ./CodeLlama-7b-Instruct-hf
```

### 3. Python 환경 설정

```bash
# Python 가상환경 생성
cd MyAIWebApp/Models
python -m venv llm_venv

# 가상환경 활성화
# Windows
llm_venv\Scripts\activate
# Linux/Mac
source llm_venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 4. .NET 패키지 복원

```bash
# 솔루션 루트에서
dotnet restore

# 또는 각 프로젝트에서
cd MyAIWebApp/Backend
dotnet restore

cd ../Frontend
dotnet restore
```

## 실행 방법

### 1. Python 서버 실행 (터미널 1)

```bash
cd MyAIWebApp/Models
# 가상환경 활성화
source llm_venv/bin/activate  # Linux/Mac
# 또는
llm_venv\Scripts\activate  # Windows

# 서버 실행
python simple_server.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

### 2. 백엔드 API 실행 (터미널 2)

```bash
cd MyAIWebApp/Backend
dotnet run
```

API가 `http://localhost:5049`에서 실행됩니다.
Swagger UI는 `http://localhost:5049/swagger`에서 확인할 수 있습니다.

### 3. 프론트엔드 실행 (터미널 3)

```bash
cd MyAIWebApp/Frontend
dotnet run
```

웹 애플리케이션이 `http://localhost:5100`에서 실행됩니다.

## 기능별 사용법

### 1. 코드 생성 (Code Llama)

1. 웹 브라우저에서 `http://localhost:5100` 접속
2. 좌측 메뉴에서 "코드 생성" 클릭
3. 텍스트 영역에 생성하고 싶은 코드에 대한 설명 입력
   - 예: "사용자 인증을 위한 JWT 토큰 생성 메서드"
4. "텍스트 생성" 버튼 클릭
5. 생성된 C# 코드 확인

### 2. 지능형 코드 검색

1. 좌측 메뉴에서 "코드 검색" 클릭
2. 검색어 입력 (예: "async", "controller", "user service")
3. "코드 검색" 버튼 클릭
4. 검색 결과 확인 (유사도 점수와 함께 표시)

#### 코드 인덱싱
1. 하단의 "코드 인덱싱" 섹션으로 이동
2. 파일명과 코드 내용 입력
3. "코드 인덱싱" 버튼 클릭
4. 인덱싱된 코드는 이후 검색에서 찾을 수 있음

### 3. RAG Q&A (README 기반 질의응답)

1. 좌측 메뉴에서 "RAG Q&A" 클릭
2. README 파일 경로 입력 또는 "현재 프로젝트 README 로드" 클릭
3. "README 인덱싱" 버튼 클릭
4. 인덱싱 완료 후, 질문 입력
   - 예: "이 프로젝트의 주요 기능은 무엇인가요?"
5. "질문하기" 버튼 클릭
6. AI가 README 내용을 기반으로 답변 생성

## API 엔드포인트

### Llama Controller
- `POST /api/llama/generate` - 코드 생성

### Search Controller
- `GET /api/search/code?query={query}&maxResults={n}` - 코드 검색
- `POST /api/search/index` - 코드 인덱싱

### RAG Controller
- `POST /api/rag/index-readme` - README 파일 인덱싱
- `POST /api/rag/query` - 질의응답
- `GET /api/rag/documents` - 인덱싱된 문서 목록

## 문제 해결

### Python 서버가 시작되지 않을 때
1. Python 버전 확인: `python --version` (3.8 이상 필요)
2. 가상환경이 활성화되어 있는지 확인
3. 모든 패키지가 설치되어 있는지 확인: `pip list`
4. GPU 메모리 부족 시 CPU 모드로 전환 고려

### 백엔드 API 연결 오류
1. 포트 5049가 사용 중인지 확인
2. `appsettings.json`에서 URL 설정 확인
3. CORS 설정이 올바른지 확인

### 프론트엔드 연결 오류
1. 백엔드 API가 실행 중인지 확인
2. `Program.cs`의 API URL이 올바른지 확인
3. 브라우저 개발자 도구에서 네트워크 오류 확인

## 성능 최적화 팁

1. **Code Llama 최적화**
   - GPU 사용 시 CUDA 설치 확인
   - 메모리 부족 시 `max_length` 파라미터 조정
   - 배치 처리로 여러 요청 동시 처리

2. **코드 검색 최적화**
   - 정기적으로 인덱스 재구축
   - 큰 코드베이스의 경우 청크 크기 조정
   - 검색 결과 캐싱 고려

3. **RAG 시스템 최적화**
   - 문서 분할 크기 조정 (기본값: 800자)
   - 임베딩 차원 조정
   - 유사도 임계값 조정

## 보안 고려사항

1. API 키 관리
   - 환경 변수 사용 권장
   - 코드에 하드코딩 금지

2. CORS 설정
   - 프로덕션 환경에서는 특정 도메인만 허용

3. 입력 검증
   - 사용자 입력 길이 제한
   - SQL 인젝션 방지

## 추가 개발 가이드

### 새로운 기능 추가
1. Backend에 새 Service 추가
2. Controller 생성 및 엔드포인트 정의
3. Frontend에 새 페이지 추가
4. NavMenu.razor에 링크 추가

### 모델 변경
1. `simple_server.py`의 모델 경로 수정
2. 토크나이저 설정 조정
3. 프롬프트 템플릿 수정

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.