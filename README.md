# AutoCI - AI 수정된 Godot과 함께하는 24시간 자동 게임 개발 시스템

WSL 환경에서 AI가 수정된 Godot을 통해 24시간 자동으로 게임을 개발하는 엔터프라이즈급 시스템입니다.

## 🚀 주요 특징

### 1️⃣ **AI 수정된 Godot 엔진**
- 소스 레벨에서 AI 제어 기능 추가
- 자동화 API 및 원격 제어 지원
- 실시간 씬 조작 및 스크립트 주입

### 2️⃣ **24시간 자동 게임 개발**
- 게임 개발 명령 입력 시 24시간 자동 개발
- 실시간 대시보드로 진행 상황 모니터링
- 자동 버그 수정 및 최적화

### 3️⃣ **24시간 C# 학습 시스템**
- 6단계 체계적 커리큘럼
- 실제 24시간 동안 진행 (버그 수정 완료)
- 학습 진도 자동 저장 및 재개

### 4️⃣ **다중 LLM 모델 통합 (NEW)**
- Llama-3.1-8B, CodeLlama-13B, Qwen2.5-Coder-32B 지원
- 자동 모델 설치 및 양자화 최적화
- 모델별 특화 기능 활용 (한글, C#, Godot)

### 5️⃣ **24시간 연속 학습 시스템 (NEW)**
- C# 프로그래밍 언어 전문 학습
- 한글 프로그래밍 용어 학습
- Godot 엔진 개발 방향성 분석
- Godot 내장 네트워킹 (AI 제어)
- Nakama 서버 개발 (AI 최적화)
- 질문-답변 자동 생성 및 분석
- 지식 베이스 자동 구축 및 확장

### 6️⃣ **Godot 내장 네트워킹 AI 통합 (NEW)**
- Godot MultiplayerAPI를 AI가 완전 제어
- AI 기반 멀티플레이어 게임 개발
- 실시간 네트워크 최적화 및 동기화
- 지능형 예측 및 지연 보상

### 7️⃣ **Nakama 오픈소스 서버 AI 통합 (NEW)**
- AI가 제어하는 Nakama 백엔드 서버
- 지능형 매치메이킹 및 로비 시스템
- AI 기반 플레이어 데이터 관리
- 실시간 채팅 및 소셜 기능 모더레이션
- 대규모 멀티플레이어 최적화

### 8️⃣ **AI 기반 Godot 엔진 개선 (NEW)**
- 24시간 학습 데이터 기반 엔진 개선
- C# 바인딩 최적화
- 한글 지원 향상
- 네트워킹 성능 개선
- Nakama 통합 최적화
- `autoci fix` 명령으로 자동 개선

### 9️⃣ **DeepSeek-Coder-v2 6.7B 통합 (NEW)**
- RTX 2080 8GB 최적화 코딩 특화 모델
- C# 및 Godot 개발에 특화된 답변 제공
- 4bit 양자화로 6GB VRAM 사용
- `autoci learn low` 명령어로 사용

## 📋 요구사항

- **OS**: Windows 10/11 + WSL2 (Ubuntu 20.04+)
- **RAM**: 16GB 이상 (32GB 권장) - LLM 모델 실행용
- **GPU**: CUDA 지원 GPU (8GB+ VRAM 권장)
- **디스크**: 100GB 이상 여유 공간 (모델 저장용)
- **도구**: Python 3.8+, Git, CUDA Toolkit

### 🧠 메모리 최적화 (NEW)
- **자동 메모리 관리**: 시스템 메모리에 따른 모델 로딩/언로딩
- **모델 순차 실행**: 3개 모델을 동시가 아닌 순차적으로 사용
- **메모리 모니터링**: 실시간 메모리 사용량 추적 및 자동 정리
- **설정 가능한 제한**: 시스템에 맞는 메모리 제한 설정 가능

## 🎯 빠른 시작 (5분)

### 1단계: 설치
```bash
# 저장소 클론
git clone https://github.com/yourusername/AutoCI.git
cd AutoCI

# 명령어 설치
sudo ./install_global_autoci.sh
chmod +x install_godot_commands.sh
./install_godot_commands.sh
```

### 2단계: Godot 엔진 빌드
```bash
# Windows 버전 Godot 빌드 (가상환경 자동 활성화)
build-godot
# → 빌드된 exe 파일: /mnt/d/AutoCI/AutoCI/godot_ai_build/output/godot.windows.editor.x86_64.exe

# Linux 버전 Godot 빌드 (가상환경 자동 활성화)
build-godot-linux
# → 빌드된 파일: .x86_64 (리눅스 실행 파일)
```

### 3단계: LLM 모델 설치 (NEW)
```bash
# 모든 LLM 모델 설치 (시간 소요) - 가상환경 자동 활성화
python install_llm_models.py

# 특정 모델만 설치 - 가상환경 자동 활성화
python install_llm_models.py llama-3.1-8b

# 메모리 제한 설정 (예: 16GB 시스템용)
python continuous_learning_system.py 24 16.0
```

### 4단계: 핵심 명령어 실행
```bash
# 사용자 명령에 맞춘 24시간 자동 게임 개발 (가상환경 자동 활성화)
autoci
> create platformer game

# AI 모델 기반 연속 학습 (5대 핵심 주제) - 가상환경 자동 활성화
autoci learn  # 5가지 핵심 주제 24시간 학습

# 학습 내용:
# 1. C# 프로그래밍 (기초부터 고급까지)
# 2. 한글 프로그래밍 용어
# 3. Godot 엔진 개발 방향성
# 4. Godot 내장 네트워킹 (AI 제어)
# 5. Nakama 서버 개발 (AI 최적화)

# 명시적 continuous learning (learn과 동일)
autoci learn continuous

# 저사양 환경 최적화 학습 (RTX 2080 GPU 8GB, 32GB 메모리) - NEW
autoci learn low

# RTX 2080 8GB 최적화 모델 설치 (가상환경 자동 활성화) - UPDATED
# 1. 현재 상태 확인 (먼저 실행)
python test_autoci_learn_low.py

# 2. DeepSeek-coder-v2 6.7B 설치 (선택사항)
python download_deepseek_coder.py  # 14GB 다운로드, 6GB VRAM 사용

# 3. autoci learn low 실행 (실시간 진행 상황 UI 포함)
autoci learn low  # 설치된 모델로 자동 실행

# ✨ 새로운 실시간 UI 기능:
# → 학습 프로세스 PID 표시
# → 실시간 모델 선택 상황 표시
# → 5가지 핵심 주제 감지 알림
# → 30초마다 진행 상황 요약
# → 메모리 상태 실시간 모니터링
# → 무응답 5분 시 자동 알림

# 메모리 최적화된 학습 (32GB 시스템)
python continuous_learning_system.py 24 32.0

# 메모리 제한된 시스템 (16GB)
python continuous_learning_system.py 24 16.0

# 짧은 시간 테스트 (1시간, 32GB)
python continuous_learning_system.py 1 32.0

# 모델이 없을 때 옵션:
# → 1. 데모 모드로 실행 (모델 없이 시뮬레이션)
# → 2. 모델 설치 안내
# → 3. 기본 학습 모드
# → 4. 취소

# 모델이 있을 때 옵션:
# → 1. 통합 학습 (전통적 + AI Q&A) - 가장 권장
# → 2. AI Q&A 학습만
# → 3. 전통적 학습만
# → 4. 빠른 AI 세션
# → 5. 사용자 지정 시간
# → 6. 데모 모드 (3분 시연) - 데모 설정시만

# 모델 설치
python install_llm_models_simple.py      # 간단한 모델
python install_llm_models_robust.py      # 전체 모델
python setup_demo_models.py              # 데모 설정

# 학습 진행 상황 모니터링
tail -f continuous_learning.log          # 실제 모델 로그
tail -f continuous_learning_demo.log     # 데모 모드 로그

# 메모리 사용량 모니터링
cat continuous_learning/memory_usage.json  # 메모리 사용 히스토리
```

## 🎯 DeepSeek-Coder-v2 6.7B 설치 및 사용 가이드 (NEW)

### 설치 전 확인
```bash
# 1. 현재 autoci learn low 상태 확인
python test_autoci_learn_low.py

# 결과:
# ✅ 현재 모델로 작동함 → DeepSeek-coder는 선택사항
# ❌ 모델 로드 실패 → 먼저 기본 모델 설치 필요
```

### 옵션 1: DeepSeek-coder-v2 6.7B 설치 (추천)
```bash
# 코딩 특화 답변을 원한다면 설치
python download_deepseek_coder.py

# 설치 후 즉시 사용 가능
autoci learn low  # DeepSeek-coder 우선 사용
```

### 옵션 2: 현재 모델로 계속 사용
```bash
# Llama-3.1-8B는 이미 RTX 2080에 최적화됨
autoci learn low  # 현재 설치된 모델로 실행
```

### DeepSeek-coder 설치 세부 정보
- **모델**: deepseek-ai/deepseek-coder-6.7b-instruct
- **크기**: 14GB 다운로드
- **VRAM**: 6GB (4bit 양자화)
- **특화 분야**: C# 코딩, Godot 개발, 한글 프로그래밍 용어
- **우선순위**: autoci learn low에서 최우선 선택

### Nakama 서버 AI 통합 (NEW)
```bash
# Nakama 서버 설치 및 설정 (가상환경 자동 활성화)
autoci nakama setup                      # Nakama 서버 설치
autoci nakama ai-server                  # AI 제어 서버 관리
autoci nakama ai-match                   # AI 매치메이킹

# AI 기능 활성화
autoci nakama ai-storage                 # 지능형 스토리지
autoci nakama ai-social                  # AI 소셜 모더레이터
autoci nakama optimize                   # 서버 최적화

# 멀티플레이어 게임 개발
autoci create multiplayer fps            # FPS 멀티플레이어
autoci create multiplayer moba           # MOBA 게임
autoci create multiplayer mmo            # MMO 게임

# 실시간 모니터링
autoci nakama monitor                    # 서버 모니터링
autoci nakama demo                       # 데모 실행
```

### Godot 엔진 AI 개선 (NEW)
```bash
# AI가 학습한 내용으로 Godot 엔진 개선 (가상환경 자동 활성화)
autoci fix                               # 5가지 핵심 학습 내용 적용

# 개선 사항:
# - C# 바인딩 최적화
# - 한글 지원 향상
# - 네트워킹 성능 개선
# - Nakama 통합 최적화
# - AI 제어 API 확장

# 개선된 엔진은 'godot_ai_improved' 디렉토리에 생성
```

## 🔧 고급 설정

### 프로덕션 모드
```bash
# 안정적인 실행 환경 (가상환경 자동 활성화)
autoci --production
```

### 모니터링
```bash
# 실시간 모니터링 활성화 (가상환경 자동 활성화)
autoci --enable-monitoring
```

## 📚 추가 문서

- [빠른 시작 가이드](QUICK_START.md)
- [AI Godot 빌드 가이드](AI_GODOT_BUILD_GUIDE.md)

## 🔍 문제 해결

### 가상환경 수동 설정 (필요시에만)
```bash
# 가상환경이 없는 경우 생성
python -m venv autoci_env

# 가상환경 활성화
source autoci_env/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### Godot 빌드 문제
```bash
# Windows 버전 빌드 (가상환경 자동 활성화)
build-godot
# → 결과: /mnt/d/AutoCI/AutoCI/godot_ai_build/output/godot.windows.editor.x86_64.exe

# Linux 버전 빌드 (가상환경 자동 활성화)
build-godot-linux
# → 결과: .x86_64 파일
```

### 빌드 실패 시
1. `godot_ai_build/logs/` 확인
2. 필수 패키지 설치:
   ```bash
   sudo apt update
   sudo apt install -y scons pkg-config libx11-dev libxcursor-dev
   ```

### LLM 모델 설치 실패 시 (NEW)
1. CUDA 설치 확인:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. 디스크 공간 확인 (100GB+ 필요)
3. Hugging Face 토큰 확인
4. 메모리 부족 시 개별 모델 설치:
   ```bash
   # 가상환경은 자동으로 활성화됩니다
   python install_llm_models.py llama-3.1-8b  # 가장 작은 모델
   ```

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 🙏 감사의 말

- Godot Engine 오픈소스 커뮤니티
- Python asyncio 개발팀
- WSL2 개발팀

## 🌟 현재 구현된 기능들

### ✅ 완료된 기능
- **AI 수정 Godot 빌드 시스템**: 소스 레벨 AI 통합
- **24시간 자동 게임 개발**: 다양한 장르 지원
- **24시간 C# 학습 시스템**: 36개 주제 완료
- **실시간 대시보드**: 웹 기반 모니터링
- **사용자 학습 추적**: 진도 저장 및 재개
- **다중 LLM 모델 통합**: 3개 주요 모델 지원
- **24시간 연속 학습**: 5대 핵심 주제 자동 학습
- **지식 베이스 구축**: 자동 패턴 추출
- **Godot 네트워킹 AI 제어**: 내장 네트워킹 완전 제어
- **Nakama 서버 통합**: AI 기반 백엔드 관리
- **AI 엔진 개선**: 학습 기반 자동 엔진 개선

### 🚧 개발 중인 기능
- Nakama-Godot 고급 동기화
- 모바일 플랫폼 지원
- 클라우드 배포 자동화
- 실시간 코드 리뷰 시스템
- Godot 실시간 대시보드 완성
- Windows 네이티브 빌드 지원
- AI 기반 자동 게임 밸런싱

---

## 🔄 autoci learn 명령어 업데이트 내역

### 최신 변경사항 (2025년 6월)
- `autoci learn low` 명령어에 **실시간 진행 상황 UI** 추가 ✨
- 더 이상 멈춘 것처럼 보이지 않고, 실제 학습 진행 상황을 실시간으로 표시
- **실시간 UI 기능**:
  - 🚀 학습 프로세스 PID 및 시작 상태 표시
  - 🔥 DeepSeek-coder 모델 선택 시 즉시 알림
  - 📚 5가지 핵심 주제 감지 시 특별 표시
  - 📊 30초마다 진행 상황 요약 (질문 수, 성공률, 모델 순환)
  - 💾 메모리 상태 실시간 모니터링
  - ⚠️ 5분 이상 무응답 시 자동 알림
- `autoci learn` 명령어가 이제 **5대 핵심 주제** AI 통합 연속 학습을 실행합니다
- C#, 한글, Godot 엔진 개발, Godot 내장 네트워킹, Nakama 서버를 24시간 동안 자동으로 학습
- **5대 핵심 주제** 세부 내용:
  1. **C# 프로그래밍**: 기초부터 고급까지 (async/await, LINQ, 제네릭 등)
  2. **한글 프로그래밍 용어**: 프로그래밍 개념의 한국어 번역 및 이해
  3. **Godot 엔진 개발**: 아키텍처, 렌더링, 성능 최적화, 미래 방향성
  4. **Godot 내장 네트워킹**: MultiplayerAPI, RPC, 동기화, AI 제어
  5. **Nakama 서버 개발**: 매치메이킹, 스토리지, 소셜, AI 최적화
- `autoci fix` 명령어 추가: 학습된 내용으로 Godot 엔진을 자동 개선

### 학습 모드
- `autoci learn` - AI 통합 연속 학습 (권장)
- `autoci learn simple` - 전통적 학습만
- `autoci learn menu` - 대화형 메뉴
- `autoci learn all` - 모든 주제 처음부터
- `autoci learn low` - RTX 2080 GPU 8GB, 32GB 메모리 최적화 (NEW)

## 📊 프로젝트 현황

### 성능 지표
- **AI Godot 빌드 시간**: 20-60분
- **24시간 학습 완료율**: 95%+
- **코드 구조**: 메인 모듈 15개, AI/학습 모듈 8개

### 알려진 이슈
- WSL Bash 환경에서 일부 명령어 직접 실행 불가 (Python 스크립트로 우회)
- Windows/Linux 경로 변환 일부 수동 처리 필요

## 🧠 메모리 최적화 시스템 (NEW)

### 스마트 메모리 관리
AutoCI의 연속 학습 시스템은 이제 시스템 메모리를 지능적으로 관리합니다:

#### 주요 기능
- **동적 모델 로딩**: 필요할 때만 모델을 메모리에 로드
- **자동 모델 언로딩**: 메모리 부족 시 자동으로 모델 해제
- **순차적 모델 사용**: 3개 모델을 동시가 아닌 순차적으로 활용
- **실시간 메모리 모니터링**: 메모리 사용량을 지속적으로 추적

#### 메모리 사용 전략
1. **llama-3.1-8b** (8GB): 한글 및 일반 질문
2. **codellama-13b** (13GB): 코드 관련 질문  
3. **qwen2.5-coder-32b** (16GB): 고급 코딩 및 아키텍처

#### 설정 방법
```bash
# 32GB 시스템 (모든 모델 사용 가능)
python continuous_learning_system.py 24 32.0

# 16GB 시스템 (작은 모델들만 사용)
python continuous_learning_system.py 24 16.0

# 8GB 시스템 (가장 작은 모델만)
python continuous_learning_system.py 24 8.0
```

#### 메모리 최적화 로직
- **85% 임계점**: 메모리 사용량이 설정 제한의 85%에 도달하면 모델 언로드
- **20사이클 로테이션**: 20개 질문마다 자동으로 모델 교체
- **가비지 컬렉션**: GPU 메모리 및 시스템 메모리 자동 정리
- **대기 시간 조정**: 메모리 상황에 따라 질문 간격 조정

#### 모니터링
- **실시간 로그**: 메모리 사용량을 지속적으로 표시
- **히스토리 저장**: `continuous_learning/memory_usage.json`에 사용 패턴 저장
- **모델 로테이션 추적**: 얼마나 자주 모델이 교체되는지 기록

## 🔧 연속 학습 시스템 세부 정보

### 통합 명령어 체계
`autoci learn`는 이제 향상된 AI 통합 연속 학습 시스템을 실행합니다:
1. **통합 시스템** (`modules/csharp_continuous_learning.py`)
   - 구조화된 24시간 커리큘럼 + Mirror 게임 서버 주제
   - 사용자 진도 추적
   - Godot-Mirror 통합 학습
   - 실시간 LLM Q&A
   
2. **독립 시스템** (`continuous_learning_system.py`)
   - 실제 LLM 모델 활용
   - 자동 Q&A 생성
   - 지식 베이스 구축

### 학습 주제 (60개+)
#### 기존 C# 주제 (36개)
- 기초: 변수, 타입, 연산자, 조건문, 반복문
- 중급: 클래스, 상속, 인터페이스, 예외처리
- 고급: 제네릭, LINQ, async/await, 리플렉션
- Godot: GDScript 연동, 씬 제어, 노드 시스템

#### Godot 엔진 개발 주제 (8개+)
- **아키텍처**: 엔진 구조, 노드 시스템, 씬 트리
- **렌더링**: 파이프라인, 셰이더, 최적화
- **성능**: 프로파일링, 최적화, 크로스 플랫폼
- **미래 방향성**: AI 통합, 에디터 개선

#### Godot 네트워킹 주제 (8개+)
- **MultiplayerAPI**: ENet, WebSocket, 서버/클라이언트
- **RPC 시스템**: @rpc, call_local, any_peer, authority
- **동기화**: MultiplayerSynchronizer, 상태 복제
- **고급 기능**: 지연 보상, 예측, AI 제어

#### Nakama 서버 주제 (10개+)
- **기초**: 설치, 설정, 인증, 세션 관리
- **매치메이킹**: AI 기반 매칭, 로비 시스템
- **데이터**: 스토리지, 리더보드, 토너먼트
- **소셜**: 친구, 그룹, 채팅, AI 모더레이션
- **고급**: 서버 확장, 대규모 최적화, AI 통합

### 학습 모드 상세
1. **통합 학습**: 전통적 학습 + LLM Q&A + 5대 핵심 주제
2. **AI Q&A 전용**: LLM과 대화형 학습
3. **전통적 학습**: 기존 24시간 커리큘럼
4. **빠른 세션**: 특정 주제 단기 학습
5. **커스텀 시간**: 원하는 시간만큼 학습
6. **데모 모드**: 모델 없이 시스템 체험

### 지식 베이스 구조
- `user_learning_data/continuous_learning/knowledge_base.json`: 축적된 지식
  - `csharp_patterns`: C# 코드 패턴
  - `korean_translations`: 한글 번역
  - `godot_integrations`: Godot 통합 정보
  - `godot_networking`: Godot 네트워킹 패턴
  - `nakama_patterns`: Nakama 서버 구현 패턴
  - `engine_improvements`: 엔진 개선 방법
  - `network_optimizations`: 네트워크 최적화 방법
- `user_learning_data/continuous_learning/qa_sessions/`: Q&A 세션 기록
- 자동 패턴 추출 및 오류 해결 방법 저장

## 🌐 Nakama 오픈소스 서버 AI 통합

### 개요
AutoCI가 이제 Nakama 오픈소스 게임 서버와 Godot을 AI가 완전히 제어하여 자동으로 대규모 멀티플레이어 게임을 개발합니다.

### 주요 기능
1. **AI 서버 관리**: 자동 Nakama 서버 설정 및 최적화
2. **지능형 매치메이킹**: AI 기반 플레이어 스킬 분석 및 매칭
3. **스마트 데이터 관리**: 플레이어 데이터 자동 최적화
4. **AI 소셜 모더레이션**: 실시간 채팅 및 행동 분석

### 아키텍처
```
┌─────────────┐     gRPC/REST       ┌─────────────┐
│   Godot     │◄────────────────────│   Nakama    │
│  (Client)   │                     │  (Server)   │
└──────┬──────┘                     └──────┬──────┘
       │                                   │
       └─────────►┌──────────┐◄───────────┘
                  │  AutoCI   │
                  │    AI     │
                  └──────────┘
```

### 지원 게임 타입
- **FPS**: 실시간 매치메이킹, 스킬 기반 매칭
- **MOBA**: 팀 밸런싱, 토너먼트 시스템
- **MMO**: 대규모 플레이어, 길드 시스템
- **Battle Royale**: 100+ 플레이어, 리더보드

## 🔧 AI 기반 Godot 엔진 개선 (`autoci fix`)

### 개요
`autoci fix` 명령어는 24시간 학습한 5대 핵심 주제의 지식을 바탕으로 Godot 엔진을 자동으로 개선합니다.

### 개선 영역
1. **C# 바인딩 최적화**
   - async/await 성능 향상
   - GC 최적화
   - Task 스케줄링 개선

2. **한글 지원 향상**
   - UTF-8 한글 처리 개선
   - IME 지원 강화
   - 한글 폰트 렌더링 최적화

3. **네트워킹 성능 개선**
   - 비동기 네트워킹 구현
   - 버퍼 풀링
   - 프로토콜 최적화

4. **Nakama 통합 최적화**
   - 네이티브 Nakama 지원
   - 실시간 매치메이킹 최적화
   - 서버 통신 효율화

5. **AI 제어 API 확장**
   - AI 컨트롤러 인터페이스
   - 에디터 자동화 API
   - 머신러닝 노드

### 사용법
```bash
# AI가 학습한 내용으로 Godot 엔진 개선
autoci fix

# 개선된 엔진은 'godot_ai_improved' 디렉토리에 생성됩니다
```

## 🗑️ 프로젝트 정리 가이드

### 삭제 가능한 중복 파일
```bash
# 중복 빌드 스크립트들 (build_ai_godot.py로 통합됨)
rm -f simple_godot_build.py quick_build_check.py start_build_now.py
rm -f start_ai_godot_build.py run_build.py test_wsl_cmd.py
```

### 핵심 파일 (반드시 유지)
- `autoci`, `autoci.py`, `autoci_terminal.py`, `autoci_production.py`
- `build_ai_godot.py`, `build-godot`, `check-godot`
- `modules/` 디렉토리의 모든 파일
- `user_learning_data/`, `game_projects/`, `godot_ai_build/`

---

## 📋 핵심 명령어 요약

💡 **필요한 명령어에는 가상환경이 자동으로 활성화됩니다**

### 기본 명령어
```bash
# 사용자 명령에 맞춘 24시간 자동 게임 개발 (가상환경 자동 활성화)
autoci

# AI 모델 기반 연속 학습 (5대 핵심 주제) - 가상환경 자동 활성화
autoci learn

# 학습을 토대로 AI의 게임 제작 능력 업데이트 (가상환경 자동 활성화)
autoci fix

# 새 게임 생성
create_game

# Godot 열기
open_godot
```

### 빌드 명령어
```bash
# Windows 버전 Godot 빌드 (가상환경 자동 활성화)
build-godot
# → /mnt/d/AutoCI/AutoCI/godot_ai_build/output/godot.windows.editor.x86_64.exe

# Linux 버전 Godot 빌드 (가상환경 자동 활성화)
build-godot-linux
# → 결과: .x86_64 파일
```

---

**버전**: 5.0 (2025년 6월)  
**문의**: GitHub Issues에 등록해주세요  
**Hugging Face Token**: 환경 변수로 설정 권장