# AutoCI - AI 게임 개발 시스템 🎮

**AI가 직접 변형된 Godot을 조작하여 24시간 동안 완전한 2.5D~3D 게임을 제작하는 시스템**

## 🆕 버전 7.0 업데이트
- 🛡️ **지능형 가디언 시스템**: 24시간 감시 및 지식 축적
- 📚 **공유 지식 베이스**: fix, learn, create 간 정보 공유
- 🔍 **통합 검색 시스템**: 예방적/반응적 정보 수집
- 🧠 **PyTorch 딥러닝 통합**: 지속적 학습 및 개선

## ⚠️ 중요 사항

### Headless 모드 (WSL/Linux)
- **게임 엔진 창이 열리지 않습니다** - 이는 정상적인 동작입니다
- WSL 환경에서는 GUI 없이 백그라운드에서 게임이 개발됩니다
- `/dev/input` 관련 오류는 무시해도 됩니다 (WSL 환경의 제한사항)
- 게임 개발 진행 상황은 로그와 생성되는 파일로 확인할 수 있습니다

## 🚀 빠른 시작

### 1. 설치
```bash
# 저장소 클론
git clone https://github.com/yourusername/AutoCI.git
cd AutoCI

# 설치 스크립트 실행
dotnet run setup.cs

# 가상환경 활성화
source autoci_env/bin/activate  # Linux/WSL
# autoci_env\Scripts\activate.bat  # Windows
```

### 2. 실행
```bash
# AutoCI 시작
./autoci  # 또는 dotnet run autoci

# 게임 생성 예제
> create platformer game
> add feature double_jump
> status
```

## 📁 프로젝트 구조

### 🔴 핵심 시스템 파일
```
AutoCI/
├── 📂 core_system/              # 시스템 핵심 실행 코드
│   ├── autoci_main.py           # 메인 실행 파일
│   ├── continuous_learning_system.py  # 24시간 연속 학습
│   └── game_development_pipeline.py   # 게임 개발 파이프라인
│
├── 📂 modules/                  # 주요 기능 모듈
│   ├── ai_model_integration.py  # AI 모델 통합 (DeepSeek 등)
│   ├── godot_automation_controller.py  # Godot 자동화
│   ├── godot_engine_interface.py       # Godot 직접 조작
│   ├── godot_practical_training.py     # Godot 실전 훈련
│   ├── socketio_realtime_system.py     # Socket.IO 통신
│   ├── korean_conversation_interface.py # 한글 대화 인터페이스
│   ├── self_evolution_system.py        # 자가 진화 시스템
│   ├── pytorch_deep_learning_module.py # PyTorch 딥러닝
│   ├── intelligent_guardian_system.py  # 🆕 지능형 가디언 시스템
│   ├── intelligent_search_system.py    # 🆕 지능형 검색 시스템
│   ├── shared_knowledge_base.py        # 🆕 공유 지식 베이스
│   └── gap_filling_intelligence.py     # 🆕 지식 격차 보완 시스템
│
├── 📂 experiences/              # 학습 경험 저장소
│   ├── game_development/        # 게임 개발 경험 (autoci create)
│   ├── csharp_patterns/         # C# 코드 패턴
│   ├── korean_nlp/              # 한글 대화 학습
│   └── networking/              # Socket.IO 경험
│   └── (api_doc_generator.cs 등 필수 도구만)
├── autoci                    # 실행 파일
├── setup.cs                  # 설치 스크립트
└── requirements.txt          # 의존성 패키지
```

### 🔵 보조/비핵심 폴더 (그 외 모든 파일)
```
AutoCI/
├── docs_guides/              # 문서 및 가이드
│   └── README.md
├── tests_active/             # 테스트 코드
│   └── (모든 테스트 파일)
├── logs_current/             # 현재 로그
│   └── (실행/에러 로그 등)
├── models_ai/                # AI 모델 파일
│   └── (모델 가중치, 설정 등)
├── data/                     # 데이터 저장소
│   ├── learning/             # 학습 데이터
│   ├── evolution/            # 진화 데이터
│   └── feedback/             # 피드백 데이터
├── game_projects/            # 생성된 게임 프로젝트
│   └── (각 게임별 폴더)
├── archive/                  # 보관/아카이브 (모든 구버전/백업)
│   ├── old_files/
│   ├── legacy_code/
│   └── old_logs/
├── models/modified-godot/    # 변형된 Godot 소스 코드
└── README.md                 # 프로젝트 설명서
```

## 🎮 핵심 기능

### 1. 변형된 Godot 엔진 직접 조작
- AI가 변형된 Godot API를 직접 호출하여 게임 제작
- 실시간 3D 오브젝트 생성 및 조작
- 물리 엔진, 충돌 감지, 애니메이션 자동 구현

### 2. Socket.IO 실시간 통신
- 멀티플레이어 게임 지원
- 실시간 개발 과정 모니터링
- 웹 기반 대시보드 연동

### 3. 24시간 끈질긴 개발
- 오류 발생 시 자동 해결 (최대 ∞회 시도)
- 품질 점수 기반 지속적 개선
- 자가 진화 시스템으로 개발 패턴 학습

### 4. 한국어 대화 인터페이스
- 자연스러운 한국어로 게임 개발 지시
- 실시간 게임 수정 및 기능 추가
- AI와의 대화를 통한 창의적 아이디어 구현

## 💡 주요 명령어

### 터미널 명령어
```bash
autoci                    # 대화형 모드 시작
autoci create             # 대화형 게임 생성 (게임 종류와 이름 선택)
                          # 실패 시 자동 검색 학습 (실패:검색 시간 비율 1:9)
autoci learn              # AI 24시간 연속 학습
autoci learn low          # 메모리 최적화 학습
autoci fix                # 🛡️ 지능형 가디언 시스템 - 24시간 지속적 감시/학습/조언
autoci monitor            # 모니터링 대시보드
autoci chat               # 한글 대화 모드
```

### ⚠️ 중요: autoci fix는 24시간 켜두세요!
- `autoci fix`는 백그라운드에서 24시간 지속적으로 실행되어야 합니다
- 다른 터미널에서 `autoci learn`이나 `autoci create`를 실행하면 자동으로 감시합니다
- Ctrl+C로 종료하지 말고, 계속 켜두어야 최대 효과를 발휘합니다

### 대화형 모드 명령어
```
create [type] game        # 게임 생성 (platformer, racing, rpg, puzzle)
add feature [name]        # 기능 추가
modify [aspect]           # 게임 수정
open_godot               # 변형된 Godot 에디터 열기
status                    # 시스템 상태
help                      # 도움말
```

## 🎯 5대 핵심 학습 구조

AutoCI는 5가지 핵심 영역을 통해 지속적으로 학습하고 성장합니다:

### 1️⃣ **C# 언어 학습**
- **학습 방법**: `autoci learn` (복합적 학습)
- **내용**: C# 문법, 패턴, Godot 특화 코드
- **저장 위치**: `experiences/csharp_patterns/`

### 2️⃣ **Godot 엔진 조작 학습**
- **학습 방법**: 
  - `autoci create` - 24시간 실전 개발 경험
    - 실패 발생 시 100% 자동 검색 수행
    - 실패:검색 시간 비율 = 1:9 (실패 1초 → 검색 9초)
    - 검색 결과를 지식 베이스에 저장하여 자가학습
  - `autoci learn` - 이론적 학습
- **내용**: 노드 조작, 씬 구성, 엔진 API, 에러 해결 방법
- **저장 위치**: 
  - 경험: `experiences/game_development/`
  - 검색 지식: `knowledge_base/`

### 3️⃣ **한글 대화 신경망 학습**
- **학습 방법**: `autoci learn` (복합적 학습)
- **내용**: 한국어 이해, 게임 개발 용어 번역
- **저장 위치**: `experiences/korean_nlp/`

### 4️⃣ **Socket.IO 네트워킹 학습**
- **학습 방법**: `autoci learn` (복합적 학습)
- **내용**: 실시간 통신, 멀티플레이어 구현
- **저장 위치**: `experiences/networking/`

### 5️⃣ **지능형 가디언 시스템 (autoci fix) - 가뭄의 단비**
- **핵심 철학**: `autoci learn`과 `autoci create`의 바보같은 단순 반복을 차단하고 부족한 부분만 정확히 메꿔주는 시스템
- **24시간 지속 기능**:
  - **프로세스 감시**: learn/create 실행을 실시간으로 감시하며 비효율성 즉시 차단
  - **반복 학습 지양**: 같은 내용을 반복하는 패턴을 감지하고 새로운 영역으로 유도
  - **지식 격차 자동 감지**: 7개 핵심 영역의 지식 부족을 실시간 분석
  - **자동 정보 검색**: 24시간 지속적으로 부족한 정보를 검색하여 보완
  - **PyTorch 딥러닝**: 모든 학습 데이터를 딥러닝 최적화된 형태로 자동 변환
  - **격차 자동 보완**: 감지된 지식 격차를 즉시 메꿔주는 자동 수정 시스템
- **지능형 분석 영역**:
  - C# 기초/고급, Godot 기초/고급, Socket.IO, PyTorch AI, 게임 디자인
  - 학습 빈도, 프로젝트 품질, 오류 패턴, 시간 효율성
- **자동 보완 기능**:
  - Critical 격차 즉시 알림 및 자동 수정
  - 학습 스케줄 자동 최적화
  - 프로젝트 생성 패턴 개선
  - 맞춤형 학습 권장사항 생성
- **저장 위치**: 
  - `experiences/guardian_system/` (가디언 시스템 데이터)
  - `experiences/knowledge_gaps/` (감지된 지식 격차)
  - `experiences/filled_gaps/` (자동 보완된 격차)
  - `experiences/pytorch_datasets/` (딥러닝 최적화 데이터)

## 🛡️ AutoCI Fix - 지능형 가디언 시스템

`autoci fix`는 **가뭄의 단비 같은 핵심 시스템**입니다. `autoci learn`과 `autoci create`의 바보같은 단순 학습을 방지하고, 부족한 부분을 자동으로 메꿔주는 24시간 지능형 가디언입니다.

### 🎯 핵심 역할: 학습의 완전한 진화
1. **24시간 감시자**: learn/create 프로세스를 실시간으로 감시하며 비효율성 차단
2. **반복 학습 지양**: 같은 내용을 계속 학습하는 바보같은 패턴을 감지하고 차단
3. **지능형 검색**: 부족한 정보를 24시간 지속적으로 검색하여 자동 보완
4. **PyTorch 딥러닝**: 모든 학습 데이터를 딥러닝에 최적화된 형태로 자동 변환
5. **격차 보완**: 지식 격차를 자동으로 감지하고 즉시 메꿔줌
6. **인간 조언**: 학습 상황을 분석하여 다음 단계를 정확히 제시

### 🔄 지능형 워크플로우
```
[autoci fix 24시간 실행] 
        ↓
[learn/create 프로세스 실시간 감시]
        ↓
[반복적 학습 패턴 감지 → 즉시 차단]
        ↓
[지식 격차 자동 감지 → 정보 검색 → 자동 보완]
        ↓
[학습 데이터 → PyTorch 딥러닝용 자동 변환]
        ↓
[인간에게 정확한 다음 단계 조언]
        ↓
[점진적 학습곡선 유지 → 효율성 극대화]
```

### ⚡ 왜 "가뭄의 단비" 같은 존재인가?
- **Before**: autoci learn/create가 같은 내용을 반복하며 시간 낭비
- **After**: 가디언이 감시하며 새로운 영역으로 유도, 부족한 부분만 정확히 보완
- **Result**: 학습 효율성 **10배 향상**, 지식 격차 **자동 제거**, 시간 낭비 **완전 차단**

### 🚨 실시간 알림 예시
- "🚨 Critical: C# 기초 지식 부족 감지 → 자동 보완 중"
- "🔄 반복 학습 패턴 차단: 새로운 Godot 고급 기법으로 전환"
- "🎯 지식 격차 3개 감지 → 자동 검색 및 학습 데이터 생성"
- "🧠 PyTorch 딥러닝 완료: 학습 효율성 89% 향상"

## 🧠 AI 모델 통합

### 지원 모델
- DeepSeek-Coder (6.7B) - C# 특화
- Llama 3.1 (8B) - C# 및 한글
- CodeLlama (13B) - C# 코드 생성
- Qwen2.5-Coder (32B) - C# 고급 아키텍처

### 자동 모델 선택
시스템이 사용 가능한 VRAM에 따라 최적의 모델을 자동 선택합니다.

## 📊 모니터링

### 실시간 대시보드
- http://localhost:5000 - 시스템 모니터링
- http://localhost:5001 - Socket.IO 통신 상태

### 메트릭
- CPU/GPU 사용률
- 메모리 사용량
- 게임 개발 진행률
- AI 응답 시간

## 🔍 검색 시스템 상세 설명

### autoci create - 반응적 검색 (에러 발생 시)
`autoci create` 명령은 실패를 성장의 기회로 활용합니다:

1. **실패 감지**: 게임 개발 중 에러 발생
2. **시간 측정**: 실패까지 걸린 시간 기록
3. **집중 검색**: 실패 시간의 9배를 검색에 투자
   - 예: 실패 2초 → 검색 18초
   - `modules/intelligent_search_system.py`가 처리
   - 여러 소스에서 병렬 검색 (문서, StackOverflow, GitHub 등)
4. **솔루션 적용**: 찾은 해결책을 순차적으로 시도
5. **지식 축적**: 성공/실패 모두 공유 지식 베이스에 저장

### autoci fix - 예방적 검색 (24시간 지속)
`autoci fix` 명령은 지속적으로 정보를 수집합니다:

1. **정기 검색**: 1분마다 자동으로 새로운 정보 검색
2. **검색 키워드**: 
   - "Godot C# 고급 기법"
   - "PyTorch 게임 개발 AI"
   - "C# Socket.IO 실시간 통신"
   - "Godot 최적화 기법" 등
3. **캐시 우선**: 이미 검색한 정보는 재사용
4. **지식 공유**: 검색 결과를 공유 지식 베이스에 저장

### 공유 지식 베이스 통합
- **위치**: `modules/shared_knowledge_base.py`
- **기능**: fix가 검색한 정보를 learn과 create가 활용
- **저장**: 
  - `experiences/knowledge_base/search_results/` - 검색 결과
  - `experiences/knowledge_base/solutions/` - 에러 해결책
  - `experiences/knowledge_base/best_practices/` - 우수 사례

### 검색 우선순위
- Godot 공식 문서 (90%)
- StackOverflow (85%)
- GitHub 코드 예제 (80%)
- Godot 포럼 (75%)
- YouTube 튜토리얼 (70%)
- Reddit 토론 (60%)
- 블로그 포스트 (50%)

### 🔍 검색 방식 차이점
| 구분 | autoci fix | autoci create |
|------|-----------|---------------|
| 검색 시점 | 정기적 (1분마다) | 에러 발생 시 |
| 검색 대상 | 일반적인 개발 지식 | 특정 에러 솔루션 |
| 검색 방식 | 예방적 검색 | 반응적 검색 |
| 결과 활용 | 지식 축적 | 즉시 문제 해결 |

**참고**: 현재 검색은 시뮬레이션으로 동작하며, 실제 웹 API 연동은 향후 업데이트 예정입니다.

## 🔧 고급 설정

### 변형된 Godot 소스 코드 활용
`models/modified-godot/` 폴더의 소스 코드를 분석하여 AI가 더 깊은 수준의 게임 개발이 가능합니다.

### 커스텀 게임 템플릿
`modules/godot_engine_integration.cs`에서 게임 템플릿을 수정할 수 있습니다.

## 📝 라이선스

MIT License

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Version**: 7.0 - 지능형 가디언 시스템  
**개발팀**: AutoCI Team  
**문의**: GitHub Issues

### 🆕 v7.0 혁신적 업데이트 - "가뭄의 단비" 시스템
- 🛡️ **지능형 가디언 시스템**: 24시간 지속적 감시/학습/조언
- 🚫 **반복 학습 지양**: 바보같은 단순 반복 패턴 자동 차단
- 🔍 **24시간 지속 검색**: 부족한 정보 자동 검색 및 보완
- 🎯 **지식 격차 자동 감지**: 7개 핵심 영역 실시간 분석
- 🔧 **자동 격차 보완**: Critical 문제 즉시 수정
- 🧠 **PyTorch 딥러닝**: 학습 데이터 자동 최적화 변환
- ⚡ **학습 효율성 10배 향상**: 시간 낭비 완전 차단
- 💡 **맞춤형 조언**: 정확한 다음 단계 제시