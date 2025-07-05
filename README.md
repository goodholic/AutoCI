# AutoCI - AI 게임 개발 시스템 🎮

**AI가 직접 변형된 Godot을 조작하여 24시간 동안 완전한 2.5D~3D 게임을 제작하는 시스템**

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

## 📁 프로젝트 구조 (핵심/보조 명확히 구분)

### 🔴 핵심/필수 폴더 (실행에 꼭 필요한 파일만)
```
AutoCI/
├── core_system/              # 시스템 핵심 실행 코드
│   ├── autoci_godot_main.cs    # 메인 실행 파일
│   ├── continuous_learning_system.cs
│   └── ... (실행에 필수적인 파일만)
├── modules/                  # 주요 기능 모듈
│   ├── ai_model_integration.cs
│   ├── godot_automation_controller.cs
│   ├── godot_engine_integration.cs    # 변형된 Godot 직접 조작
│   ├── socketio_realtime_system.cs    # Socket.IO 통신
│   ├── korean_conversation_interface.cs
│   └── ... (핵심 모듈만)
├── config_active/            # 실제 사용되는 설정 파일
│   └── (환경설정, 모델설정 등 필수 설정만)
├── tools_utilities/          # 자주 쓰는 핵심 유틸리티
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
autoci learn              # AI 학습 시작
autoci learn low          # 메모리 최적화 학습
autoci monitor            # 모니터링 대시보드
autoci chat               # 한글 대화 모드
```

### 대화형 모드 명령어
```
create [type] game        # 게임 생성 (platformer, racing, rpg, puzzle)
add feature [name]        # 기능 추가
modify [aspect]           # 게임 수정
open_godot               # 변형된 Godot 에디터 열기
status                    # 시스템 상태
help                      # 도움말
```

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

**Version**: 5.0  
**개발팀**: AutoCI Team  
**문의**: GitHub Issues