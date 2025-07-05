# AutoCI 프로젝트 구조 (정리 완료)

## 📁 최종 폴더 구조

```
AutoCI/
├── 🚀 실행 파일
│   ├── autoci                # Python 메인 실행 파일
│   ├── autoci.sh            # Linux/Mac 자동 가상환경 래퍼
│   ├── autoci.bat           # Windows 자동 가상환경 래퍼
│   ├── install.sh           # 글로벌 설치 스크립트
│   └── setup.py             # 초기 설정 스크립트
│
├── 📦 core_system/          # 핵심 시스템
│   ├── autoci_panda3d_main.py     # 메인 시스템
│   ├── ai_engine_updater.py       # AI 엔진 업데이터 (autoci fix)
│   ├── continuous_learning_system.py
│   └── (기타 핵심 파일들)
│
├── 🔧 modules/       # 활성 모듈
│   ├── ai_model_integration.py    # AI 모델 통합
│   ├── panda3d_automation_controller.py
│   ├── panda3d_engine_integration.py    # Panda3D 직접 조작
│   ├── panda3d_self_evolution_system.py # 자가 진화
│   ├── socketio_realtime_system.py      # Socket.IO 통신
│   ├── korean_conversation_interface.py # 한국어 대화
│   ├── game_development_pipeline.py     # 24시간 개발
│   ├── realtime_monitoring_system.py    # 모니터링
│   └── (기타 활성 모듈들)
│
├── ⚙️ config_active/        # 설정 파일
│   ├── autoci_config.json
│   └── production_config.py
│
├── 🛠️ tools_utilities/      # 도구 및 유틸리티
│   ├── api_doc_generator.py
│   ├── performance_profiler.py
│   └── security_auditor.py
│
├── 📚 docs_guides/          # 문서 및 가이드
│   ├── README.md
│   ├── AUTOCI_FEATURES.md
│   ├── COMMANDS.md
│   └── (기타 문서들)
│
├── 🧪 tests_active/         # 테스트 코드
│   └── (테스트 파일들)
│
├── 📊 logs_current/         # 현재 로그
│   └── (최신 로그 파일들만)
│
├── 🤖 models_ai/            # AI 모델 저장소
│   └── (다운로드된 AI 모델들)
│
├── 💾 data/                 # 데이터 저장소
│   ├── learning/            # 학습 데이터
│   ├── evolution/           # 진화 데이터
│   └── feedback/            # 피드백 데이터
│
├── 🎮 game_projects/        # 생성된 게임들
│   └── (AI가 만든 게임 프로젝트들)
│
├── 📦 models/               # Panda3D 소스
│   └── panda3d-1.10.15/    # Panda3D 소스 코드
│
├── 🗄️ archive/              # 보관소
│   ├── old_files/           # 오래된 파일
│   ├── legacy_code/         # 레거시 코드
│   ├── old_logs/            # 과거 로그
│   └── old_learning_data/   # 과거 학습 데이터
│
├── 🔒 autoci_env/           # 가상환경 (자동 생성)
│
├── 📋 requirements.txt      # 전체 패키지 목록
├── 📋 requirements_minimal.txt  # 최소 패키지 목록
├── 📖 README.md            # 프로젝트 설명서
└── 📖 PROJECT_STRUCTURE.md # 이 파일
```

## 🎯 주요 변경사항

### 1. 자동 가상환경 활성화
- `autoci.sh` (Linux/Mac) 및 `autoci.bat` (Windows) 래퍼 스크립트 추가
- 가상환경이 없으면 자동 생성 및 필수 패키지 설치
- 사용자가 가상환경을 신경쓸 필요 없음

### 2. 정리된 폴더 구조
- 필수 파일들만 남기고 나머지는 archive로 이동
- 명확한 폴더 이름으로 용도 구분
- 불필요한 중복 파일 제거

### 3. 새로운 기능
- `autoci fix`: AI 엔진 업데이터 추가 (학습 데이터 기반 개선)
- Panda3D 엔진 직접 조작 모듈
- Socket.IO 실시간 통신 시스템

### 4. 글로벌 설치
- `install.sh` 스크립트로 시스템 전역 설치 가능
- 어디서나 `autoci` 명령 사용 가능

## 💡 사용 방법

### 기본 사용 (가상환경 자동 활성화)
```bash
# Linux/Mac
./autoci.sh

# Windows
autoci.bat

# 또는 글로벌 설치 후
autoci
```

### 명령어
```bash
autoci                    # 대화형 모드 (24시간 자동 개발)
autoci learn              # AI 통합 학습
autoci learn low          # 메모리 최적화 학습
autoci fix                # AI 엔진 업데이트
autoci monitor            # 실시간 모니터링
autoci chat               # 한글 대화 모드
```

## ✅ 완료된 작업

1. ✅ 자동 가상환경 활성화 스크립트 생성
2. ✅ 불필요한 파일 정리 및 archive 이동
3. ✅ `autoci fix` 명령어 구현
4. ✅ Panda3D 엔진 직접 조작 모듈 구현
5. ✅ Socket.IO 실시간 통신 시스템 구현
6. ✅ 프로젝트 구조 최적화