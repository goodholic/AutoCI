# AutoCI - AI 게임 개발 시스템 🎮

더이상 Readme의 내용에서 더하지도 빼지도 마시길 바랍니다. 
상용 AI모델이 되도록, 구현은 Readme의 내용을 깊숙히 구현하는 방향으로 진현합니다.
  - 일단, 변형된 Godot 엔진에 대해 세부적으로 다 알아야할것같다.
  - 정보 모으는 기능도 뛰어나야할것 같다.
  - 학습 후 저장 및 검색 능력도 뛰어나야할것같다.

<PyTorch>
머신러닝의 기초가되는 인공 신경망 기술 개발, 
딥러닝은 머신러닝의 한분야로, 인공 신경망을 여러층으로 쌓아, 더 복잡한 문제를 해결하는 기술,
딥러닝 코드가 오픈소스로 공개됨.

<폴더 구조>
실행/핵심 코드: core/
문서/가이드: docs/
도구/유틸: tools/
설정: config/
관리/운영: admin/
로그: logs/
테스트: tests/
기타 데이터/모듈/진화/학습 등은 각 폴더에 정리

**AI가 직접 변형된 Godot을 조작하여 24시간 동안 완전한 2.5D~3D 게임을 제작하는 시스템**


## 🚀 핵심 기능

### 1. 24시간 자동 게임 개발
- **AI가 실제로 변형된 Godot 엔진을 조작**하는 모습을 직접 관찰
- **마우스 클릭, 키보드 입력, 메뉴 조작**을 실시간으로 확인
- **완전히 실행 가능한 2.5D~3D 게임** 생성 (MVP가 아닌 완성품)
- **24시간 끈질긴 개선**: 오류 발견 → 해결 → 기능 추가 → 반복

### 2. AI 모델 기반 학습 시스템
- **5가지 핵심 주제**: C# 프로그래밍, 한글 용어, 변형된 Godot 엔진, 네트워킹 (Socket.IO), AI 모델 최적화
- **10억+ 파라미터 AI 모델** 활용 (DeepSeek-Coder, Llama-3.1, CodeLlama 등)
- **진행 상태 자동 저장**: 언제든지 중단하고 이어서 학습 가능
- **자가 진화 시스템**: 사용자 질문을 통해 AI가 스스로 학습

### 3. 한글 대화 인터페이스
- **자연스러운 한국어**로 AI와 대화하며 게임 개발
- **실시간 게임 수정**: 개발 중인 게임을 대화로 수정 가능
- **의도 자동 분류**: 질문, 명령, 피드백을 자동으로 구분

## 🎯 빠른 시작

### 1. AutoCI 실행
```bash
# AutoCI 시작
autoci

# 게임 타입 선택 후 자동 개발 시작
> 1  # 3D 플랫폼 게임
> create racing game  # 또는 직접 명령

# → 변형된 Godot 엔진이 자동으로 실행되며 AI가 게임을 제작
```

### 2. AI 학습 시작
```bash
# 통합 학습 (권장)
autoci learn

# 메모리 최적화 학습 (RTX 2080 등)
autoci learn low

# 커스텀 시간 설정 (1시간, 16GB RAM)
dotnet run continuous_learning_system.cs 1 16.0
```

### 3. 변형된 Godot 환경 설정
```bash
# 변형된 Godot 설치 (Windows/Linux)
# Godot 엔진을 C#과 통합하여 설치

# 필요한 C# 패키지 설치
dotnet restore
```

## 🎮 지원하는 게임 타입

모든 2.5D~3D 게임

## 💬 주요 명령어

### AutoCI 터미널 명령어
```bash
autoci                      # AutoCI 시작
autoci learn                # AI 통합 학습
autoci learn low            # 메모리 최적화 학습
autoci fix                  # 학습 기반 엔진 개선
autoci chat                 # 한글 대화 모드
```

### 터미널 내 명령어
```bash
create [type] game          # 게임 생성 (racing, platformer, rpg, puzzle)
add feature [name]          # 기능 추가 (개발 중에도 가능)
modify [aspect]             # 게임 수정
open_godot                  # 변형된 Godot 에디터/뷰어 열기 (가상)
status                      # 시스템 상태
help                        # 도움말
```

### 변형된 Godot 관련 명령어
```bash
# 변형된 Godot 프로젝트 실행 (예시)
dotnet run main.cs
```

## 🧠 AI 모델 및 학습 시스템

### 지원 AI 모델
| 모델 | 파라미터 | VRAM | 특화 분야 |
|------|----------|------|-----------|
| DeepSeek-Coder-6.7B | 67억 | 8GB | C# 코딩, 변형된 Godot |
| Llama-3.1-8B | 80억 | 10GB | 일반 프로그래밍, 한글 |
| CodeLlama-13B | 130억 | 16GB | 코드 생성 |
| Qwen2.5-Coder-32B | 320억 | 40GB | 고급 아키텍처 |

### 학습 주제 (5가지 핵심)
1. **C# 프로그래밍**: 기초부터 고급까지 (비동기, 객체지향, .NET 라이브러리 활용)
2. **한글 프로그래밍 용어**: 프로그래밍 개념의 한국어 번역
3. **변형된 Godot 엔진**: 아키텍처, 렌더링, 성능 최적화, 2.5D/3D 개발
4. **네트워킹 (Socket.IO)**: 실시간 통신, 멀티플레이어 게임 구현
5. **AI 모델 최적화**: 학습 데이터, 프롬프트 엔지니어링, 모델 경량화

### 자가 진화 시스템
- **집단 지성 기반**: 사용자 질문으로 AI가 스스로 학습
- **품질 자동 평가**: 응답을 4가지 기준으로 평가
- **패턴 인식**: 성공적인 해결책을 자동으로 학습
- **지식 베이스 확장**: 검증된 솔루션을 영구 저장

## 🛠️ 설치 및 요구사항

### 시스템 요구사항
- **OS**: Windows 10/11 + WSL2 (Ubuntu 20.04+)
- **RAM**: 16GB 이상 (32GB 권장)
- **GPU**: CUDA 지원 GPU (8GB+ VRAM 권장)
- **디스크**: 100GB 이상 여유 공간

### 설치 과정
```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/AutoCI.git
cd AutoCI

# 2. C# 환경 설정 및 의존성 설치
dotnet new console -n AutoCI
cd AutoCI
dotnet add package SocketIoClientDotNet  # Socket.IO 클라이언트
# 변형된 Godot 엔진 설치

# 3. AI 모델 설치
dotnet run install_llm_models.cs

# 4. AutoCI 전역 명령어 설치 (선택 사항)
sudo ./install_global_autoci.sh
```

## 🔄 24시간 끈질긴 개발 과정

### 개발 철학: "절대 포기하지 않는 AI"
```
오류 발견 → 기본 수정 시도 (1-10회)
     ↓ 실패시
웹 검색 + AI 솔루션 (11-50회)
     ↓ 실패시  
창의적 우회 + 커뮤니티 (51-100회)
     ↓ 실패시
실험적 접근 + 양자 디버깅 (101-∞회)
```

### 품질 향상 단계
- **기본 기능** (0-30점): 플레이어 제어, 물리, 충돌
- **중급 기능** (30-50점): 사운드, 점수, 메커니즘
- **고급 기능** (50-70점): UI/UX, 저장, 레벨 디자인
- **폴리싱** (70-100점): 파티클, 애니메이션, 최적화

## 🎨 실시간 개발 과정 시각화

### 단계별 게임 제작
```
🎯 1단계: 기획 (게임 컨셉 정의)
🎨 2단계: 디자인 (아트 방향성 결정)  
🔧 3단계: 프로토타입 (기본 시스템 구현)
⚙️ 4단계: 메커니즘 (게임플레이 구현)
🗺️ 5단계: 레벨 디자인 (콘텐츠 제작)
🎵 6단계: 오디오 (사운드 및 음악)
🎨 7단계: 비주얼 (그래픽 폴리싱)
🧪 8단계: 테스트 (버그 수정)
📦 9단계: 빌드 (배포 준비)
📝 10단계: 문서화 (완성 보고서)
```

### 실시간 상태 표시
```
⏱️ 경과: 2:34:15 | 남은 시간: 21:25:45
🔄 반복: 147 | 수정: 23 | 개선: 18  
📊 현재 게임 품질 점수: 67/100
🔧 현재 작업: 점프 메커니즘 개선 중...
```

## 🔧 문제 해결

### 자주 발생하는 문제

#### 1. 모듈 import 오류
```bash
# .NET 환경 확인
dotnet --version

# 또는 C# 프로젝트 경로 확인
dotnet --list-sdks
```

#### 2. AI 모델 로딩 실패
```bash
# CUDA 설치 확인
nvidia-smi

# 메모리 부족 시 가벼운 모델 사용
autoci learn low
```

#### 3. 변형된 Godot 실행 오류
```bash
# 변형된 Godot 설치 확인
# Godot 실행 파일 경로 확인

# 필요한 패키지 설치 확인
dotnet restore

# 로그 확인 (변형된 Godot 자체 로그 또는 AutoCI 로그)
# 예: cat logs/godot_errors.log
```

## 📁 프로젝트 구조

```
AutoCI/
├── core/                    # 시스템 핵심 코드 (autoci.cs, autoci_command.cs 등)
├── modules/                 # AI/게임/학습 관련 모듈
├── user_learning_data/      # 사용자별 학습 데이터
├── continuous_learning/     # AI 연속 학습 시스템 및 데이터
├── mvp_games/               # 완성된 게임 프로젝트
├── game_projects/           # 생성 중인 변형된 Godot 게임 프로젝트
├── tools/                   # 부가 도구 (api_doc_generator.cs 등)
├── tests/                   # 테스트 코드
├── docs/                    # 문서 및 가이드
├── config/                  # 환경설정 파일
├── legacy/                  # 이전 버전/실험 코드
├── logs/                    # 로그 파일
├── models/                  # AI 모델 관련 파일
├── evolution_data/          # 집단지성/진화 관련 데이터
├── requirements.txt
├── README.md
└── 기타 설정/스크립트 파일
```

## 📈 성능 지표

- **게임 제작 시간**: 5분 (MVP) → 24시간 (완성품)
- **AI 응답 속도**: 초당 10-50 토큰
- **학습 완료율**: 95%+
- **오류 해결률**: 99%+ (끈질긴 모드)

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 라이선스

AutoCI License 

---

**버전**: 5.0 (2025년 7월)  
**개발팀**: AutoCI Team  
**문의**: GitHub Issues
