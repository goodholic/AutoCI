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

## 📋 요구사항

- **OS**: Windows 10/11 + WSL2 (Ubuntu 20.04+)
- **RAM**: 8GB 이상 (16GB 권장)
- **디스크**: 30GB 이상 여유 공간
- **도구**: Python 3.8+, Git

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

### 2단계: AI Godot 빌드
```bash
# AI 수정된 Godot 빌드 (권장)
build-godot
# → 1번 선택: AI 수정된 Godot 빌드 (Linux)

# 상태 확인
check-godot
```

### 3단계: 실행
```bash
# 24시간 자동 게임 개발
autoci
> create platformer game

# 24시간 C# 학습
autoci learn
```

## 🛠️ AI Godot 빌드 옵션

### 🔨 옵션 1: WSL에서 AI 빌드 (권장)
```bash
build-godot  # → 1번 선택
```
- 소스코드 자동 다운로드 및 패치
- AI 제어 기능 추가
- 예상 시간: 20-60분

### ⚡ 옵션 2: 빠른 Windows 설정
```bash
build-godot  # → 2번 선택
```
- 일반 Godot 다운로드
- 기본 AI 기능만 제공
- 예상 시간: 5-10분

## 📂 프로젝트 구조

```
AutoCI/
├── 🎮 메인 시스템
│   ├── autoci                      # 전역 실행 파일
│   ├── autoci.py                   # 메인 진입점
│   ├── autoci_terminal.py          # 터미널 인터페이스
│   └── autoci_production.py        # 프로덕션 모드
│
├── 🤖 AI Godot 빌드
│   ├── build_ai_godot.py           # AI Godot 빌드 시스템
│   ├── build-godot                 # 빌드 명령어
│   ├── check-godot                 # 상태 확인 명령어
│   └── godot_ai_build/             # 빌드 출력 디렉토리
│       ├── output/                 # 빌드된 실행 파일
│       └── logs/                   # 빌드 로그
│
├── 📚 모듈
│   ├── modules/
│   │   ├── csharp_24h_learning.py     # 24시간 학습 시스템
│   │   ├── godot_ai_integration.py    # Godot AI 통합
│   │   ├── godot_realtime_dashboard.py # 실시간 대시보드
│   │   └── monitoring_system.py        # 모니터링
│   │
│   └── godot_ai/                   # AI 플러그인 및 템플릿
│
└── 💾 데이터
    ├── user_learning_data/         # 학습 데이터
    ├── game_projects/              # 생성된 게임 프로젝트
    └── logs/                       # 실행 로그
```

## 🎮 사용 예시

### 게임 개발
```bash
autoci
> create platformer game     # 플랫포머 게임
> create racing game         # 레이싱 게임
> create rpg game            # RPG 게임
> create puzzle game         # 퍼즐 게임
```

### C# 학습
```bash
# 전체 24시간 학습
autoci learn

# 특정 주제 학습
autoci --csharp-session "async/await"

# 데모 모드 (1시간)
autoci --learn-demo
```

## 🔧 고급 설정

### 프로덕션 모드
```bash
# 안정적인 실행 환경
autoci --production
```

### 모니터링
```bash
# 실시간 모니터링 활성화
autoci --enable-monitoring
```

## 📚 문서

- [빠른 시작 가이드](QUICK_START.md)
- [AI Godot 빌드 가이드](AI_GODOT_BUILD_GUIDE.md)
- [API 문서](docs/API.md)

## 🔍 문제 해결

### Godot을 찾을 수 없을 때
```bash
# 상태 확인
check-godot

# 재빌드
build-godot
```

### 빌드 실패 시
1. `godot_ai_build/logs/` 확인
2. 필수 패키지 설치:
   ```bash
   sudo apt update
   sudo apt install -y scons pkg-config libx11-dev libxcursor-dev
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

---

**버전**: 4.0 (2025년 6월)  
**문의**: GitHub Issues에 등록해주세요