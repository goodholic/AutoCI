# 🤖 AutoCI - 24시간 학습하는 한국어 AI 코딩 어시스턴트

<div align="center">
  <img src="https://img.shields.io/badge/AI-자가학습-brightgreen?style=for-the-badge" alt="자가학습">
  <img src="https://img.shields.io/badge/한국어-ChatGPT수준-blue?style=for-the-badge" alt="한국어">
  <img src="https://img.shields.io/badge/Unity-C%23전문가-purple?style=for-the-badge" alt="Unity">
  <img src="https://img.shields.io/badge/모니터링-1분간격-orange?style=for-the-badge" alt="모니터링">
</div>

## 🌟 AutoCI란?

AutoCI는 **실제로 학습하는 AI** 기반 코딩 어시스턴트입니다. ChatGPT처럼 자연스러운 한국어로 대화하며, 24시간 스스로 학습하여 계속 똑똑해집니다.

### 🚀 핵심 특징

1. **🧠 진짜 학습하는 AI**
   - 사용자와의 대화에서 학습
   - 코드 패턴을 스스로 분석하고 개선
   - 실수를 기억하고 다음에는 더 나은 답변 제공

2. **💬 ChatGPT 수준 한국어 대화**
   - "안녕! 오늘 뭐 만들까?" 같은 자연스러운 대화
   - 감정을 이해하고 공감하는 응답
   - 존댓말/반말 자동 감지 및 맞춤 응답

3. **📊 1분마다 학습 상태 모니터링**
   - 실시간 학습 진행률 확인
   - 웹 대시보드로 시각적 모니터링
   - AI가 무엇을 배우고 있는지 투명하게 공개

4. **🎮 Unity/C# 전문가 수준**
   - Unity 프로젝트 구조 자동 분석
   - 베스트 프랙티스 기반 코드 개선
   - 실시간 에러 해결 도움

## 📦 설치 방법 (5분 완성)

### 1️⃣ 프로젝트 클론
```bash
git clone https://github.com/yourusername/AutoCI.git
cd AutoCI
```

### 2️⃣ 자동 설치 (WSL/Linux/Mac)
```bash
chmod +x install_autoci_wsl.sh
./install_autoci_wsl.sh
```

### 3️⃣ 설치 확인
```bash
# 터미널 재시작 후
autoci --version
```

## 🎯 사용 방법

### 🗣️ 한국어 대화 모드
```bash
autoci
```

실행하면 바로 한국어로 대화할 수 있습니다:
```
🤖: 안녕하세요! 오늘은 어떤 프로젝트를 진행하고 계신가요?
👤: Unity에서 플레이어 이동 스크립트 만들고 싶어
🤖: 좋아요! 2D 게임인가요, 3D 게임인가요? 어떤 스타일의 이동을 원하시나요?
```

### 📊 학습 모니터링
```bash
# 1분마다 자동 모니터링 시작
autoci monitor start

# 웹 대시보드 열기 (http://localhost:8888)
autoci monitor web
```

### 🧠 24시간 자가 학습
```bash
# 백그라운드 학습 시작
autoci learn start

# 학습 상태 확인
autoci learn status
```

## 💡 실제 사용 예시

### 1. Unity 프로젝트 정리
```bash
autoci
```
```
👤: 내 Unity 프로젝트 Scripts 폴더가 너무 복잡해
🤖: 프로젝트 경로를 알려주시면 제가 분석해서 정리해드릴게요!
👤: C:/MyUnityProject
🤖: 분석 완료! 다음과 같이 정리하면 좋을 것 같아요:
    - Scripts/Player (플레이어 관련)
    - Scripts/UI (UI 관련)
    - Scripts/Managers (게임 매니저)
    자동으로 정리할까요? (y/n)
```

### 2. 코드 에러 해결
```
👤: PlayerController.cs에서 NullReferenceException이 계속 나 ㅠㅠ
🤖: 힘드시죠? 제가 도와드릴게요! 
    에러가 어느 줄에서 발생하나요? 
    아니면 제가 파일을 분석해볼까요?
```

### 3. 학습 진행 상황
```
👤: 너 지금 뭐 배우고 있어?
🤖: 지금 Unity의 새로운 Input System에 대해 학습 중이에요!
    오늘 3시간 동안 152개의 코드 패턴을 분석했고,
    특히 모바일 터치 입력 처리 부분을 집중적으로 공부했어요. 📚
```

## 🧠 실제 학습 시스템 작동 원리

### 1. 대화 학습
- 모든 대화를 분석하여 패턴 추출
- 사용자 피드백으로 응답 품질 개선
- 자주 나오는 질문은 더 빠르고 정확하게 답변

### 2. 코드 패턴 학습
- 사용자 프로젝트의 코드 스타일 학습
- 반복되는 에러 패턴 기억
- 성공적인 해결책 데이터베이스화

### 3. 지속적 개선
- 매일 밤 학습 내용 정리 및 최적화
- 새로운 Unity 버전/C# 기능 자동 학습
- 커뮤니티 베스트 프랙티스 수집

## 📊 모니터링 대시보드

웹 브라우저에서 http://localhost:8888 접속:

```
┌─────────────────────────────────────┐
│ 🧠 AutoCI 학습 상태                  │
├─────────────────────────────────────┤
│ 학습률: ████████░░ 82%              │
│ 대화 수: 1,247개                    │
│ 학습한 패턴: 3,891개                │
│ 정확도: 94.3% ↑2.1%                │
│                                     │
│ 최근 학습 주제:                     │
│ • Unity 2023 새 기능               │
│ • async/await 패턴                 │
│ • 모바일 최적화 기법               │
└─────────────────────────────────────┘
```

## 🛠️ 고급 기능

### 프로젝트 분석
```bash
autoci analyze /path/to/project
```

### 코드 자동 개선
```bash
autoci improve PlayerController.cs
```

### 학습 데이터 추가
```bash
autoci learn add https://github.com/awesome/unity-project
```

## 🔧 시스템 요구사항

- **OS**: Windows (WSL2), Linux, macOS
- **Python**: 3.8+
- **메모리**: 8GB+ (16GB 권장)
- **저장공간**: 10GB+

## 🤝 기여하기

AutoCI는 오픈소스입니다! 기여를 환영합니다:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 라이선스

MIT License - 자유롭게 사용하세요!

## 🙏 감사의 말

- OpenAI의 GPT 기술에서 영감을 받았습니다
- Unity 커뮤니티의 지속적인 피드백
- 모든 기여자들께 감사드립니다

---

<div align="center">
  <h3>🚀 지금 바로 시작하세요!</h3>
  <p>AutoCI와 함께라면 코딩이 즐거워집니다</p>
  <code>autoci</code>
</div>