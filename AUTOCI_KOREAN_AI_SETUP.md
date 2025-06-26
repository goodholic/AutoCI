# 🇰🇷 AutoCI ChatGPT 수준 한국어 AI 설치 가이드

## 🚀 한 번에 설치하기 (WSL 환경)

### 1. 빠른 설치
```bash
# AutoCI 디렉토리에서 실행
bash install_autoci_wsl.sh
```

### 2. 설치 완료 후 사용
```bash
# 새 터미널을 열거나
source ~/.bashrc

# 이제 어디서든 사용 가능
autoci
```

## 🌟 주요 기능

### ChatGPT 수준 한국어 AI
- **자연스러운 한국어 대화**: "너 나랑 대화할 수 있어?"
- **격식체/반말 자동 감지**: 사용자의 말투에 맞춰 응답
- **감정 인식**: 사용자의 감정 상태를 파악하고 공감적 응답
- **Unity 전문 지식**: 게임 개발 관련 전문적인 도움
- **문맥 이해**: 이전 대화 내용을 기억하며 연속적인 대화

## 📋 명령어 가이드

### 기본 사용법
```bash
autoci                    # ChatGPT 수준 한국어 AI 대화 모드
autoci korean             # 한국어 대화 모드
autoci k                  # 한국어 모드 (단축키)
autoci 한국어             # 한글 명령어
```

### 한국어 alias (별명)
```bash
오토시아이                 # autoci와 동일
한국어ai                  # autoci korean과 동일
ai대화                    # autoci korean과 동일
코드생성 PlayerController  # autoci create PlayerController
코드수정 script.cs        # autoci modify script.cs
코드개선 MyScript.cs      # autoci improve MyScript.cs
버그수정 NetworkManager   # autoci fix NetworkManager
도움말                    # autoci help
```

### 빠른 코드 작업
```bash
autoci c PlayerController        # 코드 생성
autoci m Assets/Scripts/Game.cs  # 코드 수정
autoci i MyScript.cs            # 코드 개선
autoci f "네트워크 연결 오류"    # 버그 수정
```

## 💬 대화 예시

### 자연스러운 한국어 대화
```
$ autoci

🤖 AutoCI ChatGPT 수준 한국어 AI 통합 시스템
✨ 자연스러운 한국어로 대화하세요!

🤖 autoci> 안녕하세요! Unity 게임 개발을 도와드릴 준비가 되었습니다.

사용자: 너 나랑 대화할 수 있어?

🤖 autoci> 네, 당연히 대화할 수 있어요! 😊 
저는 ChatGPT 수준의 한국어 AI로 업그레이드되어서 
자연스러운 한국어로 대화가 가능합니다.

Unity 게임 개발, C# 프로그래밍, 또는 다른 궁금한 것들
편하게 물어보세요!

사용자: PlayerController 스크립트 만들어줘

🤖 autoci> PlayerController 스크립트를 만들어드릴게요! 
기본적인 이동과 점프 기능이 포함된 스크립트를 작성하겠습니다.

[PlayerController.cs 파일이 생성됩니다]
```

### 격식체 자동 감지
```
사용자: 안녕하십니까, Unity에서 최적화 방법을 알려주시겠습니까?

🤖 autoci> 안녕하십니까! Unity 최적화에 대해 자세히 설명드리겠습니다.

[격식체로 전문적인 답변 제공]

사용자: 고마워! 더 간단한 방법 있어?

🤖 autoci> 그럼! 더 쉬운 방법들 알려줄게 😊

[반말로 친근한 답변 제공]
```

## 🛠️ 수동 설치 (고급 사용자)

### 1. PATH 설정
```bash
# ~/.bashrc에 추가
echo 'export PATH="$PATH:/path/to/AutoCI"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 실행 권한 부여
```bash
chmod +x autoci
```

### 3. 한국어 alias 설정 (선택사항)
```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'
alias 오토시아이='autoci'
alias 한국어ai='autoci korean'
alias ai대화='autoci korean'
alias 코드생성='autoci create'
alias 코드수정='autoci modify'
alias 코드개선='autoci improve'
alias 버그수정='autoci fix'
alias 도움말='autoci help'
EOF
```

## 🔧 문제 해결

### 의존성 오류 (ModuleNotFoundError)
```bash
# 오류: ModuleNotFoundError: No module named 'rich'
# 해결: 의존성 설치 스크립트 실행
bash install_dependencies_wsl.sh

# 또는 수동 설치
source llm_venv_wsl/bin/activate  # 가상환경 활성화
pip install rich colorama psutil requests
```

### 명령어를 찾을 수 없는 경우
```bash
# PATH 확인
echo $PATH

# bashrc 다시 로드
source ~/.bashrc

# 직접 실행으로 테스트
./autoci help
```

## 🎯 사용 팁

### 1. 자연스러운 대화
- "안녕하세요", "안녕", "뭐해?" 등 자연스럽게 인사
- "~해줘", "~할 수 있어?", "~방법 알려줘" 등 편한 말투
- 감정 표현도 인식: "답답해", "고마워", "어려워" 등

### 2. Unity 전문 질문
- "PlayerController 만들어줘"
- "Object Pool 패턴 구현하는 방법"
- "네트워크 동기화 문제 해결"
- "성능 최적화 방법"

### 3. 파일 작업
- 코드 생성, 수정, 개선 모두 자동화
- 파일 경로 지정 가능
- Unity Assets 폴더 구조 이해

## 🚀 고급 기능

### 다른 AutoCI 모드들
```bash
autoci terminal          # 기존 터미널 모드
autoci dual start        # 고급 RAG 시스템
autoci enhance start     # 24시간 자동 시스템
autoci help              # 전체 명령어 보기
```

### 웹 인터페이스
```bash
autoci dual start        # 웹 UI: http://localhost:8080
```

---

🎉 **이제 어디서든 `autoci` 명령어로 ChatGPT 수준의 한국어 AI와 대화하며 Unity 개발을 진행할 수 있습니다!** 