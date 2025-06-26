# 🔧 AutoCI 의존성 오류 해결 가이드

## 문제 상황
```
ModuleNotFoundError: No module named 'rich'
```

## 🚀 빠른 해결 방법

### 1. WSL 터미널에서 AutoCI 디렉토리로 이동
```bash
cd "/mnt/c/Users/super/Desktop/Unity Project(25년도 제작)/26.AutoCI/AutoCI"
```

### 2. 의존성 설치 스크립트 실행
```bash
bash install_dependencies_wsl.sh
```

### 3. 수동 설치 (위 방법이 안 될 경우)
```bash
# 가상환경 활성화
source llm_venv_wsl/bin/activate

# 필수 패키지 설치
pip install rich colorama psutil requests

# 설치 확인
pip list | grep -E "(rich|colorama|psutil|requests)"
```

### 4. AutoCI 실행 테스트
```bash
./autoci help
```

## 🎯 완전 자동 설치 (추천)

### 한 번에 설치하기
```bash
# AutoCI 디렉토리에서
bash install_autoci_wsl.sh
```

### 새 터미널에서 사용
```bash
# bashrc 다시 로드
source ~/.bashrc

# 어디서든 사용 가능
autoci
```

## 🐛 추가 문제 해결

### 가상환경이 없는 경우
```bash
# 새 가상환경 생성
python3 -m venv llm_venv_wsl

# 가상환경 활성화
source llm_venv_wsl/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install rich colorama psutil requests
```

### Python3가 없는 경우
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

## ✅ 성공 확인

설치가 완료되면 다음 명령어들이 작동해야 합니다:

```bash
autoci                    # ChatGPT 수준 한국어 AI
autoci korean             # 한국어 모드
autoci help              # 도움말
```

## 💬 테스트 대화

```bash
autoci
🤖 autoci> 안녕하세요! Unity 개발 도와주세요
🤖 autoci> 너랑 대화할 수 있어?
🤖 autoci> PlayerController 만들어줘
```

---

**문제가 계속되면 이 파일의 명령어를 순서대로 실행해보세요!** 🚀 