# AutoCI WSL 환경 설정 가이드 🐧

## 빠른 설치 (WSL)

WSL 환경에서 AutoCI를 설치하고 전역 명령어로 사용하는 방법입니다.

### 1. 원클릭 설치

```bash
# AutoCI 디렉토리로 이동
cd /mnt/d/AutoCI/AutoCI

# 설치 스크립트 실행
./install_autoci_wsl.sh
```

### 2. 설치 확인

```bash
# 새 터미널을 열거나
source ~/.bashrc

# AutoCI 실행
autoci
```

## 수동 설치

설치 스크립트가 작동하지 않는 경우:

### 1. 가상환경 생성

```bash
cd /mnt/d/AutoCI/AutoCI
python3 -m venv autoci_env
source autoci_env/bin/activate
pip install -r requirements.txt
```

### 2. 전역 명령어 설정

```bash
# 실행 권한 부여
chmod +x core/autoci_wsl_launcher.py

# 전역 명령어 설정
sudo python3 core/autoci_wsl_launcher.py --setup
```

## 사용 방법

### 기본 명령어

```bash
# AutoCI 터미널 시작 (가상환경 자동 활성화)
autoci

# Panda3D 게임 개발 시작
autoci chat

# AI 학습 모드
autoci learn

# 실시간 모니터링
autoci monitor

# 도움말
autoci --help
```

### 주요 기능

1. **자동 가상환경 활성화**
   - `autoci` 명령어만 입력하면 자동으로 가상환경이 활성화됩니다
   - 프로젝트 디렉토리를 자동으로 찾아 이동합니다

2. **WSL 최적화**
   - WSL 환경을 자동으로 감지합니다
   - Linux 경로와 Windows 경로를 자동으로 처리합니다

3. **전역 접근**
   - 어느 디렉토리에서든 `autoci` 명령어를 사용할 수 있습니다
   - 프로젝트 경로를 기억할 필요가 없습니다

## 문제 해결

### 1. 명령어를 찾을 수 없음

```bash
# PATH 확인
echo $PATH | grep "/usr/local/bin"

# PATH에 추가 (없는 경우)
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 권한 오류

```bash
# 실행 권한 부여
chmod +x /mnt/d/AutoCI/AutoCI/core/*.py
chmod +x /mnt/d/AutoCI/AutoCI/*.sh
```

### 3. Python 모듈 오류

```bash
# 가상환경 활성화 후 패키지 재설치
cd /mnt/d/AutoCI/AutoCI
source autoci_env/bin/activate
pip install -r requirements.txt
```

### 4. WSL 버전 확인

```bash
# WSL 버전 확인
wsl --version

# WSL2로 업그레이드 (권장)
wsl --set-default-version 2
```

## 개발 환경 구성

### VSCode에서 사용

1. WSL 확장 설치
2. WSL 터미널에서 프로젝트 열기:
   ```bash
   cd /mnt/d/AutoCI/AutoCI
   code .
   ```

### PyCharm에서 사용

1. WSL Python 인터프리터 설정
2. 프로젝트 인터프리터: `/mnt/d/AutoCI/AutoCI/autoci_env/bin/python`

## 고급 설정

### 별칭 추가

```bash
# ~/.bashrc에 추가
alias aci='autoci'
alias aci-chat='autoci chat'
alias aci-learn='autoci learn'
alias aci-monitor='autoci monitor'
```

### 자동 완성 설정

```bash
# bash 자동완성 (향후 추가 예정)
# complete -W "chat learn monitor help" autoci
```

## 시스템 요구사항

- **OS**: Windows 10/11 + WSL2 (Ubuntu 20.04+)
- **Python**: 3.8 이상
- **RAM**: 8GB 이상 (16GB 권장)
- **디스크**: 10GB 이상 여유 공간

## 다음 단계

1. `autoci` 실행하여 터미널 시작
2. `autoci chat`으로 한글 대화 모드 시작
3. "플랫폼 게임 만들어줘"라고 입력하여 24시간 게임 개발 시작

---

💡 **팁**: WSL2는 WSL1보다 성능이 훨씬 좋습니다. 가능하면 WSL2를 사용하세요!