# AutoCI 명령어 가이드 🚀

가상 환경 활성화 없이 AutoCI를 바로 사용할 수 있습니다!

## 설치 방법

### Windows
```batch
# 관리자 권한으로 실행
install_autoci_command.bat
```

### Linux/Mac
```bash
# 설치 스크립트 실행
chmod +x install_autoci_command.sh
./install_autoci_command.sh
```

## 사용 가능한 명령어

### 1. 기본 명령어
```bash
# 대화형 모드로 AutoCI 시작
autoci

# 도움말 보기
autoci --help
```

### 2. 게임 개발 명령어
```bash
# 24시간 자동 게임 개발
autoci create --name MyGame --type platformer

# 게임 타입 옵션:
# - platformer : 플랫폼 게임
# - racing     : 레이싱 게임  
# - rpg        : RPG 게임
# - puzzle     : 퍼즐 게임
# - shooter    : 슈팅 게임
# - adventure  : 어드벤처 게임
# - simulation : 시뮬레이션 게임

# 개발 시간 지정 (기본 24시간)
autoci create --name MyGame --type rpg --hours 12
```

### 3. AI 학습 명령어
```bash
# AI 모델 기반 연속 학습
autoci learn

# 메모리 최적화 학습 (RTX 2080 등 8GB GPU)
autoci learn low

# 커스텀 학습 설정
autoci learn --hours 2 --memory 16
```

### 4. 엔진 개선 명령어
```bash
# 학습을 토대로 AI의 게임 엔진 능력 업데이트
autoci fix

# 특정 기능 개선
autoci fix --feature physics
autoci fix --feature rendering
```

### 5. 모니터링 명령어
```bash
# 실시간 개발 모니터링
autoci monitor

# 포트 지정
autoci monitor --port 5001
```

### 6. 분석 명령어
```bash
# 게임 프로젝트 분석
autoci analyze game_projects/MyGame
```

### 7. 데모 명령어
```bash
# 5분 빠른 데모
autoci demo
```

## 사용 예시

### 플랫폼 게임 만들기
```bash
# 슈퍼마리오 스타일의 플랫폼 게임 생성
autoci create --name SuperPlatformer --type platformer
```

### RPG 게임 만들기 (12시간)
```bash
# 12시간 동안 RPG 게임 개발
autoci create --name MyRPGAdventure --type rpg --hours 12
```

### AI 학습 후 게임 개발
```bash
# 1. AI 학습 실행
autoci learn

# 2. 학습 완료 후 게임 개발
autoci create --name SmartGame --type puzzle
```

### 멀티 게임 동시 개발
```bash
# 터미널 1
autoci create --name Game1 --type platformer

# 터미널 2  
autoci create --name Game2 --type racing

# 터미널 3 - 모니터링
autoci monitor
```

## 고급 사용법

### 환경 변수 설정
```bash
# GPU 메모리 제한
export AUTOCI_GPU_MEMORY=8

# 디바이스 지정
export AUTOCI_DEVICE=cuda:0

# 모델 경로 지정
export AUTOCI_MODEL_PATH=/path/to/models
```

### 배치 스크립트
```bash
#!/bin/bash
# batch_create_games.sh

games=(
    "Platformer:platformer:24"
    "RacingPro:racing:12"
    "PuzzleMaster:puzzle:6"
    "SpaceShooter:shooter:18"
)

for game in "${games[@]}"; do
    IFS=':' read -r name type hours <<< "$game"
    echo "Creating $name..."
    autoci create --name "$name" --type "$type" --hours "$hours" &
    sleep 60  # 1분 간격으로 시작
done

wait
echo "All games created!"
```

### PowerShell 스크립트 (Windows)
```powershell
# batch_create_games.ps1

$games = @(
    @{Name="Platformer"; Type="platformer"; Hours=24},
    @{Name="RacingPro"; Type="racing"; Hours=12},
    @{Name="PuzzleMaster"; Type="puzzle"; Hours=6}
)

foreach ($game in $games) {
    Write-Host "Creating $($game.Name)..."
    Start-Process -NoNewWindow autoci -ArgumentList "create", "--name", $game.Name, "--type", $game.Type, "--hours", $game.Hours
    Start-Sleep -Seconds 60
}
```

## 문제 해결

### 명령어를 찾을 수 없음
```bash
# PATH 확인
echo $PATH

# 수동으로 PATH 추가
export PATH="$HOME/.local/bin:$PATH"

# Windows
set PATH=%USERPROFILE%\AppData\Local\autoci;%PATH%
```

### Python 버전 오류
```bash
# Python 버전 확인
python --version

# Python 3.8+ 필요
# pyenv 사용 시
pyenv install 3.8.10
pyenv global 3.8.10
```

### GPU 인식 오류
```bash
# CUDA 확인
nvidia-smi

# PyTorch GPU 테스트
python -c "import torch; print(torch.cuda.is_available())"
```

## 팁과 트릭

### 1. 빠른 프로토타입
```bash
# 30분 빠른 프로토타입
autoci create --name QuickProto --type platformer --hours 0.5
```

### 2. 야간 개발
```bash
# 밤새 게임 개발 (8시간)
autoci create --name NightGame --type rpg --hours 8
```

### 3. 주말 프로젝트
```bash
# 48시간 대작 게임
autoci create --name WeekendMasterpiece --type adventure --hours 48
```

## 자주 묻는 질문

**Q: 가상 환경을 활성화해야 하나요?**
A: 아니요! `autoci` 명령어가 자동으로 가상 환경을 관리합니다.

**Q: 여러 게임을 동시에 만들 수 있나요?**
A: 네! 각각 다른 터미널에서 실행하면 됩니다.

**Q: 개발 중인 게임을 중단하려면?**
A: Ctrl+C를 누르면 안전하게 중단됩니다.

**Q: 생성된 게임은 어디에 저장되나요?**
A: `game_projects/` 폴더에 저장됩니다.

**Q: AI 모델은 어떤 것을 사용하나요?**
A: DeepSeek-Coder, Llama-3.1, CodeLlama 등을 자동 선택합니다.