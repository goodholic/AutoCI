# AutoCI 실시간 모니터링 가이드

## 개요
AutoCI의 24시간 백그라운드 프로세스를 실시간으로 모니터링할 수 있는 시스템을 구현했습니다.

## 사용 방법

### 1. AutoCI 백그라운드 실행
```bash
# AutoCI를 백그라운드에서 실행
autoci &

# 또는 nohup으로 실행
nohup autoci > autoci.log 2>&1 &
```

### 2. 실시간 모니터링 시작
새 터미널을 열고 다음 명령어를 실행하세요:

```bash
# 기본 모니터링 (curses UI)
autoci-monitor

# Simple 모드 (curses 없이)
autoci-monitor --simple

# 특정 프로젝트 모니터링
autoci-monitor --project my_game
```

## 모니터링 화면 구성

### Curses 모드 (기본)
- **상단**: 프로젝트 정보
- **중앙**: 실시간 상태 정보
  - 경과 시간 / 남은 시간
  - 진행률
  - 반복/수정/개선 횟수
  - 품질 점수
  - 현재 작업
  - 끈질김 레벨
  - 창의성 레벨
- **하단**: 실시간 로그 (최근 활동)

### Simple 모드
- 1초마다 화면이 갱신됩니다
- Ctrl+C로 종료할 수 있습니다

## 모니터링 정보

### 끈질김 레벨
- **NORMAL**: 일반 모드
- **DETERMINED**: 결연한 모드
- **STUBBORN**: 고집스러운 모드
- **OBSESSIVE**: 집착적 모드
- **INFINITE**: 무한 모드

### 창의성 레벨
- 0-10 스케일로 표시
- 오류 해결 시도가 많을수록 증가

### 절망 모드
- 50번 이상 실패 시 활성화
- 극단적인 해결 방법 시도

## 로그 파일 위치
- 상태 파일: `logs/24h_improvement/{프로젝트명}_status.json`
- 진행 파일: `logs/24h_improvement/{프로젝트명}_progress.json`
- 로그 파일: `logs/24h_improvement/latest_improvement.log`

## 주의 사항
- WSL 환경에서는 기본적으로 curses 모드를 사용합니다
- Windows에서는 자동으로 simple 모드로 전환됩니다
- 모니터링은 읽기 전용이며 프로세스에 영향을 주지 않습니다