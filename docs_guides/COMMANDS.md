# AutoCI 명령어 가이드 📚

## 주요 명령어

### 1. 기본 실행
```bash
autoci
```
- 대화형 메뉴를 표시합니다
- 원하는 기능을 선택하여 사용할 수 있습니다

### 2. 24시간 자동 게임 개발
```bash
autoci create [게임타입]
```
- **게임 타입**: `platformer`, `racing`, `rpg`, `puzzle`
- AI가 24시간 동안 자동으로 완전한 게임을 개발합니다
- 예시:
  ```bash
  autoci create platformer    # 플랫폼 게임 자동 개발
  autoci create racing        # 레이싱 게임 자동 개발
  ```

### 3. 한글 대화 모드
```bash
autoci chat
```
- 자연스러운 한국어로 게임 개발을 진행합니다
- "플랫폼 게임 만들어줘"와 같은 명령을 사용할 수 있습니다
- 개발 중인 게임을 실시간으로 수정할 수 있습니다

### 4. AI 학습 모드
```bash
autoci learn         # 기본 학습 (24시간)
autoci learn low     # 메모리 최적화 학습 (8GB GPU)
```
- **학습 주제**: Python, 한글 용어, Panda3D, Socket.IO, AI 최적화
- 학습한 내용은 지식 베이스에 저장되어 재사용됩니다
- `low` 옵션: RTX 2080 8GB, 32GB RAM 환경에 최적화

### 5. AI 능력 업데이트
```bash
autoci fix
```
- 학습한 내용을 바탕으로 AI의 게임 제작 능력을 개선합니다
- 진화 알고리즘을 통해 최적의 패턴을 찾습니다
- 베스트 프랙티스가 자동으로 저장됩니다

### 6. 실시간 모니터링
```bash
autoci monitor
```
- 웹 기반 대시보드를 실행합니다
- http://localhost:5555 에서 접속 가능
- 시스템 리소스, 개발 진행 상황, AI 모델 상태를 실시간으로 확인

### 7. 시스템 상태 확인
```bash
autoci status
```
- AI 모델 상태
- 진화 시스템 통계
- 게임 프로젝트 현황
- 시스템 건강도 체크

### 8. 도움말
```bash
autoci help
```
- 사용 가능한 모든 명령어와 옵션을 표시합니다

## 사용 예시

### 시나리오 1: 플랫폼 게임 만들기
```bash
# 방법 1: 자동 개발
autoci create platformer

# 방법 2: 대화형
autoci chat
> 플랫폼 게임 만들어줘
> 점프 높이 더 높게 해줘
> 적 캐릭터 추가해줘
```

### 시나리오 2: AI 학습 후 개선
```bash
# 1. 학습 실행
autoci learn low    # 메모리가 적은 경우

# 2. 학습 내용으로 능력 개선
autoci fix

# 3. 개선된 AI로 게임 개발
autoci create rpg
```

### 시나리오 3: 모니터링하며 개발
```bash
# 터미널 1: 게임 개발
autoci create racing

# 터미널 2: 모니터링
autoci monitor

# 브라우저에서 http://localhost:5555 접속
```

## 고급 사용법

### 백그라운드 실행
```bash
# 24시간 개발을 백그라운드로
nohup autoci create platformer > game_dev.log 2>&1 &

# 로그 확인
tail -f game_dev.log
```

### 스크립트로 자동화
```bash
#!/bin/bash
# auto_develop.sh

# 학습 실행
autoci learn low

# 능력 업데이트
autoci fix

# 게임 개발
for game_type in platformer racing rpg puzzle; do
    autoci create $game_type
    sleep 300  # 5분 대기
done
```

### 상태 모니터링 자동화
```bash
# 5초마다 상태 확인
watch -n 5 autoci status
```

## 문제 해결

### 명령어를 찾을 수 없음
```bash
# 설치 스크립트 재실행
cd /mnt/d/AutoCI/AutoCI
./install_autoci_wsl.sh
```

### 메모리 부족
```bash
# 메모리 최적화 모드 사용
autoci learn low
```

### GPU 오류
```bash
# CPU 모드로 실행 (환경변수 설정)
export CUDA_VISIBLE_DEVICES=""
autoci create platformer
```

## 팁과 트릭

1. **빠른 테스트**: `autoci chat`에서 간단한 게임부터 시작
2. **성능 최적화**: `autoci learn low`로 메모리 절약
3. **진행 상황 저장**: 개발 중간에 중단해도 자동 저장됨
4. **병렬 실행**: 여러 터미널에서 다른 게임 동시 개발 가능
5. **커스텀 설정**: `config/` 폴더의 설정 파일 수정

---

💡 **Pro Tip**: `autoci`만 입력하면 대화형 메뉴가 나타나므로, 명령어를 외울 필요가 없습니다!