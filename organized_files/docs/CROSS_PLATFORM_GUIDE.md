# AutoCI Cross-Platform Guide

AutoCI는 이제 Windows와 WSL 환경 모두에서 작동합니다!

## 🚀 빠른 시작

### Windows (Command Prompt)
```batch
# 가상환경 활성화
autoci_env\Scripts\activate.bat

# AutoCI 실행
autoci.bat learn
autoci.bat create platformer
autoci.bat fix
```

### Windows (PowerShell)
```powershell
# 가상환경 활성화
autoci_env\Scripts\Activate.ps1

# AutoCI 실행
.\autoci.ps1 learn
.\autoci.ps1 create platformer  
.\autoci.ps1 fix
```

### WSL/Linux
```bash
# 가상환경 활성화
source autoci_env/bin/activate

# AutoCI 실행
./autoci learn
./autoci create platformer
./autoci fix
```

## 📁 파일 구조

- `autoci` - 메인 Python 스크립트 (cross-platform)
- `autoci.bat` - Windows Command Prompt용
- `autoci.ps1` - Windows PowerShell용
- `autoci_cross_platform.py` - 대체 cross-platform 스크립트

## 🔧 주요 변경사항

1. **자동 플랫폼 감지**: Windows/WSL 자동 인식
2. **경로 자동 변환**: OS에 맞는 경로 사용
3. **Python 실행 최적화**: 각 OS에 최적화된 실행 방식

## 💻 지원 명령어

모든 플랫폼에서 동일하게 작동:
- `autoci learn` - AI 통합 학습
- `autoci learn low` - 메모리 최적화 학습  
- `autoci create [game_type]` - 게임 생성/이어서 개발
- `autoci resume` - 일시 정지된 게임 개발 재개
- `autoci sessions` - 모든 게임 개발 세션 보기
- `autoci fix` - 학습 기반 엔진 개선
- `autoci chat` - 한꺈 대화 모드
- `autoci monitor` - 실시간 모니터링

### 🎮 게임 개발 이어서 하기

#### 방법 1: 자동 인식
```bash
# 기존 platformer 게임이 있으면 자동으로 물어봄
autoci create platformer
# > 기존 platformer 게임 개발을 발견했습니다!
# > 이어서 개발하시겠습니까? (y/n):
```

#### 방법 2: 세션 선택
```bash
# 일시 정지된 세션 목록 보기
autoci resume
# > 1. platformer_game_20250106_143022 (platformer)
# >    상태: paused
# >    진행률: 45%
# > 어떤 세션을 이어서 개발하시겠습니까? (번호 입력):
```

#### 방법 3: 세션 현황 확인
```bash
# 모든 세션 현황 보기
autoci sessions
# > 🔴 platformer_game_20250106_143022 (platformer)
# >    상태: active
# >    진혁률: 45%
# > 🟡 racing_game_20250105_210155 (racing)  
# >    상태: paused
# >    진행률: 72%
```

## ⚠️ 주의사항

### Windows
- Python 3.8+ 필요
- 경로에 공백이 있어도 자동 처리됨
- PowerShell 실행 정책 확인 필요

### WSL
- X11 forwarding 설정 권장 (GUI 앱용)
- Windows 드라이브는 `/mnt/` 아래 마운트됨

## 🐛 문제 해결

### "Python not found" 오류
Windows에서 Python 설치 확인:
```batch
python --version
```

### PowerShell 실행 오류  
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### WSL에서 디스플레이 오류
```bash
export DISPLAY=:0
```

## 📝 개발자 노트

코드에서 cross-platform 지원:
```python
import platform
from pathlib import Path

def get_project_root():
    if platform.system() == "Windows":
        return Path(os.path.dirname(os.path.abspath(__file__)))
    else:
        return Path("/mnt/d/AutoCI/AutoCI")
```

## 💾 세션 데이터 위치

게임 개발 세션은 다음 위치에 저장됩니다:
- Windows: `C:\AutoCI\AutoCI\game_sessions\`
- WSL: `/mnt/d/AutoCI/AutoCI/game_sessions/`

각 세션은 다음 정보를 포함:
- `session.json` - 세션 메타데이터
- 게임 파일 경로 목록
- 개발 진행 상황
- 추가된 기능 목록

이제 Windows와 WSL 모두에서 AutoCI를 즐기세요! 🎮