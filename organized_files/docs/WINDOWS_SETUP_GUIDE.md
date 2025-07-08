# AutoCI Windows 설정 가이드

## 🚀 빠른 설정 (권장)

### 1. 설치 스크립트 실행
관리자 권한으로 Command Prompt를 열고:
```batch
cd D:\AutoCI\AutoCI
install-windows.bat
```

### 2. 새 Command Prompt 열기
설치 후 **반드시 새로운 Command Prompt**를 열어야 PATH가 적용됩니다.

### 3. 명령어 사용
이제 어디서든 사용 가능:
```batch
autoci learn
autoci create platformer
autoci fix
```

## 🛠️ 수동 설정 (설치 스크립트가 작동하지 않을 때)

### 방법 1: 전체 경로 사용
```batch
D:\AutoCI\AutoCI\autoci.bat learn
D:\AutoCI\AutoCI\autoci.bat create platformer
D:\AutoCI\AutoCI\autoci.bat fix
```

### 방법 2: 디렉토리 이동 후 실행
```batch
cd D:\AutoCI\AutoCI
autoci.bat learn
autoci.bat create platformer
autoci.bat fix
```

### 방법 3: 개별 배치 파일 사용
```batch
D:\AutoCI\AutoCI\autoci-learn.bat
D:\AutoCI\AutoCI\autoci-create.bat platformer
D:\AutoCI\AutoCI\autoci-fix.bat
```

### 방법 4: PowerShell 사용
PowerShell을 열고:
```powershell
cd D:\AutoCI\AutoCI
.\autoci.ps1 learn
.\autoci.ps1 create platformer
.\autoci.ps1 fix
```

## 📝 PATH 수동 추가

### Windows 10/11에서 PATH 추가:
1. Windows 키 + X → 시스템
2. 고급 시스템 설정 → 환경 변수
3. 시스템 변수에서 "Path" 선택 → 편집
4. 새로 만들기 → `D:\AutoCI\AutoCI` 추가
5. 확인 → 확인 → 확인
6. **새 Command Prompt 열기**

## 🔧 문제 해결

### "python이 내부 또는 외부 명령... 아닙니다"
Python이 설치되지 않았거나 PATH에 없습니다:
1. https://www.python.org 에서 Python 3.8+ 설치
2. 설치 시 "Add Python to PATH" 체크

### "가상환경이 활성화되지 않았습니다"
```batch
cd D:\AutoCI\AutoCI
autoci_env\Scripts\activate.bat
autoci learn
```

### PowerShell 실행 정책 오류
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 한글 깨짐 문제
Command Prompt에서:
```batch
chcp 65001
```

## 📂 파일 구조 설명

```
D:\AutoCI\AutoCI\
├── autoci.bat          # 메인 실행 파일
├── autoci.ps1          # PowerShell 버전
├── autoci-learn.bat    # learn 전용
├── autoci-create.bat   # create 전용
├── autoci-fix.bat      # fix 전용
├── install-windows.bat # 설치 스크립트
└── autoci              # Python 스크립트 (직접 실행 X)
```

## ✅ 권장 사용법

1. **처음 한 번만**: `install-windows.bat` 실행 (관리자 권한)
2. **이후**: 어디서든 `autoci learn`, `autoci create platformer` 등 사용

## 💡 팁

- WSL을 사용 중이라면 WSL 터미널에서는 `./autoci` 사용
- Windows 터미널에서는 `autoci.bat` 또는 `autoci` 사용 (PATH 추가 후)
- 가상환경은 자동으로 감지되므로 별도 활성화 불필요

이제 Windows에서 AutoCI를 즐기세요! 🎮