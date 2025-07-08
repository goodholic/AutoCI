# AutoCI Windows Commands - WSL과 동일하게 사용하기

## 🚀 빠른 시작

### 1. 필요한 패키지 설치 (처음 한 번만)
```cmd
.\install-requirements.cmd
```

### 2. 3대 핵심 명령어 사용

#### 방법 1: UI 모드 (추천)
```cmd
.\autoci-ui.cmd
```
메뉴에서 선택하여 사용

#### 방법 2: 직접 명령어 (WSL과 동일)
```cmd
.\learn-now.cmd
.\create-now.cmd  
.\fix-now.cmd
```

#### 방법 3: 더 짧은 명령어
```cmd
.\_learn
.\_create
.\_fix
```

## 🎯 WSL과 동일한 경험을 위한 설정

### PowerShell 별칭 설정
PowerShell에서 한 번만 실행:
```powershell
# PowerShell 프로필 열기
notepad $PROFILE

# 다음 내용 추가
function learn { & "D:\AutoCI\AutoCI\learn-now.cmd" }
function create { & "D:\AutoCI\AutoCI\create-now.cmd" }
function fix { & "D:\AutoCI\AutoCI\fix-now.cmd" }

# 저장 후 PowerShell 재시작
```

이제 PowerShell 어디서든:
```powershell
learn
create
fix
```

## 🔧 문제 해결

### "accelerate" 오류
```cmd
py -m pip install accelerate
```

### CUDA/GPU 오류
```cmd
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### UI/색상이 안 보일 때
```cmd
py -m pip install colorama rich
```

## ✨ 특징

- WSL과 동일한 명령어 체계
- 가상환경 프롬프트 없음
- Windows 네이티브 실행
- UI 모드 지원