# Windows에서 AutoCI 사용하기 - 완전 해결 가이드

## 🔍 문제 진단

1. **Python이 설치되지 않음** - 가장 큰 문제입니다
2. **PowerShell 프로필이 로드되지 않음**

## 🚀 해결 방법

### 1단계: Python 설치 확인
```powershell
.\check-python.bat
```

### 2단계: Python이 없다면 설치
1. https://www.python.org/downloads/ 에서 Python 3.8+ 다운로드
2. 설치 시 **반드시** "Add Python to PATH" 체크 ✅
3. 설치 완료 후 **새 터미널 열기**

### 3단계: 전역 명령어 설치 (권장)
```powershell
.\install-global-commands.bat
```
이후 **새 터미널**을 열면 어디서든 사용 가능:
```powershell
autoci learn
autoci create platformer
autoci fix
```

### 4단계: PowerShell 함수 사용 (선택)
```powershell
# 프로필 다시 로드
. $PROFILE

# 이제 사용 가능
autoci-learn
autoci-create platformer
autoci-fix
```

## 💡 즉시 사용하는 방법 (Python만 있다면)

### 방법 1: 가상환경 생성 후 사용
```powershell
# Python이 설치되어 있다면
python -m venv autoci_env
.\autoci_env\Scripts\activate
pip install -r requirements.txt
.\autoci.bat learn
```

### 방법 2: 직접 Python 실행
```powershell
# Python이 있다면 직접 실행
python autoci learn
python autoci create platformer
python autoci fix
```

### 방법 3: py 런처 사용 (Windows Python 설치 시 포함)
```powershell
py autoci learn
py autoci create platformer
py autoci fix
```

## 🎯 최종 해결책

1. **Python 설치** (필수!)
2. **install-global-commands.bat 실행**
3. **새 터미널 열기**
4. **autoci learn 사용**

## ✅ 작동 확인

새 PowerShell/CMD를 열고:
```
autoci learn
autoci create platformer
autoci fix
```

이제 WSL과 동일하게 사용할 수 있습니다! 🎉