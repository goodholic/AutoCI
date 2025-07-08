# Microsoft Store Python 앱 실행 별칭 비활성화 방법

Windows에서 `python` 명령어가 Microsoft Store로 리디렉션되는 문제를 해결합니다.

## 해결 방법:

1. **Windows 설정 열기**
   - Windows 키 + I

2. **앱 → 앱 및 기능 → 앱 실행 별칭**으로 이동
   - 또는 설정에서 "앱 실행 별칭" 검색

3. **Python 관련 항목 끄기**
   - "python.exe" - 끄기
   - "python3.exe" - 끄기

4. **PowerShell 재시작**

## 대체 방법 - 실제 Python 설치:

1. https://www.python.org/downloads/ 에서 Python 다운로드
2. 설치 시 "Add Python to PATH" 체크 ✅
3. 설치 완료 후 새 터미널 열기

## 임시 해결책 - py 런처 사용:

py 런처가 설치되어 있다면:
```powershell
py -m pip install -r requirements.txt
py autoci learn
py autoci create
py autoci fix
```