# AutoCI Godot 자동화 시스템 가이드

## 🎮 개요

AutoCI의 Godot 자동화 시스템은 화면 인식과 가상 입력을 통해 Godot 에디터를 직접 제어합니다.

## 📋 주요 구성 요소

### 1. **화면 인식 시스템**
- OpenCV 기반 템플릿 매칭
- PyTesseract OCR 텍스트 인식
- PyTorch 딥러닝 UI 요소 감지

### 2. **가상 입력 시스템**
- PyAutoGUI 마우스/키보드 제어
- Windows API 정밀 제어
- 자연스러운 베지어 곡선 마우스 움직임

### 3. **강화학습 시스템**
- PyTorch A3C 알고리즘
- 경험 재생 버퍼
- 실시간 학습 및 개선

## 🚀 빠른 시작

### 1. 필수 패키지 설치

```bash
pip install opencv-python pyautogui pillow pytesseract torch torchvision mss

# Windows 사용자
pip install pywin32
```

### 2. Tesseract OCR 설치

- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- 설치 후 PATH에 추가 또는 코드에서 경로 지정

### 3. 테스트 실행

```bash
python test_godot_automation.py
```

## 🧪 테스트 방법

### 기본 테스트

1. **화면 캡처 테스트**
   ```python
   python test_godot_automation.py
   # 메뉴에서 1번 선택
   ```

2. **가상 입력 테스트**
   ```python
   # 메뉴에서 2번 선택
   # 마우스가 자동으로 움직입니다
   ```

3. **Godot 감지 테스트**
   ```python
   # Godot을 먼저 실행
   # 메뉴에서 3번 선택
   ```

### 자동화 데모

```python
python demo_godot_automation.py
```

## 📁 모듈 구조

```
modules/
├── godot_automation_system.py      # 기본 자동화 시스템
├── advanced_godot_controller.py    # 고급 정밀 제어
├── vision_based_godot_controller.py # 비전 기반 제어
├── realtime_godot_automation.py   # 실시간 모니터링
└── templates/                      # UI 템플릿 이미지
    ├── file_menu.png
    ├── scene_panel.png
    ├── inspector.png
    └── ...
```

## 🎯 주요 기능

### 1. 노드 생성
```python
controller.create_2d_platformer_player()
```

### 2. UI 생성
```python
controller.create_ui_menu()
```

### 3. 커스텀 작업
```python
task = AutomationTask(
    task_id="custom_task",
    task_type="create_scene",
    description="커스텀 씬 생성",
    steps=[
        {"type": "create_node", "node_type": "Node2D"},
        {"type": "set_property", "property": "position", "value": "100, 100"}
    ]
)
controller.executor.add_task(task)
```

## ⚙️ 고급 설정

### 템플릿 이미지 설정

1. Godot 에디터 열기
2. 각 UI 요소 스크린샷 캡처
3. `modules/templates/` 폴더에 저장

### 화면 영역 조정

```python
# godot_automation_system.py에서 수정
self.regions = {
    "menu_bar": (0, 0, 1920, 50),      # 메뉴바
    "scene_panel": (0, 50, 400, 600),   # 씬 패널
    "viewport": (400, 50, 1520, 900),   # 뷰포트
    "inspector": (1520, 50, 400, 900),  # 인스펙터
}
```

## 🤖 강화학습 활용

시스템은 자동으로 학습하여 작업 효율성을 개선합니다:

```python
# 학습 데이터 저장
automation.save_model("godot_automation_model.pth")

# 학습된 모델 로드
automation.load_model("godot_automation_model.pth")
```

## 📊 성능 모니터링

```python
# 실시간 모니터링
monitor = RealtimeScreenMonitor(fps=10)
monitor.start_monitoring()

# 변화 감지 콜백
def on_change(event_type, data):
    print(f"변화 감지: {event_type}")
    
monitor.register_callback(on_change)
```

## ⚠️ 주의사항

1. **안전 장치**: 마우스를 화면 왼쪽 상단으로 이동하면 자동화 중단
2. **권한**: Windows에서는 관리자 권한이 필요할 수 있음
3. **해상도**: 1920x1080 기준으로 설정됨 (조정 필요 시 수정)

## 🔧 문제 해결

### OCR이 작동하지 않을 때
- Tesseract 설치 확인
- 경로 설정: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

### 템플릿 매칭 실패
- Godot 테마가 일치하는지 확인
- 템플릿 이미지 업데이트

### 가상 입력 차단
- 안티바이러스 예외 추가
- Windows Defender 설정 확인

## 📚 추가 자료

- [PyAutoGUI 문서](https://pyautogui.readthedocs.io/)
- [OpenCV 템플릿 매칭](https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html)
- [PyTorch 강화학습](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)