# AutoCI 검색 시스템 상세 설명

## 개요
AutoCI의 `fix`와 `create` 명령어는 서로 다른 방식으로 정보를 검색하고 활용합니다.

## 1. autoci fix - 지능형 가디언 시스템의 검색

### 검색 방식
- **위치**: `modules/intelligent_guardian_system.py`의 `_perform_intelligent_search()` 메서드
- **방법**: 시뮬레이션된 검색 (실제 웹 검색 API 대신 모의 결과 생성)
- **주기**: 1분마다 자동 검색
- **검색 소스**: Godot 공식 문서, StackOverflow, GitHub, Reddit, YouTube 튜토리얼

### 검색 키워드
```python
search_keywords = [
    "Godot C# 고급 기법",
    "PyTorch 게임 개발 AI", 
    "C# Socket.IO 실시간 통신",
    "Godot 자동화 스크립팅",
    "AI 게임 개발 최신 기술",
    "C# 딥러닝 통합",
    "Godot 최적화 기법",
    "게임 AI 행동 패턴"
]
```

### 검색 프로세스
1. 캐시된 결과 확인 (공유 지식 베이스)
2. 없으면 새로운 검색 수행
3. 검색 결과를 JSON으로 저장
4. 공유 지식 베이스에 저장

## 2. autoci create - 게임 개발 파이프라인의 검색

### 검색 방식
- **위치**: `modules/intelligent_search_system.py` 클래스 사용
- **방법**: 에러 기반 검색 (에러 발생 시 솔루션 검색)
- **트리거**: 개발 중 에러 발생 시
- **검색 소스**: Documentation, StackOverflow, GitHub, Godot Forums, YouTube, Reddit, Blog

### 검색 프로세스
1. **에러 발생 시**:
   - `search_for_solution()` 호출
   - 에러 타입과 메시지 분석
   - 컨텍스트 기반 쿼리 생성

2. **쿼리 생성 예시**:
   ```python
   # ImportError 발생 시
   - "how to fix {module} import error godot"
   - "{module} not found godot engine"
   - "godot {module} module missing solution"
   ```

3. **병렬 검색**:
   - 모든 소스에서 동시에 검색
   - 상위 5개 쿼리로 100% 검색
   - 실패/검색 비율 1:9 유지

## 3. 검색 시스템 차이점

| 구분 | autoci fix | autoci create |
|------|-----------|---------------|
| 검색 시점 | 정기적 (1분마다) | 에러 발생 시 |
| 검색 대상 | 일반적인 개발 지식 | 특정 에러 솔루션 |
| 검색 방식 | 예방적 검색 | 반응적 검색 |
| 결과 활용 | 지식 축적 | 즉시 문제 해결 |

## 4. 공유 지식 베이스 통합

두 시스템 모두 `SharedKnowledgeBase`를 통해:
- 검색 결과 캐싱
- 솔루션 공유
- 베스트 프랙티스 저장

## 5. 실제 구현 vs 시뮬레이션

현재 AutoCI의 검색은 **시뮬레이션**입니다:
- 실제 웹 API 호출 없음
- 랜덤 결과 생성
- 템플릿 기반 솔루션

### 실제 구현을 위한 확장 포인트:
1. `_search_source()` 메서드에 실제 API 호출 추가
2. Google Custom Search API 통합
3. Stack Exchange API 연동
4. GitHub API 활용

## 6. 검색 효율성 전략

### autoci fix (예방적 전략)
- 자주 필요한 정보 미리 수집
- 광범위한 주제 커버
- 시간대별 다른 키워드 검색

### autoci create (대응적 전략)
- 에러별 맞춤 검색
- 높은 관련성 우선
- 즉시 적용 가능한 솔루션 중심