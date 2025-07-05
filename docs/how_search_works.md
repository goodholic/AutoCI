# AutoCI 검색 시스템 작동 원리

## 🔍 검색이 수행되는 위치

### 1. autoci fix - 지능형 가디언 시스템
- **파일**: `modules/intelligent_guardian_system.py`
- **메서드**: `_perform_intelligent_search(keyword)`
- **검색 주기**: 1분마다 자동
- **검색 대상**: 미리 정의된 키워드 리스트

### 2. autoci create - 게임 개발 파이프라인
- **파일**: `modules/intelligent_search_system.py`
- **메서드**: `search_for_solution(error, context)`
- **검색 시점**: 에러 발생 시
- **검색 대상**: 에러 메시지 기반 동적 쿼리

## 🔍 검색 프로세스 상세

### autoci fix의 검색 흐름
```python
# 1. 키워드 순환
search_keywords = [
    "Godot C# 고급 기법",
    "PyTorch 게임 개발 AI",
    "C# Socket.IO 실시간 통신",
    ...
]

# 2. 캐시 확인
cached_result = await shared_kb.get_cached_search(keyword)
if cached_result:
    return cached_result

# 3. 새로운 검색 수행 (시뮬레이션)
search_results = {
    "sources": {
        "Godot 공식 문서": {...},
        "StackOverflow": {...},
        "GitHub": {...}
    }
}

# 4. 결과 저장
await shared_kb.save_search_result(keyword, search_results)
```

### autoci create의 검색 흐름
```python
# 1. 에러 분석
error_type = type(error).__name__
error_message = str(error)

# 2. 쿼리 생성
queries = [
    f"godot {game_type} {error_type} solution",
    f"fix {error_message} godot engine"
]

# 3. 병렬 검색
for query in queries:
    for source in SearchSource:
        tasks.append(search_source(source, query))

# 4. 솔루션 적용
for solution in search_results:
    if apply_solution(solution):
        break
```

## 🌐 검색 소스

### 시뮬레이션된 소스 (현재)
1. Godot 공식 문서 (가중치: 90%)
2. StackOverflow (가중치: 85%)
3. GitHub (가중치: 80%)
4. Godot Forums (가중치: 75%)
5. YouTube Tutorials (가중치: 70%)
6. Reddit (가중치: 60%)
7. Blog Posts (가중치: 50%)

### 실제 구현 시 필요한 API
- Google Custom Search API
- Stack Exchange API
- GitHub Search API
- YouTube Data API
- Reddit API

## 📊 검색 결과 구조

### fix의 검색 결과
```json
{
    "keyword": "Godot C# 고급 기법",
    "timestamp": "2025-01-06T...",
    "sources": {
        "Godot 공식 문서": {
            "status": "검색 완료",
            "results_count": 15,
            "quality_score": 0.92
        }
    },
    "summary": "Godot C# 고급 기법에 대한 최신 정보 수집 완료",
    "actionable_insights": [
        "새로운 접근법 발견",
        "최적화 방법 업데이트"
    ]
}
```

### create의 검색 결과
```python
SearchResult(
    source=SearchSource.DOCUMENTATION,
    title="Godot 4 Module Import Solution",
    content="In Godot 4, use preload()...",
    relevance_score=0.87,
    solution_code="var MyScene = preload(...)",
    tags=["godot4", "import", "preload"]
)
```

## 🔄 공유 지식 베이스 통합

모든 검색 결과는 `SharedKnowledgeBase`에 저장되어:
- fix가 검색한 내용을 create가 재사용
- create가 발견한 솔루션을 learn이 학습
- 중복 검색 방지 및 효율성 향상

## 🚀 향후 개선 방향

1. **실제 API 통합**: 시뮬레이션 대신 실제 웹 검색
2. **LLM 기반 쿼리 개선**: 더 정확한 검색어 생성
3. **검색 결과 평가**: 품질 점수 자동 계산
4. **다국어 검색**: 한국어 문서 검색 지원
5. **실시간 업데이트**: 최신 정보 자동 반영