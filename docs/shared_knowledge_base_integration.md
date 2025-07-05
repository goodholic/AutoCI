# AutoCI 공유 지식 베이스 통합 문서

## 개요
AutoCI의 모든 구성 요소(fix, learn, create)가 지식을 공유하는 통합 시스템이 구현되었습니다.

## 주요 기능

### 1. 공유 지식 베이스 (`modules/shared_knowledge_base.py`)
- **검색 결과 캐싱**: autoci fix가 검색한 정보를 저장하고 재사용
- **솔루션 데이터베이스**: 오류 해결 방법을 저장하고 공유
- **베스트 프랙티스**: 고품질 학습 결과를 저장

### 2. 통합된 구성 요소

#### autoci fix (지능형 가디언 시스템)
- 24시간 지속적으로 정보를 검색
- 검색 결과를 공유 지식 베이스에 저장
- 1분마다 지식 베이스 통계 표시

#### autoci learn (연속 학습 시스템)
- 질문 생성 전 캐시된 검색 결과 확인
- 고품질 답변(0.8 이상)을 베스트 프랙티스로 저장
- 캐시된 정보를 컨텍스트로 활용

#### autoci create (게임 개발 파이프라인)
- 각 개발 단계 전 캐시된 지식 검색
- 오류 해결책을 공유 지식 베이스에 저장
- 다른 프로젝트의 솔루션 재활용

## 데이터 흐름

```
autoci fix (검색) → 공유 지식 베이스 → autoci learn/create (활용)
                          ↑
                    솔루션/베스트 프랙티스 저장
```

## 저장 구조

```
experiences/knowledge_base/
├── search_results/      # 검색 결과 캐시
├── solutions/          # 오류 해결 방법
└── best_practices/     # 베스트 프랙티스
```

## 통계 정보

지식 베이스는 다음 통계를 제공합니다:
- 총 검색 결과 수
- 캐시된 검색 수
- 저장된 솔루션 수
- 베스트 프랙티스 수

## 사용 예시

```python
# 공유 지식 베이스 접근
from modules.shared_knowledge_base import get_shared_knowledge_base
shared_kb = get_shared_knowledge_base()

# 캐시된 검색 결과 확인
cached = await shared_kb.get_cached_search("Godot C# player movement")

# 솔루션 저장
await shared_kb.save_solution("NullReferenceException", error_msg, solution)

# 베스트 프랙티스 저장
await shared_kb.save_best_practice("Godot Optimization", practice_data)
```

## 효과

1. **중복 검색 방지**: 한 번 검색한 정보는 재사용
2. **학습 효율 향상**: 과거 경험을 바탕으로 더 나은 답변 생성
3. **오류 해결 속도 향상**: 이전에 해결한 오류는 즉시 해결
4. **지식 축적**: 시간이 지날수록 더 많은 지식이 축적되어 성능 향상

## 유지 관리

- 30일 이상 된 데이터는 자동으로 정리됨
- 24시간 이내 검색 결과만 캐시로 사용
- 메모리 캐시는 최근 7일 데이터만 유지