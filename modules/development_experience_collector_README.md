# Development Experience Collector

## 개요

Development Experience Collector는 24시간 게임 개발 과정에서 발견되는 모든 가치있는 정보와 패턴을 자동으로 수집하고 학습하는 시스템입니다.

## 주요 기능

### 1. 오류 해결책 모니터링
- 모든 오류와 성공적인 해결 방법을 자동 추적
- 실패한 시도도 학습하여 향후 개선에 활용
- 유사한 오류에 대한 검색 기능 제공

### 2. 게임 메카닉 구현 추적
- 성공적으로 구현된 게임 메카닉 수집
- 성능 메트릭과 함께 저장
- 복잡도 자동 평가

### 3. 코드 패턴 발견
- 웹 검색이나 AI를 통해 발견한 유용한 코드 패턴 저장
- 패턴 사용 빈도 추적
- 효과성 점수 계산

### 4. 성능 최적화 캡처
- 작동한 모든 성능 최적화 방법 기록
- 개선율 자동 계산
- 최적화 전후 메트릭 비교

### 5. 리소스 생성 패턴 저장
- 자동 생성된 리소스의 패턴 학습
- 성공률 추적
- 재사용 가능한 템플릿 구축

### 6. 커뮤니티 솔루션 학습
- Discord, Reddit, Forums 등에서 발견한 해결책 수집
- 검증된 솔루션 우선순위 부여
- 투표수 기반 신뢰도 평가

### 7. 실시간 지식 베이스 구축
- 모든 경험을 JSON 형식으로 영구 저장
- 카테고리별 모범 사례 자동 추출
- 학습 인사이트 생성

## 사용 방법

### 기본 사용법

```python
from modules.development_experience_collector import get_experience_collector

# 수집기 인스턴스 가져오기
collector = get_experience_collector()

# 프로젝트 모니터링 시작
await collector.start_monitoring(project_path)
```

### 기존 시스템과 통합

```python
# persistent_game_improver와 통합
await collector.integrate_with_improver(improver)

# extreme_persistence_engine과 통합
await collector.integrate_with_extreme_engine(extreme_engine)

# ai_model_controller와 통합
await collector.integrate_with_ai_controller(ai_controller)
```

### 수동으로 경험 수집

```python
# 오류 해결책 수집
await collector.collect_error_solution(error, solution, success=True)

# 게임 메카닉 수집
await collector.collect_game_mechanic("dash_system", implementation, metrics)

# 성능 최적화 수집
await collector.collect_performance_optimization(optimization_data)
```

### 지식 검색 및 활용

```python
# 유사한 문제 해결책 검색
similar_solutions = collector.search_similar_problems(current_problem)

# 모범 사례 조회
best_practices = collector.get_best_practices(category="performance")

# 학습 인사이트 얻기
insights = collector.get_learning_insights()
```

## 데이터 구조

### 저장되는 경험 타입

1. **오류 해결책**
   - 오류 타입과 설명
   - 적용된 해결 전략
   - 시도 횟수
   - 성공 여부

2. **게임 메카닉**
   - 메카닉 이름과 설명
   - 구현 코드
   - 성능 영향
   - 복잡도 점수

3. **코드 패턴**
   - 패턴 이름
   - 코드 스니펫
   - 사용 사례
   - 효과성 점수

4. **성능 최적화**
   - 최적화 타입
   - 개선 전후 메트릭
   - 적용 방법
   - 개선율

5. **리소스 패턴**
   - 리소스 타입
   - 생성 방법
   - 파라미터
   - 성공률

## 자동 학습 기능

### 패턴 인식
- 5회 이상 반복되는 해결책을 자동으로 패턴으로 인식
- 효과적인 전략을 자동으로 우선순위화

### 실시간 분석
- 5분마다 자동으로 지식 저장
- 주기적인 패턴 분석 수행
- 새로운 인사이트 발견

### 품질 평가
- 모든 수집된 경험에 대한 효과성 점수 계산
- 성공률 기반 자동 순위 매기기

## 생성되는 파일

- `continuous_learning/development_knowledge/collected_knowledge.json` - 메인 지식 베이스
- `continuous_learning/development_knowledge/learning_report_*.md` - 학습 보고서

## 통합 아키텍처

```
development_experience_collector
        │
        ├── persistent_game_improver (오류 해결 모니터링)
        ├── extreme_persistence_engine (창의적 해결책 수집)
        └── ai_model_controller (고품질 AI 응답 패턴 수집)
```

## 활용 예시

1. **반복되는 오류 빠른 해결**
   - 이전에 해결했던 유사한 오류 즉시 검색
   - 검증된 해결책 우선 적용

2. **게임 메카닉 라이브러리 구축**
   - 성공적인 메카닉들의 재사용 가능한 라이브러리
   - 성능 영향 미리 파악

3. **최적화 가이드라인**
   - 효과적인 최적화 방법 목록
   - 개선율 기반 우선순위

4. **AI 학습 데이터**
   - 수집된 경험을 AI 학습에 활용
   - 더 나은 자동화 시스템 구축