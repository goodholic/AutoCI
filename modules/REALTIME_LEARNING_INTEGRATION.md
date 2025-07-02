# 실시간 학습 통합 시스템

AutoCI의 모든 개발 경험을 실시간으로 AI 학습 데이터로 변환하는 시스템입니다.

## 🎯 주요 기능

### 1. 경험 수집 및 변환
- **오류 해결책**: 발생한 오류와 성공적인 해결 방법을 Q&A로 변환
- **게임 메카닉**: 구현된 게임 기능을 학습 가능한 패턴으로 저장
- **코드 패턴**: 효과적인 코드 패턴을 재사용 가능한 지식으로 변환
- **성능 최적화**: 성능 개선 방법을 측정 가능한 데이터로 저장
- **AI 발견**: AI가 찾은 창의적인 해결책을 영구 지식으로 보존

### 2. 자동 Q&A 생성
- 각 경험에서 다양한 관점의 질문-답변 쌍 자동 생성
- 난이도 자동 계산 (2-5단계)
- 한글/영어 이중 언어 지원

### 3. 지식 베이스 실시간 업데이트
- C# 패턴, 한글 번역, Godot 통합, 오류 해결책 등 즉시 반영
- 효과성 기반 모범 사례 자동 선정

### 4. 특화 학습 데이터셋 생성
- 카테고리별 전문 데이터셋 자동 구성
- 품질 점수 기반 필터링

## 📁 시스템 구조

```
modules/
├── realtime_learning_integrator.py    # 핵심 통합 엔진
├── development_experience_collector.py # 경험 수집기
├── autoci_learning_integration.py     # AutoCI 통합 래퍼
└── ai_model_controller.py             # AI 품질 제어

continuous_learning/
├── realtime_integration/
│   ├── integration_state.json         # 통합 상태
│   ├── qa_pairs/                      # 생성된 Q&A 쌍
│   ├── training_datasets/             # 특화 데이터셋
│   └── reports/                       # 통합 보고서
└── development_knowledge/             # 수집된 개발 지식
```

## 🚀 사용 방법

### 기본 사용법

```python
from modules.autoci_learning_integration import init_learning, learn_from_error

# 초기화
await init_learning(continuous_learning_system, ai_controller)

# 오류 해결 학습
error = {
    'type': 'NullReferenceException',
    'description': 'Player 노드 참조 오류'
}
solution = {
    'steps': 'null 체크 추가',
    'code': 'if (player != null) { ... }'
}
await learn_from_error(error, solution)
```

### AutoCI와 통합

```python
from modules.autoci_learning_integration import with_learning

@with_learning
async def fix_build_error(error):
    # 오류 해결 로직
    solution = await solve_error(error)
    return {'error': error, 'solution': solution}
    # 자동으로 학습됨!
```

### 직접 경험 추가

```python
from modules.realtime_learning_integrator import add_experience

# 게임 메카닉 추가
await add_experience('game_mechanic', {
    'name': '더블 점프',
    'code_snippet': '...',
    'effectiveness': 0.9
})

# 성능 최적화 추가
await add_experience('performance_opt', {
    'type': 'draw_call_reduction',
    'before': {'fps': 30},
    'after': {'fps': 60},
    'improvement': 100
})
```

## 📊 생성되는 Q&A 예시

### 오류 해결 Q&A
**Q**: "C#에서 NullReferenceException 오류가 발생할 때 어떻게 해결하나요?"

**A**: 
```
오류: Godot 노드가 null 참조 예외 발생

해결 방법:
1. 노드 경로 확인
2. GetNode() 호출 전 null 체크
3. 노드 존재 확인

코드 예제:
```csharp
var healthBar = GetNode<ProgressBar>("UI/HealthBar");
if (healthBar != null)
{
    healthBar.Value = 100;
}
```

설명: 노드를 사용하기 전에 항상 null 체크를 수행하여 안전성을 확보합니다.
```

### 게임 메카닉 Q&A
**Q**: "Godot에서 대시 시스템 기능을 구현하는 방법은?"

**A**:
```
구현 방법:

1. 개념 설명:
플레이어가 빠르게 이동하는 대시 메카닉

2. 코드 구현:
```csharp
public async void PerformDash(Vector2 direction)
{
    isDashing = true;
    player.Velocity = direction * dashSpeed;
    await ToSignal(GetTree().CreateTimer(dashDuration), "timeout");
    isDashing = false;
}
```

3. 사용 예시:
Input 이벤트에서 대시 키 감지 시 호출

4. 성능 고려사항:
FPS 영향 최소, 메모리 사용량 < 1MB
```

## 📈 통합 보고서

시스템은 자동으로 다음과 같은 보고서를 생성합니다:

- **통합 통계**: 변환된 경험 수, 생성된 Q&A 수
- **경험 인사이트**: 가장 효과적인 전략, 일반적인 패턴
- **품질 지표**: Q&A 품질 점수 분석
- **학습 효과**: 새로운 패턴 발견, 성공률 개선

## 🔧 고급 설정

### 변환 규칙 커스터마이징

```python
integrator.conversion_rules['custom_type'] = {
    'question_templates': [...],
    'answer_format': '...',
    'difficulty_calculator': lambda exp: ...
}
```

### 카테고리 매핑 수정

```python
integrator.category_mapping['new_experience'] = 'core_custom_topic'
```

## 💡 모범 사례

1. **즉시 수집**: 경험이 발생하는 즉시 수집하여 컨텍스트 보존
2. **상세 정보**: 코드, 설명, 성능 지표 등 최대한 상세히 기록
3. **효과성 측정**: 각 경험의 효과성을 수치로 기록
4. **정기 보고서**: 주기적으로 학습 보고서 확인 및 분석

## 🚨 주의사항

- 민감한 정보나 개인 정보가 포함되지 않도록 주의
- 생성된 Q&A의 품질을 주기적으로 검토
- 디스크 공간 관리 (오래된 데이터 정리)

## 📞 문제 해결

문제 발생 시 다음을 확인하세요:

1. `integration_state.json` 파일 확인
2. 로그 파일에서 오류 메시지 확인
3. 큐 크기가 너무 크지 않은지 확인 (최대 1000)
4. 메모리 사용량 모니터링

---

이 시스템을 통해 AutoCI는 모든 개발 경험에서 지속적으로 학습하며, 
시간이 지날수록 더 똑똑하고 효과적인 AI 어시스턴트가 됩니다! 🚀