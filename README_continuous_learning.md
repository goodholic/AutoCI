# AutoCI 24시간 연속 학습 시스템

## 📚 개요

AutoCI가 ChatGPT, Gemini, Claude와 같은 수준의 한국어 AI가 되기 위해 백그라운드에서 24시간 지속적으로 C# 전문 내용을 크롤링하고 학습하는 시스템입니다.

## 🚀 주요 기능

### 1. 24시간 백그라운드 학습
- **연속 크롤링**: Microsoft Docs, Unity Docs, GitHub, StackOverflow에서 C# 전문 지식 수집
- **실시간 학습**: 수집된 데이터를 실시간으로 분석하고 지식베이스 구축
- **자동 분류**: 내용을 Unity, 비동기 프로그래밍, LINQ, 성능 최적화 등으로 자동 분류
- **품질 평가**: AI가 자동으로 내용의 품질을 평가하여 고품질 지식만 저장

### 2. 지능형 크롤링
- **다중 소스**: GitHub API, Microsoft Docs, Unity 릴리스, StackExchange API
- **중복 제거**: 해시 기반 중복 컨텐츠 자동 제거
- **오류 복구**: 네트워크 오류 시 자동 재시도 및 복구
- **Rate Limiting**: API 제한을 고려한 지능형 요청 관리

### 3. 실제 학습 시뮬레이션
- **신경망 가중치**: 가상의 신경망 가중치 업데이트 시뮬레이션
- **학습 통계**: 세션 수, 학습 시간, 정확도 향상 추적
- **메모리 시스템**: 카테고리별 지식 저장 및 관리
- **학습 효율성**: 시간당 학습 속도 및 지식 증가량 측정

## 🛠️ 시스템 구성

### 고급 버전 (의존성 필요)
```bash
# 고급 기능이 포함된 연속 학습 시스템
python autoci_continuous_learning.py
```

**필요 라이브러리:**
- `aiohttp`: 비동기 HTTP 요청
- `schedule`: 작업 스케줄링
- `sqlite3`: 데이터베이스 (내장)

### 간단 버전 (의존성 없음)
```bash
# 순수 Python만 사용하는 간단한 버전
python autoci_simple_continuous_learning.py
```

**특징:**
- 외부 라이브러리 없이 순수 Python만 사용
- urllib.request로 HTTP 요청
- 간소화된 크롤링 및 학습 기능

## 💻 사용법

### 1. 백그라운드 서비스 관리

```bash
# 연속 학습 시작
python start_continuous_learning.py start

# 학습 상태 확인
python start_continuous_learning.py status

# 로그 보기
python start_continuous_learning.py logs

# 연속 학습 중지
python start_continuous_learning.py stop

# 연속 학습 재시작
python start_continuous_learning.py restart
```

### 2. 대화형 학습 모니터링

```bash
# 간단한 버전으로 대화형 모니터링
python autoci_simple_continuous_learning.py
```

**명령어:**
- `start`: 연속 학습 시작
- `stop`: 연속 학습 중지
- `status`: 현재 학습 상태 확인
- `search [쿼리]`: 학습된 지식 검색
- `quit`: 프로그램 종료

## 📊 학습 모니터링

### 실시간 상태 정보
```
🧠 세션: 42 | 📚 지식: 1,247 | 🎯 정확도: 0.876 | 🔄 활성: ✅
```

### 상세 학습 통계
```
📊 AutoCI 연속 학습 상태
========================================
🔄 실행 상태: 🟢 실행중
🆔 프로세스 ID: 12345
📄 로그 파일: logs/continuous_learning.log (2.3MB)
📈 마지막 업데이트: 2024-01-15T14:30:45
🧠 학습 세션: 156
📚 수집된 문서: 3,421
🗄️ 지식 데이터베이스: 15.7MB
```

### 지식베이스 카테고리별 현황
```
📚 총 지식량: 1,847
🎮 Unity 팁: 423
⚡ C# 패턴: 612
🚀 성능 팁: 284
📰 최신 업데이트: 528
🔄 크롤링 사이클: 78
📈 학습 효율: 15.32 지식/분
```

## 🗃️ 데이터베이스 구조

### knowledge_base 테이블
```sql
CREATE TABLE knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,           -- 출처 (github, microsoft_docs, unity_docs, stackoverflow)
    title TEXT NOT NULL,            -- 제목
    content TEXT NOT NULL,          -- 내용
    category TEXT,                  -- 카테고리 (unity, async, linq, performance, oop)
    difficulty REAL DEFAULT 0.5,   -- 난이도 (0.0-1.0)
    quality_score REAL DEFAULT 0.0,-- 품질 점수 (0.0-1.0)
    crawled_at TIMESTAMP,           -- 크롤링 시간
    updated_at TIMESTAMP,           -- 업데이트 시간
    hash TEXT UNIQUE                -- 중복 방지용 해시
);
```

### learning_sessions 테이블
```sql
CREATE TABLE learning_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_start TIMESTAMP,       -- 세션 시작 시간
    session_end TIMESTAMP,         -- 세션 종료 시간
    documents_processed INTEGER,   -- 처리된 문서 수
    knowledge_gained REAL,         -- 획득한 지식량
    model_updates INTEGER          -- 모델 업데이트 횟수
);
```

### code_patterns 테이블
```sql
CREATE TABLE code_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,         -- 코드 패턴
    usage_count INTEGER DEFAULT 1, -- 사용 횟수
    success_rate REAL DEFAULT 0.5, -- 성공률
    context TEXT,                  -- 컨텍스트
    learned_at TIMESTAMP           -- 학습 시간
);
```

## 🔧 설정 및 커스터마이징

### 크롤링 간격 조정
```python
# autoci_continuous_learning.py에서 수정
schedule.every(1).hours.do(self._run_crawling_cycle)      # 1시간마다 크롤링
schedule.every(30).minutes.do(self._run_learning_cycle)   # 30분마다 학습
schedule.every(6).hours.do(self._save_learning_progress)  # 6시간마다 진행상황 저장
```

### 크롤링 소스 추가
```python
self.sources = {
    "microsoft_docs": [...],
    "unity_docs": [...],
    "github_repos": [...],
    "stackoverflow": [...],
    # 새로운 소스 추가
    "your_custom_source": ["https://your-api-endpoint.com"]
}
```

### 품질 평가 기준 조정
```python
def _assess_quality(self, content: str) -> float:
    score = 0.5  # 기본 점수
    
    # 커스텀 품질 평가 로직
    if "best practices" in content.lower():
        score += 0.3
    
    return min(score, 1.0)
```

## 🔍 지식 검색 및 활용

### 학습된 지식 검색
```bash
AutoCI> search async await
🔍 'async await' 검색 결과:

1. [async] C# Async/Await 패턴
   async/await는 C#에서 비동기 프로그래밍을 위한 핵심 패턴입니다...
   품질: 0.85

2. [unity] Unity에서 비동기 처리
   Unity에서 async/await 사용 시 주의사항과 모범 사례...
   품질: 0.78
```

### API를 통한 지식 활용
```python
# 학습된 지식을 AutoCI 대화에 활용
results = learning_ai.query_knowledge("Unity MonoBehaviour")
for result in results:
    print(f"카테고리: {result['category']}")
    print(f"제목: {result['title']}")
    print(f"내용: {result['content']}")
```

## 📈 성능 최적화

### 메모리 관리
- 각 카테고리당 최대 100개 항목 유지
- 품질 점수 기반 자동 정리
- 정기적인 데이터베이스 최적화

### 네트워크 최적화
- 비동기 HTTP 요청으로 병렬 처리
- API Rate Limiting 자동 처리
- 실패한 요청 자동 재시도

### 데이터베이스 최적화
- 인덱스 활용으로 빠른 검색
- 중복 데이터 자동 제거
- 압축 저장으로 공간 절약

## 🚨 문제 해결

### 의존성 오류
```bash
# 필요한 라이브러리 설치
pip install aiohttp schedule

# 또는 간단한 버전 사용
python autoci_simple_continuous_learning.py
```

### 데이터베이스 오류
```bash
# 데이터베이스 재생성
rm learning_data/csharp_knowledge.db
python autoci_continuous_learning.py
```

### 네트워크 오류
- GitHub API Rate Limit: 1시간 후 자동 재시도
- 네트워크 연결 오류: 자동 재연결 시도
- 타임아웃: 요청 시간 초과 설정 조정

## 🔮 향후 계획

### 1. 실제 신경망 통합
```python
# PyTorch/TensorFlow 기반 실제 학습 모델
import torch
import torch.nn as nn

class AutoCILearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(...)
        self.classifier = nn.Linear(...)
```

### 2. 고급 자연어 처리
- 한국어 토크나이저 최적화
- 문맥 이해 개선
- 감정 분석 고도화

### 3. 클라우드 학습
- AWS/Azure 클라우드 학습 환경
- 분산 처리로 학습 속도 향상
- 실시간 모델 배포

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여

AutoCI 발전에 기여하고 싶으시다면:
1. Fork 후 브랜치 생성
2. 새로운 기능 개발 또는 버그 수정
3. Pull Request 제출

---

**💡 팁**: 처음 사용할 때는 간단한 버전(`autoci_simple_continuous_learning.py`)부터 시작하여 동작을 확인한 후, 고급 버전으로 업그레이드하는 것을 권장합니다! 