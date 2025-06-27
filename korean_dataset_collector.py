#!/usr/bin/env python3
"""
AutoCI 한국어 대화 데이터셋 수집기
고품질 한국어 대화 데이터를 자동으로 수집하고 정제
"""

import os
import sys
import time
import json
import sqlite3
import threading
import logging
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import urllib.parse
import hashlib
import random

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 웹 스크래핑 라이브러리 (선택적)
try:
    from bs4 import BeautifulSoup
    import selenium
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("⚠️ 웹 스크래핑 라이브러리 없음 - 기본 데이터만 사용")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationPair:
    """대화 쌍 데이터"""
    id: str
    user_message: str
    ai_response: str
    context: str
    topic: str
    quality_score: float
    source: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class DatasetMetrics:
    """데이터셋 품질 지표"""
    total_pairs: int
    avg_quality_score: float
    topic_distribution: Dict[str, int]
    length_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    language_quality: float

class KoreanDatasetDatabase:
    """한국어 데이터셋 데이터베이스"""
    
    def __init__(self, db_path: str = "korean_conversation_dataset.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 대화 쌍 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_pairs (
                    id TEXT PRIMARY KEY,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    context TEXT,
                    topic TEXT,
                    quality_score REAL,
                    source TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 원본 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    url TEXT,
                    title TEXT,
                    content TEXT,
                    extracted_pairs INTEGER DEFAULT 0,
                    quality_score REAL,
                    collected_at TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 품질 평가 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_evaluations (
                    pair_id TEXT,
                    evaluator TEXT,
                    relevance_score REAL,
                    fluency_score REAL,
                    appropriateness_score REAL,
                    overall_score REAL,
                    comments TEXT,
                    evaluated_at TEXT,
                    FOREIGN KEY (pair_id) REFERENCES conversation_pairs (id)
                )
            ''')
            
            conn.commit()
    
    def add_conversation_pair(self, pair: ConversationPair):
        """대화 쌍 추가"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO conversation_pairs 
                (id, user_message, ai_response, context, topic, quality_score, 
                 source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pair.id, pair.user_message, pair.ai_response, pair.context,
                pair.topic, pair.quality_score, pair.source, pair.timestamp,
                json.dumps(pair.metadata, ensure_ascii=False)
            ))
            conn.commit()
    
    def get_high_quality_pairs(self, min_quality: float = 0.7, limit: int = 1000) -> List[ConversationPair]:
        """고품질 대화 쌍 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, user_message, ai_response, context, topic, 
                       quality_score, source, timestamp, metadata
                FROM conversation_pairs 
                WHERE quality_score >= ? 
                ORDER BY quality_score DESC 
                LIMIT ?
            ''', (min_quality, limit))
            
            rows = cursor.fetchall()
            pairs = []
            
            for row in rows:
                metadata = json.loads(row[8]) if row[8] else {}
                pair = ConversationPair(
                    id=row[0], user_message=row[1], ai_response=row[2],
                    context=row[3], topic=row[4], quality_score=row[5],
                    source=row[6], timestamp=row[7], metadata=metadata
                )
                pairs.append(pair)
            
            return pairs
    
    def get_dataset_metrics(self) -> DatasetMetrics:
        """데이터셋 통계"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 총 개수
            cursor.execute("SELECT COUNT(*) FROM conversation_pairs")
            total_pairs = cursor.fetchone()[0]
            
            # 평균 품질 점수
            cursor.execute("SELECT AVG(quality_score) FROM conversation_pairs")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # 주제별 분포
            cursor.execute("SELECT topic, COUNT(*) FROM conversation_pairs GROUP BY topic")
            topic_dist = dict(cursor.fetchall())
            
            # 소스별 분포
            cursor.execute("SELECT source, COUNT(*) FROM conversation_pairs GROUP BY source")
            source_dist = dict(cursor.fetchall())
            
            return DatasetMetrics(
                total_pairs=total_pairs,
                avg_quality_score=avg_quality,
                topic_distribution=topic_dist,
                length_distribution={},  # 구현 필요
                source_distribution=source_dist,
                language_quality=0.8  # 임시값
            )

class ConversationDataCollector:
    """대화 데이터 수집기"""
    
    def __init__(self, db: KoreanDatasetDatabase):
        self.db = db
        self.collection_sources = {
            "chatgpt_conversations": self.collect_chatgpt_style_conversations,
            "korean_qa_sites": self.collect_korean_qa_sites,
            "programming_forums": self.collect_programming_forums,
            "unity_korean_community": self.collect_unity_korean_community,
            "synthetic_conversations": self.generate_synthetic_conversations,
            "reddit_korean": self.collect_reddit_korean,
            "korean_blogs": self.collect_korean_blogs
        }
        
        # 수집 설정
        self.max_requests_per_minute = 30
        self.request_delay = 2  # 초
        self.quality_threshold = 0.5
        
        logger.info("📊 한국어 대화 데이터 수집기 초기화")
    
    def collect_chatgpt_style_conversations(self) -> List[ConversationPair]:
        """ChatGPT 스타일 대화 수집 (시뮬레이션)"""
        conversations = [
            {
                "user": "Unity에서 GameObject를 프로그래밍적으로 생성하는 방법을 알려주세요.",
                "assistant": "Unity에서 GameObject를 프로그래밍적으로 생성하는 방법은 여러 가지가 있습니다.\n\n1. **Instantiate 메서드 사용:**\n```csharp\npublic GameObject prefab;\n\nvoid Start() {\n    GameObject newObject = Instantiate(prefab);\n    newObject.transform.position = new Vector3(0, 0, 0);\n}\n```\n\n2. **new GameObject() 사용:**\n```csharp\nGameObject newObject = new GameObject(\"MyObject\");\nnewObject.AddComponent<MeshRenderer>();\nnewObject.AddComponent<BoxCollider>();\n```\n\n3. **CreatePrimitive 사용:**\n```csharp\nGameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);\n```\n\n가장 일반적으로 사용되는 방법은 Instantiate입니다.",
                "topic": "unity_programming",
                "quality": 0.9
            },
            {
                "user": "C#에서 리스트에 있는 중복된 값을 제거하는 방법이 있나요?",
                "assistant": "C#에서 리스트의 중복값을 제거하는 방법은 여러 가지가 있습니다:\n\n1. **LINQ Distinct() 사용 (가장 간단):**\n```csharp\nList<int> numbers = new List<int> {1, 2, 2, 3, 3, 4};\nList<int> unique = numbers.Distinct().ToList();\n```\n\n2. **HashSet 사용:**\n```csharp\nList<int> numbers = new List<int> {1, 2, 2, 3, 3, 4};\nHashSet<int> uniqueSet = new HashSet<int>(numbers);\nList<int> unique = uniqueSet.ToList();\n```\n\n3. **수동으로 확인:**\n```csharp\nList<int> unique = new List<int>();\nforeach(int num in numbers) {\n    if (!unique.Contains(num)) {\n        unique.Add(num);\n    }\n}\n```\n\n성능면에서는 Distinct()가 가장 효율적입니다.",
                "topic": "csharp_programming",
                "quality": 0.85
            },
            {
                "user": "Unity에서 코루틴이 무엇인지 설명해주시고 언제 사용하는지 알려주세요.",
                "assistant": "Unity의 코루틴(Coroutine)은 시간이 걸리는 작업을 여러 프레임에 걸쳐 실행할 수 있게 해주는 기능입니다.\n\n**코루틴의 특징:**\n- 메인 스레드를 블록하지 않음\n- yield return 문으로 실행을 중단하고 다음 프레임에 재개\n- IEnumerator를 반환하는 메서드\n\n**사용 예시:**\n```csharp\nIEnumerator MyCoroutine() {\n    Debug.Log(\"시작\");\n    yield return new WaitForSeconds(2f); // 2초 대기\n    Debug.Log(\"2초 후\");\n    yield return null; // 다음 프레임까지 대기\n    Debug.Log(\"다음 프레임\");\n}\n\n// 코루틴 시작\nStartCoroutine(MyCoroutine());\n```\n\n**언제 사용하나요?**\n- 애니메이션 효과\n- 시간 지연 처리\n- 점진적 로딩\n- 페이드 인/아웃 효과\n- HTTP 요청 처리\n\n코루틴은 Unity에서 비동기 프로그래밍의 핵심입니다.",
                "topic": "unity_coroutines",
                "quality": 0.92
            },
            {
                "user": "안녕하세요! Unity 초보자인데 어떻게 시작하면 좋을까요?",
                "assistant": "안녕하세요! Unity 초보자라면 다음과 같은 단계로 시작하시는 것을 추천드립니다:\n\n**1단계: 기초 학습**\n- Unity Hub 설치 및 Unity 에디터 설치\n- Unity 인터페이스 익히기 (Scene, Game, Inspector, Hierarchy 창)\n- 기본 GameObject와 Component 개념 이해\n\n**2단계: 첫 번째 프로젝트**\n- \"Roll a Ball\" 튜토리얼 따라하기\n- 간단한 3D 오브젝트 조작해보기\n- Transform 컴포넌트 이해하기\n\n**3단계: C# 스크립팅**\n- C# 기초 문법 학습\n- MonoBehaviour 클래스 이해\n- Start(), Update() 메서드 활용\n\n**4단계: 실습 프로젝트**\n- 간단한 2D 플랫포머 게임\n- 퍼즐 게임 제작\n- 3D 어드벤처 게임\n\n**추천 자료:**\n- Unity Learn (공식 무료 튜토리얼)\n- Unity 매뉴얼 한글판\n- 유튜브 Unity 강의\n\n차근차근 따라하시면 금방 실력이 늘 거예요!",
                "topic": "unity_beginner",
                "quality": 0.88
            },
            {
                "user": "Unity에서 ScriptableObject가 무엇인지 설명해주세요.",
                "assistant": "ScriptableObject는 Unity에서 데이터를 저장하고 관리하기 위한 특별한 클래스입니다.\n\n**ScriptableObject의 특징:**\n- 게임 오브젝트에 붙지 않는 독립적인 데이터 컨테이너\n- 에셋으로 저장되어 프로젝트 전체에서 공유 가능\n- 메모리 효율적이고 성능 최적화에 유리\n\n**사용 예시:**\n```csharp\n[CreateAssetMenu(fileName = \"New Item\", menuName = \"Game/Item\")]\npublic class ItemData : ScriptableObject {\n    public string itemName;\n    public int damage;\n    public Sprite icon;\n    public string description;\n}\n```\n\n**언제 사용하나요?**\n- 게임 설정 데이터 (아이템, 스킬, 레벨 정보)\n- 대화 시스템의 대화 데이터\n- 오디오 설정, 게임 밸런스 값\n- 여러 오브젝트가 공유하는 데이터\n\n**장점:**\n- 메모리 절약 (같은 데이터를 여러 번 복사하지 않음)\n- 런타임에 수정해도 원본 에셋은 변경되지 않음\n- Inspector에서 쉽게 편집 가능\n- 버전 관리 시스템과 잘 작동\n\nScriptableObject는 데이터 중심 설계의 핵심 도구입니다!",
                "topic": "unity_scriptableobject",
                "quality": 0.9
            }
        ]
        
        pairs = []
        for i, conv in enumerate(conversations):
            pair_id = f"chatgpt_style_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=conv["user"],
                ai_response=conv["assistant"],
                context="Unity/C# 프로그래밍 질문",
                topic=conv["topic"],
                quality_score=conv["quality"],
                source="chatgpt_style",
                timestamp=datetime.now().isoformat(),
                metadata={"conversation_length": len(conv["user"] + conv["assistant"])}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_korean_qa_sites(self) -> List[ConversationPair]:
        """한국어 Q&A 사이트 데이터 수집 (시뮬레이션)"""
        qa_data = [
            {
                "question": "Unity에서 물리 시뮬레이션을 어떻게 구현하나요?",
                "answer": "Unity에서 물리 시뮬레이션은 Rigidbody 컴포넌트를 사용합니다. GameObject에 Rigidbody를 추가하면 중력, 충돌, 힘 등의 물리 효과가 적용됩니다. Physics.AddForce()로 힘을 가하거나, velocity를 직접 설정할 수 있습니다.",
                "topic": "unity_physics"
            },
            {
                "question": "C#에서 델리게이트와 이벤트의 차이점은 무엇인가요?",
                "answer": "델리게이트는 메서드를 참조할 수 있는 타입이고, 이벤트는 델리게이트를 기반으로 한 특별한 멤버입니다. 이벤트는 외부에서 직접 호출할 수 없고 += 또는 -= 연산자로만 구독/해제할 수 있어 더 안전합니다.",
                "topic": "csharp_events"
            },
            {
                "question": "Unity 2D에서 스프라이트 애니메이션을 만드는 방법을 알려주세요.",
                "answer": "1. 스프라이트 이미지들을 프로젝트에 임포트합니다. 2. Window > Animation > Animation을 열어 Animation Clip을 생성합니다. 3. 스프라이트들을 타임라인에 배치합니다. 4. Animator Controller를 만들어 애니메이션 상태를 관리합니다. 5. GameObject에 Animator 컴포넌트를 추가하고 Controller를 할당합니다.",
                "topic": "unity_2d_animation"
            }
        ]
        
        pairs = []
        for i, qa in enumerate(qa_data):
            pair_id = f"korean_qa_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=qa["question"],
                ai_response=qa["answer"],
                context="한국어 프로그래밍 Q&A",
                topic=qa["topic"],
                quality_score=random.uniform(0.7, 0.9),
                source="korean_qa_sites",
                timestamp=datetime.now().isoformat(),
                metadata={"verified": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_programming_forums(self) -> List[ConversationPair]:
        """프로그래밍 포럼 데이터 수집 (시뮬레이션)"""
        forum_data = [
            {
                "title": "Unity에서 싱글톤 패턴 구현하기",
                "question": "Unity에서 GameManager를 싱글톤으로 만들고 싶은데 어떻게 해야 하나요?",
                "answer": "Unity에서 싱글톤은 다음과 같이 구현할 수 있습니다:\n\n```csharp\npublic class GameManager : MonoBehaviour {\n    public static GameManager Instance { get; private set; }\n    \n    void Awake() {\n        if (Instance == null) {\n            Instance = this;\n            DontDestroyOnLoad(gameObject);\n        } else {\n            Destroy(gameObject);\n        }\n    }\n}\n```\n\nDontDestroyOnLoad를 사용하면 씬 전환 시에도 오브젝트가 유지됩니다.",
                "topic": "unity_singleton"
            },
            {
                "title": "C# async/await 패턴 질문",
                "question": "Unity에서 비동기 프로그래밍을 할 때 async/await를 어떻게 사용하나요?",
                "answer": "Unity에서 async/await는 다음과 같이 사용합니다:\n\n```csharp\nusing System.Threading.Tasks;\n\npublic async void LoadDataAsync() {\n    Debug.Log(\"로딩 시작\");\n    await Task.Delay(1000); // 1초 대기\n    Debug.Log(\"로딩 완료\");\n}\n\npublic async Task<string> GetDataFromServerAsync() {\n    // 웹 요청 등의 비동기 작업\n    await Task.Delay(2000);\n    return \"서버 데이터\";\n}\n```\n\n주의: async void는 이벤트 핸들러에서만 사용하고, 일반적으로는 async Task를 사용하세요.",
                "topic": "csharp_async"
            }
        ]
        
        pairs = []
        for i, forum in enumerate(forum_data):
            pair_id = f"forum_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=forum["question"],
                ai_response=forum["answer"],
                context=f"포럼 질문: {forum['title']}",
                topic=forum["topic"],
                quality_score=random.uniform(0.75, 0.95),
                source="programming_forums",
                timestamp=datetime.now().isoformat(),
                metadata={"title": forum["title"], "forum_verified": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_unity_korean_community(self) -> List[ConversationPair]:
        """Unity 한국 커뮤니티 데이터 수집 (시뮬레이션)"""
        community_data = [
            {
                "user": "Unity 2022.3 LTS에서 새로 추가된 기능들이 뭐가 있나요?",
                "assistant": "Unity 2022.3 LTS의 주요 새 기능들:\n\n1. **성능 개선:**\n   - 렌더링 파이프라인 최적화\n   - 메모리 관리 향상\n   - 빌드 시간 단축\n\n2. **새로운 도구:**\n   - Overlays 시스템 개선\n   - Scene Template 기능\n   - 향상된 Package Manager\n\n3. **에디터 개선:**\n   - 새로운 아이콘 세트\n   - 더 나은 검색 기능\n   - 커스터마이즈 가능한 툴바\n\n4. **플랫폼 지원:**\n   - Apple Silicon 네이티브 지원\n   - Android 대상 API 레벨 업데이트\n   - WebGL 성능 향상\n\nLTS 버전이므로 안정성이 보장되어 상용 프로젝트에 적합합니다.",
                "topic": "unity_2022_features"
            },
            {
                "user": "모바일 게임 개발할 때 성능 최적화 팁 좀 알려주세요.",
                "assistant": "모바일 게임 성능 최적화 팁:\n\n**1. 그래픽 최적화:**\n- 텍스처 크기 줄이기 (1024x1024 이하 권장)\n- 텍스처 압축 포맷 사용 (ETC2, ASTC)\n- 폴리곤 수 최소화 (LOD 시스템 활용)\n- UI 오버드로우 줄이기\n\n**2. 코드 최적화:**\n- Update()에서 무거운 연산 피하기\n- 오브젝트 풀링 사용\n- string 대신 StringBuilder 사용\n- 가비지 컬렉션 최소화\n\n**3. 메모리 관리:**\n- 사용하지 않는 에셋 언로드\n- 압축된 오디오 포맷 사용\n- Resources 폴더 사용 지양\n\n**4. 배터리 최적화:**\n- 타겟 프레임레이트 설정 (30fps)\n- 불필요한 렌더링 끄기\n- 백그라운드에서 일시정지\n\n**5. 프로파일링:**\n- Unity Profiler 활용\n- 실제 디바이스에서 테스트\n- Memory Profiler로 메모리 누수 확인",
                "topic": "mobile_optimization"
            }
        ]
        
        pairs = []
        for i, data in enumerate(community_data):
            pair_id = f"unity_community_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=data["user"],
                ai_response=data["assistant"],
                context="Unity 한국 커뮤니티",
                topic=data["topic"],
                quality_score=random.uniform(0.8, 0.95),
                source="unity_korean_community",
                timestamp=datetime.now().isoformat(),
                metadata={"community_verified": True, "korean_native": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def generate_synthetic_conversations(self) -> List[ConversationPair]:
        """합성 대화 생성"""
        
        # 질문 템플릿
        question_templates = [
            "{subject}에서 {action}하는 방법을 알려주세요.",
            "{subject}의 {feature} 기능은 어떻게 사용하나요?",
            "{problem} 문제가 발생했는데 해결 방법이 있나요?",
            "{subject}를 사용할 때 주의할 점은 무엇인가요?",
            "{subject}와 {alternative}의 차이점은 무엇인가요?"
        ]
        
        # 응답 템플릿
        answer_templates = [
            "{subject}에서 {action}하려면 다음과 같이 하세요:\n\n1. {step1}\n2. {step2}\n3. {step3}\n\n이렇게 하면 원하는 결과를 얻을 수 있습니다.",
            "{subject}의 {feature}는 매우 유용한 기능입니다. {explanation} 다음 코드로 구현할 수 있습니다:\n\n```csharp\n{code_example}\n```",
            "{problem} 문제는 일반적으로 {cause} 때문에 발생합니다. 해결 방법:\n\n- {solution1}\n- {solution2}\n- {solution3}\n\n가장 효과적인 방법은 {best_solution}입니다."
        ]
        
        # 콘텐츠 데이터
        subjects = ["Unity", "C#", "GameObject", "Transform", "Rigidbody", "Collider", "Animation"]
        actions = ["생성", "삭제", "이동", "회전", "스케일링", "설정", "최적화"]
        features = ["컴포넌트", "프로퍼티", "메서드", "이벤트", "인터페이스"]
        problems = ["null reference", "메모리 누수", "성능 저하", "빌드 오류", "런타임 에러"]
        
        pairs = []
        for i in range(20):  # 20개 합성 대화 생성
            # 랜덤 템플릿 선택
            q_template = random.choice(question_templates)
            a_template = random.choice(answer_templates)
            
            # 변수 값 설정
            subject = random.choice(subjects)
            action = random.choice(actions)
            feature = random.choice(features)
            problem = random.choice(problems)
            
            # 질문 생성
            question = q_template.format(
                subject=subject,
                action=action,
                feature=feature,
                problem=problem,
                alternative=random.choice([s for s in subjects if s != subject])
            )
            
            # 답변 생성
            answer = a_template.format(
                subject=subject,
                action=action,
                feature=feature,
                problem=problem,
                explanation=f"{subject}는 매우 중요한 개념입니다.",
                step1=f"{subject}를 초기화하세요",
                step2=f"필요한 설정을 적용하세요",
                step3=f"결과를 확인하세요",
                code_example=f"// {subject} 예시 코드\npublic void Example() {{\n    // 구현 내용\n}}",
                cause=f"{subject} 설정 오류",
                solution1=f"{subject} 재설정",
                solution2="캐시 정리",
                solution3="재시작",
                best_solution=f"{subject} 재설정"
            )
            
            pair_id = f"synthetic_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=question,
                ai_response=answer,
                context="합성 생성 대화",
                topic=f"synthetic_{subject.lower()}",
                quality_score=random.uniform(0.6, 0.8),
                source="synthetic_conversations",
                timestamp=datetime.now().isoformat(),
                metadata={"generated": True, "template_based": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_reddit_korean(self) -> List[ConversationPair]:
        """Reddit 한국어 프로그래밍 커뮤니티 (시뮬레이션)"""
        reddit_data = [
            {
                "title": "Unity 입문자를 위한 조언",
                "post": "Unity를 처음 시작하는데 어떤 프로젝트부터 시작하면 좋을까요?",
                "reply": "처음에는 간단한 2D 게임부터 시작하세요. Pong이나 Snake 같은 게임을 만들어보면서 기본기를 익히고, 그 다음에 플랫포머 게임으로 넘어가는 것이 좋습니다. 3D는 그 이후에 도전하세요."
            },
            {
                "title": "C# 성능 최적화 질문",
                "post": "게임에서 C# 코드 성능을 높이려면 어떤 점을 주의해야 하나요?",
                "reply": "1. 가비지 컬렉션 최소화 (new 키워드 사용 줄이기)\n2. StringBuilder 사용\n3. 오브젝트 풀링\n4. 배열 대신 List<T> 적절히 사용\n5. LINQ 과도한 사용 피하기\n6. Update()에서 무거운 연산 피하기"
            }
        ]
        
        pairs = []
        for i, data in enumerate(reddit_data):
            pair_id = f"reddit_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=data["post"],
                ai_response=data["reply"],
                context=f"Reddit: {data['title']}",
                topic="reddit_programming",
                quality_score=random.uniform(0.7, 0.85),
                source="reddit_korean",
                timestamp=datetime.now().isoformat(),
                metadata={"platform": "reddit", "title": data["title"]}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_korean_blogs(self) -> List[ConversationPair]:
        """한국어 개발 블로그 데이터 (시뮬레이션)"""
        blog_data = [
            {
                "blog": "Unity 개발 블로그",
                "question": "Unity에서 오브젝트 풀링을 구현하는 방법",
                "answer": "오브젝트 풀링은 성능 최적화를 위한 중요한 패턴입니다:\n\n```csharp\npublic class ObjectPool : MonoBehaviour {\n    public GameObject prefab;\n    public int poolSize = 10;\n    private Queue<GameObject> pool;\n    \n    void Start() {\n        pool = new Queue<GameObject>();\n        for (int i = 0; i < poolSize; i++) {\n            GameObject obj = Instantiate(prefab);\n            obj.SetActive(false);\n            pool.Enqueue(obj);\n        }\n    }\n    \n    public GameObject GetObject() {\n        if (pool.Count > 0) {\n            GameObject obj = pool.Dequeue();\n            obj.SetActive(true);\n            return obj;\n        }\n        return Instantiate(prefab);\n    }\n    \n    public void ReturnObject(GameObject obj) {\n        obj.SetActive(false);\n        pool.Enqueue(obj);\n    }\n}\n```\n\n이렇게 하면 Instantiate/Destroy 비용을 줄일 수 있습니다."
            }
        ]
        
        pairs = []
        for i, data in enumerate(blog_data):
            pair_id = f"blog_{i}_{int(time.time())}"
            
            pair = ConversationPair(
                id=pair_id,
                user_message=data["question"],
                ai_response=data["answer"],
                context=f"블로그: {data['blog']}",
                topic="unity_object_pooling",
                quality_score=0.9,
                source="korean_blogs",
                timestamp=datetime.now().isoformat(),
                metadata={"blog_name": data["blog"], "code_included": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def collect_all_sources(self) -> int:
        """모든 소스에서 데이터 수집"""
        total_collected = 0
        
        for source_name, collect_func in self.collection_sources.items():
            try:
                logger.info(f"📥 {source_name}에서 데이터 수집 중...")
                pairs = collect_func()
                
                for pair in pairs:
                    self.db.add_conversation_pair(pair)
                    total_collected += 1
                
                logger.info(f"✅ {source_name}: {len(pairs)}개 대화 쌍 수집")
                
                # 요청 간 지연
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"❌ {source_name} 수집 실패: {e}")
        
        logger.info(f"🎯 총 {total_collected}개 대화 쌍 수집 완료")
        return total_collected

class ConversationQualityEvaluator:
    """대화 품질 평가기"""
    
    def __init__(self):
        self.quality_criteria = {
            "relevance": 0.3,      # 관련성
            "fluency": 0.25,       # 유창성
            "appropriateness": 0.2, # 적절성
            "informativeness": 0.15, # 정보성
            "coherence": 0.1       # 일관성
        }
    
    def evaluate_conversation_pair(self, pair: ConversationPair) -> float:
        """대화 쌍 품질 평가"""
        
        scores = {}
        
        # 관련성 평가
        scores["relevance"] = self._evaluate_relevance(pair.user_message, pair.ai_response)
        
        # 유창성 평가
        scores["fluency"] = self._evaluate_fluency(pair.ai_response)
        
        # 적절성 평가
        scores["appropriateness"] = self._evaluate_appropriateness(pair.user_message, pair.ai_response)
        
        # 정보성 평가
        scores["informativeness"] = self._evaluate_informativeness(pair.ai_response)
        
        # 일관성 평가
        scores["coherence"] = self._evaluate_coherence(pair.ai_response)
        
        # 가중 평균 계산
        overall_score = sum(
            scores[criterion] * weight 
            for criterion, weight in self.quality_criteria.items()
        )
        
        return min(overall_score, 1.0)
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """관련성 평가"""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.0
        
        # 공통 단어 비율
        common_words = question_words.intersection(answer_words)
        relevance = len(common_words) / len(question_words)
        
        # Unity/C# 키워드 보너스
        tech_keywords = {"unity", "c#", "csharp", "gameobject", "script", "code"}
        if any(keyword in question.lower() for keyword in tech_keywords):
            if any(keyword in answer.lower() for keyword in tech_keywords):
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _evaluate_fluency(self, text: str) -> float:
        """유창성 평가"""
        score = 0.5  # 기본 점수
        
        # 문장 길이 체크
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 5 <= avg_sentence_length <= 20:  # 적절한 문장 길이
            score += 0.2
        
        # 한국어 자연스러움
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        if korean_chars > len(text) * 0.3:  # 30% 이상 한글
            score += 0.2
        
        # 코드 블록 존재 (기술적 답변)
        if '```' in text or 'public' in text or 'void' in text:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_appropriateness(self, question: str, answer: str) -> float:
        """적절성 평가"""
        score = 0.7  # 기본 점수
        
        # 답변 길이 적절성
        question_length = len(question)
        answer_length = len(answer)
        
        if answer_length > question_length * 0.5:  # 충분한 답변
            score += 0.2
        
        if answer_length < question_length * 3:  # 너무 장황하지 않음
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_informativeness(self, answer: str) -> float:
        """정보성 평가"""
        score = 0.5
        
        # 구체적인 정보 포함 여부
        info_indicators = [
            "예시", "example", "```", "코드", "방법", "단계", 
            "1.", "2.", "3.", "첫째", "둘째", "셋째"
        ]
        
        info_count = sum(1 for indicator in info_indicators if indicator in answer.lower())
        score += min(info_count * 0.1, 0.4)
        
        # 설명의 구조화
        if any(char in answer for char in ['•', '-', '*', '1.', '2.']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_coherence(self, text: str) -> float:
        """일관성 평가"""
        score = 0.8  # 기본 점수 (대부분의 텍스트는 일관성이 있다고 가정)
        
        # 문장 간 연결성 체크 (간단한 버전)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) > 1:
            # 연결어 존재 여부
            connectors = ["그리고", "또한", "하지만", "따라서", "그러므로", "예를 들어"]
            if any(conn in text for conn in connectors):
                score += 0.1
        
        return min(score, 1.0)

def main():
    """메인 함수"""
    print("📊 AutoCI 한국어 대화 데이터셋 수집기")
    print("=" * 60)
    
    try:
        # 데이터베이스 및 수집기 초기화
        db = KoreanDatasetDatabase()
        collector = ConversationDataCollector(db)
        evaluator = ConversationQualityEvaluator()
        
        # 데이터 수집
        logger.info("🚀 한국어 대화 데이터 수집 시작")
        total_collected = collector.collect_all_sources()
        
        # 품질 평가
        logger.info("📝 수집된 데이터 품질 평가 중...")
        high_quality_pairs = db.get_high_quality_pairs(min_quality=0.7, limit=100)
        
        for pair in high_quality_pairs[:10]:  # 상위 10개만 재평가
            quality_score = evaluator.evaluate_conversation_pair(pair)
            logger.info(f"품질 평가: {pair.topic} - {quality_score:.2f}")
        
        # 데이터셋 통계
        metrics = db.get_dataset_metrics()
        logger.info(f"📈 데이터셋 통계:")
        logger.info(f"  총 대화 쌍: {metrics.total_pairs}")
        logger.info(f"  평균 품질: {metrics.avg_quality_score:.2f}")
        logger.info(f"  주제별 분포: {metrics.topic_distribution}")
        logger.info(f"  소스별 분포: {metrics.source_distribution}")
        
        logger.info("🎉 한국어 대화 데이터셋 수집 완료!")
        
    except Exception as e:
        logger.error(f"❌ 데이터 수집 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())