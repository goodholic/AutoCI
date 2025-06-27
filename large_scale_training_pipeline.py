#!/usr/bin/env python3
"""
Large-Scale Training Data Pipeline for Neural AutoCI
대규모 학습 데이터 파이프라인 - ChatGPT 수준의 데이터 처리
"""

import os
import sys
import time
import json
import sqlite3
import threading
import logging
import hashlib
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import random
import re

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """대규모 학습용 예제 데이터"""
    id: str
    input_text: str
    target_output: str
    context: str
    topic: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    quality_score: float
    source: str
    language: str
    created_at: str
    tokens: int
    metadata: Dict[str, Any]

@dataclass
class DatasetStatistics:
    """데이터셋 통계"""
    total_examples: int
    total_tokens: int
    avg_quality_score: float
    topic_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    avg_tokens_per_example: float

class LargeScaleDataGenerator:
    """대규모 학습 데이터 생성기"""
    
    def __init__(self, target_examples: int = 1000000):
        self.target_examples = target_examples
        self.generated_examples = 0
        self.quality_threshold = 0.7
        
        # Unity/C# 전문 지식 베이스
        self.unity_knowledge_base = {
            "basic_concepts": [
                "GameObject", "Transform", "Component", "MonoBehaviour", 
                "Unity Engine", "Scene", "Prefab", "Inspector"
            ],
            "scripting": [
                "C# Script", "Start", "Update", "Awake", "OnEnable", 
                "Coroutine", "Invoke", "Events", "Delegates"
            ],
            "physics": [
                "Rigidbody", "Collider", "Physics", "Raycast", 
                "Trigger", "Joints", "Forces", "Gravity"
            ],
            "ui": [
                "Canvas", "UI Elements", "Button", "Text", 
                "Image", "Slider", "Toggle", "Layout"
            ],
            "animation": [
                "Animator", "Animation Clip", "State Machine", 
                "Blend Trees", "Timeline", "Playable API"
            ],
            "rendering": [
                "Camera", "Light", "Material", "Shader", 
                "Texture", "Mesh", "Renderer", "Post Processing"
            ]
        }
        
        # 한국어 질문 패턴
        self.korean_question_patterns = [
            "{topic}을/를 어떻게 {action}하나요?",
            "{topic}의 {property}에 대해 설명해주세요",
            "{topic}을/를 사용할 때 주의사항이 있나요?",
            "{topic}과/와 {related_topic}의 차이점은 무엇인가요?",
            "{topic}을/를 최적화하는 방법을 알려주세요",
            "{topic}에서 {issue} 문제를 해결하는 방법은?",
            "{topic}의 생명주기에 대해 설명해주세요",
            "{topic}을/를 코드로 어떻게 구현하나요?"
        ]
        
        # 답변 템플릿 (구조화된 형태)
        self.answer_templates = {
            "explanation": [
                "{topic}는 {definition}입니다.\n\n주요 특징:\n1. {feature1}\n2. {feature2}\n3. {feature3}\n\n사용 예시:\n```csharp\n{code_example}\n```",
                "{topic}에 대해 설명드리겠습니다.\n\n{topic}는 {purpose}을 위해 사용됩니다. {detailed_explanation}\n\n기본 사용법:\n```csharp\n{code_example}\n```\n\n참고사항: {note}"
            ],
            "tutorial": [
                "{topic}을 {action}하는 방법:\n\n단계 1: {step1}\n단계 2: {step2}\n단계 3: {step3}\n\n완성된 코드:\n```csharp\n{complete_code}\n```",
                "{topic} {action} 가이드:\n\n🎯 목표: {goal}\n\n📋 준비사항:\n- {requirement1}\n- {requirement2}\n\n💻 구현:\n```csharp\n{implementation}\n```\n\n✅ 결과: {expected_result}"
            ],
            "troubleshooting": [
                "{issue} 문제 해결 방법:\n\n🔍 원인: {cause}\n\n💡 해결책:\n1. {solution1}\n2. {solution2}\n3. {solution3}\n\n📝 예방법: {prevention}",
                "{issue} 문제가 발생했을 때:\n\n일반적인 원인들:\n- {common_cause1}\n- {common_cause2}\n\n해결 코드:\n```csharp\n{fix_code}\n```\n\n추가 팁: {additional_tip}"
            ]
        }

    def generate_synthetic_examples(self, count: int) -> Generator[TrainingExample, None, None]:
        """합성 학습 데이터 생성"""
        logger.info(f"합성 데이터 {count:,}개 생성 시작")
        
        for i in range(count):
            try:
                # 주제와 카테고리 선택
                category = random.choice(list(self.unity_knowledge_base.keys()))
                topic = random.choice(self.unity_knowledge_base[category])
                
                # 질문 생성
                question_pattern = random.choice(self.korean_question_patterns)
                input_text = self._generate_question(question_pattern, topic, category)
                
                # 답변 생성
                answer_type = random.choice(list(self.answer_templates.keys()))
                target_output = self._generate_answer(answer_type, topic, category)
                
                # 메타데이터 생성
                difficulty = random.choices(
                    ["beginner", "intermediate", "advanced"],
                    weights=[0.4, 0.4, 0.2]
                )[0]
                
                quality_score = self._calculate_quality_score(input_text, target_output)
                
                if quality_score >= self.quality_threshold:
                    example = TrainingExample(
                        id=f"synthetic_{i:08d}",
                        input_text=input_text,
                        target_output=target_output,
                        context=f"Unity {category} 관련 질문",
                        topic=f"unity_{category}",
                        difficulty=difficulty,
                        quality_score=quality_score,
                        source="synthetic_generation",
                        language="korean",
                        created_at=datetime.now().isoformat(),
                        tokens=len(input_text.split()) + len(target_output.split()),
                        metadata={
                            "category": category,
                            "topic": topic,
                            "answer_type": answer_type,
                            "generated_by": "LargeScaleDataGenerator"
                        }
                    )
                    
                    self.generated_examples += 1
                    yield example
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"진행률: {i+1:,}/{count:,} ({(i+1)/count*100:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"데이터 생성 오류 (인덱스 {i}): {e}")
                continue
        
        logger.info(f"합성 데이터 생성 완료: {self.generated_examples:,}개")

    def _generate_question(self, pattern: str, topic: str, category: str) -> str:
        """질문 생성"""
        # 패턴 기반 질문 생성
        actions = {
            "basic_concepts": ["사용", "생성", "설정", "관리"],
            "scripting": ["작성", "구현", "호출", "사용"],
            "physics": ["적용", "설정", "계산", "처리"],
            "ui": ["생성", "배치", "연결", "스타일링"],
            "animation": ["제어", "생성", "재생", "편집"],
            "rendering": ["설정", "최적화", "구성", "렌더링"]
        }
        
        properties = {
            "basic_concepts": ["구조", "생명주기", "계층구조", "속성"],
            "scripting": ["메서드", "이벤트", "변수", "실행순서"],
            "physics": ["물리법칙", "충돌감지", "힘", "속도"],
            "ui": ["레이아웃", "이벤트", "스타일", "반응성"],
            "animation": ["키프레임", "곡선", "상태", "전환"],
            "rendering": ["품질", "성능", "효과", "최적화"]
        }
        
        # 관련 주제
        related_topics = self.unity_knowledge_base[category]
        related_topic = random.choice([t for t in related_topics if t != topic])
        
        # 일반적인 이슈들
        issues = ["성능저하", "오류발생", "예상과 다른 동작", "메모리 누수", "렌더링 문제"]
        
        # 패턴 채우기
        filled_pattern = pattern.format(
            topic=topic,
            action=random.choice(actions.get(category, ["사용"])),
            property=random.choice(properties.get(category, ["속성"])),
            related_topic=related_topic,
            issue=random.choice(issues)
        )
        
        return filled_pattern

    def _generate_answer(self, answer_type: str, topic: str, category: str) -> str:
        """답변 생성"""
        template = random.choice(self.answer_templates[answer_type])
        
        # 카테고리별 세부 정보
        definitions = {
            "GameObject": "Unity에서 모든 객체의 기본이 되는 클래스",
            "Transform": "객체의 위치, 회전, 크기를 관리하는 컴포넌트",
            "MonoBehaviour": "Unity 스크립트의 기본이 되는 클래스",
            "Rigidbody": "물리 시뮬레이션을 위한 컴포넌트"
        }
        
        code_examples = {
            "GameObject": "GameObject obj = new GameObject(\"MyObject\");\nobj.transform.position = Vector3.zero;",
            "Transform": "transform.position = new Vector3(0, 1, 0);\ntransform.Rotate(0, 90, 0);",
            "MonoBehaviour": "public class MyScript : MonoBehaviour\n{\n    void Start() { Debug.Log(\"Hello!\"); }\n}",
            "Rigidbody": "Rigidbody rb = GetComponent<Rigidbody>();\nrb.AddForce(Vector3.up * 10);"
        }
        
        # 템플릿 채우기
        filled_template = template.format(
            topic=topic,
            definition=definitions.get(topic, f"{topic}는 Unity의 핵심 기능 중 하나"),
            purpose=f"{category} 개발",
            detailed_explanation=f"{topic}는 게임 개발에서 중요한 역할을 합니다.",
            feature1=f"{topic}의 첫 번째 특징",
            feature2=f"{topic}의 두 번째 특징", 
            feature3=f"{topic}의 세 번째 특징",
            code_example=code_examples.get(topic, f"// {topic} 사용 예시\n// 코드 구현"),
            step1=f"{topic} 준비하기",
            step2=f"{topic} 설정하기",
            step3=f"{topic} 테스트하기",
            complete_code=code_examples.get(topic, f"// 완성된 {topic} 코드"),
            goal=f"{topic} 마스터하기",
            requirement1=f"{topic} 기본 지식",
            requirement2="Unity 에디터 사용법",
            implementation=code_examples.get(topic, f"// {topic} 구현"),
            expected_result=f"{topic}가 정상적으로 작동함",
            issue=f"{topic} 문제",
            cause=f"{topic} 설정 오류",
            solution1="설정 확인",
            solution2="코드 검토",
            solution3="Unity 재시작",
            prevention="정기적인 테스트",
            common_cause1="잘못된 설정",
            common_cause2="버전 충돌",
            fix_code=code_examples.get(topic, f"// {topic} 수정 코드"),
            additional_tip=f"{topic} 사용 시 주의사항",
            action="구현",
            note=f"{topic} 사용 시 성능을 고려하세요"
        )
        
        return filled_template

    def _calculate_quality_score(self, input_text: str, output_text: str) -> float:
        """품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 길이 기반 점수
        if len(input_text) > 10:
            score += 0.1
        if len(output_text) > 50:
            score += 0.1
        
        # 기술적 키워드 포함
        tech_keywords = ["Unity", "GameObject", "Transform", "C#", "Script", "Component"]
        if any(keyword in input_text for keyword in tech_keywords):
            score += 0.1
        if any(keyword in output_text for keyword in tech_keywords):
            score += 0.1
        
        # 코드 블록 포함
        if "```" in output_text:
            score += 0.15
        
        # 구조화된 답변 (번호, 단계)
        if any(marker in output_text for marker in ["1.", "단계", "방법:"]):
            score += 0.05
        
        return min(1.0, score)

class LargeScaleDatasetDatabase:
    """대규모 데이터셋 데이터베이스"""
    
    def __init__(self, db_path: str = "large_scale_training_dataset.db"):
        self.db_path = db_path
        self.batch_size = 10000
        self.init_database()
        
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 메인 학습 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_examples (
                    id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    target_output TEXT NOT NULL,
                    context TEXT,
                    topic TEXT,
                    difficulty TEXT,
                    quality_score REAL,
                    source TEXT,
                    language TEXT,
                    created_at TEXT,
                    tokens INTEGER,
                    metadata TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    training_set TEXT DEFAULT 'train'
                )
            ''')
            
            # 인덱스 생성
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_topic ON training_examples(topic)",
                "CREATE INDEX IF NOT EXISTS idx_difficulty ON training_examples(difficulty)", 
                "CREATE INDEX IF NOT EXISTS idx_quality ON training_examples(quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_source ON training_examples(source)",
                "CREATE INDEX IF NOT EXISTS idx_training_set ON training_examples(training_set)",
                "CREATE INDEX IF NOT EXISTS idx_processed ON training_examples(processed)"
            ]
            
            for index_sql in indices:
                cursor.execute(index_sql)
            
            # 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_examples INTEGER,
                    total_tokens INTEGER,
                    avg_quality_score REAL,
                    topic_distribution TEXT,
                    difficulty_distribution TEXT,
                    language_distribution TEXT,
                    source_distribution TEXT,
                    updated_at TEXT
                )
            ''')
            
            conn.commit()
            logger.info("✅ 대규모 데이터셋 데이터베이스 초기화 완료")

    def insert_examples_batch(self, examples: List[TrainingExample]) -> int:
        """배치 단위로 예제 삽입"""
        inserted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                batch_data = []
                for example in examples:
                    batch_data.append((
                        example.id,
                        example.input_text,
                        example.target_output,
                        example.context,
                        example.topic,
                        example.difficulty,
                        example.quality_score,
                        example.source,
                        example.language,
                        example.created_at,
                        example.tokens,
                        json.dumps(example.metadata, ensure_ascii=False),
                        False,  # processed
                        'train'  # training_set
                    ))
                
                cursor.executemany('''
                    INSERT OR REPLACE INTO training_examples 
                    (id, input_text, target_output, context, topic, difficulty, 
                     quality_score, source, language, created_at, tokens, metadata, processed, training_set)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch_data)
                
                inserted_count = cursor.rowcount
                conn.commit()
                
            except Exception as e:
                logger.error(f"배치 삽입 오류: {e}")
                conn.rollback()
        
        return inserted_count

    def get_dataset_statistics(self) -> DatasetStatistics:
        """데이터셋 통계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기본 통계
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_examples,
                    SUM(tokens) as total_tokens,
                    AVG(quality_score) as avg_quality
                FROM training_examples
            ''')
            
            basic_stats = cursor.fetchone()
            
            # 주제별 분포
            cursor.execute('''
                SELECT topic, COUNT(*) 
                FROM training_examples 
                GROUP BY topic
            ''')
            topic_dist = dict(cursor.fetchall())
            
            # 난이도별 분포
            cursor.execute('''
                SELECT difficulty, COUNT(*) 
                FROM training_examples 
                GROUP BY difficulty
            ''')
            difficulty_dist = dict(cursor.fetchall())
            
            # 언어별 분포
            cursor.execute('''
                SELECT language, COUNT(*) 
                FROM training_examples 
                GROUP BY language
            ''')
            language_dist = dict(cursor.fetchall())
            
            # 소스별 분포
            cursor.execute('''
                SELECT source, COUNT(*) 
                FROM training_examples 
                GROUP BY source
            ''')
            source_dist = dict(cursor.fetchall())
            
            return DatasetStatistics(
                total_examples=basic_stats[0] or 0,
                total_tokens=basic_stats[1] or 0,
                avg_quality_score=basic_stats[2] or 0.0,
                topic_distribution=topic_dist,
                difficulty_distribution=difficulty_dist,
                language_distribution=language_dist,
                source_distribution=source_dist,
                avg_tokens_per_example=(basic_stats[1] or 0) / max(basic_stats[0] or 1, 1)
            )

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """데이터셋 분할 (train/validation/test)"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 전체 데이터 수 조회
            cursor.execute("SELECT COUNT(*) FROM training_examples")
            total_count = cursor.fetchone()[0]
            
            if total_count == 0:
                logger.warning("분할할 데이터가 없습니다")
                return
            
            # 랜덤 순서로 데이터 조회
            cursor.execute("SELECT id FROM training_examples ORDER BY RANDOM()")
            all_ids = [row[0] for row in cursor.fetchall()]
            
            # 분할 계산
            train_count = int(total_count * train_ratio)
            val_count = int(total_count * val_ratio)
            
            train_ids = all_ids[:train_count]
            val_ids = all_ids[train_count:train_count + val_count]
            test_ids = all_ids[train_count + val_count:]
            
            # 분할 적용
            splits = [
                ('train', train_ids),
                ('validation', val_ids),
                ('test', test_ids)
            ]
            
            for split_name, ids in splits:
                if ids:
                    cursor.executemany(
                        "UPDATE training_examples SET training_set = ? WHERE id = ?",
                        [(split_name, id_) for id_ in ids]
                    )
            
            conn.commit()
            
            logger.info(f"데이터셋 분할 완료:")
            logger.info(f"  Train: {len(train_ids):,} ({len(train_ids)/total_count*100:.1f}%)")
            logger.info(f"  Validation: {len(val_ids):,} ({len(val_ids)/total_count*100:.1f}%)")
            logger.info(f"  Test: {len(test_ids):,} ({len(test_ids)/total_count*100:.1f}%)")

    def get_training_batch(self, batch_size: int = 32, split: str = 'train') -> List[TrainingExample]:
        """학습용 배치 데이터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM training_examples 
                WHERE training_set = ? AND processed = FALSE
                ORDER BY RANDOM()
                LIMIT ?
            ''', (split, batch_size))
            
            batch_examples = []
            for row in cursor.fetchall():
                example = TrainingExample(
                    id=row[0],
                    input_text=row[1],
                    target_output=row[2],
                    context=row[3],
                    topic=row[4],
                    difficulty=row[5],
                    quality_score=row[6],
                    source=row[7],
                    language=row[8],
                    created_at=row[9],
                    tokens=row[10],
                    metadata=json.loads(row[11]) if row[11] else {}
                )
                batch_examples.append(example)
            
            return batch_examples

class LargeScaleDataPipeline:
    """대규모 데이터 파이프라인 관리자"""
    
    def __init__(self, target_examples: int = 1000000):
        self.target_examples = target_examples
        self.data_generator = LargeScaleDataGenerator(target_examples)
        self.database = LargeScaleDatasetDatabase()
        self.worker_threads = 4
        self.batch_size = 10000
        
    def run_pipeline(self):
        """전체 파이프라인 실행"""
        logger.info(f"🚀 대규모 데이터 파이프라인 시작 (목표: {self.target_examples:,}개)")
        start_time = time.time()
        
        try:
            # 1. 데이터 생성
            self._generate_training_data()
            
            # 2. 데이터셋 분할
            self._split_dataset()
            
            # 3. 통계 업데이트
            self._update_statistics()
            
            # 4. 품질 검증
            self._validate_dataset()
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 파이프라인 완료 (소요시간: {elapsed_time/60:.1f}분)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return False
    
    def _generate_training_data(self):
        """학습 데이터 생성"""
        logger.info("📊 합성 학습 데이터 생성 중...")
        
        examples_per_batch = self.batch_size
        total_batches = (self.target_examples + examples_per_batch - 1) // examples_per_batch
        
        current_batch = []
        batch_count = 0
        
        for example in self.data_generator.generate_synthetic_examples(self.target_examples):
            current_batch.append(example)
            
            if len(current_batch) >= examples_per_batch:
                # 배치 저장
                inserted = self.database.insert_examples_batch(current_batch)
                batch_count += 1
                
                logger.info(f"배치 {batch_count}/{total_batches} 완료 ({inserted:,}개 삽입)")
                current_batch = []
        
        # 남은 데이터 저장
        if current_batch:
            inserted = self.database.insert_examples_batch(current_batch)
            logger.info(f"최종 배치 완료 ({inserted:,}개 삽입)")
    
    def _split_dataset(self):
        """데이터셋 분할"""
        logger.info("📂 데이터셋 train/validation/test 분할...")
        self.database.split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    def _update_statistics(self):
        """통계 업데이트"""
        logger.info("📈 데이터셋 통계 업데이트...")
        stats = self.database.get_dataset_statistics()
        
        logger.info(f"📊 데이터셋 통계:")
        logger.info(f"  총 예제 수: {stats.total_examples:,}")
        logger.info(f"  총 토큰 수: {stats.total_tokens:,}")
        logger.info(f"  평균 품질 점수: {stats.avg_quality_score:.3f}")
        logger.info(f"  평균 토큰/예제: {stats.avg_tokens_per_example:.1f}")
        
        logger.info(f"  주제 분포: {list(stats.topic_distribution.keys())[:5]}...")
        logger.info(f"  난이도 분포: {stats.difficulty_distribution}")
    
    def _validate_dataset(self):
        """데이터셋 품질 검증"""
        logger.info("🔍 데이터셋 품질 검증...")
        
        stats = self.database.get_dataset_statistics()
        
        # 품질 기준 확인
        quality_checks = []
        
        if stats.total_examples >= self.target_examples * 0.8:
            quality_checks.append("✅ 충분한 데이터 수량")
        else:
            quality_checks.append("❌ 부족한 데이터 수량")
        
        if stats.avg_quality_score >= 0.7:
            quality_checks.append("✅ 높은 평균 품질")
        else:
            quality_checks.append("❌ 낮은 평균 품질")
        
        if len(stats.topic_distribution) >= 5:
            quality_checks.append("✅ 다양한 주제 분포")
        else:
            quality_checks.append("❌ 제한적인 주제 분포")
        
        if stats.avg_tokens_per_example >= 50:
            quality_checks.append("✅ 적절한 예제 길이")
        else:
            quality_checks.append("❌ 짧은 예제 길이")
        
        for check in quality_checks:
            logger.info(f"  {check}")
        
        passed_checks = sum(1 for check in quality_checks if check.startswith("✅"))
        total_checks = len(quality_checks)
        
        logger.info(f"품질 검증 결과: {passed_checks}/{total_checks} 통과 ({passed_checks/total_checks*100:.1f}%)")
        
        return passed_checks >= total_checks * 0.75

def main():
    """메인 실행 함수"""
    print("🚀 대규모 학습 데이터 파이프라인")
    print("=" * 60)
    
    try:
        # 파이프라인 설정
        target_examples = 100000  # 테스트용으로 10만개
        pipeline = LargeScaleDataPipeline(target_examples)
        
        # 파이프라인 실행
        success = pipeline.run_pipeline()
        
        if success:
            print("🎉 대규모 데이터 파이프라인 성공적으로 완료!")
            return 0
        else:
            print("❌ 파이프라인 실행 실패")
            return 1
            
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())