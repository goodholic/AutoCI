#!/usr/bin/env python3
"""
Knowledge Base System for Failed Attempts
실패한 시도들을 저장하고 검색하여 미래의 개발을 돕는 지식 베이스
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """지식 유형"""
    FAILED_ATTEMPT = "failed_attempt"
    SUCCESSFUL_SOLUTION = "successful_solution"
    BEST_PRACTICE = "best_practice"
    ANTI_PATTERN = "anti_pattern"
    WORKAROUND = "workaround"
    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    DEBUGGING = "debugging"

@dataclass
class KnowledgeEntry:
    """지식 항목"""
    id: Optional[int]
    timestamp: str
    type: KnowledgeType
    title: str
    description: str
    context: Dict[str, Any]
    problem: str
    attempted_solution: str
    outcome: str
    lessons_learned: List[str]
    tags: List[str]
    code_snippets: List[Dict[str, str]]
    related_entries: List[int]
    success_rate: float
    reusability_score: float
    search_vector: Optional[str] = None

class KnowledgeBaseSystem:
    """지식 베이스 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.kb_path = self.project_root / "knowledge_base"
        self.kb_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.kb_path / "knowledge.db"
        self._init_database()
        
        # 텍스트 검색을 위한 벡터라이저
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.search_vectors = {}
        
        # 캐시
        self.cache = {}
        self.tag_index = defaultdict(list)
        self._build_indices()
        
        # AI 모델 연동
        self.ai_model = None
        try:
            from modules.ai_model_integration import get_ai_integration
            self.ai_model = get_ai_integration()
        except ImportError:
            logger.warning("AI 모델을 로드할 수 없습니다.")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 지식 항목 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                context TEXT,
                problem TEXT NOT NULL,
                attempted_solution TEXT NOT NULL,
                outcome TEXT NOT NULL,
                lessons_learned TEXT,
                tags TEXT,
                code_snippets TEXT,
                related_entries TEXT,
                success_rate REAL DEFAULT 0.0,
                reusability_score REAL DEFAULT 0.0,
                search_vector TEXT,
                view_count INTEGER DEFAULT 0,
                usefulness_rating REAL DEFAULT 0.0
            )
        """)
        
        # 검색 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON knowledge_entries(type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags ON knowledge_entries(tags)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_title ON knowledge_entries(title)
        """)
        
        # 사용 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT,
                was_helpful BOOLEAN,
                feedback TEXT,
                FOREIGN KEY(knowledge_id) REFERENCES knowledge_entries(id)
            )
        """)
        
        # 태그 통계 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_statistics (
                tag TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_used TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _build_indices(self):
        """인덱스 구축"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 태그 인덱스 구축
        cursor.execute("SELECT id, tags FROM knowledge_entries")
        for entry_id, tags_str in cursor.fetchall():
            if tags_str:
                tags = json.loads(tags_str)
                for tag in tags:
                    self.tag_index[tag].append(entry_id)
        
        # 검색 벡터 로드
        cursor.execute("SELECT id, search_vector FROM knowledge_entries WHERE search_vector IS NOT NULL")
        for entry_id, vector_str in cursor.fetchall():
            self.search_vectors[entry_id] = json.loads(vector_str)
        
        conn.close()
    
    async def add_failed_attempt(
        self,
        title: str,
        problem: str,
        attempted_solution: str,
        outcome: str,
        context: Dict[str, Any],
        code_snippets: List[Dict[str, str]] = None,
        tags: List[str] = None
    ) -> int:
        """실패한 시도 추가"""
        # 교훈 추출
        lessons = await self._extract_lessons(problem, attempted_solution, outcome)
        
        # 유사한 항목 찾기
        similar_entries = await self.search_similar(problem + " " + attempted_solution)
        related_ids = [e['id'] for e in similar_entries[:3]]
        
        # 지식 항목 생성
        entry = KnowledgeEntry(
            id=None,
            timestamp=datetime.now().isoformat(),
            type=KnowledgeType.FAILED_ATTEMPT,
            title=title,
            description=f"Failed attempt: {title}",
            context=context,
            problem=problem,
            attempted_solution=attempted_solution,
            outcome=outcome,
            lessons_learned=lessons,
            tags=tags or [],
            code_snippets=code_snippets or [],
            related_entries=related_ids,
            success_rate=0.0,  # 실패한 시도
            reusability_score=0.5  # 실패에서도 배울 수 있음
        )
        
        # 데이터베이스에 저장
        entry_id = self._save_entry(entry)
        
        # 태그 통계 업데이트
        self._update_tag_statistics(tags, success=False)
        
        logger.info(f"✅ 실패 시도 저장: ID={entry_id}, Title={title}")
        
        return entry_id
    
    async def add_successful_solution(
        self,
        title: str,
        problem: str,
        solution: str,
        context: Dict[str, Any],
        code_snippets: List[Dict[str, str]] = None,
        tags: List[str] = None,
        reusability_score: float = 0.8
    ) -> int:
        """성공적인 해결책 추가"""
        # 교훈 추출
        lessons = await self._extract_lessons(problem, solution, "Success")
        
        # 지식 항목 생성
        entry = KnowledgeEntry(
            id=None,
            timestamp=datetime.now().isoformat(),
            type=KnowledgeType.SUCCESSFUL_SOLUTION,
            title=title,
            description=f"Successful solution: {title}",
            context=context,
            problem=problem,
            attempted_solution=solution,
            outcome="Success",
            lessons_learned=lessons,
            tags=tags or [],
            code_snippets=code_snippets or [],
            related_entries=[],
            success_rate=1.0,
            reusability_score=reusability_score
        )
        
        # 데이터베이스에 저장
        entry_id = self._save_entry(entry)
        
        # 태그 통계 업데이트
        self._update_tag_statistics(tags, success=True)
        
        logger.info(f"✅ 성공 솔루션 저장: ID={entry_id}, Title={title}")
        
        return entry_id
    
    async def _extract_lessons(self, problem: str, solution: str, outcome: str) -> List[str]:
        """교훈 추출"""
        lessons = []
        
        # AI 모델 사용
        if self.ai_model:
            prompt = f"""
            다음 개발 시도에서 배울 수 있는 교훈을 3-5개 추출해주세요:
            
            문제: {problem}
            시도한 해결책: {solution}
            결과: {outcome}
            
            교훈은 미래의 개발에 도움이 되는 구체적이고 실용적인 내용이어야 합니다.
            """
            
            try:
                response = await self.ai_model.generate_response(prompt)
                if response:
                    # 응답에서 교훈 추출
                    lines = response.strip().split('\n')
                    for line in lines:
                        if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                            lessons.append(line.strip().lstrip('-•').strip())
            except:
                pass
        
        # 기본 교훈 추출
        if not lessons:
            if "Success" in outcome:
                lessons.append(f"이 접근 방법이 {problem} 문제 해결에 효과적임")
            else:
                lessons.append(f"이 접근 방법은 {problem} 문제에 적합하지 않음")
                if "error" in outcome.lower():
                    lessons.append("오류 처리 개선 필요")
                if "performance" in outcome.lower():
                    lessons.append("성능 최적화 필요")
        
        return lessons[:5]  # 최대 5개
    
    def _save_entry(self, entry: KnowledgeEntry) -> int:
        """항목 저장"""
        # 검색 벡터 생성
        search_text = f"{entry.title} {entry.problem} {entry.attempted_solution} {' '.join(entry.tags)}"
        search_vector = self._create_search_vector(search_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO knowledge_entries (
                timestamp, type, title, description, context,
                problem, attempted_solution, outcome, lessons_learned,
                tags, code_snippets, related_entries, success_rate,
                reusability_score, search_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp,
            entry.type.value,
            entry.title,
            entry.description,
            json.dumps(entry.context),
            entry.problem,
            entry.attempted_solution,
            entry.outcome,
            json.dumps(entry.lessons_learned),
            json.dumps(entry.tags),
            json.dumps(entry.code_snippets),
            json.dumps(entry.related_entries),
            entry.success_rate,
            entry.reusability_score,
            json.dumps(search_vector)
        ))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 인덱스 업데이트
        for tag in entry.tags:
            self.tag_index[tag].append(entry_id)
        self.search_vectors[entry_id] = search_vector
        
        return entry_id
    
    def _create_search_vector(self, text: str) -> List[float]:
        """검색 벡터 생성"""
        # 간단한 TF-IDF 벡터 생성
        # 실제로는 더 정교한 방법 필요
        words = text.lower().split()
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 상위 100개 단어의 빈도를 벡터로
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100]
        vector = [freq for word, freq in top_words]
        
        # 정규화
        if vector:
            max_freq = max(vector)
            vector = [f / max_freq for f in vector]
        
        return vector
    
    def _update_tag_statistics(self, tags: List[str], success: bool):
        """태그 통계 업데이트"""
        if not tags:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tag in tags:
            cursor.execute("""
                INSERT INTO tag_statistics (tag, usage_count, success_rate, last_used)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(tag) DO UPDATE SET
                    usage_count = usage_count + 1,
                    success_rate = (success_rate * usage_count + ?) / (usage_count + 1),
                    last_used = ?
            """, (
                tag,
                1.0 if success else 0.0,
                datetime.now().isoformat(),
                1.0 if success else 0.0,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """유사한 항목 검색"""
        # 쿼리 벡터 생성
        query_vector = self._create_search_vector(query)
        
        # 모든 항목과 유사도 계산
        similarities = []
        for entry_id, entry_vector in self.search_vectors.items():
            similarity = self._calculate_similarity(query_vector, entry_vector)
            if similarity >= min_similarity:
                similarities.append((entry_id, similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 결과 반환
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entry_id, similarity in similarities[:limit]:
            cursor.execute("""
                SELECT * FROM knowledge_entries WHERE id = ?
            """, (entry_id,))
            
            row = cursor.fetchone()
            if row:
                entry = self._row_to_entry(row)
                results.append({
                    'id': entry.id,
                    'title': entry.title,
                    'type': entry.type.value,
                    'problem': entry.problem,
                    'similarity': similarity,
                    'success_rate': entry.success_rate,
                    'reusability_score': entry.reusability_score
                })
        
        conn.close()
        return results
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """벡터 유사도 계산"""
        if not vec1 or not vec2:
            return 0.0
        
        # 코사인 유사도
        dot_product = sum(a * b for a, b in zip(vec1[:len(vec2)], vec2[:len(vec1)]))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[Dict[str, Any]]:
        """태그로 검색"""
        if not tags:
            return []
        
        # 태그별 항목 ID 수집
        entry_ids_per_tag = []
        for tag in tags:
            entry_ids_per_tag.append(set(self.tag_index.get(tag, [])))
        
        # 교집합 또는 합집합
        if match_all:
            # 모든 태그를 포함하는 항목
            result_ids = entry_ids_per_tag[0]
            for ids in entry_ids_per_tag[1:]:
                result_ids &= ids
        else:
            # 하나 이상의 태그를 포함하는 항목
            result_ids = set()
            for ids in entry_ids_per_tag:
                result_ids |= ids
        
        # 항목 정보 조회
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entry_id in result_ids:
            cursor.execute("""
                SELECT id, title, type, success_rate, reusability_score
                FROM knowledge_entries WHERE id = ?
            """, (entry_id,))
            
            row = cursor.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'type': row[2],
                    'success_rate': row[3],
                    'reusability_score': row[4]
                })
        
        conn.close()
        
        # 성공률 순으로 정렬
        results.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return results
    
    async def get_entry(self, entry_id: int) -> Optional[KnowledgeEntry]:
        """항목 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM knowledge_entries WHERE id = ?
        """, (entry_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # 조회수 증가
            self._increment_view_count(entry_id)
            return self._row_to_entry(row)
        
        return None
    
    def _row_to_entry(self, row: tuple) -> KnowledgeEntry:
        """데이터베이스 행을 항목으로 변환"""
        return KnowledgeEntry(
            id=row[0],
            timestamp=row[1],
            type=KnowledgeType(row[2]),
            title=row[3],
            description=row[4],
            context=json.loads(row[5]) if row[5] else {},
            problem=row[6],
            attempted_solution=row[7],
            outcome=row[8],
            lessons_learned=json.loads(row[9]) if row[9] else [],
            tags=json.loads(row[10]) if row[10] else [],
            code_snippets=json.loads(row[11]) if row[11] else [],
            related_entries=json.loads(row[12]) if row[12] else [],
            success_rate=row[13],
            reusability_score=row[14],
            search_vector=row[15]
        )
    
    def _increment_view_count(self, entry_id: int):
        """조회수 증가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE knowledge_entries
            SET view_count = view_count + 1
            WHERE id = ?
        """, (entry_id,))
        
        conn.commit()
        conn.close()
    
    async def record_usage(
        self,
        entry_id: int,
        context: Dict[str, Any],
        was_helpful: bool,
        feedback: str = ""
    ):
        """사용 기록"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO usage_history (
                knowledge_id, timestamp, context, was_helpful, feedback
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            entry_id,
            datetime.now().isoformat(),
            json.dumps(context),
            was_helpful,
            feedback
        ))
        
        # 유용성 평가 업데이트
        cursor.execute("""
            UPDATE knowledge_entries
            SET usefulness_rating = (
                SELECT AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END)
                FROM usage_history
                WHERE knowledge_id = ?
            )
            WHERE id = ?
        """, (entry_id, entry_id))
        
        conn.commit()
        conn.close()
    
    async def get_recommendations(
        self,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """상황에 맞는 추천"""
        recommendations = []
        
        # 컨텍스트 분석
        problem_keywords = self._extract_keywords(context.get('problem', ''))
        technology = context.get('technology', 'general')
        error_type = context.get('error_type', '')
        
        # 1. 유사한 문제의 성공 사례 찾기
        if problem_keywords:
            query = ' '.join(problem_keywords)
            similar = await self.search_similar(query, limit=limit*2)
            
            # 성공률 높은 항목 우선
            successful = [s for s in similar if s['success_rate'] > 0.7]
            recommendations.extend(successful[:limit//2])
        
        # 2. 같은 기술 스택의 베스트 프랙티스
        best_practices = await self.search_by_tags([technology, "best_practice"])
        recommendations.extend(best_practices[:limit//2])
        
        # 3. 비슷한 오류의 해결책
        if error_type:
            error_solutions = await self.search_by_tags([error_type, "solution"])
            recommendations.extend(error_solutions[:limit//2])
        
        # 중복 제거 및 정렬
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['id'] not in seen:
                seen.add(rec['id'])
                unique_recommendations.append(rec)
        
        # 점수 계산 및 정렬
        for rec in unique_recommendations:
            rec['score'] = (
                rec.get('similarity', 0) * 0.4 +
                rec.get('success_rate', 0) * 0.3 +
                rec.get('reusability_score', 0) * 0.3
            )
        
        unique_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return unique_recommendations[:limit]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출
        # 실제로는 더 정교한 NLP 필요
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:10]
    
    async def generate_insights_report(self) -> Dict[str, Any]:
        """인사이트 보고서 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        cursor.execute("""
            SELECT 
                COUNT(*) as total_entries,
                AVG(success_rate) as avg_success_rate,
                AVG(reusability_score) as avg_reusability,
                SUM(view_count) as total_views
            FROM knowledge_entries
        """)
        stats = cursor.fetchone()
        
        # 가장 많이 사용된 태그
        cursor.execute("""
            SELECT tag, usage_count, success_rate
            FROM tag_statistics
            ORDER BY usage_count DESC
            LIMIT 10
        """)
        top_tags = cursor.fetchall()
        
        # 가장 유용한 항목
        cursor.execute("""
            SELECT id, title, type, usefulness_rating, view_count
            FROM knowledge_entries
            WHERE usefulness_rating > 0
            ORDER BY usefulness_rating DESC, view_count DESC
            LIMIT 10
        """)
        most_useful = cursor.fetchall()
        
        # 실패 패턴 분석
        cursor.execute("""
            SELECT outcome, COUNT(*) as count
            FROM knowledge_entries
            WHERE type = 'failed_attempt'
            GROUP BY outcome
            ORDER BY count DESC
            LIMIT 10
        """)
        failure_patterns = cursor.fetchall()
        
        conn.close()
        
        # 인사이트 생성
        insights = {
            "summary": {
                "total_knowledge_entries": stats[0] or 0,
                "average_success_rate": round(stats[1] or 0, 2),
                "average_reusability": round(stats[2] or 0, 2),
                "total_knowledge_views": stats[3] or 0
            },
            "top_technologies": [
                {"tag": tag, "usage": count, "success_rate": round(rate, 2)}
                for tag, count, rate in top_tags
            ],
            "most_useful_knowledge": [
                {
                    "id": id,
                    "title": title,
                    "type": type,
                    "usefulness": round(rating, 2),
                    "views": views
                }
                for id, title, type, rating, views in most_useful
            ],
            "common_failure_patterns": [
                {"pattern": pattern, "occurrences": count}
                for pattern, count in failure_patterns
            ],
            "recommendations": []
        }
        
        # AI 기반 추천 생성
        if self.ai_model and insights["common_failure_patterns"]:
            prompt = f"""
            다음 실패 패턴들을 분석하여 개선 방안을 제안해주세요:
            {insights["common_failure_patterns"]}
            
            각 패턴에 대해 1-2줄의 구체적인 개선 방안을 제시해주세요.
            """
            
            try:
                response = await self.ai_model.generate_response(prompt)
                if response:
                    insights["recommendations"] = response.strip().split('\n')
            except:
                pass
        
        return insights
    
    def export_knowledge(self, output_path: Path, format: str = "json"):
        """지식 내보내기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 모든 지식 항목 조회
        cursor.execute("SELECT * FROM knowledge_entries")
        columns = [description[0] for description in cursor.description]
        entries = []
        
        for row in cursor.fetchall():
            entry_dict = dict(zip(columns, row))
            # JSON 필드 파싱
            for field in ['context', 'lessons_learned', 'tags', 'code_snippets', 'related_entries']:
                if entry_dict.get(field):
                    entry_dict[field] = json.loads(entry_dict[field])
            entries.append(entry_dict)
        
        conn.close()
        
        # 내보내기
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_date": datetime.now().isoformat(),
                    "total_entries": len(entries),
                    "entries": entries
                }, f, ensure_ascii=False, indent=2)
        elif format == "markdown":
            self._export_as_markdown(entries, output_path)
        
        logger.info(f"✅ 지식 베이스 내보내기 완료: {output_path}")
    
    def _export_as_markdown(self, entries: List[Dict], output_path: Path):
        """마크다운으로 내보내기"""
        content = f"""# Knowledge Base Export

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Entries: {len(entries)}

## Table of Contents

"""
        
        # 유형별 그룹화
        by_type = defaultdict(list)
        for entry in entries:
            by_type[entry['type']].append(entry)
        
        # 목차 생성
        for type_name, type_entries in by_type.items():
            content += f"- [{type_name}](#{type_name.replace(' ', '-').lower()}) ({len(type_entries)} entries)\n"
        
        content += "\n---\n\n"
        
        # 각 유형별 내용
        for type_name, type_entries in by_type.items():
            content += f"## {type_name}\n\n"
            
            for entry in sorted(type_entries, key=lambda x: x['success_rate'], reverse=True):
                content += f"### {entry['title']}\n\n"
                content += f"**Problem:** {entry['problem']}\n\n"
                content += f"**Solution:** {entry['attempted_solution']}\n\n"
                content += f"**Outcome:** {entry['outcome']}\n\n"
                
                if entry.get('lessons_learned'):
                    content += "**Lessons Learned:**\n"
                    for lesson in entry['lessons_learned']:
                        content += f"- {lesson}\n"
                    content += "\n"
                
                if entry.get('code_snippets'):
                    content += "**Code Snippets:**\n"
                    for snippet in entry['code_snippets']:
                        content += f"```{snippet.get('language', '')}\n"
                        content += snippet.get('code', '') + "\n"
                        content += "```\n\n"
                
                content += f"**Success Rate:** {entry['success_rate']*100:.0f}%\n"
                content += f"**Reusability:** {entry['reusability_score']*100:.0f}%\n"
                
                if entry.get('tags'):
                    content += f"**Tags:** {', '.join(entry['tags'])}\n"
                
                content += "\n---\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


# 싱글톤 인스턴스
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBaseSystem:
    """지식 베이스 싱글톤 인스턴스 반환"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBaseSystem()
    return _knowledge_base


# 테스트 및 예제
async def test_knowledge_base():
    """테스트 함수"""
    kb = get_knowledge_base()
    
    # 실패 시도 추가
    failure_id = await kb.add_failed_attempt(
        title="Godot 더블 점프 구현 실패",
        problem="플레이어가 공중에서 두 번째 점프를 할 수 없음",
        attempted_solution="is_on_floor() 체크 후 점프 카운터 증가",
        outcome="점프 카운터가 제대로 리셋되지 않아 실패",
        context={"game_type": "platformer", "engine": "godot"},
        code_snippets=[{
            "language": "gdscript",
            "code": "if is_on_floor():\n    jump_count = 0\nif Input.is_action_just_pressed('jump'):\n    velocity.y = JUMP_VELOCITY"
        }],
        tags=["godot", "platformer", "jump", "physics"]
    )
    
    print(f"Added failed attempt: ID={failure_id}")
    
    # 성공 솔루션 추가
    success_id = await kb.add_successful_solution(
        title="Godot 더블 점프 성공적 구현",
        problem="플레이어가 공중에서 두 번째 점프를 할 수 없음",
        solution="점프 상태를 별도로 추적하고 _physics_process에서 관리",
        context={"game_type": "platformer", "engine": "godot"},
        code_snippets=[{
            "language": "gdscript",
            "code": """var max_jumps = 2
var current_jumps = 0

func _physics_process(delta):
    if is_on_floor():
        current_jumps = 0
    
    if Input.is_action_just_pressed('jump') and current_jumps < max_jumps:
        velocity.y = JUMP_VELOCITY
        current_jumps += 1"""
        }],
        tags=["godot", "platformer", "jump", "physics", "solution"]
    )
    
    print(f"Added successful solution: ID={success_id}")
    
    # 유사 항목 검색
    similar = await kb.search_similar("godot double jump implementation")
    print("\nSimilar entries:")
    for entry in similar:
        print(f"- {entry['title']} (similarity: {entry['similarity']:.2f})")
    
    # 추천 받기
    recommendations = await kb.get_recommendations({
        "problem": "double jump not working",
        "technology": "godot",
        "error_type": "physics"
    })
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- {rec['title']} (score: {rec['score']:.2f})")
    
    # 인사이트 보고서
    insights = await kb.generate_insights_report()
    print("\nInsights Report:")
    print(json.dumps(insights, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_knowledge_base())