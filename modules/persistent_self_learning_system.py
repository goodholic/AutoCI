#!/usr/bin/env python3
"""
영구 자가 학습 시스템

24시간 게임 개발 과정에서 학습한 모든 데이터를 저장하고
지속적으로 진화하는 자가 학습 시스템
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import sqlite3


@dataclass
class LearningEntry:
    """학습 엔트리"""
    id: str
    timestamp: str
    category: str  # error_fix, feature_add, optimization, user_feedback 등
    context: Dict[str, Any]
    solution: Dict[str, Any]
    outcome: Dict[str, Any]
    quality_score: float
    confidence: float
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class Pattern:
    """학습된 패턴"""
    id: str
    pattern_type: str
    occurrences: int
    success_rate: float
    context_similarity: float
    solutions: List[str]
    first_seen: str
    last_seen: str
    evolution_history: List[Dict[str, Any]]


class PersistentSelfLearningSystem:
    """영구 자가 학습 시스템"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/mnt/d/AutoCI/AutoCI/learning_data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 경로
        self.db_path = self.base_path / "learning_database.db"
        self.knowledge_base_path = self.base_path / "knowledge_base.json"
        self.patterns_path = self.base_path / "patterns.json"
        self.insights_path = self.base_path / "insights.json"
        
        # 메모리 캐시
        self.knowledge_base: Dict[str, Any] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.insights: List[Dict[str, Any]] = []
        self.learning_queue: List[LearningEntry] = []
        
        # 통계
        self.stats = {
            "total_entries": 0,
            "successful_solutions": 0,
            "failed_attempts": 0,
            "patterns_discovered": 0,
            "insights_generated": 0,
            "quality_improvements": 0,
            "learning_cycles": 0
        }
        
        # 학습 설정
        self.config = {
            "min_pattern_occurrences": 3,
            "quality_threshold": 0.7,
            "confidence_threshold": 0.8,
            "evolution_interval": 100,  # 100개 엔트리마다 진화
            "insight_generation_interval": 50
        }
        
        # 초기화
        self._initialize_database()
        self._load_existing_data()
        
        # 백그라운드 학습 스레드
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._background_learning)
        self.learning_thread.daemon = True
        self.learning_thread.start()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 학습 엔트리 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                category TEXT,
                context TEXT,
                solution TEXT,
                outcome TEXT,
                quality_score REAL,
                confidence REAL,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # 패턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                occurrences INTEGER,
                success_rate REAL,
                context_similarity REAL,
                solutions TEXT,
                first_seen TEXT,
                last_seen TEXT,
                evolution_history TEXT
            )
        """)
        
        # 인사이트 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                type TEXT,
                title TEXT,
                description TEXT,
                impact_score REAL,
                related_patterns TEXT,
                metadata TEXT
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON learning_entries(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON learning_entries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON learning_entries(quality_score)")
        
        conn.commit()
        conn.close()
    
    def _load_existing_data(self):
        """기존 데이터 로드"""
        # 지식 베이스 로드
        if self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        
        # 패턴 로드
        if self.patterns_path.exists():
            with open(self.patterns_path, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
                self.patterns = {
                    pid: Pattern(**pdata) for pid, pdata in patterns_data.items()
                }
        
        # 인사이트 로드
        if self.insights_path.exists():
            with open(self.insights_path, 'r', encoding='utf-8') as f:
                self.insights = json.load(f)
        
        # 통계 업데이트
        self._update_stats_from_db()
    
    def add_learning_entry(
        self,
        category: str,
        context: Dict[str, Any],
        solution: Dict[str, Any],
        outcome: Dict[str, Any],
        quality_score: float = 0.5,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """학습 엔트리 추가"""
        # 엔트리 생성
        entry_id = self._generate_id(category, context, solution)
        entry = LearningEntry(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            category=category,
            context=context,
            solution=solution,
            outcome=outcome,
            quality_score=quality_score,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 큐에 추가
        self.learning_queue.append(entry)
        
        # 즉시 처리가 필요한 경우
        if category in ["critical_error", "user_feedback"]:
            self._process_entry_immediate(entry)
        
        return entry_id
    
    def _process_entry_immediate(self, entry: LearningEntry):
        """엔트리 즉시 처리"""
        # 데이터베이스에 저장
        self._save_entry_to_db(entry)
        
        # 패턴 분석
        patterns = self._analyze_patterns(entry)
        
        # 지식 베이스 업데이트
        self._update_knowledge_base(entry, patterns)
        
        # 품질이 높은 경우 인사이트 생성 시도
        if entry.quality_score >= self.config["quality_threshold"]:
            self._try_generate_insight(entry, patterns)
    
    def _save_entry_to_db(self, entry: LearningEntry):
        """데이터베이스에 엔트리 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO learning_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.timestamp,
            entry.category,
            json.dumps(entry.context),
            json.dumps(entry.solution),
            json.dumps(entry.outcome),
            entry.quality_score,
            entry.confidence,
            json.dumps(entry.tags),
            json.dumps(entry.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        self.stats["total_entries"] += 1
        if entry.outcome.get("success", False):
            self.stats["successful_solutions"] += 1
        else:
            self.stats["failed_attempts"] += 1
    
    def _analyze_patterns(self, entry: LearningEntry) -> List[Pattern]:
        """패턴 분석"""
        discovered_patterns = []
        
        # 컨텍스트 기반 패턴 찾기
        context_hash = self._hash_context(entry.context)
        similar_entries = self._find_similar_entries(entry)
        
        if len(similar_entries) >= self.config["min_pattern_occurrences"]:
            # 패턴 생성 또는 업데이트
            pattern_id = f"pattern_{context_hash}"
            
            if pattern_id in self.patterns:
                # 기존 패턴 업데이트
                pattern = self.patterns[pattern_id]
                pattern.occurrences += 1
                pattern.last_seen = entry.timestamp
                
                # 성공률 재계산
                success_count = sum(1 for e in similar_entries if e.outcome.get("success", False))
                pattern.success_rate = success_count / len(similar_entries)
                
                # 솔루션 추가
                if entry.id not in pattern.solutions:
                    pattern.solutions.append(entry.id)
                
                # 진화 히스토리 추가
                pattern.evolution_history.append({
                    "timestamp": entry.timestamp,
                    "change": "occurrence_added",
                    "quality_score": entry.quality_score
                })
            else:
                # 새 패턴 생성
                pattern = Pattern(
                    id=pattern_id,
                    pattern_type=entry.category,
                    occurrences=len(similar_entries),
                    success_rate=sum(1 for e in similar_entries if e.outcome.get("success", False)) / len(similar_entries),
                    context_similarity=self._calculate_similarity(similar_entries),
                    solutions=[e.id for e in similar_entries],
                    first_seen=min(e.timestamp for e in similar_entries),
                    last_seen=entry.timestamp,
                    evolution_history=[{
                        "timestamp": entry.timestamp,
                        "change": "pattern_discovered",
                        "initial_occurrences": len(similar_entries)
                    }]
                )
                
                self.patterns[pattern_id] = pattern
                self.stats["patterns_discovered"] += 1
                discovered_patterns.append(pattern)
        
        return discovered_patterns
    
    def _update_knowledge_base(self, entry: LearningEntry, patterns: List[Pattern]):
        """지식 베이스 업데이트"""
        # 카테고리별 지식 구조화
        if entry.category not in self.knowledge_base:
            self.knowledge_base[entry.category] = {
                "solutions": {},
                "patterns": [],
                "best_practices": [],
                "common_errors": [],
                "optimization_tips": []
            }
        
        category_kb = self.knowledge_base[entry.category]
        
        # 솔루션 추가
        solution_key = self._generate_solution_key(entry.context)
        if solution_key not in category_kb["solutions"]:
            category_kb["solutions"][solution_key] = []
        
        category_kb["solutions"][solution_key].append({
            "id": entry.id,
            "solution": entry.solution,
            "quality_score": entry.quality_score,
            "outcome": entry.outcome
        })
        
        # 고품질 솔루션을 베스트 프랙티스로 추가
        if entry.quality_score >= 0.9 and entry.outcome.get("success", False):
            category_kb["best_practices"].append({
                "context": entry.context,
                "solution": entry.solution,
                "reason": entry.outcome.get("reason", "High quality solution")
            })
        
        # 일반적인 오류 추적
        if not entry.outcome.get("success", False):
            error_type = entry.outcome.get("error_type", "unknown")
            category_kb["common_errors"].append({
                "error_type": error_type,
                "context": entry.context,
                "attempted_solution": entry.solution,
                "lesson_learned": entry.outcome.get("lesson", "")
            })
        
        # 최적화 팁 추가
        if "optimization" in entry.tags:
            category_kb["optimization_tips"].append({
                "tip": entry.solution.get("optimization", ""),
                "impact": entry.outcome.get("performance_improvement", 0),
                "context": entry.context
            })
        
        # 패턴 참조 추가
        for pattern in patterns:
            if pattern.id not in category_kb["patterns"]:
                category_kb["patterns"].append(pattern.id)
    
    def _try_generate_insight(self, entry: LearningEntry, patterns: List[Pattern]):
        """인사이트 생성 시도"""
        # 인사이트 생성 조건 확인
        if len(self.learning_queue) % self.config["insight_generation_interval"] != 0:
            return
        
        # 최근 엔트리들 분석
        recent_entries = self._get_recent_entries(100)
        
        # 트렌드 분석
        trends = self._analyze_trends(recent_entries)
        
        # 인사이트 생성
        for trend in trends:
            if trend["significance"] > 0.7:
                insight = {
                    "id": self._generate_id("insight", trend, {}),
                    "timestamp": datetime.now().isoformat(),
                    "type": trend["type"],
                    "title": trend["title"],
                    "description": trend["description"],
                    "impact_score": trend["significance"],
                    "related_patterns": [p.id for p in patterns],
                    "metadata": {
                        "entries_analyzed": len(recent_entries),
                        "trend_data": trend
                    }
                }
                
                self.insights.append(insight)
                self._save_insight_to_db(insight)
                self.stats["insights_generated"] += 1
    
    def _analyze_trends(self, entries: List[LearningEntry]) -> List[Dict[str, Any]]:
        """트렌드 분석"""
        trends = []
        
        # 성공률 트렌드
        success_rates = [e.outcome.get("success", False) for e in entries]
        if len(success_rates) > 10:
            recent_success_rate = sum(success_rates[-10:]) / 10
            overall_success_rate = sum(success_rates) / len(success_rates)
            
            if recent_success_rate > overall_success_rate * 1.2:
                trends.append({
                    "type": "improvement",
                    "title": "성공률 향상 감지",
                    "description": f"최근 성공률이 {recent_success_rate:.1%}로 전체 평균 {overall_success_rate:.1%}보다 높습니다.",
                    "significance": min((recent_success_rate - overall_success_rate) * 5, 1.0)
                })
        
        # 품질 트렌드
        quality_scores = [e.quality_score for e in entries]
        if len(quality_scores) > 10:
            recent_quality = np.mean(quality_scores[-10:])
            overall_quality = np.mean(quality_scores)
            
            if recent_quality > overall_quality * 1.1:
                trends.append({
                    "type": "quality_improvement",
                    "title": "품질 향상 감지",
                    "description": f"최근 품질 점수가 {recent_quality:.2f}로 향상되었습니다.",
                    "significance": min((recent_quality - overall_quality) * 10, 1.0)
                })
        
        # 패턴 활용도
        pattern_usage = Counter(tag for e in entries for tag in e.tags if tag.startswith("pattern_"))
        if pattern_usage:
            most_common_pattern = pattern_usage.most_common(1)[0]
            usage_rate = most_common_pattern[1] / len(entries)
            
            if usage_rate > 0.3:
                trends.append({
                    "type": "pattern_effectiveness",
                    "title": "효과적인 패턴 발견",
                    "description": f"패턴 '{most_common_pattern[0]}'이 {usage_rate:.1%}의 경우에 사용되었습니다.",
                    "significance": usage_rate
                })
        
        return trends
    
    def query_knowledge(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """지식 베이스 쿼리"""
        if category not in self.knowledge_base:
            return {"found": False, "suggestions": []}
        
        category_kb = self.knowledge_base[category]
        solution_key = self._generate_solution_key(context)
        
        # 직접 매칭 솔루션 찾기
        direct_solutions = category_kb["solutions"].get(solution_key, [])
        
        # 유사한 컨텍스트의 솔루션 찾기
        similar_solutions = []
        for key, solutions in category_kb["solutions"].items():
            similarity = self._calculate_context_similarity(context, self._parse_solution_key(key))
            if similarity > 0.7:
                similar_solutions.extend([
                    {**sol, "similarity": similarity} for sol in solutions
                ])
        
        # 품질 점수로 정렬
        all_solutions = direct_solutions + similar_solutions
        all_solutions.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # 관련 패턴 찾기
        related_patterns = []
        for pattern_id in category_kb["patterns"]:
            if pattern_id in self.patterns:
                pattern = self.patterns[pattern_id]
                if pattern.success_rate > 0.7:
                    related_patterns.append(pattern)
        
        # 베스트 프랙티스 찾기
        relevant_practices = [
            bp for bp in category_kb["best_practices"]
            if self._calculate_context_similarity(context, bp["context"]) > 0.6
        ]
        
        return {
            "found": len(all_solutions) > 0,
            "solutions": all_solutions[:5],  # 상위 5개
            "patterns": related_patterns,
            "best_practices": relevant_practices,
            "optimization_tips": category_kb["optimization_tips"][:3]
        }
    
    def get_learning_report(self) -> Dict[str, Any]:
        """학습 보고서 생성"""
        return {
            "statistics": self.stats,
            "knowledge_categories": list(self.knowledge_base.keys()),
            "total_patterns": len(self.patterns),
            "total_insights": len(self.insights),
            "recent_insights": self.insights[-5:],
            "top_patterns": self._get_top_patterns(5),
            "learning_progress": self._calculate_learning_progress(),
            "quality_metrics": self._calculate_quality_metrics()
        }
    
    def _get_top_patterns(self, count: int) -> List[Dict[str, Any]]:
        """상위 패턴 가져오기"""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.success_rate * p.occurrences,
            reverse=True
        )
        
        return [
            {
                "id": p.id,
                "type": p.pattern_type,
                "occurrences": p.occurrences,
                "success_rate": p.success_rate,
                "last_seen": p.last_seen
            }
            for p in sorted_patterns[:count]
        ]
    
    def _calculate_learning_progress(self) -> Dict[str, float]:
        """학습 진행도 계산"""
        if self.stats["total_entries"] == 0:
            return {"overall": 0.0}
        
        return {
            "overall": min(self.stats["total_entries"] / 1000, 1.0),  # 1000개 엔트리 기준
            "pattern_discovery": min(self.stats["patterns_discovered"] / 50, 1.0),  # 50개 패턴 기준
            "insight_generation": min(self.stats["insights_generated"] / 20, 1.0),  # 20개 인사이트 기준
            "success_rate": self.stats["successful_solutions"] / self.stats["total_entries"] if self.stats["total_entries"] > 0 else 0
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """품질 메트릭 계산"""
        recent_entries = self._get_recent_entries(100)
        
        if not recent_entries:
            return {"average_quality": 0.0, "average_confidence": 0.0}
        
        return {
            "average_quality": np.mean([e.quality_score for e in recent_entries]),
            "average_confidence": np.mean([e.confidence for e in recent_entries]),
            "quality_trend": self._calculate_trend([e.quality_score for e in recent_entries])
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """트렌드 계산 (양수: 상승, 음수: 하락)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # 선형 회귀
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # 기울기
    
    def _background_learning(self):
        """백그라운드 학습 프로세스"""
        while self.is_running:
            try:
                # 큐에서 엔트리 처리
                if self.learning_queue:
                    batch_size = min(10, len(self.learning_queue))
                    batch = self.learning_queue[:batch_size]
                    self.learning_queue = self.learning_queue[batch_size:]
                    
                    for entry in batch:
                        self._process_entry_immediate(entry)
                    
                    self.stats["learning_cycles"] += 1
                
                # 주기적 진화
                if self.stats["total_entries"] % self.config["evolution_interval"] == 0:
                    self._evolve_knowledge()
                
                # 주기적 저장
                if self.stats["learning_cycles"] % 10 == 0:
                    self._save_all_data()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"백그라운드 학습 오류: {e}")
                time.sleep(5)
    
    def _evolve_knowledge(self):
        """지식 진화"""
        # 패턴 통합
        self._consolidate_patterns()
        
        # 오래된 저품질 데이터 정리
        self._prune_low_quality_data()
        
        # 새로운 연결 발견
        self._discover_connections()
        
        self.stats["quality_improvements"] += 1
    
    def _consolidate_patterns(self):
        """유사한 패턴 통합"""
        pattern_groups = defaultdict(list)
        
        # 유사도 기반 그룹화
        for pattern in self.patterns.values():
            group_found = False
            for group_key, group_patterns in pattern_groups.items():
                if self._patterns_similar(pattern, group_patterns[0]):
                    group_patterns.append(pattern)
                    group_found = True
                    break
            
            if not group_found:
                pattern_groups[pattern.id] = [pattern]
        
        # 그룹 통합
        for group_patterns in pattern_groups.values():
            if len(group_patterns) > 1:
                # 가장 성공률이 높은 패턴을 기준으로 통합
                best_pattern = max(group_patterns, key=lambda p: p.success_rate)
                
                for pattern in group_patterns:
                    if pattern.id != best_pattern.id:
                        # 솔루션 병합
                        best_pattern.solutions.extend(pattern.solutions)
                        best_pattern.occurrences += pattern.occurrences
                        
                        # 패턴 제거
                        del self.patterns[pattern.id]
    
    def _save_all_data(self):
        """모든 데이터 저장"""
        # 지식 베이스 저장
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        
        # 패턴 저장
        patterns_data = {
            pid: asdict(pattern) for pid, pattern in self.patterns.items()
        }
        with open(self.patterns_path, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)
        
        # 인사이트 저장
        with open(self.insights_path, 'w', encoding='utf-8') as f:
            json.dump(self.insights, f, ensure_ascii=False, indent=2)
    
    # 유틸리티 메서드들
    def _generate_id(self, *args) -> str:
        """유니크 ID 생성"""
        content = json.dumps(args, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트 해시"""
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()[:8]
    
    def _generate_solution_key(self, context: Dict[str, Any]) -> str:
        """솔루션 키 생성"""
        key_parts = []
        for k, v in sorted(context.items()):
            if isinstance(v, (str, int, float)):
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def _parse_solution_key(self, key: str) -> Dict[str, Any]:
        """솔루션 키 파싱"""
        context = {}
        for part in key.split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                context[k] = v
        return context
    
    def _find_similar_entries(self, entry: LearningEntry) -> List[LearningEntry]:
        """유사한 엔트리 찾기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 같은 카테고리의 최근 엔트리 조회
        cursor.execute("""
            SELECT * FROM learning_entries 
            WHERE category = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        """, (entry.category,))
        
        similar_entries = []
        for row in cursor.fetchall():
            db_entry = self._row_to_entry(row)
            similarity = self._calculate_context_similarity(entry.context, db_entry.context)
            if similarity > 0.7:
                similar_entries.append(db_entry)
        
        conn.close()
        return similar_entries
    
    def _row_to_entry(self, row) -> LearningEntry:
        """데이터베이스 행을 엔트리로 변환"""
        return LearningEntry(
            id=row[0],
            timestamp=row[1],
            category=row[2],
            context=json.loads(row[3]),
            solution=json.loads(row[4]),
            outcome=json.loads(row[5]),
            quality_score=row[6],
            confidence=row[7],
            tags=json.loads(row[8]),
            metadata=json.loads(row[9])
        )
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """컨텍스트 유사도 계산"""
        if not context1 or not context2:
            return 0.0
        
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        # 키 중첩도
        key_overlap = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 0
        
        # 값 유사도
        value_similarity = 0
        common_keys = keys1 & keys2
        
        if common_keys:
            for key in common_keys:
                if context1[key] == context2[key]:
                    value_similarity += 1
            value_similarity /= len(common_keys)
        
        return (key_overlap + value_similarity) / 2
    
    def _calculate_similarity(self, entries: List[LearningEntry]) -> float:
        """엔트리 그룹의 평균 유사도"""
        if len(entries) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = self._calculate_context_similarity(
                    entries[i].context, 
                    entries[j].context
                )
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _patterns_similar(self, p1: Pattern, p2: Pattern) -> bool:
        """패턴 유사도 확인"""
        # 타입이 같고 성공률이 비슷한 경우
        return (
            p1.pattern_type == p2.pattern_type and
            abs(p1.success_rate - p2.success_rate) < 0.1 and
            p1.context_similarity > 0.8 and p2.context_similarity > 0.8
        )
    
    def _get_recent_entries(self, count: int) -> List[LearningEntry]:
        """최근 엔트리 가져오기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM learning_entries 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (count,))
        
        entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        conn.close()
        
        return entries
    
    def _save_insight_to_db(self, insight: Dict[str, Any]):
        """인사이트 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight["id"],
            insight["timestamp"],
            insight["type"],
            insight["title"],
            insight["description"],
            insight["impact_score"],
            json.dumps(insight["related_patterns"]),
            json.dumps(insight["metadata"])
        ))
        
        conn.commit()
        conn.close()
    
    def _update_stats_from_db(self):
        """데이터베이스에서 통계 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 총 엔트리 수
        cursor.execute("SELECT COUNT(*) FROM learning_entries")
        self.stats["total_entries"] = cursor.fetchone()[0]
        
        # 패턴 수
        cursor.execute("SELECT COUNT(*) FROM patterns")
        self.stats["patterns_discovered"] = cursor.fetchone()[0]
        
        # 인사이트 수
        cursor.execute("SELECT COUNT(*) FROM insights")
        self.stats["insights_generated"] = cursor.fetchone()[0]
        
        conn.close()
    
    def _prune_low_quality_data(self):
        """저품질 데이터 정리"""
        # 구현 예정: 오래되고 품질이 낮은 데이터 제거
        pass
    
    def _discover_connections(self):
        """새로운 연결 발견"""
        # 구현 예정: 패턴 간의 새로운 연결 발견
        pass
    
    def shutdown(self):
        """시스템 종료"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        self._save_all_data()


def demo():
    """데모 실행"""
    system = PersistentSelfLearningSystem()
    
    # 샘플 학습 데이터 추가
    for i in range(10):
        system.add_learning_entry(
            category="game_development",
            context={
                "task": "add_feature",
                "feature": f"feature_{i}",
                "game_type": "platformer"
            },
            solution={
                "approach": "implementation",
                "code": f"// Feature {i} implementation"
            },
            outcome={
                "success": i % 3 != 0,
                "performance": 0.8 + (i % 5) * 0.02
            },
            quality_score=0.7 + (i % 4) * 0.05,
            confidence=0.8,
            tags=["feature", "platformer"]
        )
    
    # 지식 쿼리
    time.sleep(2)  # 백그라운드 처리 대기
    
    result = system.query_knowledge(
        "game_development",
        {"task": "add_feature", "game_type": "platformer"}
    )
    
    print("쿼리 결과:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 학습 보고서
    report = system.get_learning_report()
    print("\n학습 보고서:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    system.shutdown()


if __name__ == "__main__":
    demo()