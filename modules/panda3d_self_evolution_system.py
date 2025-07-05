"""
Panda3D 자가 진화 시스템
집단지성 기반으로 게임 개발 패턴을 학습하고 진화
"""

import os
import json
import time
import hashlib
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# AI 모델 통합
from .ai_model_integration import get_ai_integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvolutionPattern:
    """진화 패턴"""
    pattern_id: str
    pattern_type: str  # code, design, optimization, bug_fix
    description: str
    solution: str
    context: Dict[str, Any]
    success_rate: float
    usage_count: int
    created_at: str
    last_used: str
    fitness_score: float = 0.0


@dataclass
class CollectiveKnowledge:
    """집단 지식"""
    topic: str
    patterns: List[EvolutionPattern]
    insights: List[str]
    best_practices: List[str]
    common_mistakes: List[str]
    evolution_history: List[Dict[str, Any]]


class Panda3DSelfEvolutionSystem:
    """Panda3D 자가 진화 시스템"""
    
    def __init__(self):
        self.ai_model = get_ai_integration()
        
        # 진화 데이터 경로
        self.evolution_path = Path("evolution_data")
        self.evolution_path.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = self.evolution_path / "evolution.db"
        self._init_database()
        
        # 학습 주제 (Panda3D 게임 개발)
        self.learning_topics = [
            "panda3d_basics",
            "game_mechanics",
            "rendering_optimization",
            "physics_integration",
            "networking",
            "ui_development",
            "asset_management",
            "performance_tuning"
        ]
        
        # 집단 지식 저장소
        self.collective_knowledge: Dict[str, CollectiveKnowledge] = {}
        self._load_collective_knowledge()
        
        # 패턴 인식 시스템
        self.pattern_vectorizer = TfidfVectorizer(max_features=1000)
        self.pattern_vectors = None
        
        # 진화 설정
        self.evolution_config = {
            "min_fitness_threshold": 0.7,
            "mutation_rate": 0.1,
            "crossover_rate": 0.3,
            "population_size": 100,
            "generations": 10,
            "elite_ratio": 0.2
        }
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 패턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                description TEXT,
                solution TEXT,
                context TEXT,
                success_rate REAL,
                usage_count INTEGER,
                created_at TEXT,
                last_used TEXT,
                fitness_score REAL
            )
        """)
        
        # 진화 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                timestamp TEXT,
                best_fitness REAL,
                average_fitness REAL,
                patterns_evolved INTEGER,
                insights TEXT
            )
        """)
        
        # 사용자 피드백 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT,
                feedback_type TEXT,
                rating INTEGER,
                comment TEXT,
                timestamp TEXT,
                FOREIGN KEY (pattern_id) REFERENCES patterns (pattern_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_collective_knowledge(self):
        """집단 지식 로드"""
        knowledge_path = self.evolution_path / "collective_knowledge"
        knowledge_path.mkdir(exist_ok=True)
        
        for topic in self.learning_topics:
            topic_file = knowledge_path / f"{topic}.json"
            if topic_file.exists():
                with open(topic_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    patterns = [EvolutionPattern(**p) for p in data.get("patterns", [])]
                    self.collective_knowledge[topic] = CollectiveKnowledge(
                        topic=topic,
                        patterns=patterns,
                        insights=data.get("insights", []),
                        best_practices=data.get("best_practices", []),
                        common_mistakes=data.get("common_mistakes", []),
                        evolution_history=data.get("evolution_history", [])
                    )
            else:
                self.collective_knowledge[topic] = CollectiveKnowledge(
                    topic=topic,
                    patterns=[],
                    insights=[],
                    best_practices=[],
                    common_mistakes=[],
                    evolution_history=[]
                )
    
    def _save_collective_knowledge(self):
        """집단 지식 저장"""
        knowledge_path = self.evolution_path / "collective_knowledge"
        
        for topic, knowledge in self.collective_knowledge.items():
            topic_file = knowledge_path / f"{topic}.json"
            data = {
                "patterns": [asdict(p) for p in knowledge.patterns],
                "insights": knowledge.insights,
                "best_practices": knowledge.best_practices,
                "common_mistakes": knowledge.common_mistakes,
                "evolution_history": knowledge.evolution_history
            }
            with open(topic_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def learn_from_interaction(self, user_input: str, ai_response: str, 
                              feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사용자 상호작용으로부터 학습"""
        # 패턴 추출
        patterns = self._extract_patterns(user_input, ai_response)
        
        # 품질 평가
        quality_score = self._evaluate_quality(user_input, ai_response, feedback)
        
        # 고품질 패턴은 저장
        if quality_score >= self.evolution_config["min_fitness_threshold"]:
            for pattern in patterns:
                pattern.fitness_score = quality_score
                self._store_pattern(pattern)
        
        # 패턴 인식 및 개선
        similar_patterns = self._find_similar_patterns(patterns)
        improvements = await self._suggest_improvements(patterns, similar_patterns)
        
        # 진화 사이클 트리거 체크
        if self._should_evolve():
            asyncio.create_task(self.run_evolution_cycle())
        
        return {
            "patterns_extracted": len(patterns),
            "quality_score": quality_score,
            "similar_patterns": len(similar_patterns),
            "improvements": improvements
        }
    
    def _extract_patterns(self, user_input: str, ai_response: str) -> List[EvolutionPattern]:
        """패턴 추출"""
        patterns = []
        
        # 코드 패턴 추출
        if "```" in ai_response:
            code_blocks = self._extract_code_blocks(ai_response)
            for code in code_blocks:
                pattern = EvolutionPattern(
                    pattern_id=self._generate_pattern_id(code),
                    pattern_type="code",
                    description=f"Code pattern from: {user_input[:50]}...",
                    solution=code,
                    context={"user_input": user_input, "language": "python"},
                    success_rate=0.0,
                    usage_count=1,
                    created_at=datetime.now().isoformat(),
                    last_used=datetime.now().isoformat()
                )
                patterns.append(pattern)
        
        # 디자인 패턴 추출
        design_keywords = ["architecture", "design", "pattern", "structure"]
        if any(keyword in user_input.lower() for keyword in design_keywords):
            pattern = EvolutionPattern(
                pattern_id=self._generate_pattern_id(ai_response),
                pattern_type="design",
                description=f"Design pattern: {user_input[:50]}...",
                solution=ai_response,
                context={"user_input": user_input},
                success_rate=0.0,
                usage_count=1,
                created_at=datetime.now().isoformat(),
                last_used=datetime.now().isoformat()
            )
            patterns.append(pattern)
        
        # 최적화 패턴 추출
        optimization_keywords = ["optimize", "performance", "faster", "efficient"]
        if any(keyword in user_input.lower() for keyword in optimization_keywords):
            pattern = EvolutionPattern(
                pattern_id=self._generate_pattern_id(ai_response),
                pattern_type="optimization",
                description=f"Optimization: {user_input[:50]}...",
                solution=ai_response,
                context={"user_input": user_input},
                success_rate=0.0,
                usage_count=1,
                created_at=datetime.now().isoformat(),
                last_used=datetime.now().isoformat()
            )
            patterns.append(pattern)
        
        return patterns
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """코드 블록 추출"""
        code_blocks = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def _generate_pattern_id(self, content: str) -> str:
        """패턴 ID 생성"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _evaluate_quality(self, user_input: str, ai_response: str, 
                         feedback: Optional[Dict[str, Any]] = None) -> float:
        """품질 평가"""
        score = 0.0
        
        # 응답 길이와 구조
        if len(ai_response) > 100:
            score += 0.1
        if "```" in ai_response:  # 코드 포함
            score += 0.2
        if any(marker in ai_response for marker in ["1.", "2.", "-", "*"]):  # 구조화
            score += 0.1
        
        # Panda3D 관련 키워드
        panda3d_keywords = ["ShowBase", "render", "NodePath", "camera", "loader", "task"]
        keyword_count = sum(1 for kw in panda3d_keywords if kw in ai_response)
        score += min(0.3, keyword_count * 0.05)
        
        # 사용자 피드백
        if feedback:
            if feedback.get("helpful", False):
                score += 0.3
            if feedback.get("rating", 0) > 3:
                score += 0.2
        
        return min(1.0, score)
    
    def _store_pattern(self, pattern: EvolutionPattern):
        """패턴 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO patterns 
            (pattern_id, pattern_type, description, solution, context, 
             success_rate, usage_count, created_at, last_used, fitness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id, pattern.pattern_type, pattern.description,
            pattern.solution, json.dumps(pattern.context), pattern.success_rate,
            pattern.usage_count, pattern.created_at, pattern.last_used,
            pattern.fitness_score
        ))
        
        conn.commit()
        conn.close()
        
        # 집단 지식에도 추가
        topic = self._classify_pattern_topic(pattern)
        if topic in self.collective_knowledge:
            self.collective_knowledge[topic].patterns.append(pattern)
    
    def _classify_pattern_topic(self, pattern: EvolutionPattern) -> str:
        """패턴 주제 분류"""
        # 간단한 키워드 기반 분류
        content = pattern.description + " " + pattern.solution
        
        topic_keywords = {
            "panda3d_basics": ["ShowBase", "render", "camera", "loader"],
            "game_mechanics": ["player", "enemy", "collision", "physics"],
            "rendering_optimization": ["shader", "texture", "LOD", "culling"],
            "physics_integration": ["bullet", "rigid", "collision", "force"],
            "networking": ["socket", "multiplayer", "server", "client"],
            "ui_development": ["DirectGUI", "button", "label", "menu"],
            "asset_management": ["model", "texture", "sound", "load"],
            "performance_tuning": ["optimize", "fps", "performance", "profiling"]
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in content.lower())
            topic_scores[topic] = score
        
        # 가장 높은 점수의 주제 반환
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic if topic_scores[best_topic] > 0 else "panda3d_basics"
    
    def _find_similar_patterns(self, patterns: List[EvolutionPattern]) -> List[EvolutionPattern]:
        """유사 패턴 찾기"""
        similar_patterns = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            # 같은 타입의 패턴 검색
            cursor.execute("""
                SELECT * FROM patterns 
                WHERE pattern_type = ? AND pattern_id != ?
                ORDER BY fitness_score DESC
                LIMIT 5
            """, (pattern.pattern_type, pattern.pattern_id))
            
            rows = cursor.fetchall()
            for row in rows:
                similar = EvolutionPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    description=row[2],
                    solution=row[3],
                    context=json.loads(row[4]),
                    success_rate=row[5],
                    usage_count=row[6],
                    created_at=row[7],
                    last_used=row[8],
                    fitness_score=row[9]
                )
                similar_patterns.append(similar)
        
        conn.close()
        return similar_patterns
    
    async def _suggest_improvements(self, patterns: List[EvolutionPattern], 
                            similar_patterns: List[EvolutionPattern]) -> List[str]:
        """개선 제안"""
        improvements = []
        
        # 높은 적합도의 유사 패턴에서 학습
        high_fitness_patterns = [p for p in similar_patterns if p.fitness_score > 0.8]
        
        for pattern in patterns:
            for similar in high_fitness_patterns:
                if similar.pattern_type == pattern.pattern_type:
                    improvement = f"Consider using approach from '{similar.description}' " \
                                f"which has {similar.fitness_score:.2f} fitness score"
                    improvements.append(improvement)
        
        # AI 모델을 통한 개선 제안
        if patterns and self.ai_model.is_model_loaded():
            prompt = f"""
            Analyze these patterns and suggest improvements:
            {[p.description for p in patterns]}
            
            Focus on Panda3D best practices and performance optimization.
            """
            context = {
                "task": "analyze_patterns",
                "patterns": [p.description for p in patterns],
                "focus": "Panda3D best practices and performance optimization"
            }
            ai_result = await self.ai_model.generate_code(prompt, context, max_length=300)
            ai_suggestions = ai_result.get('code', '') if isinstance(ai_result, dict) else str(ai_result)
            if ai_suggestions:
                improvements.append(f"AI Suggestion: {ai_suggestions}")
        
        return improvements[:5]  # 최대 5개 제안
    
    def _should_evolve(self) -> bool:
        """진화 사이클 실행 여부 결정"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 마지막 진화 시간 확인
        cursor.execute("""
            SELECT MAX(timestamp) FROM evolution_history
        """)
        last_evolution = cursor.fetchone()[0]
        
        if last_evolution:
            last_time = datetime.fromisoformat(last_evolution)
            if datetime.now() - last_time < timedelta(hours=1):
                conn.close()
                return False
        
        # 새로운 패턴 수 확인
        cursor.execute("""
            SELECT COUNT(*) FROM patterns 
            WHERE created_at > ?
        """, (last_evolution or "2000-01-01",))
        
        new_patterns = cursor.fetchone()[0]
        conn.close()
        
        return new_patterns >= 10  # 10개 이상의 새 패턴이 있으면 진화
    
    async def run_evolution_cycle(self):
        """진화 사이클 실행"""
        logger.info("🧬 진화 사이클 시작...")
        
        generation_results = []
        
        for generation in range(self.evolution_config["generations"]):
            # 현재 세대의 패턴들
            population = self._get_population()
            
            # 적합도 평가
            fitness_scores = self._evaluate_population_fitness(population)
            
            # 선택 (엘리트 + 토너먼트)
            selected = self._selection(population, fitness_scores)
            
            # 교차 (크로스오버)
            offspring = self._crossover(selected)
            
            # 돌연변이
            mutated = await self._mutation(offspring)
            
            # 새로운 세대 구성
            new_population = self._create_new_generation(selected, mutated)
            
            # 결과 기록
            best_fitness = max(fitness_scores.values()) if fitness_scores else 0
            avg_fitness = sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0
            
            generation_results.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "population_size": len(new_population)
            })
            
            logger.info(f"세대 {generation}: 최고 적합도={best_fitness:.2f}, "
                       f"평균 적합도={avg_fitness:.2f}")
        
        # 진화 인사이트 생성
        insights = await self._generate_evolution_insights(generation_results)
        
        # 진화 기록 저장
        self._save_evolution_history(generation_results[-1], insights)
        
        # 집단 지식 업데이트
        self._update_collective_knowledge(insights)
        
        logger.info("🧬 진화 사이클 완료!")
    
    def _get_population(self) -> List[EvolutionPattern]:
        """현재 패턴 모집단 가져오기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM patterns 
            ORDER BY fitness_score DESC
            LIMIT ?
        """, (self.evolution_config["population_size"],))
        
        population = []
        for row in cursor.fetchall():
            pattern = EvolutionPattern(
                pattern_id=row[0],
                pattern_type=row[1],
                description=row[2],
                solution=row[3],
                context=json.loads(row[4]),
                success_rate=row[5],
                usage_count=row[6],
                created_at=row[7],
                last_used=row[8],
                fitness_score=row[9]
            )
            population.append(pattern)
        
        conn.close()
        return population
    
    def _evaluate_population_fitness(self, population: List[EvolutionPattern]) -> Dict[str, float]:
        """모집단 적합도 평가"""
        fitness_scores = {}
        
        for pattern in population:
            # 기본 적합도
            fitness = pattern.fitness_score
            
            # 사용 빈도 보너스
            fitness += min(0.1, pattern.usage_count * 0.01)
            
            # 최근 사용 보너스
            last_used = datetime.fromisoformat(pattern.last_used)
            days_ago = (datetime.now() - last_used).days
            if days_ago < 7:
                fitness += 0.05
            
            # 성공률 보너스
            fitness += pattern.success_rate * 0.2
            
            fitness_scores[pattern.pattern_id] = min(1.0, fitness)
        
        return fitness_scores
    
    def _selection(self, population: List[EvolutionPattern], 
                  fitness_scores: Dict[str, float]) -> List[EvolutionPattern]:
        """선택 (엘리트 + 토너먼트)"""
        selected = []
        
        # 엘리트 선택
        elite_count = int(len(population) * self.evolution_config["elite_ratio"])
        sorted_population = sorted(population, 
                                 key=lambda p: fitness_scores.get(p.pattern_id, 0), 
                                 reverse=True)
        selected.extend(sorted_population[:elite_count])
        
        # 토너먼트 선택
        tournament_size = 3
        while len(selected) < len(population) // 2:
            tournament = np.random.choice(population, tournament_size, replace=False)
            winner = max(tournament, key=lambda p: fitness_scores.get(p.pattern_id, 0))
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parents: List[EvolutionPattern]) -> List[EvolutionPattern]:
        """교차 (크로스오버)"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            if np.random.random() < self.evolution_config["crossover_rate"]:
                parent1, parent2 = parents[i], parents[i + 1]
                
                # 솔루션 교차
                child_solution = self._combine_solutions(parent1.solution, parent2.solution)
                
                # 새로운 패턴 생성
                child = EvolutionPattern(
                    pattern_id=self._generate_pattern_id(child_solution),
                    pattern_type=parent1.pattern_type,
                    description=f"Evolved from {parent1.description} and {parent2.description}",
                    solution=child_solution,
                    context={**parent1.context, **parent2.context},
                    success_rate=(parent1.success_rate + parent2.success_rate) / 2,
                    usage_count=0,
                    created_at=datetime.now().isoformat(),
                    last_used=datetime.now().isoformat(),
                    fitness_score=(parent1.fitness_score + parent2.fitness_score) / 2
                )
                offspring.append(child)
        
        return offspring
    
    def _combine_solutions(self, solution1: str, solution2: str) -> str:
        """두 솔루션 결합"""
        # 간단한 결합 전략
        lines1 = solution1.split('\n')
        lines2 = solution2.split('\n')
        
        # 번갈아가며 라인 선택
        combined = []
        for i in range(max(len(lines1), len(lines2))):
            if i < len(lines1) and (i >= len(lines2) or i % 2 == 0):
                combined.append(lines1[i])
            elif i < len(lines2):
                combined.append(lines2[i])
        
        return '\n'.join(combined)
    
    async def _mutation(self, population: List[EvolutionPattern]) -> List[EvolutionPattern]:
        """돌연변이"""
        mutated = []
        
        for pattern in population:
            if np.random.random() < self.evolution_config["mutation_rate"]:
                # AI를 통한 돌연변이
                mutation_prompt = f"""
                Improve or modify this Panda3D code pattern:
                {pattern.solution}
                
                Make a small but meaningful change.
                """
                
                if self.ai_model.is_model_loaded():
                    context = {
                        "task": "mutate_pattern",
                        "original_solution": pattern.solution,
                        "pattern_type": pattern.pattern_type
                    }
                    mutated_result = await self.ai_model.generate_code(mutation_prompt, context, max_length=500)
                    mutated_solution = mutated_result.get('code', '') if isinstance(mutated_result, dict) else str(mutated_result)
                    if mutated_solution:
                        pattern.solution = mutated_solution
                        pattern.description += " [Mutated]"
                        pattern.pattern_id = self._generate_pattern_id(mutated_solution)
                
                mutated.append(pattern)
        
        return mutated
    
    def _create_new_generation(self, selected: List[EvolutionPattern], 
                              offspring: List[EvolutionPattern]) -> List[EvolutionPattern]:
        """새로운 세대 생성"""
        new_generation = selected + offspring
        
        # 중복 제거
        unique_patterns = {}
        for pattern in new_generation:
            if pattern.pattern_id not in unique_patterns:
                unique_patterns[pattern.pattern_id] = pattern
        
        return list(unique_patterns.values())
    
    async def _generate_evolution_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        """진화 인사이트 생성"""
        insights = []
        
        # 적합도 향상 분석
        if len(results) > 1:
            fitness_improvement = results[-1]["best_fitness"] - results[0]["best_fitness"]
            if fitness_improvement > 0:
                insights.append(f"적합도가 {fitness_improvement:.2%} 향상되었습니다.")
        
        # 패턴 타입별 분석
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pattern_type, AVG(fitness_score), COUNT(*)
            FROM patterns
            GROUP BY pattern_type
            ORDER BY AVG(fitness_score) DESC
        """)
        
        for row in cursor.fetchall():
            pattern_type, avg_fitness, count = row
            insights.append(f"{pattern_type} 패턴: 평균 적합도 {avg_fitness:.2f}, 총 {count}개")
        
        conn.close()
        
        # AI 기반 인사이트
        if self.ai_model.is_model_loaded() and results:
            insight_prompt = f"""
            Analyze these evolution results and provide insights:
            {results}
            
            Focus on Panda3D game development patterns and improvements.
            """
            context = {
                "task": "analyze_evolution",
                "results": results,
                "focus": "Panda3D game development patterns"
            }
            ai_result = await self.ai_model.generate_code(insight_prompt, context, max_length=300)
            ai_insights = ai_result.get('code', '') if isinstance(ai_result, dict) else str(ai_result)
            if ai_insights:
                insights.append(f"AI 분석: {ai_insights}")
        
        return insights
    
    def _save_evolution_history(self, result: Dict[str, Any], insights: List[str]):
        """진화 기록 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evolution_history 
            (generation, timestamp, best_fitness, average_fitness, patterns_evolved, insights)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            result["generation"],
            datetime.now().isoformat(),
            result["best_fitness"],
            result["average_fitness"],
            result["population_size"],
            json.dumps(insights, ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_collective_knowledge(self, insights: List[str]):
        """집단 지식 업데이트"""
        # 각 주제별로 인사이트 분류
        for topic in self.learning_topics:
            knowledge = self.collective_knowledge[topic]
            
            # 관련 인사이트 추가
            for insight in insights:
                if topic.replace("_", " ") in insight.lower():
                    knowledge.insights.append(insight)
            
            # 베스트 프랙티스 업데이트
            self._update_best_practices(topic)
            
            # 일반적인 실수 패턴 식별
            self._identify_common_mistakes(topic)
        
        # 저장
        self._save_collective_knowledge()
    
    def _update_best_practices(self, topic: str):
        """베스트 프랙티스 업데이트"""
        knowledge = self.collective_knowledge[topic]
        
        # 높은 적합도 패턴에서 추출
        high_fitness_patterns = [p for p in knowledge.patterns if p.fitness_score > 0.8]
        
        for pattern in high_fitness_patterns[:5]:  # 상위 5개
            practice = f"{pattern.description}: {pattern.solution[:100]}..."
            if practice not in knowledge.best_practices:
                knowledge.best_practices.append(practice)
    
    def _identify_common_mistakes(self, topic: str):
        """일반적인 실수 패턴 식별"""
        knowledge = self.collective_knowledge[topic]
        
        # 낮은 적합도 패턴 분석
        low_fitness_patterns = [p for p in knowledge.patterns if p.fitness_score < 0.3]
        
        # 패턴에서 공통 요소 찾기
        if low_fitness_patterns:
            common_issues = []
            for pattern in low_fitness_patterns[:5]:
                issue = f"Avoid: {pattern.description}"
                if issue not in knowledge.common_mistakes:
                    knowledge.common_mistakes.append(issue)
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """진화 보고서 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(fitness_score) FROM patterns")
        avg_fitness = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM evolution_history")
        total_evolutions = cursor.fetchone()[0]
        
        # 주제별 통계
        topic_stats = {}
        for topic in self.learning_topics:
            knowledge = self.collective_knowledge[topic]
            topic_stats[topic] = {
                "patterns": len(knowledge.patterns),
                "insights": len(knowledge.insights),
                "best_practices": len(knowledge.best_practices),
                "common_mistakes": len(knowledge.common_mistakes)
            }
        
        conn.close()
        
        return {
            "total_patterns": total_patterns,
            "average_fitness": avg_fitness,
            "total_evolutions": total_evolutions,
            "topic_statistics": topic_stats,
            "last_updated": datetime.now().isoformat()
        }


# 테스트 및 예제
if __name__ == "__main__":
    evolution_system = Panda3DSelfEvolutionSystem()
    
    # 상호작용 학습 예제
    user_input = "How do I create a jumping mechanic in Panda3D?"
    ai_response = """
    Here's how to implement jumping in Panda3D:
    
    ```python
    def jump(self):
        if not self.is_jumping:
            self.velocity_z = 10
            self.is_jumping = True
    
    def update(self, task):
        # Gravity
        self.velocity_z -= 20 * dt
        
        # Update position
        self.player.setZ(self.player.getZ() + self.velocity_z * dt)
        
        # Ground check
        if self.player.getZ() <= 0:
            self.player.setZ(0)
            self.is_jumping = False
            self.velocity_z = 0
    ```
    """
    
    result = evolution_system.learn_from_interaction(
        user_input, 
        ai_response,
        {"helpful": True, "rating": 5}
    )
    
    print(f"학습 결과: {result}")
    
    # 진화 보고서
    report = evolution_system.get_evolution_report()
    print(f"\n진화 보고서: {json.dumps(report, indent=2)}")