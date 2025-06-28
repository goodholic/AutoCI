"""
AutoCI 24시간 학습 시스템
지속적으로 학습하고 개선하는 AI Agent 시스템
"""

import asyncio
import json
import sqlite3
import logging
import time
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import numpy as np
from collections import defaultdict
import threading
import signal
import sys

from modules.autoci_orchestrator import AutoCIOrchestrator, Task, TaskType, TaskPriority

@dataclass
class LearningEntry:
    """학습 데이터 항목"""
    id: str
    timestamp: datetime
    task_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    execution_time: float
    error: Optional[str] = None
    user_feedback: Optional[str] = None
    quality_score: float = 0.0

@dataclass
class Pattern:
    """학습된 패턴"""
    pattern_type: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_used: datetime

class ContinuousLearningSystem:
    """24시간 지속 학습 시스템"""
    
    def __init__(self, orchestrator: AutoCIOrchestrator, db_path: str = "autoci_learning.db"):
        self.orchestrator = orchestrator
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # 학습 데이터
        self.learning_entries: List[LearningEntry] = []
        self.patterns: Dict[str, Pattern] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # 학습 설정
        self.learning_rate = 0.1
        self.pattern_threshold = 0.8  # 패턴 인식 임계값
        self.improvement_cycle = 3600  # 1시간마다 개선 사이클
        
        # 상태
        self.running = False
        self.learning_thread = None
        self.last_improvement = datetime.now()
        
        # 데이터베이스 초기화
        self._init_database()
        
    def _init_database(self):
        """학습 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 학습 항목 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                task_type TEXT,
                input_data TEXT,
                output_data TEXT,
                success BOOLEAN,
                execution_time REAL,
                error TEXT,
                user_feedback TEXT,
                quality_score REAL
            )
        """)
        
        # 패턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                condition TEXT,
                action TEXT,
                success_rate REAL,
                usage_count INTEGER,
                last_used TEXT
            )
        """)
        
        # 성능 메트릭 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT,
                metric_name TEXT,
                value REAL
            )
        """)
        
        conn.commit()
        conn.close()
        
    async def start(self):
        """24시간 학습 시스템 시작"""
        self.running = True
        self.logger.info("24시간 학습 시스템 시작...")
        
        # 이전 학습 데이터 로드
        self._load_learning_data()
        
        # 학습 루프 시작
        self.learning_thread = asyncio.create_task(self._learning_loop())
        
        # 신호 처리기 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("학습 시스템이 성공적으로 시작되었습니다")
        
    async def stop(self):
        """학습 시스템 중지"""
        self.running = False
        
        if self.learning_thread:
            self.learning_thread.cancel()
            
        # 학습 데이터 저장
        self._save_learning_data()
        
        self.logger.info("학습 시스템이 중지되었습니다")
        
    async def _learning_loop(self):
        """메인 학습 루프"""
        while self.running:
            try:
                # 최근 작업 모니터링
                await self._monitor_recent_tasks()
                
                # 패턴 분석
                await self._analyze_patterns()
                
                # 성능 최적화
                if (datetime.now() - self.last_improvement).seconds >= self.improvement_cycle:
                    await self._perform_improvement_cycle()
                    self.last_improvement = datetime.now()
                    
                # 자동 실험
                await self._conduct_experiments()
                
                # 대기
                await asyncio.sleep(60)  # 1분마다 확인
                
            except Exception as e:
                self.logger.error(f"학습 루프 오류: {e}")
                
    async def _monitor_recent_tasks(self):
        """최근 작업 모니터링 및 학습"""
        # Orchestrator에서 완료된 작업 가져오기
        recent_tasks = self.orchestrator.completed_tasks[-10:]  # 최근 10개
        
        for task in recent_tasks:
            # 이미 학습한 작업인지 확인
            if not any(entry.id == task.id for entry in self.learning_entries):
                # 학습 항목 생성
                entry = LearningEntry(
                    id=task.id,
                    timestamp=task.created_at,
                    task_type=task.type.value,
                    input_data=task.parameters,
                    output_data=task.result or {},
                    success=task.status == "completed",
                    execution_time=(datetime.now() - task.created_at).total_seconds(),
                    error=task.error
                )
                
                # 품질 점수 계산
                entry.quality_score = self._calculate_quality_score(entry)
                
                # 학습 데이터에 추가
                self.learning_entries.append(entry)
                
                # 데이터베이스에 저장
                self._save_learning_entry(entry)
                
                # 성공/실패에서 학습
                await self._learn_from_entry(entry)
                
    async def _analyze_patterns(self):
        """패턴 분석 및 인식"""
        # 작업 유형별 그룹화
        task_groups = defaultdict(list)
        for entry in self.learning_entries:
            task_groups[entry.task_type].append(entry)
            
        # 각 그룹에서 패턴 찾기
        for task_type, entries in task_groups.items():
            if len(entries) >= 5:  # 최소 5개 이상의 샘플
                patterns = self._find_patterns_in_group(entries)
                
                for pattern in patterns:
                    pattern_id = f"{task_type}_{len(self.patterns)}"
                    self.patterns[pattern_id] = pattern
                    self._save_pattern(pattern_id, pattern)
                    
    def _find_patterns_in_group(self, entries: List[LearningEntry]) -> List[Pattern]:
        """그룹에서 패턴 찾기"""
        patterns = []
        
        # 성공한 작업들의 공통점 찾기
        successful_entries = [e for e in entries if e.success]
        if len(successful_entries) >= 3:
            # 입력 파라미터의 공통 패턴 분석
            common_params = self._find_common_parameters(successful_entries)
            
            if common_params:
                pattern = Pattern(
                    pattern_type="success_pattern",
                    condition=common_params,
                    action={"strategy": "use_successful_approach"},
                    success_rate=len(successful_entries) / len(entries),
                    usage_count=0,
                    last_used=datetime.now()
                )
                patterns.append(pattern)
                
        # 실패한 작업들의 공통점 찾기
        failed_entries = [e for e in entries if not e.success]
        if len(failed_entries) >= 2:
            common_errors = self._find_common_errors(failed_entries)
            
            if common_errors:
                pattern = Pattern(
                    pattern_type="failure_pattern",
                    condition=common_errors,
                    action={"strategy": "avoid_failure_approach"},
                    success_rate=0.0,
                    usage_count=0,
                    last_used=datetime.now()
                )
                patterns.append(pattern)
                
        return patterns
        
    def _find_common_parameters(self, entries: List[LearningEntry]) -> Dict[str, Any]:
        """공통 파라미터 찾기"""
        if not entries:
            return {}
            
        # 모든 입력 데이터에서 공통 키 찾기
        common_keys = set(entries[0].input_data.keys())
        for entry in entries[1:]:
            common_keys &= set(entry.input_data.keys())
            
        common_params = {}
        for key in common_keys:
            # 값들의 분포 분석
            values = [entry.input_data[key] for entry in entries]
            
            # 문자열인 경우 가장 빈번한 값
            if all(isinstance(v, str) for v in values):
                most_common = max(set(values), key=values.count)
                if values.count(most_common) > len(values) * 0.6:
                    common_params[key] = most_common
                    
        return common_params
        
    def _find_common_errors(self, entries: List[LearningEntry]) -> Dict[str, Any]:
        """공통 오류 패턴 찾기"""
        error_patterns = {}
        
        errors = [e.error for e in entries if e.error]
        if errors:
            # 오류 메시지에서 공통 키워드 추출
            common_words = self._extract_common_words(errors)
            if common_words:
                error_patterns["common_error_keywords"] = common_words
                
        return error_patterns
        
    def _extract_common_words(self, texts: List[str]) -> List[str]:
        """텍스트에서 공통 단어 추출"""
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
            
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.items() 
                       if count > len(texts) * 0.5]
        
        return common_words
        
    async def _perform_improvement_cycle(self):
        """개선 사이클 수행"""
        self.logger.info("개선 사이클 시작...")
        
        # 성능 메트릭 계산
        metrics = self._calculate_performance_metrics()
        
        # 메트릭 저장
        for metric_name, value in metrics.items():
            self.performance_metrics[metric_name].append(value)
            self._save_metric(metric_name, value)
            
        # 개선이 필요한 영역 식별
        improvement_areas = self._identify_improvement_areas(metrics)
        
        # 각 영역에 대한 개선 작업 생성
        for area in improvement_areas:
            await self._create_improvement_task(area)
            
        self.logger.info(f"개선 사이클 완료. 개선 영역: {improvement_areas}")
        
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 계산"""
        metrics = {}
        
        if self.learning_entries:
            # 전체 성공률
            total_success = sum(1 for e in self.learning_entries if e.success)
            metrics["overall_success_rate"] = total_success / len(self.learning_entries)
            
            # 작업 유형별 성공률
            for task_type in TaskType:
                type_entries = [e for e in self.learning_entries 
                              if e.task_type == task_type.value]
                if type_entries:
                    type_success = sum(1 for e in type_entries if e.success)
                    metrics[f"{task_type.value}_success_rate"] = type_success / len(type_entries)
                    
            # 평균 실행 시간
            avg_time = sum(e.execution_time for e in self.learning_entries) / len(self.learning_entries)
            metrics["average_execution_time"] = avg_time
            
            # 평균 품질 점수
            avg_quality = sum(e.quality_score for e in self.learning_entries) / len(self.learning_entries)
            metrics["average_quality_score"] = avg_quality
            
        return metrics
        
    def _identify_improvement_areas(self, metrics: Dict[str, float]) -> List[str]:
        """개선이 필요한 영역 식별"""
        improvement_areas = []
        
        # 낮은 성공률 확인
        for metric_name, value in metrics.items():
            if "success_rate" in metric_name and value < 0.7:
                improvement_areas.append(metric_name.replace("_success_rate", ""))
                
        # 느린 실행 시간
        if metrics.get("average_execution_time", 0) > 30:
            improvement_areas.append("execution_speed")
            
        # 낮은 품질 점수
        if metrics.get("average_quality_score", 0) < 0.7:
            improvement_areas.append("quality")
            
        return improvement_areas
        
    async def _create_improvement_task(self, area: str):
        """개선 작업 생성"""
        task = Task(
            id=f"improvement_{area}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type=TaskType.CONTINUOUS_IMPROVEMENT,
            description=f"{area} 영역 자동 개선",
            parameters={
                "improvement_area": area,
                "current_metrics": self.performance_metrics.get(f"{area}_success_rate", [])
            },
            priority=TaskPriority.BACKGROUND
        )
        
        await self.orchestrator.queue_task(task)
        
    async def _conduct_experiments(self):
        """자동 실험 수행"""
        # 랜덤하게 새로운 접근 방식 시도
        if np.random.random() < 0.1:  # 10% 확률로 실험
            experiment_task = Task(
                id=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=TaskType.CREATE_CONTENT,
                description="실험적 콘텐츠 생성",
                parameters={
                    "experimental": True,
                    "approach": np.random.choice(["creative", "efficient", "minimal"])
                },
                priority=TaskPriority.BACKGROUND
            )
            
            await self.orchestrator.queue_task(experiment_task)
            
    async def _learn_from_entry(self, entry: LearningEntry):
        """개별 항목에서 학습"""
        # 성공한 경우 관련 패턴 강화
        if entry.success:
            for pattern_id, pattern in self.patterns.items():
                if self._matches_pattern(entry, pattern):
                    pattern.success_rate = (pattern.success_rate * pattern.usage_count + 1) / (pattern.usage_count + 1)
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
                    
        # 실패한 경우 관련 패턴 약화
        else:
            for pattern_id, pattern in self.patterns.items():
                if self._matches_pattern(entry, pattern):
                    pattern.success_rate = (pattern.success_rate * pattern.usage_count) / (pattern.usage_count + 1)
                    pattern.usage_count += 1
                    
    def _matches_pattern(self, entry: LearningEntry, pattern: Pattern) -> bool:
        """항목이 패턴과 일치하는지 확인"""
        # 간단한 일치 확인 - 실제로는 더 복잡한 로직 필요
        for key, value in pattern.condition.items():
            if key in entry.input_data and entry.input_data[key] != value:
                return False
        return True
        
    def _calculate_quality_score(self, entry: LearningEntry) -> float:
        """품질 점수 계산"""
        score = 0.0
        
        # 성공 여부 (40%)
        if entry.success:
            score += 0.4
            
        # 실행 시간 (30%)
        if entry.execution_time < 10:
            score += 0.3
        elif entry.execution_time < 30:
            score += 0.2
        elif entry.execution_time < 60:
            score += 0.1
            
        # 오류 없음 (30%)
        if not entry.error:
            score += 0.3
            
        return score
        
    def get_recommendation(self, task_type: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """주어진 작업에 대한 추천 제공"""
        # 관련 패턴 찾기
        relevant_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            if pattern.success_rate > self.pattern_threshold:
                # 패턴이 현재 작업과 관련있는지 확인
                match_score = self._calculate_pattern_match(parameters, pattern.condition)
                if match_score > 0.5:
                    relevant_patterns.append((pattern, match_score))
                    
        if relevant_patterns:
            # 가장 관련성 높은 패턴 선택
            best_pattern = max(relevant_patterns, key=lambda x: x[1])[0]
            return {
                "recommendation": best_pattern.action,
                "confidence": best_pattern.success_rate,
                "based_on": f"{best_pattern.usage_count} previous successes"
            }
            
        return None
        
    def _calculate_pattern_match(self, parameters: Dict[str, Any], 
                                condition: Dict[str, Any]) -> float:
        """패턴 일치도 계산"""
        if not condition:
            return 0.0
            
        matches = 0
        total = len(condition)
        
        for key, value in condition.items():
            if key in parameters and parameters[key] == value:
                matches += 1
                
        return matches / total if total > 0 else 0.0
        
    # 데이터베이스 작업
    def _save_learning_entry(self, entry: LearningEntry):
        """학습 항목 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO learning_entries 
            (id, timestamp, task_type, input_data, output_data, success, 
             execution_time, error, user_feedback, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.timestamp.isoformat(),
            entry.task_type,
            json.dumps(entry.input_data),
            json.dumps(entry.output_data),
            entry.success,
            entry.execution_time,
            entry.error,
            entry.user_feedback,
            entry.quality_score
        ))
        
        conn.commit()
        conn.close()
        
    def _save_pattern(self, pattern_id: str, pattern: Pattern):
        """패턴 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO patterns 
            (id, pattern_type, condition, action, success_rate, usage_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern_id,
            pattern.pattern_type,
            json.dumps(pattern.condition),
            json.dumps(pattern.action),
            pattern.success_rate,
            pattern.usage_count,
            pattern.last_used.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def _save_metric(self, metric_name: str, value: float):
        """성능 메트릭 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics (timestamp, metric_name, value)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), metric_name, value))
        
        conn.commit()
        conn.close()
        
    def _load_learning_data(self):
        """학습 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 학습 항목 로드
        cursor.execute("SELECT * FROM learning_entries ORDER BY timestamp DESC LIMIT 1000")
        for row in cursor.fetchall():
            entry = LearningEntry(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                task_type=row[2],
                input_data=json.loads(row[3]),
                output_data=json.loads(row[4]),
                success=bool(row[5]),
                execution_time=row[6],
                error=row[7],
                user_feedback=row[8],
                quality_score=row[9]
            )
            self.learning_entries.append(entry)
            
        # 패턴 로드
        cursor.execute("SELECT * FROM patterns")
        for row in cursor.fetchall():
            pattern = Pattern(
                pattern_type=row[1],
                condition=json.loads(row[2]),
                action=json.loads(row[3]),
                success_rate=row[4],
                usage_count=row[5],
                last_used=datetime.fromisoformat(row[6])
            )
            self.patterns[row[0]] = pattern
            
        conn.close()
        
        self.logger.info(f"로드됨: {len(self.learning_entries)}개 학습 항목, {len(self.patterns)}개 패턴")
        
    def _save_learning_data(self):
        """모든 학습 데이터 저장"""
        # 이미 개별적으로 저장하고 있으므로 여기서는 최종 정리만
        self.logger.info(f"저장됨: {len(self.learning_entries)}개 학습 항목, {len(self.patterns)}개 패턴")
        
    def _signal_handler(self, signum, frame):
        """신호 처리"""
        self.logger.info("종료 신호 받음. 학습 시스템 종료 중...")
        asyncio.create_task(self.stop())
        
    def get_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            "total_entries": len(self.learning_entries),
            "total_patterns": len(self.patterns),
            "metrics": {}
        }
        
        # 최신 메트릭
        for metric_name, values in self.performance_metrics.items():
            if values:
                stats["metrics"][metric_name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[-2] else "stable"
                }
                
        return stats

async def main():
    """메인 함수"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Orchestrator 생성
    orchestrator = AutoCIOrchestrator()
    await orchestrator.start()
    
    # 학습 시스템 생성 및 시작
    learning_system = ContinuousLearningSystem(orchestrator)
    await learning_system.start()
    
    print("AutoCI 24시간 학습 시스템이 시작되었습니다.")
    print("종료하려면 Ctrl+C를 누르세요.")
    
    try:
        # 무한 루프
        while True:
            # 통계 출력
            stats = learning_system.get_statistics()
            print(f"\n--- 학습 통계 ---")
            print(f"총 학습 항목: {stats['total_entries']}")
            print(f"발견된 패턴: {stats['total_patterns']}")
            
            for metric, data in stats['metrics'].items():
                print(f"{metric}: {data['current']:.2f} (추세: {data['trend']})")
                
            # 30분 대기
            await asyncio.sleep(1800)
            
    except KeyboardInterrupt:
        print("\n종료 중...")
        await learning_system.stop()
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())