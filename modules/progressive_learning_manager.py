#!/usr/bin/env python3
"""
Progressive Learning Manager - 진행형 학습 관리자
학습 진행 상태를 추적하고 난이도를 점진적으로 높입니다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ProgressiveLearningManager:
    """진행형 학습 관리자"""
    
    def __init__(self, learning_dir: Path):
        self.learning_dir = learning_dir
        self.progress_dir = learning_dir / "progress"
        self.progress_dir.mkdir(exist_ok=True)
        
        # 학습 진행 상태 로드
        self.progress = self._load_progress()
        
        # 난이도별 주제 분류
        self.difficulty_levels = {
            2: "기초 (Basics)",
            3: "중급 (Intermediate)", 
            4: "고급 (Advanced)",
            5: "전문가 (Expert)"
        }
        
        # 주제별 마스터리 기준
        self.mastery_threshold = {
            2: 0.7,  # 기초는 70% 이상 성공률
            3: 0.65, # 중급은 65% 이상
            4: 0.6,  # 고급은 60% 이상
            5: 0.55  # 전문가는 55% 이상
        }
        
        # 최소 학습 질문 수
        self.min_questions_per_topic = 20
    
    def _load_progress(self) -> Dict:
        """학습 진행 상태 로드"""
        progress_file = self.progress_dir / "learning_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "current_difficulty": 2,  # 기초부터 시작
            "topic_mastery": {},
            "difficulty_progress": {
                2: {"total": 0, "success": 0, "rate": 0.0},
                3: {"total": 0, "success": 0, "rate": 0.0},
                4: {"total": 0, "success": 0, "rate": 0.0},
                5: {"total": 0, "success": 0, "rate": 0.0}
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def save_progress(self):
        """진행 상태 저장"""
        self.progress["last_updated"] = datetime.now().isoformat()
        progress_file = self.progress_dir / "learning_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def get_current_difficulty(self) -> int:
        """현재 학습해야 할 난이도 반환"""
        current = self.progress["current_difficulty"]
        
        # 현재 난이도의 마스터리 확인
        if self._is_difficulty_mastered(current):
            # 다음 난이도로 진급 가능한지 확인
            if current < 5:  # 최대 난이도가 5
                next_difficulty = current + 1
                logger.info(f"🎓 난이도 {current} 마스터! 난이도 {next_difficulty}로 진급합니다.")
                self.progress["current_difficulty"] = next_difficulty
                self.save_progress()
                return next_difficulty
        
        return current
    
    def _is_difficulty_mastered(self, difficulty: int) -> bool:
        """특정 난이도를 마스터했는지 확인"""
        diff_progress = self.progress["difficulty_progress"].get(str(difficulty), {})
        
        total = diff_progress.get("total", 0)
        success = diff_progress.get("success", 0)
        
        # 최소 질문 수 확인
        if total < self.min_questions_per_topic * 3:  # 난이도별로 최소 3개 주제
            return False
        
        # 성공률 확인
        success_rate = success / total if total > 0 else 0.0
        threshold = self.mastery_threshold.get(difficulty, 0.6)
        
        return success_rate >= threshold
    
    def select_topic_by_difficulty(self, topics: List, current_difficulty: int) -> Optional[object]:
        """현재 난이도에 맞는 주제 선택"""
        # 현재 난이도의 주제들만 필터링
        difficulty_topics = [t for t in topics if t.difficulty == current_difficulty]
        
        if not difficulty_topics:
            logger.warning(f"난이도 {current_difficulty}에 해당하는 주제가 없습니다.")
            return None
        
        # 가장 적게 학습한 주제 선택
        topic_stats = {}
        for topic in difficulty_topics:
            topic_id = topic.id
            mastery = self.progress["topic_mastery"].get(topic_id, {"total": 0, "success": 0})
            topic_stats[topic] = mastery["total"]
        
        # 학습 횟수가 가장 적은 주제 선택
        selected_topic = min(topic_stats.items(), key=lambda x: x[1])[0]
        
        logger.info(f"📚 선택된 주제: {selected_topic.topic} (난이도: {current_difficulty}, 학습 횟수: {topic_stats[selected_topic]})")
        
        return selected_topic
    
    def update_topic_progress(self, topic_id: str, difficulty: int, success: bool):
        """주제별 진행 상태 업데이트"""
        # 주제별 마스터리 업데이트
        if topic_id not in self.progress["topic_mastery"]:
            self.progress["topic_mastery"][topic_id] = {
                "total": 0,
                "success": 0,
                "rate": 0.0,
                "last_studied": None
            }
        
        mastery = self.progress["topic_mastery"][topic_id]
        mastery["total"] += 1
        if success:
            mastery["success"] += 1
        mastery["rate"] = mastery["success"] / mastery["total"]
        mastery["last_studied"] = datetime.now().isoformat()
        
        # 난이도별 진행 상태 업데이트
        diff_key = str(difficulty)
        if diff_key not in self.progress["difficulty_progress"]:
            self.progress["difficulty_progress"][diff_key] = {
                "total": 0,
                "success": 0,
                "rate": 0.0
            }
        
        diff_progress = self.progress["difficulty_progress"][diff_key]
        diff_progress["total"] += 1
        if success:
            diff_progress["success"] += 1
        diff_progress["rate"] = diff_progress["success"] / diff_progress["total"] if diff_progress["total"] > 0 else 0.0
        
        # 진행 상태 저장
        if mastery["total"] % 5 == 0:  # 5개 질문마다 저장
            self.save_progress()
    
    def get_progress_summary(self) -> Dict:
        """학습 진행 상태 요약"""
        summary = {
            "current_difficulty": self.progress["current_difficulty"],
            "difficulty_name": self.difficulty_levels.get(self.progress["current_difficulty"], "Unknown"),
            "difficulties": {}
        }
        
        for diff, name in self.difficulty_levels.items():
            diff_progress = self.progress["difficulty_progress"].get(str(diff), {})
            summary["difficulties"][diff] = {
                "name": name,
                "total": diff_progress.get("total", 0),
                "success": diff_progress.get("success", 0),
                "rate": diff_progress.get("rate", 0.0),
                "mastered": self._is_difficulty_mastered(diff)
            }
        
        # 전체 통계
        total_questions = sum(d.get("total", 0) for d in self.progress["difficulty_progress"].values())
        total_success = sum(d.get("success", 0) for d in self.progress["difficulty_progress"].values())
        overall_rate = total_success / total_questions if total_questions > 0 else 0.0
        
        summary["overall"] = {
            "total_questions": total_questions,
            "total_success": total_success,
            "success_rate": overall_rate,
            "topics_studied": len(self.progress["topic_mastery"])
        }
        
        return summary
    
    def should_advance_difficulty(self) -> bool:
        """난이도를 올려야 하는지 확인"""
        current = self.progress["current_difficulty"]
        return self._is_difficulty_mastered(current) and current < 5
    
    def get_recommended_focus_topics(self) -> List[str]:
        """집중 학습이 필요한 주제 추천"""
        weak_topics = []
        
        for topic_id, mastery in self.progress["topic_mastery"].items():
            if mastery["total"] >= 10 and mastery["rate"] < 0.5:
                weak_topics.append({
                    "id": topic_id,
                    "rate": mastery["rate"],
                    "total": mastery["total"]
                })
        
        # 성공률이 낮은 순으로 정렬
        weak_topics.sort(key=lambda x: x["rate"])
        
        return [t["id"] for t in weak_topics[:5]]  # 상위 5개 반환