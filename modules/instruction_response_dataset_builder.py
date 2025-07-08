#!/usr/bin/env python3
"""
지시-응답(Instruction-Response) 데이터셋 빌더
Gemini의 조언에 따라 고품질의 정제된 학습 데이터를 구축하는 시스템
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class InstructionResponsePair:
    """지시-응답 쌍"""
    instruction: str
    input: Optional[str] = None  # 추가 컨텍스트
    output: str = ""
    category: str = ""
    difficulty: str = "medium"  # easy, medium, hard, expert
    quality_score: float = 0.0
    verified: bool = False
    source: str = ""  # 데이터 출처
    timestamp: str = ""
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            # 고유 ID 생성
            content = f"{self.instruction}{self.input or ''}{self.output}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def to_alpaca_format(self) -> Dict:
        """Alpaca 포맷으로 변환"""
        if self.input:
            return {
                "instruction": self.instruction,
                "input": self.input,
                "output": self.output
            }
        else:
            return {
                "instruction": self.instruction,
                "output": self.output
            }

class InstructionResponseDatasetBuilder:
    """고품질 지시-응답 데이터셋 구축 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dataset_dir = self.project_root / "continuous_learning" / "instruction_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 파일 경로들
        self.raw_dataset_path = self.dataset_dir / "raw_dataset.jsonl"
        self.curated_dataset_path = self.dataset_dir / "curated_dataset.jsonl"
        self.training_dataset_path = self.dataset_dir / "training_dataset.jsonl"
        
        # 품질 기준값
        self.quality_threshold = 0.7
        self.max_dataset_size = 50000  # Alpaca 데이터셋 크기 참고
        
        # Godot 관련 고품질 템플릿
        self.godot_templates = self._load_godot_templates()
        
        # 데이터 통계
        self.stats = {
            "total_collected": 0,
            "total_curated": 0,
            "total_verified": 0,
            "by_category": {},
            "by_difficulty": {}
        }
        
        self.load_existing_stats()
    
    def _load_godot_templates(self) -> Dict[str, List[Dict]]:
        """Godot 관련 고품질 템플릿 로드"""
        return {
            "concept_explanation": [
                {
                    "template": "Godot에서 {concept}이란 무엇이고 언제 사용해야 해?",
                    "concepts": ["시그널(Signal)", "노드(Node)", "씬(Scene)", "리소스(Resource)", 
                               "스크립트(Script)", "Area2D", "RigidBody2D", "AnimationPlayer"]
                },
                {
                    "template": "GDScript에서 {feature}를 사용하는 방법을 설명해줘.",
                    "features": ["딕셔너리", "배열", "커스텀 시그널", "export 변수", "setget", 
                               "yield", "coroutine", "match 문"]
                }
            ],
            "code_generation": [
                {
                    "template": "GDScript로 {task}하는 코드를 작성해줘.",
                    "tasks": ["플레이어가 스페이스바를 누르면 점프", "적이 플레이어를 추적",
                            "아이템 수집 시스템", "체력바 UI 업데이트", "세이브/로드 시스템",
                            "파티클 이펙트 생성", "카메라 흔들림 효과", "대화 시스템"]
                }
            ],
            "debugging": [
                {
                    "template": "이 GDScript 코드에서 {error_type} 오류가 발생해. 어떻게 고쳐야 할까?",
                    "error_types": ["null 참조", "시그널 연결 실패", "무한 루프", "메모리 누수",
                                  "물리 충돌 미작동", "애니메이션 재생 안됨"]
                }
            ],
            "optimization": [
                {
                    "template": "이 Godot 프로젝트의 {aspect}를 최적화하려면 어떻게 해야 해?",
                    "aspects": ["프레임레이트", "메모리 사용량", "로딩 시간", "물리 연산",
                              "렌더링 성능", "모바일 성능"]
                }
            ],
            "architecture": [
                {
                    "template": "{game_type} 게임을 Godot으로 만들 때 추천하는 프로젝트 구조는?",
                    "game_types": ["플랫포머", "RPG", "퍼즐", "슈팅", "전략", "로그라이크"]
                }
            ]
        }
    
    def generate_instruction_from_template(self, category: str) -> Optional[str]:
        """템플릿에서 지시문 생성"""
        if category not in self.godot_templates:
            return None
        
        template_group = random.choice(self.godot_templates[category])
        template = template_group["template"]
        
        # 템플릿에서 변수 추출
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        if variables:
            var_name = variables[0]
            if var_name + "s" in template_group:  # 복수형 체크
                value = random.choice(template_group[var_name + "s"])
            elif var_name in template_group:
                value = random.choice(template_group[var_name])
            else:
                return None
            
            return template.format(**{var_name: value})
        
        return template
    
    def add_instruction_response_pair(self, instruction: str, output: str, 
                                    input: Optional[str] = None,
                                    category: str = "general",
                                    difficulty: str = "medium",
                                    source: str = "generated",
                                    verified: bool = False) -> InstructionResponsePair:
        """새로운 지시-응답 쌍 추가"""
        pair = InstructionResponsePair(
            instruction=instruction,
            input=input,
            output=output,
            category=category,
            difficulty=difficulty,
            source=source,
            verified=verified
        )
        
        # 원시 데이터셋에 추가
        with open(self.raw_dataset_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')
        
        self.stats["total_collected"] += 1
        self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
        self.stats["by_difficulty"][difficulty] = self.stats["by_difficulty"].get(difficulty, 0) + 1
        
        logger.info(f"새 지시-응답 쌍 추가: {pair.id} ({category}/{difficulty})")
        
        return pair
    
    def curate_dataset(self, min_quality: float = 0.7) -> int:
        """데이터셋 큐레이션 - 고품질 데이터만 선별"""
        if not self.raw_dataset_path.exists():
            logger.warning("원시 데이터셋이 없습니다")
            return 0
        
        curated_pairs = []
        seen_instructions = set()  # 중복 방지
        
        # 원시 데이터 로드
        with open(self.raw_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    pair = InstructionResponsePair(**data)
                    
                    # 품질 검사
                    if pair.quality_score < min_quality:
                        continue
                    
                    # 중복 검사
                    instruction_key = pair.instruction.lower().strip()
                    if instruction_key in seen_instructions:
                        continue
                    seen_instructions.add(instruction_key)
                    
                    # 출력 길이 검사 (너무 짧거나 긴 답변 제외)
                    if len(pair.output) < 50 or len(pair.output) > 2000:
                        continue
                    
                    # 기본 품질 검사 통과
                    curated_pairs.append(pair)
                    
                except Exception as e:
                    logger.error(f"데이터 파싱 오류: {e}")
                    continue
        
        # 카테고리별 균형 맞추기
        balanced_pairs = self._balance_dataset(curated_pairs)
        
        # 큐레이션된 데이터셋 저장
        with open(self.curated_dataset_path, 'w', encoding='utf-8') as f:
            for pair in balanced_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')
        
        self.stats["total_curated"] = len(balanced_pairs)
        logger.info(f"큐레이션 완료: {len(balanced_pairs)}개 선별")
        
        return len(balanced_pairs)
    
    def _balance_dataset(self, pairs: List[InstructionResponsePair], 
                        max_per_category: int = 5000) -> List[InstructionResponsePair]:
        """카테고리별로 데이터셋 균형 맞추기"""
        category_buckets = {}
        
        # 카테고리별로 분류
        for pair in pairs:
            if pair.category not in category_buckets:
                category_buckets[pair.category] = []
            category_buckets[pair.category].append(pair)
        
        # 각 카테고리에서 최대 개수만큼 선택
        balanced = []
        for category, items in category_buckets.items():
            # 품질 점수로 정렬 후 상위 선택
            sorted_items = sorted(items, key=lambda x: x.quality_score, reverse=True)
            balanced.extend(sorted_items[:max_per_category])
        
        # 전체 섞기
        random.shuffle(balanced)
        
        return balanced[:self.max_dataset_size]
    
    def export_for_training(self, format: str = "alpaca") -> Path:
        """학습용 데이터셋 내보내기"""
        if not self.curated_dataset_path.exists():
            logger.error("큐레이션된 데이터셋이 없습니다. 먼저 curate_dataset()을 실행하세요.")
            return None
        
        training_data = []
        
        # 큐레이션된 데이터 로드
        with open(self.curated_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    pair = InstructionResponsePair(**data)
                    
                    if format == "alpaca":
                        training_data.append(pair.to_alpaca_format())
                    else:
                        training_data.append(pair.to_dict())
                        
                except Exception as e:
                    logger.error(f"데이터 변환 오류: {e}")
                    continue
        
        # 학습 데이터셋 저장
        output_path = self.dataset_dir / f"training_{format}_format.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"학습 데이터셋 내보내기 완료: {output_path} ({len(training_data)}개)")
        
        return output_path
    
    def import_from_existing_knowledge(self, knowledge_base_path: Path) -> int:
        """기존 지식 베이스에서 고품질 데이터 추출"""
        if not knowledge_base_path.exists():
            logger.error(f"지식 베이스를 찾을 수 없습니다: {knowledge_base_path}")
            return 0
        
        imported_count = 0
        
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            # 카테고리별로 처리
            for category, items in knowledge_data.items():
                if not isinstance(items, list):
                    continue
                
                for item in items:
                    # 지시-응답 형식으로 변환
                    if 'question' in item and 'answer' in item:
                        quality_score = item.get('confidence', 0.5)
                        
                        # 품질이 높은 것만 선택
                        if quality_score >= self.quality_threshold:
                            self.add_instruction_response_pair(
                                instruction=item['question'],
                                output=item['answer'],
                                category=category,
                                difficulty=self._estimate_difficulty(item),
                                source="knowledge_base",
                                verified=quality_score > 0.8
                            )
                            imported_count += 1
            
        except Exception as e:
            logger.error(f"지식 베이스 임포트 오류: {e}")
        
        logger.info(f"지식 베이스에서 {imported_count}개 데이터 임포트 완료")
        return imported_count
    
    def _estimate_difficulty(self, item: Dict) -> str:
        """항목의 난이도 추정"""
        text_length = len(item.get('answer', ''))
        
        if text_length < 200:
            return "easy"
        elif text_length < 500:
            return "medium"
        elif text_length < 1000:
            return "hard"
        else:
            return "expert"
    
    def validate_dataset(self) -> Dict[str, Any]:
        """데이터셋 검증 및 통계"""
        validation_results = {
            "total_raw": 0,
            "total_curated": 0,
            "quality_distribution": {},
            "category_distribution": {},
            "difficulty_distribution": {},
            "average_instruction_length": 0,
            "average_output_length": 0,
            "duplicate_count": 0,
            "validation_passed": False
        }
        
        # 원시 데이터 검증
        if self.raw_dataset_path.exists():
            seen_instructions = set()
            instruction_lengths = []
            output_lengths = []
            
            with open(self.raw_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        validation_results["total_raw"] += 1
                        
                        # 중복 체크
                        if data['instruction'] in seen_instructions:
                            validation_results["duplicate_count"] += 1
                        seen_instructions.add(data['instruction'])
                        
                        # 길이 수집
                        instruction_lengths.append(len(data['instruction']))
                        output_lengths.append(len(data.get('output', '')))
                        
                    except:
                        continue
            
            if instruction_lengths:
                validation_results["average_instruction_length"] = sum(instruction_lengths) / len(instruction_lengths)
            if output_lengths:
                validation_results["average_output_length"] = sum(output_lengths) / len(output_lengths)
        
        # 큐레이션된 데이터 검증
        if self.curated_dataset_path.exists():
            with open(self.curated_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        validation_results["total_curated"] += 1
                        
                        # 분포 수집
                        category = data.get('category', 'unknown')
                        difficulty = data.get('difficulty', 'unknown')
                        quality = round(data.get('quality_score', 0), 1)
                        
                        validation_results["category_distribution"][category] = \
                            validation_results["category_distribution"].get(category, 0) + 1
                        validation_results["difficulty_distribution"][difficulty] = \
                            validation_results["difficulty_distribution"].get(difficulty, 0) + 1
                        validation_results["quality_distribution"][str(quality)] = \
                            validation_results["quality_distribution"].get(str(quality), 0) + 1
                        
                    except:
                        continue
        
        # 검증 통과 여부
        validation_results["validation_passed"] = (
            validation_results["total_curated"] >= 1000 and  # 최소 1000개
            validation_results["duplicate_count"] < validation_results["total_raw"] * 0.1 and  # 중복 10% 미만
            len(validation_results["category_distribution"]) >= 3  # 최소 3개 카테고리
        )
        
        return validation_results
    
    def save_stats(self):
        """통계 저장"""
        stats_path = self.dataset_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
    
    def load_existing_stats(self):
        """기존 통계 로드"""
        stats_path = self.dataset_dir / "dataset_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except:
                pass

# 싱글톤 인스턴스
_dataset_builder = None

def get_dataset_builder() -> InstructionResponseDatasetBuilder:
    """데이터셋 빌더 싱글톤 인스턴스 반환"""
    global _dataset_builder
    if _dataset_builder is None:
        _dataset_builder = InstructionResponseDatasetBuilder()
    return _dataset_builder