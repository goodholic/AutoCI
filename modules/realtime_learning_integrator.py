#!/usr/bin/env python3
"""
실시간 학습 통합기 - 개발 경험을 AI 학습 데이터로 실시간 변환
개발 중 수집된 모든 성공적인 솔루션, 패턴, 최적화를 AI의 영구적인 지식으로 통합합니다.

주요 기능:
1. 개발 경험을 AI 학습용 Q&A 쌍으로 변환
2. 성공적인 솔루션에서 자동으로 학습 데이터 생성
3. 실시간으로 지식 베이스 업데이트
4. 학습 시스템과 완벽한 통합
5. 게임 개발 경험을 전문 데이터셋으로 구성
"""

import os
import sys
import json
import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LearningData:
    """학습 데이터 구조"""
    id: str
    category: str
    topic: str
    question: str
    answer: str
    keywords: List[str]
    difficulty: int
    source: str  # 'development', 'community', 'ai_discovery'
    effectiveness: float
    metadata: Dict[str, Any]
    timestamp: str

@dataclass
class QAPair:
    """질문-답변 쌍"""
    question_id: str
    question_text: str
    question_type: str
    answer_text: str
    quality_score: float
    source_experience: str
    generated_at: str

class RealtimeLearningIntegrator:
    """실시간 학습 통합기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.learning_base_path = self.project_root / "continuous_learning"
        self.integration_path = self.learning_base_path / "realtime_integration"
        self.integration_path.mkdir(parents=True, exist_ok=True)
        
        # 통합 상태
        self.integration_stats = {
            'total_experiences_converted': 0,
            'qa_pairs_generated': 0,
            'knowledge_updates': 0,
            'training_datasets_created': 0,
            'last_integration': None
        }
        
        # 변환 규칙
        self.conversion_rules = self._load_conversion_rules()
        
        # 카테고리 매핑 (개발 경험 -> 학습 주제)
        self.category_mapping = {
            'error_solution': 'core_csharp_basics',
            'game_mechanic': 'core_godot_architecture',
            'code_pattern': 'core_csharp_advanced',
            'performance_opt': 'core_godot_ai_network',
            'resource_generation': 'core_godot_architecture',
            'community_solution': 'core_korean_concepts',
            'ai_discovery': 'core_nakama_ai',
            'networking': 'core_godot_networking',
            'nakama_integration': 'core_nakama_basics'
        }
        
        # 학습 시스템 연결
        self.continuous_learning_system = None
        self.experience_collector = None
        self.ai_model_controller = None
        
        # 실시간 큐
        self.experience_queue = asyncio.Queue(maxsize=1000)
        self.processing_active = False
        
        # 통합 로그
        self.integration_log = []
        
        # 기존 통합 상태 로드
        self._load_integration_state()
    
    def _load_conversion_rules(self) -> Dict[str, Any]:
        """경험 타입별 변환 규칙 로드"""
        return {
            'error_solution': {
                'question_templates': [
                    "C#에서 {error_type} 오류가 발생할 때 어떻게 해결하나요?",
                    "{error_description}가 발생하는 이유와 해결 방법을 설명해주세요.",
                    "Godot에서 {error_context} 관련 오류를 해결하는 방법은?"
                ],
                'answer_format': "오류: {error_description}\n\n해결 방법:\n{solution_steps}\n\n코드 예제:\n```csharp\n{code_example}\n```\n\n설명: {explanation}",
                'difficulty_calculator': lambda exp: min(5, 2 + exp.get('attempts', 1) // 3)
            },
            'game_mechanic': {
                'question_templates': [
                    "Godot에서 {mechanic_name} 기능을 구현하는 방법은?",
                    "{mechanic_name}을 C#으로 구현하는 최선의 방법을 보여주세요.",
                    "게임에서 {mechanic_description}을 만들려면 어떻게 해야 하나요?"
                ],
                'answer_format': "구현 방법:\n\n1. 개념 설명:\n{description}\n\n2. 코드 구현:\n```csharp\n{code_snippet}\n```\n\n3. 사용 예시:\n{usage_example}\n\n4. 성능 고려사항:\n{performance_notes}",
                'difficulty_calculator': lambda exp: min(5, 3 + exp.get('complexity', 0) // 20)
            },
            'code_pattern': {
                'question_templates': [
                    "C#에서 {pattern_name} 패턴을 어떻게 활용하나요?",
                    "{use_case}에 적합한 코드 패턴을 보여주세요.",
                    "Godot 프로젝트에서 {pattern_name}을 사용하는 모범 사례는?"
                ],
                'answer_format': "패턴 이름: {pattern_name}\n\n사용 사례: {use_case}\n\n구현:\n```csharp\n{code}\n```\n\n장점:\n- 효과성: {effectiveness}\n- 적용 횟수: {applications}회\n\n주의사항: {considerations}",
                'difficulty_calculator': lambda exp: 4  # 패턴은 보통 고급
            },
            'performance_opt': {
                'question_templates': [
                    "Godot에서 {optimization_type} 성능을 최적화하는 방법은?",
                    "{before_metrics}에서 {after_metrics}로 개선하는 방법을 설명해주세요.",
                    "게임 성능을 {improvement}% 향상시키는 최적화 기법은?"
                ],
                'answer_format': "최적화 방법: {method}\n\n이전 성능:\n{before_metrics}\n\n이후 성능:\n{after_metrics}\n\n개선율: {improvement}%\n\n코드 변경사항:\n```csharp\n{code_changes}\n```\n\n핵심 포인트: {key_points}",
                'difficulty_calculator': lambda exp: min(5, 3 + int(exp.get('improvement', 0) / 20))
            },
            'ai_discovery': {
                'question_templates': [
                    "{discovery_type}에 대한 혁신적인 접근 방법은?",
                    "AI가 발견한 {description}을 설명해주세요.",
                    "{context}에서 창의적인 해결책은 무엇인가요?"
                ],
                'answer_format': "AI 발견: {discovery_type}\n\n설명:\n{description}\n\n구현 코드:\n```csharp\n{code}\n```\n\n맥락: {context}\n\n창의성 점수: {creativity_score}/10\n효과성: {effectiveness}",
                'difficulty_calculator': lambda exp: min(5, 4 + exp.get('creativity_score', 0) // 3)
            }
        }
    
    def _load_integration_state(self):
        """통합 상태 로드"""
        state_file = self.integration_path / "integration_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    self.integration_stats = json.load(f)
                logger.info(f"통합 상태 로드: {self.integration_stats['qa_pairs_generated']}개 Q&A 쌍 생성됨")
            except Exception as e:
                logger.error(f"통합 상태 로드 실패: {e}")
    
    def _save_integration_state(self):
        """통합 상태 저장"""
        state_file = self.integration_path / "integration_state.json"
        self.integration_stats['last_integration'] = datetime.now().isoformat()
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.integration_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"통합 상태 저장 실패: {e}")
    
    async def connect_systems(self, continuous_learning_system=None, experience_collector=None, ai_model_controller=None):
        """학습 시스템들과 연결"""
        if continuous_learning_system:
            self.continuous_learning_system = continuous_learning_system
            logger.info("✅ 연속 학습 시스템과 연결됨")
        
        if experience_collector:
            self.experience_collector = experience_collector
            logger.info("✅ 경험 수집기와 연결됨")
            
            # 기존 경험 데이터 동기화
            await self._sync_existing_experiences()
        
        if ai_model_controller:
            self.ai_model_controller = ai_model_controller
            logger.info("✅ AI 모델 컨트롤러와 연결됨")
    
    async def _sync_existing_experiences(self):
        """기존 수집된 경험 동기화"""
        if not self.experience_collector:
            return
        
        logger.info("기존 경험 데이터 동기화 시작...")
        
        # 오류 해결책
        for error_hash, solutions in self.experience_collector.error_solutions.items():
            for solution in solutions:
                if solution.get('success', False):
                    await self.convert_experience_to_learning_data('error_solution', solution)
        
        # 성공적인 게임 메카닉
        for mechanic in self.experience_collector.successful_mechanics:
            await self.convert_experience_to_learning_data('game_mechanic', mechanic)
        
        # 코드 패턴
        for pattern_hash, pattern in self.experience_collector.code_patterns.items():
            if pattern.get('effectiveness', 0) > 0.7:
                await self.convert_experience_to_learning_data('code_pattern', pattern)
        
        # 성능 최적화
        for optimization in self.experience_collector.performance_optimizations:
            if optimization.get('improvement', 0) > 10:  # 10% 이상 개선
                await self.convert_experience_to_learning_data('performance_opt', optimization)
        
        # AI 발견
        for discovery in self.experience_collector.ai_discoveries:
            if discovery.get('effectiveness', 0) > 0.5:
                await self.convert_experience_to_learning_data('ai_discovery', discovery)
        
        logger.info(f"동기화 완료: {self.integration_stats['total_experiences_converted']}개 경험 변환됨")
    
    async def start_realtime_processing(self):
        """실시간 처리 시작"""
        self.processing_active = True
        logger.info("🚀 실시간 학습 통합 시작")
        
        # 처리 태스크 시작
        asyncio.create_task(self._process_experience_queue())
        
        # 주기적 통합 태스크
        asyncio.create_task(self._periodic_integration())
    
    async def stop_realtime_processing(self):
        """실시간 처리 중지"""
        self.processing_active = False
        
        # 큐에 남은 경험 처리
        while not self.experience_queue.empty():
            await asyncio.sleep(0.1)
        
        # 최종 통합
        await self._integrate_with_continuous_learning()
        
        # 상태 저장
        self._save_integration_state()
        
        logger.info("🛑 실시간 학습 통합 중지")
    
    async def _process_experience_queue(self):
        """경험 큐 처리"""
        while self.processing_active:
            try:
                # 큐에서 경험 가져오기
                experience_type, experience_data = await asyncio.wait_for(
                    self.experience_queue.get(), 
                    timeout=1.0
                )
                
                # 학습 데이터로 변환
                learning_data = await self.convert_experience_to_learning_data(
                    experience_type, 
                    experience_data
                )
                
                if learning_data:
                    # Q&A 쌍 생성
                    qa_pairs = await self.generate_qa_pairs(learning_data)
                    
                    # 지식 베이스 업데이트
                    await self.update_knowledge_base(learning_data)
                    
                    # 통계 업데이트
                    self.integration_stats['total_experiences_converted'] += 1
                    self.integration_stats['qa_pairs_generated'] += len(qa_pairs)
                    
                    logger.info(f"✅ 경험 변환 완료: {experience_type} -> {len(qa_pairs)}개 Q&A 생성")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"경험 처리 오류: {e}")
    
    async def convert_experience_to_learning_data(self, experience_type: str, experience_data: Dict[str, Any]) -> Optional[LearningData]:
        """개발 경험을 학습 데이터로 변환"""
        try:
            # 변환 규칙 가져오기
            rules = self.conversion_rules.get(experience_type)
            if not rules:
                logger.warning(f"변환 규칙 없음: {experience_type}")
                return None
            
            # 카테고리 결정
            category = self.category_mapping.get(experience_type, 'core_csharp_basics')
            
            # 키워드 추출
            keywords = self._extract_keywords_from_experience(experience_type, experience_data)
            
            # 난이도 계산
            difficulty = rules['difficulty_calculator'](experience_data)
            
            # 학습 데이터 생성
            learning_data = LearningData(
                id=f"{experience_type}_{int(datetime.now().timestamp())}_{hashlib.md5(str(experience_data).encode()).hexdigest()[:8]}",
                category=category,
                topic=self._generate_topic_from_experience(experience_type, experience_data),
                question="",  # generate_qa_pairs에서 생성
                answer="",    # generate_qa_pairs에서 생성
                keywords=keywords,
                difficulty=difficulty,
                source='development',
                effectiveness=experience_data.get('effectiveness', 
                                                experience_data.get('success_rate', 
                                                                  experience_data.get('improvement', 50) / 100)),
                metadata=experience_data,
                timestamp=datetime.now().isoformat()
            )
            
            return learning_data
            
        except Exception as e:
            logger.error(f"경험 변환 오류: {e}")
            return None
    
    def _extract_keywords_from_experience(self, experience_type: str, experience_data: Dict[str, Any]) -> List[str]:
        """경험에서 키워드 추출"""
        keywords = []
        
        # 타입별 키워드 추출
        if experience_type == 'error_solution':
            keywords.extend(['오류', '해결', experience_data.get('error', {}).get('type', '오류')])
        elif experience_type == 'game_mechanic':
            keywords.extend(['게임', '메카닉', experience_data.get('name', '기능')])
        elif experience_type == 'code_pattern':
            keywords.extend(['패턴', '코드', experience_data.get('name', '패턴')])
        elif experience_type == 'performance_opt':
            keywords.extend(['성능', '최적화', experience_data.get('type', '최적화')])
        elif experience_type == 'ai_discovery':
            keywords.extend(['AI', '발견', experience_data.get('discovery_type', 'AI')])
        
        # 코드에서 추가 키워드 추출
        code = experience_data.get('code', experience_data.get('code_snippet', ''))
        if code:
            # C# 키워드 찾기
            csharp_keywords = re.findall(r'\b(class|public|private|void|int|string|bool|async|await)\b', code)
            keywords.extend(list(set(csharp_keywords)))
            
            # Godot 키워드 찾기
            godot_keywords = re.findall(r'\b(Node|GDScript|signal|export|ready|process)\b', code)
            keywords.extend(list(set(godot_keywords)))
        
        return list(set(keywords))[:10]  # 최대 10개
    
    def _generate_topic_from_experience(self, experience_type: str, experience_data: Dict[str, Any]) -> str:
        """경험에서 주제 생성"""
        if experience_type == 'error_solution':
            return f"{experience_data.get('error', {}).get('type', '오류')} 해결"
        elif experience_type == 'game_mechanic':
            return f"{experience_data.get('name', '게임 기능')} 구현"
        elif experience_type == 'code_pattern':
            return f"{experience_data.get('name', '코드 패턴')} 활용"
        elif experience_type == 'performance_opt':
            return f"{experience_data.get('type', '성능')} 최적화"
        elif experience_type == 'ai_discovery':
            return f"AI {experience_data.get('discovery_type', '발견')}"
        else:
            return "개발 경험"
    
    async def generate_qa_pairs(self, learning_data: LearningData) -> List[QAPair]:
        """학습 데이터에서 Q&A 쌍 생성"""
        qa_pairs = []
        
        # 경험 타입 결정
        experience_type = learning_data.id.split('_')[0]
        rules = self.conversion_rules.get(experience_type, {})
        
        if not rules:
            return qa_pairs
        
        # 다양한 질문 생성
        question_templates = rules.get('question_templates', [])
        answer_format = rules.get('answer_format', '')
        
        for i, template in enumerate(question_templates[:3]):  # 최대 3개 질문
            try:
                # 질문 생성
                question_text = self._format_template(template, learning_data.metadata)
                
                # 답변 생성
                answer_text = self._format_template(answer_format, learning_data.metadata)
                
                # Q&A 쌍 생성
                qa_pair = QAPair(
                    question_id=f"{learning_data.id}_q{i+1}",
                    question_text=question_text,
                    question_type=self._determine_question_type(experience_type),
                    answer_text=answer_text,
                    quality_score=learning_data.effectiveness,
                    source_experience=experience_type,
                    generated_at=datetime.now().isoformat()
                )
                
                qa_pairs.append(qa_pair)
                
                # 학습 시스템에 직접 추가
                if self.continuous_learning_system:
                    await self._add_to_learning_system(qa_pair, learning_data)
                
            except Exception as e:
                logger.error(f"Q&A 생성 오류: {e}")
        
        # Q&A 쌍 저장
        await self._save_qa_pairs(qa_pairs)
        
        return qa_pairs
    
    def _format_template(self, template: str, data: Dict[str, Any]) -> str:
        """템플릿 포맷팅"""
        # 데이터에서 변수 추출
        variables = {}
        
        # 기본 변수들
        variables.update({
            'error_type': data.get('error', {}).get('type', '일반 오류'),
            'error_description': data.get('error', {}).get('description', data.get('error', '오류')),
            'error_context': data.get('error', {}).get('context', 'Godot'),
            'solution_steps': data.get('solution', {}).get('steps', data.get('solution', '해결 방법')),
            'code_example': data.get('code', data.get('code_snippet', '// 코드 예제')),
            'explanation': data.get('explanation', data.get('description', '설명')),
            'mechanic_name': data.get('name', '게임 메카닉'),
            'mechanic_description': data.get('description', '게임 기능'),
            'pattern_name': data.get('name', '디자인 패턴'),
            'use_case': data.get('use_case', '일반적인 사용'),
            'code': data.get('code', '// 패턴 코드'),
            'effectiveness': f"{data.get('effectiveness', 1.0):.1f}",
            'applications': data.get('applications', 1),
            'considerations': data.get('considerations', '주의사항 없음'),
            'optimization_type': data.get('type', '일반'),
            'before_metrics': json.dumps(data.get('before_metrics', data.get('before', {})), ensure_ascii=False),
            'after_metrics': json.dumps(data.get('after_metrics', data.get('after', {})), ensure_ascii=False),
            'improvement': f"{data.get('improvement', 0):.1f}",
            'method': data.get('method', '최적화 방법'),
            'code_changes': data.get('code_changes', '// 변경된 코드'),
            'key_points': data.get('key_points', '핵심 포인트'),
            'discovery_type': data.get('discovery_type', data.get('type', 'general')),
            'description': data.get('description', '설명'),
            'context': data.get('context', '일반 맥락'),
            'creativity_score': data.get('creativity_score', 5),
            'usage_example': data.get('usage_example', '// 사용 예시'),
            'performance_notes': data.get('performance_notes', data.get('performance', '성능 고려사항'))
        })
        
        # 템플릿 포맷팅
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"템플릿 변수 누락: {e}")
            return template
    
    def _determine_question_type(self, experience_type: str) -> str:
        """경험 타입에서 질문 타입 결정"""
        type_mapping = {
            'error_solution': 'error',
            'game_mechanic': 'example',
            'code_pattern': 'example',
            'performance_opt': 'optimize',
            'ai_discovery': 'integrate'
        }
        return type_mapping.get(experience_type, 'explain')
    
    async def _add_to_learning_system(self, qa_pair: QAPair, learning_data: LearningData):
        """학습 시스템에 Q&A 추가"""
        if not self.continuous_learning_system:
            return
        
        try:
            # 질문 형식 맞추기
            question = {
                "id": qa_pair.question_id,
                "topic": learning_data.topic,
                "type": qa_pair.question_type,
                "language": "korean",
                "difficulty": learning_data.difficulty,
                "question": qa_pair.question_text,
                "keywords": learning_data.keywords
            }
            
            # 답변 형식 맞추기
            answer = {
                "model": "realtime_integration",
                "question_id": qa_pair.question_id,
                "answer": qa_pair.answer_text,
                "response_time": 0.1,
                "timestamp": qa_pair.generated_at
            }
            
            # 분석 결과
            analysis = {
                "success": True,
                "quality_score": qa_pair.quality_score,
                "extracted_knowledge": {
                    "source": qa_pair.source_experience,
                    "effectiveness": learning_data.effectiveness
                },
                "new_patterns": [],
                "improvements": []
            }
            
            # 학습 시스템에 저장
            self.continuous_learning_system.save_qa_pair(question, answer, analysis)
            
            logger.debug(f"학습 시스템에 Q&A 추가: {qa_pair.question_id}")
            
        except Exception as e:
            logger.error(f"학습 시스템 추가 오류: {e}")
    
    async def _save_qa_pairs(self, qa_pairs: List[QAPair]):
        """Q&A 쌍 저장"""
        if not qa_pairs:
            return
        
        # 날짜별 디렉토리
        today = datetime.now().strftime("%Y%m%d")
        qa_dir = self.integration_path / "qa_pairs" / today
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        # 배치 파일로 저장
        batch_id = f"batch_{int(datetime.now().timestamp())}"
        batch_file = qa_dir / f"{batch_id}.json"
        
        batch_data = {
            "batch_id": batch_id,
            "generated_at": datetime.now().isoformat(),
            "qa_pairs": [asdict(qa) for qa in qa_pairs],
            "count": len(qa_pairs)
        }
        
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Q&A 배치 저장: {batch_file}")
        except Exception as e:
            logger.error(f"Q&A 저장 오류: {e}")
    
    async def update_knowledge_base(self, learning_data: LearningData):
        """지식 베이스 실시간 업데이트"""
        if not self.continuous_learning_system:
            return
        
        try:
            kb = self.continuous_learning_system.knowledge_base
            
            # 카테고리별 업데이트
            if learning_data.category.startswith('core_csharp'):
                # C# 패턴 업데이트
                pattern_key = f"{learning_data.topic}_{learning_data.id}"
                kb["csharp_patterns"][pattern_key] = {
                    "topic": learning_data.topic,
                    "code": learning_data.metadata.get('code', ''),
                    "language": "korean",
                    "effectiveness": learning_data.effectiveness
                }
                
            elif learning_data.category.startswith('core_korean'):
                # 한글 번역 업데이트
                for keyword in learning_data.keywords:
                    if re.match(r'[가-힣]+', keyword):
                        kb["korean_translations"][keyword] = learning_data.metadata.get('description', '')
                
            elif learning_data.category.startswith('core_godot'):
                # Godot 통합 업데이트
                integration_key = f"{learning_data.topic}_{learning_data.id}"
                kb["godot_integrations"][integration_key] = {
                    "topic": learning_data.topic,
                    "implementation": learning_data.metadata,
                    "effectiveness": learning_data.effectiveness
                }
            
            # 공통 오류 패턴 업데이트
            if learning_data.source == 'error_solution':
                error_type = learning_data.metadata.get('error', {}).get('type', 'unknown')
                kb["common_errors"][error_type] = learning_data.metadata.get('solution', '')
            
            # 모범 사례 업데이트
            if learning_data.effectiveness > 0.8:
                practice_key = f"best_{learning_data.category}_{learning_data.id}"
                kb["best_practices"][practice_key] = {
                    "category": learning_data.category,
                    "practice": learning_data.topic,
                    "effectiveness": learning_data.effectiveness,
                    "source": learning_data.source
                }
            
            # 지식 베이스 저장
            self.continuous_learning_system._save_knowledge_base()
            
            self.integration_stats['knowledge_updates'] += 1
            logger.debug(f"지식 베이스 업데이트: {learning_data.category}")
            
        except Exception as e:
            logger.error(f"지식 베이스 업데이트 오류: {e}")
    
    async def create_specialized_training_dataset(self, category: str = None) -> Dict[str, Any]:
        """특화된 학습 데이터셋 생성"""
        dataset = {
            "name": f"game_dev_specialized_{category or 'all'}",
            "created_at": datetime.now().isoformat(),
            "category": category,
            "qa_pairs": [],
            "statistics": {}
        }
        
        # Q&A 파일들 수집
        qa_files = list(self.integration_path.glob("qa_pairs/**/*.json"))
        
        total_pairs = 0
        category_counts = defaultdict(int)
        quality_scores = []
        
        for qa_file in qa_files:
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    
                    for qa_data in batch_data.get('qa_pairs', []):
                        # 카테고리 필터링
                        if category and not qa_data.get('source_experience', '').startswith(category):
                            continue
                        
                        dataset['qa_pairs'].append(qa_data)
                        total_pairs += 1
                        category_counts[qa_data.get('source_experience', 'unknown')] += 1
                        quality_scores.append(qa_data.get('quality_score', 0))
                        
            except Exception as e:
                logger.error(f"데이터셋 파일 읽기 오류: {e}")
        
        # 통계 계산
        dataset['statistics'] = {
            "total_pairs": total_pairs,
            "category_distribution": dict(category_counts),
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "min_quality": min(quality_scores) if quality_scores else 0,
            "max_quality": max(quality_scores) if quality_scores else 0
        }
        
        # 데이터셋 저장
        dataset_file = self.integration_path / "training_datasets" / f"{dataset['name']}_{int(datetime.now().timestamp())}.json"
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            self.integration_stats['training_datasets_created'] += 1
            logger.info(f"특화 데이터셋 생성: {dataset_file.name} ({total_pairs}개 Q&A)")
            
        except Exception as e:
            logger.error(f"데이터셋 저장 오류: {e}")
        
        return dataset
    
    async def _periodic_integration(self):
        """주기적 통합 작업"""
        while self.processing_active:
            await asyncio.sleep(300)  # 5분마다
            
            try:
                # 학습 시스템과 통합
                await self._integrate_with_continuous_learning()
                
                # 특화 데이터셋 생성
                await self.create_specialized_training_dataset()
                
                # 통합 상태 저장
                self._save_integration_state()
                
                # 통합 보고서 생성
                await self._generate_integration_report()
                
            except Exception as e:
                logger.error(f"주기적 통합 오류: {e}")
    
    async def _integrate_with_continuous_learning(self):
        """연속 학습 시스템과 완전 통합"""
        if not self.continuous_learning_system:
            return
        
        logger.info("연속 학습 시스템과 통합 시작...")
        
        # 새로운 주제 추가
        new_topics_added = 0
        
        # 수집된 경험에서 새로운 학습 주제 생성
        if self.experience_collector:
            # 자주 발생하는 오류 패턴을 새 주제로
            for error_hash, solutions in self.experience_collector.error_solutions.items():
                if len(solutions) > 10:  # 10번 이상 발생
                    topic_id = f"frequent_error_{error_hash}"
                    if not any(t.id == topic_id for t in self.continuous_learning_system.learning_topics):
                        # 새 주제 추가
                        from continuous_learning_system import LearningTopic
                        new_topic = LearningTopic(
                            id=topic_id,
                            category="자주 발생하는 오류",
                            topic=f"오류 패턴 {error_hash}",
                            difficulty=3,
                            korean_keywords=["오류", "해결", "패턴"],
                            csharp_concepts=["error", "exception", "handling"],
                            godot_integration="Godot 오류 처리"
                        )
                        self.continuous_learning_system.learning_topics.append(new_topic)
                        new_topics_added += 1
        
        # 학습 진행 상태 업데이트
        if hasattr(self.continuous_learning_system, 'progressive_manager') and self.continuous_learning_system.progressive_manager:
            # 실제 개발 경험 기반으로 난이도 조정
            insights = self.experience_collector.get_learning_insights() if self.experience_collector else {}
            
            if insights.get('success_rate', 0) > 0.8:
                # 성공률이 높으면 난이도 상승
                current = self.continuous_learning_system.progressive_manager.progress['current_difficulty']
                if current < 5:
                    self.continuous_learning_system.progressive_manager.progress['current_difficulty'] = current + 1
                    logger.info(f"난이도 상승: {current} -> {current + 1}")
        
        logger.info(f"통합 완료: {new_topics_added}개 새 주제 추가")
    
    async def _generate_integration_report(self):
        """통합 보고서 생성"""
        report_dir = self.integration_path / "reports"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 경험 수집기 인사이트
        experience_insights = self.experience_collector.get_learning_insights() if self.experience_collector else {}
        
        report = f"""# 실시간 학습 통합 보고서

## 📊 통합 통계
- 변환된 경험: {self.integration_stats['total_experiences_converted']}
- 생성된 Q&A 쌍: {self.integration_stats['qa_pairs_generated']}
- 지식 베이스 업데이트: {self.integration_stats['knowledge_updates']}
- 생성된 학습 데이터셋: {self.integration_stats['training_datasets_created']}
- 마지막 통합: {self.integration_stats.get('last_integration', 'N/A')}

## 💡 경험 수집 인사이트
"""
        
        if experience_insights:
            report += f"""- 총 경험: {experience_insights.get('total_experiences', 0)}
- 성공률: {experience_insights.get('success_rate', 0):.1%}
- AI 창의성 점수: {experience_insights.get('ai_creativity_score', 0):.1f}/10
- 커뮤니티 기여: {experience_insights.get('community_contribution', 0)}

### 가장 효과적인 전략
"""
            for strategy, score in experience_insights.get('most_effective_strategies', [])[:5]:
                report += f"- {strategy}: {score:.1f}점\n"
            
            report += "\n### 일반적인 패턴\n"
            for pattern, freq in experience_insights.get('common_patterns', [])[:10]:
                report += f"- {pattern}: {freq}회\n"
        
        report += f"""
## 🎯 학습 효과
- 새로운 오류 해결 패턴: {len(self.experience_collector.error_solutions) if self.experience_collector else 0}
- 성공적인 게임 메카닉: {len(self.experience_collector.successful_mechanics) if self.experience_collector else 0}
- 발견된 코드 패턴: {len(self.experience_collector.code_patterns) if self.experience_collector else 0}

## 📈 품질 지표
"""
        
        # Q&A 품질 분석
        qa_files = list(self.integration_path.glob("qa_pairs/**/*.json"))
        quality_scores = []
        
        for qa_file in qa_files[-10:]:  # 최근 10개 파일
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    for qa in batch_data.get('qa_pairs', []):
                        quality_scores.append(qa.get('quality_score', 0))
            except:
                pass
        
        if quality_scores:
            report += f"""- 평균 Q&A 품질 점수: {sum(quality_scores)/len(quality_scores):.2f}
- 최고 품질 점수: {max(quality_scores):.2f}
- 최저 품질 점수: {min(quality_scores):.2f}
"""
        
        report += f"""
## 🔄 다음 단계
1. 더 많은 개발 경험 수집
2. Q&A 품질 개선
3. 특화 모델 훈련
4. 실시간 피드백 강화

---
생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"통합 보고서 생성: {report_file}")
        except Exception as e:
            logger.error(f"보고서 생성 오류: {e}")
    
    async def on_new_experience(self, experience_type: str, experience_data: Dict[str, Any]):
        """새로운 경험 수신 (외부 연동용)"""
        if not self.processing_active:
            logger.warning("실시간 처리가 비활성화 상태입니다")
            return
        
        try:
            # 큐에 추가
            await self.experience_queue.put((experience_type, experience_data))
            logger.debug(f"새 경험 큐에 추가: {experience_type}")
        except asyncio.QueueFull:
            logger.error("경험 큐가 가득 참")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 반환"""
        return {
            "active": self.processing_active,
            "stats": self.integration_stats,
            "queue_size": self.experience_queue.qsize() if self.experience_queue else 0,
            "connected_systems": {
                "continuous_learning": self.continuous_learning_system is not None,
                "experience_collector": self.experience_collector is not None,
                "ai_controller": self.ai_model_controller is not None
            },
            "last_report": max(
                [f for f in (self.integration_path / "reports").glob("*.md")],
                key=lambda x: x.stat().st_mtime,
                default=None
            ) if (self.integration_path / "reports").exists() else None
        }

# 싱글톤 인스턴스
_integrator_instance = None

def get_realtime_integrator():
    """실시간 통합기 인스턴스 반환"""
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = RealtimeLearningIntegrator()
    return _integrator_instance

# 간편 사용을 위한 함수들
async def start_integration(continuous_learning_system=None, experience_collector=None, ai_controller=None):
    """통합 시작"""
    integrator = get_realtime_integrator()
    await integrator.connect_systems(continuous_learning_system, experience_collector, ai_controller)
    await integrator.start_realtime_processing()
    return integrator

async def stop_integration():
    """통합 중지"""
    integrator = get_realtime_integrator()
    await integrator.stop_realtime_processing()

async def add_experience(experience_type: str, experience_data: Dict[str, Any]):
    """경험 추가"""
    integrator = get_realtime_integrator()
    await integrator.on_new_experience(experience_type, experience_data)

def get_status():
    """상태 조회"""
    integrator = get_realtime_integrator()
    return integrator.get_integration_status()