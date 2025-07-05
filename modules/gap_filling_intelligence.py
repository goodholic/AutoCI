"""
AutoCI 격차 보완 지능 시스템
autoci learn과 autoci create에서 부족한 부분을 자동으로 감지하고 메꿔주는 핵심 모듈
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
try:
    import numpy as np
except ImportError:
    # numpy가 없을 때 대체 구현
    class np:
        @staticmethod
        def random():
            import random
            class Random:
                @staticmethod
                def randint(a, b):
                    return random.randint(a, b)
                @staticmethod
                def uniform(a, b):
                    return random.uniform(a, b)
            return Random()
        
        @staticmethod
        def var(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            return sum((x - mean) ** 2 for x in data) / len(data)
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGap:
    """지식 격차 정보"""
    category: str
    severity: float  # 0.0 ~ 1.0
    description: str
    suggested_actions: List[str]
    search_keywords: List[str]
    priority: str  # 'low', 'medium', 'high', 'critical'
    detected_at: datetime
    auto_fix_possible: bool

@dataclass
class LearningDeficiency:
    """학습 부족 부분"""
    skill_area: str
    deficiency_score: float
    evidence: List[str]
    recommended_resources: List[str]
    estimated_learning_time: int  # minutes
    
class GapFillingIntelligence:
    """격차 보완 지능 시스템"""
    
    def __init__(self):
        self.gaps_dir = Path("experiences/knowledge_gaps")
        self.gaps_dir.mkdir(parents=True, exist_ok=True)
        
        self.filled_gaps_dir = Path("experiences/filled_gaps")
        self.filled_gaps_dir.mkdir(parents=True, exist_ok=True)
        
        # 지식 영역 정의
        self.knowledge_areas = {
            "csharp_basics": {
                "keywords": ["class", "method", "property", "namespace", "using"],
                "min_threshold": 10,
                "importance": 0.9
            },
            "csharp_advanced": {
                "keywords": ["async", "await", "generic", "linq", "delegate"],
                "min_threshold": 5,
                "importance": 0.8
            },
            "godot_basics": {
                "keywords": ["node", "scene", "signal", "script", "_ready"],
                "min_threshold": 15,
                "importance": 0.9
            },
            "godot_advanced": {
                "keywords": ["shader", "gdnative", "multiplayer", "physics", "animation"],
                "min_threshold": 8,
                "importance": 0.7
            },
            "socketio_networking": {
                "keywords": ["socket", "emit", "on", "connect", "disconnect"],
                "min_threshold": 6,
                "importance": 0.6
            },
            "pytorch_ai": {
                "keywords": ["tensor", "model", "train", "loss", "optimizer"],
                "min_threshold": 5,
                "importance": 0.7
            },
            "game_design": {
                "keywords": ["gameplay", "mechanics", "level", "ui", "player"],
                "min_threshold": 12,
                "importance": 0.8
            }
        }
        
        # 자동 수정 가능한 문제 패턴
        self.auto_fixable_patterns = {
            "missing_using_statements": {
                "pattern": r"The type or namespace name '(\w+)' could not be found",
                "fix_template": "using {namespace};"
            },
            "missing_async_await": {
                "pattern": r"Cannot await|async method",
                "fix_template": "async/await 패턴 적용 필요"
            },
            "godot_node_access": {
                "pattern": r"GetNode.*null|Node not found",
                "fix_template": "노드 경로 확인 및 null 체크 추가"
            }
        }
        
        logger.info("🎯 격차 보완 지능 시스템 초기화 완료")
    
    async def analyze_comprehensive_gaps(self) -> List[KnowledgeGap]:
        """종합적인 지식 격차 분석"""
        logger.info("🔍 종합적인 지식 격차 분석 시작")
        
        gaps = []
        
        # 1. 학습 데이터 기반 격차 분석
        learning_gaps = await self._analyze_learning_data_gaps()
        gaps.extend(learning_gaps)
        
        # 2. 프로젝트 생성 결과 기반 격차 분석
        project_gaps = await self._analyze_project_creation_gaps()
        gaps.extend(project_gaps)
        
        # 3. 오류 패턴 기반 격차 분석
        error_gaps = await self._analyze_error_pattern_gaps()
        gaps.extend(error_gaps)
        
        # 4. 시간 효율성 기반 격차 분석
        efficiency_gaps = await self._analyze_efficiency_gaps()
        gaps.extend(efficiency_gaps)
        
        # 격차 우선순위 정렬
        gaps.sort(key=lambda g: (g.severity, g.priority), reverse=True)
        
        logger.info(f"🎯 총 {len(gaps)}개의 지식 격차 감지됨")
        
        # 격차 정보 저장
        await self._save_gap_analysis(gaps)
        
        return gaps
    
    async def _analyze_learning_data_gaps(self) -> List[KnowledgeGap]:
        """학습 데이터 기반 격차 분석"""
        gaps = []
        
        try:
            # 연속 학습 데이터 분석
            continuous_learning_dir = Path("continuous_learning")
            if not continuous_learning_dir.exists():
                return gaps
            
            # 각 지식 영역별 데이터 수집
            area_data = defaultdict(list)
            
            for file_path in continuous_learning_dir.rglob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = str(data).lower()
                        
                        # 각 지식 영역에 매칭
                        for area, config in self.knowledge_areas.items():
                            keyword_count = sum(1 for keyword in config["keywords"] 
                                             if keyword in content)
                            if keyword_count > 0:
                                area_data[area].append({
                                    'file': str(file_path),
                                    'keyword_count': keyword_count,
                                    'content_length': len(content)
                                })
                except:
                    continue
            
            # 격차 분석
            for area, config in self.knowledge_areas.items():
                data_points = area_data.get(area, [])
                total_keywords = sum(d['keyword_count'] for d in data_points)
                
                if total_keywords < config["min_threshold"]:
                    severity = 1.0 - (total_keywords / config["min_threshold"])
                    severity = min(severity * config["importance"], 1.0)
                    
                    gap = KnowledgeGap(
                        category=f"learning_data_{area}",
                        severity=severity,
                        description=f"{area} 영역의 학습 데이터 부족 (현재: {total_keywords}, 필요: {config['min_threshold']})",
                        suggested_actions=[
                            f"{area} 관련 autoci learn 세션 증가",
                            f"{area} 전용 학습 자료 검색",
                            f"{area} 실습 프로젝트 생성"
                        ],
                        search_keywords=config["keywords"],
                        priority="high" if severity > 0.7 else "medium" if severity > 0.4 else "low",
                        detected_at=datetime.now(),
                        auto_fix_possible=True
                    )
                    gaps.append(gap)
        
        except Exception as e:
            logger.error(f"학습 데이터 격차 분석 오류: {e}")
        
        return gaps
    
    async def _analyze_project_creation_gaps(self) -> List[KnowledgeGap]:
        """프로젝트 생성 결과 기반 격차 분석"""
        gaps = []
        
        try:
            game_projects_dir = Path("game_projects")
            if not game_projects_dir.exists():
                return gaps
            
            # 최근 프로젝트 분석
            recent_projects = []
            one_week_ago = datetime.now() - timedelta(days=7)
            
            for project_dir in game_projects_dir.iterdir():
                if project_dir.is_dir():
                    creation_time = datetime.fromtimestamp(project_dir.stat().st_ctime)
                    if creation_time > one_week_ago:
                        recent_projects.append(project_dir)
            
            if len(recent_projects) < 3:
                gap = KnowledgeGap(
                    category="project_creation_frequency",
                    severity=0.8,
                    description="최근 1주일간 생성된 프로젝트가 부족함",
                    suggested_actions=[
                        "더 자주 autoci create 실행",
                        "다양한 게임 타입 실험",
                        "프로젝트 생성 자동화 고려"
                    ],
                    search_keywords=["game development", "project templates", "rapid prototyping"],
                    priority="high",
                    detected_at=datetime.now(),
                    auto_fix_possible=True
                )
                gaps.append(gap)
            
            # 프로젝트 품질 분석
            failed_projects = 0
            for project_dir in recent_projects:
                config_file = project_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            if config.get('status') == 'failed' or config.get('quality_score', 0) < 0.5:
                                failed_projects += 1
                    except:
                        failed_projects += 1
                else:
                    failed_projects += 1
            
            if failed_projects > len(recent_projects) * 0.5:  # 50% 이상 실패
                gap = KnowledgeGap(
                    category="project_quality",
                    severity=0.9,
                    description=f"프로젝트 실패율이 높음 ({failed_projects}/{len(recent_projects)})",
                    suggested_actions=[
                        "기초 학습 강화",
                        "단순한 프로젝트부터 시작",
                        "오류 패턴 분석 및 해결"
                    ],
                    search_keywords=["game development best practices", "common errors", "debugging"],
                    priority="critical",
                    detected_at=datetime.now(),
                    auto_fix_possible=False
                )
                gaps.append(gap)
        
        except Exception as e:
            logger.error(f"프로젝트 격차 분석 오류: {e}")
        
        return gaps
    
    async def _analyze_error_pattern_gaps(self) -> List[KnowledgeGap]:
        """오류 패턴 기반 격차 분석"""
        gaps = []
        
        try:
            # 경험 데이터에서 오류 패턴 수집
            experiences_dir = Path("experiences")
            error_patterns = Counter()
            
            for file_path in experiences_dir.rglob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = str(data).lower()
                        
                        # 오류 키워드 검색
                        if any(keyword in content for keyword in ['error', 'exception', 'failed', 'null']):
                            # 구체적인 오류 패턴 식별
                            if 'null reference' in content or 'nullreferenceexception' in content:
                                error_patterns['null_reference'] += 1
                            if 'compile' in content and 'error' in content:
                                error_patterns['compilation'] += 1
                            if 'socket' in content and 'error' in content:
                                error_patterns['networking'] += 1
                            if 'godot' in content and 'error' in content:
                                error_patterns['godot_api'] += 1
                            if 'async' in content and 'error' in content:
                                error_patterns['async_await'] += 1
                except:
                    continue
            
            # 빈번한 오류에 대한 격차 생성
            for error_type, count in error_patterns.items():
                if count >= 3:  # 3회 이상 발생한 오류
                    severity = min(count / 10.0, 1.0)  # 최대 10회를 1.0으로 정규화
                    
                    gap = KnowledgeGap(
                        category=f"error_pattern_{error_type}",
                        severity=severity,
                        description=f"{error_type} 오류가 빈번히 발생함 ({count}회)",
                        suggested_actions=[
                            f"{error_type} 오류 해결 방법 학습",
                            f"{error_type} 예방 패턴 습득",
                            f"{error_type} 관련 모범 사례 검색"
                        ],
                        search_keywords=[error_type, "solution", "best practices"],
                        priority="high" if count >= 5 else "medium",
                        detected_at=datetime.now(),
                        auto_fix_possible=error_type in [p.split('_')[0] for p in self.auto_fixable_patterns.keys()]
                    )
                    gaps.append(gap)
        
        except Exception as e:
            logger.error(f"오류 패턴 격차 분석 오류: {e}")
        
        return gaps
    
    async def _analyze_efficiency_gaps(self) -> List[KnowledgeGap]:
        """시간 효율성 기반 격차 분석"""
        gaps = []
        
        try:
            # 학습 세션 시간 분석
            continuous_learning_dir = Path("continuous_learning")
            if not continuous_learning_dir.exists():
                return gaps
            
            session_times = []
            recent_files = []
            
            # 최근 세션들의 시간 분석
            for file_path in continuous_learning_dir.rglob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time > datetime.now() - timedelta(days=3):  # 최근 3일
                        recent_files.append((file_path, file_time))
                except:
                    continue
            
            # 세션 간격 분석
            recent_files.sort(key=lambda x: x[1])
            
            if len(recent_files) >= 2:
                intervals = []
                for i in range(1, len(recent_files)):
                    interval = (recent_files[i][1] - recent_files[i-1][1]).total_seconds() / 3600  # hours
                    intervals.append(interval)
                
                avg_interval = sum(intervals) / len(intervals)
                
                # 학습 간격이 너무 길면 격차로 간주
                if avg_interval > 24:  # 24시간 이상
                    gap = KnowledgeGap(
                        category="learning_frequency",
                        severity=min(avg_interval / 48, 1.0),  # 48시간을 최대로 정규화
                        description=f"학습 세션 간격이 너무 김 (평균 {avg_interval:.1f}시간)",
                        suggested_actions=[
                            "더 자주 autoci learn 실행",
                            "자동 학습 스케줄링 활성화",
                            "짧은 세션으로 학습 빈도 증가"
                        ],
                        search_keywords=["continuous learning", "spaced repetition", "learning schedule"],
                        priority="medium",
                        detected_at=datetime.now(),
                        auto_fix_possible=True
                    )
                    gaps.append(gap)
                
                # 학습 시간 일관성 분석
                if len(intervals) >= 3:
                    interval_variance = np.var(intervals)
                    if interval_variance > 100:  # 높은 분산
                        gap = KnowledgeGap(
                            category="learning_consistency",
                            severity=min(interval_variance / 200, 1.0),
                            description="학습 패턴이 일관성이 없음",
                            suggested_actions=[
                                "정기적인 학습 스케줄 설정",
                                "알림 시스템 활용",
                                "학습 루틴 개발"
                            ],
                            search_keywords=["learning habits", "consistent study", "routine"],
                            priority="low",
                            detected_at=datetime.now(),
                            auto_fix_possible=True
                        )
                        gaps.append(gap)
        
        except Exception as e:
            logger.error(f"효율성 격차 분석 오류: {e}")
        
        return gaps
    
    async def auto_fill_gaps(self, gaps: List[KnowledgeGap]) -> Dict[str, Any]:
        """자동으로 메꿀 수 있는 격차들을 처리"""
        logger.info("🔧 자동 격차 보완 시작")
        
        filled_count = 0
        auto_actions = []
        manual_actions = []
        
        for gap in gaps:
            if gap.auto_fix_possible:
                try:
                    success = await self._attempt_auto_fix(gap)
                    if success:
                        filled_count += 1
                        auto_actions.append({
                            'gap': gap.category,
                            'action': 'auto_fixed',
                            'timestamp': datetime.now().isoformat()
                        })
                        logger.info(f"✅ 자동 보완 성공: {gap.category}")
                    else:
                        manual_actions.append({
                            'gap': gap.category,
                            'reason': 'auto_fix_failed',
                            'suggested_actions': gap.suggested_actions
                        })
                except Exception as e:
                    logger.error(f"자동 보완 실패 {gap.category}: {e}")
                    manual_actions.append({
                        'gap': gap.category,
                        'reason': 'exception',
                        'error': str(e)
                    })
            else:
                manual_actions.append({
                    'gap': gap.category,
                    'reason': 'manual_intervention_required',
                    'suggested_actions': gap.suggested_actions
                })
        
        result = {
            'total_gaps': len(gaps),
            'auto_filled': filled_count,
            'auto_actions': auto_actions,
            'manual_actions': manual_actions,
            'success_rate': filled_count / len(gaps) if gaps else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # 결과 저장
        result_file = self.filled_gaps_dir / f"auto_fill_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎯 자동 격차 보완 완료: {filled_count}/{len(gaps)} 성공")
        
        return result
    
    async def _attempt_auto_fix(self, gap: KnowledgeGap) -> bool:
        """개별 격차 자동 보완 시도"""
        try:
            if gap.category.startswith('learning_data_'):
                # 학습 데이터 부족 -> 검색 키워드 추가
                from modules.intelligent_guardian_system import get_guardian_system
                guardian = get_guardian_system()
                
                for keyword in gap.search_keywords:
                    await guardian.search_queue.put(f"{keyword} 고급 기법")
                
                return True
                
            elif gap.category == 'project_creation_frequency':
                # 프로젝트 생성 빈도 부족 -> 자동 생성 제안
                suggestion_file = self.filled_gaps_dir / "auto_create_suggestion.json"
                suggestion = {
                    'timestamp': datetime.now().isoformat(),
                    'suggestion': 'autoci create 자동 실행 고려',
                    'recommended_types': ['platformer', 'puzzle', 'rpg'],
                    'schedule': 'daily'
                }
                
                with open(suggestion_file, 'w', encoding='utf-8') as f:
                    json.dump(suggestion, f, indent=2, ensure_ascii=False)
                
                return True
                
            elif gap.category == 'learning_frequency':
                # 학습 빈도 부족 -> 스케줄링 제안
                schedule_file = self.filled_gaps_dir / "learning_schedule.json"
                schedule = {
                    'timestamp': datetime.now().isoformat(),
                    'recommended_interval': '12 hours',
                    'auto_reminder': True,
                    'suggested_times': ['09:00', '21:00']
                }
                
                with open(schedule_file, 'w', encoding='utf-8') as f:
                    json.dump(schedule, f, indent=2, ensure_ascii=False)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"자동 보완 시도 오류: {e}")
            return False
    
    async def _save_gap_analysis(self, gaps: List[KnowledgeGap]):
        """격차 분석 결과 저장"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'total_gaps': len(gaps),
            'critical_gaps': len([g for g in gaps if g.priority == 'critical']),
            'high_priority_gaps': len([g for g in gaps if g.priority == 'high']),
            'auto_fixable_gaps': len([g for g in gaps if g.auto_fix_possible]),
            'gaps': [
                {
                    'category': gap.category,
                    'severity': gap.severity,
                    'description': gap.description,
                    'suggested_actions': gap.suggested_actions,
                    'search_keywords': gap.search_keywords,
                    'priority': gap.priority,
                    'detected_at': gap.detected_at.isoformat(),
                    'auto_fix_possible': gap.auto_fix_possible
                }
                for gap in gaps
            ]
        }
        
        analysis_file = self.gaps_dir / f"gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 격차 분석 결과 저장: {analysis_file}")
    
    async def generate_learning_recommendations(self, gaps: List[KnowledgeGap]) -> Dict[str, Any]:
        """격차 기반 학습 권장사항 생성"""
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'resource_suggestions': [],
            'priority_order': []
        }
        
        # 우선순위별 분류
        critical_gaps = [g for g in gaps if g.priority == 'critical']
        high_gaps = [g for g in gaps if g.priority == 'high']
        medium_gaps = [g for g in gaps if g.priority == 'medium']
        
        # 즉시 조치 (Critical)
        for gap in critical_gaps:
            recommendations['immediate_actions'].extend(gap.suggested_actions)
        
        # 단기 목표 (High)
        for gap in high_gaps:
            recommendations['short_term_goals'].append({
                'goal': f"{gap.category} 개선",
                'actions': gap.suggested_actions,
                'keywords': gap.search_keywords
            })
        
        # 장기 목표 (Medium)
        for gap in medium_gaps:
            recommendations['long_term_goals'].append({
                'goal': f"{gap.category} 완성",
                'description': gap.description
            })
        
        # 리소스 제안
        all_keywords = set()
        for gap in gaps:
            all_keywords.update(gap.search_keywords)
        
        recommendations['resource_suggestions'] = [
            f"{keyword} 튜토리얼" for keyword in list(all_keywords)[:10]
        ]
        
        # 우선순위 순서
        recommendations['priority_order'] = [
            f"{gap.priority}: {gap.category}" for gap in gaps[:10]
        ]
        
        return recommendations

# 싱글톤 인스턴스
_gap_intelligence = None

def get_gap_filling_intelligence() -> GapFillingIntelligence:
    """격차 보완 지능 시스템 싱글톤 반환"""
    global _gap_intelligence
    if _gap_intelligence is None:
        _gap_intelligence = GapFillingIntelligence()
    return _gap_intelligence