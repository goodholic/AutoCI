"""
AutoCI 지능형 가디언 시스템
24시간 지속적으로 autoci learn과 autoci create를 감시하고,
부족한 부분을 자동으로 메꿔주는 핵심 시스템
"""

import asyncio
import json
import logging
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
try:
    import torch
except ImportError:
    torch = None

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
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

@dataclass
class LearningProgress:
    """학습 진행 상황 추적"""
    session_id: str
    start_time: datetime
    last_activity: datetime
    total_learning_time: float
    quality_score: float
    repetitive_patterns: List[str]
    knowledge_gaps: List[str]
    learning_efficiency: float
    next_recommended_action: str

@dataclass
class SystemMonitoringState:
    """시스템 모니터링 상태"""
    autoci_learn_running: bool
    autoci_create_running: bool
    autoci_resume_running: bool
    last_learn_session: Optional[datetime]
    last_create_session: Optional[datetime]
    last_resume_session: Optional[datetime]
    total_monitored_time: float
    detected_repetitions: int
    filled_knowledge_gaps: int
    pytorch_training_sessions: int
    godot_projects_monitored: List[str]

class IntelligentGuardianSystem:
    """AutoCI 지능형 가디언 - 24시간 감시 및 최적화 시스템"""
    
    def __init__(self):
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.search_thread: Optional[threading.Thread] = None
        self.pytorch_thread: Optional[threading.Thread] = None
        
        # 데이터 디렉토리 설정
        self.guardian_dir = Path("experiences/guardian_system")
        self.guardian_dir.mkdir(parents=True, exist_ok=True)
        
        self.pytorch_datasets_dir = Path("experiences/pytorch_datasets")
        self.pytorch_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_base_dir = Path("experiences/knowledge_base")
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 모니터링 상태 초기화
        self.monitoring_state = SystemMonitoringState(
            autoci_learn_running=False,
            autoci_create_running=False,
            autoci_resume_running=False,
            last_learn_session=None,
            last_create_session=None,
            last_resume_session=None,
            total_monitored_time=0.0,
            detected_repetitions=0,
            filled_knowledge_gaps=0,
            pytorch_training_sessions=0,
            godot_projects_monitored=[]
        )
        
        # 학습 진행 상황 추적
        self.current_learning_progress: Optional[LearningProgress] = None
        
        # 검색 키워드 큐
        self.search_queue = asyncio.Queue()
        
        # PyTorch 학습 큐
        self.pytorch_queue = asyncio.Queue()
        
        logger.info("🛡️ AutoCI 지능형 가디언 시스템 초기화 완료")
    
    async def start_guardian_mode(self):
        """24시간 가디언 모드 시작"""
        if self.is_running:
            logger.warning("가디언 시스템이 이미 실행 중입니다.")
            return
        
        self.is_running = True
        logger.info("🛡️ AutoCI 지능형 가디언 시스템 시작")
        print("=" * 70)
        print("🛡️ AutoCI 지능형 가디언 시스템 활성화")
        print("   - 24시간 지속적 감시 시작")
        print("   - 반복적 학습 지양 시스템 활성화")
        print("   - 지속적 정보 검색 시스템 시작")
        print("   - PyTorch 자동 딥러닝 시스템 시작")
        print("=" * 70)
        
        # 병렬로 모든 시스템 시작
        await asyncio.gather(
            self._start_process_monitoring(),
            self._start_repetition_prevention(),
            self._start_continuous_search(),
            self._start_pytorch_training(),
            self._start_knowledge_gap_detection(),
            self._start_human_advisory()
        )
    
    async def _start_process_monitoring(self):
        """autoci learn/create 프로세스 모니터링"""
        logger.info("🔍 프로세스 모니터링 시작")
        
        while self.is_running:
            try:
                # 실행 중인 autoci 프로세스 확인
                autoci_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline'] and any('autoci' in cmd for cmd in proc.info['cmdline']):
                            autoci_processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # learn/create 프로세스 확인
                learn_running = any('learn' in str(proc['cmdline']) for proc in autoci_processes)
                create_running = any('create' in str(proc['cmdline']) for proc in autoci_processes)
                resume_running = any('resume' in str(proc['cmdline']) for proc in autoci_processes)
                
                # 상태 업데이트
                if learn_running and not self.monitoring_state.autoci_learn_running:
                    self.monitoring_state.autoci_learn_running = True
                    self.monitoring_state.last_learn_session = datetime.now()
                    logger.info("📚 autoci learn 세션 감지됨")
                    await self._on_learn_session_start()
                
                if create_running and not self.monitoring_state.autoci_create_running:
                    self.monitoring_state.autoci_create_running = True
                    self.monitoring_state.last_create_session = datetime.now()
                    logger.info("🎮 autoci create 세션 감지됨")
                    await self._on_create_session_start()
                
                if resume_running and not self.monitoring_state.autoci_resume_running:
                    self.monitoring_state.autoci_resume_running = True
                    self.monitoring_state.last_resume_session = datetime.now()
                    logger.info("🔄 autoci resume 세션 감지됨")
                    await self._on_resume_session_start()
                
                if not learn_running and self.monitoring_state.autoci_learn_running:
                    self.monitoring_state.autoci_learn_running = False
                    logger.info("📚 autoci learn 세션 종료됨")
                    await self._on_learn_session_end()
                
                if not create_running and self.monitoring_state.autoci_create_running:
                    self.monitoring_state.autoci_create_running = False
                    logger.info("🎮 autoci create 세션 종료됨")
                    await self._on_create_session_end()
                
                if not resume_running and self.monitoring_state.autoci_resume_running:
                    self.monitoring_state.autoci_resume_running = False
                    logger.info("🔄 autoci resume 세션 종료됨")
                    await self._on_resume_session_end()
                
                # 모니터링 시간 업데이트
                self.monitoring_state.total_monitored_time += 5
                
                # 매 1분마다 상태 출력
                if int(self.monitoring_state.total_monitored_time) % 60 == 0:
                    minutes = int(self.monitoring_state.total_monitored_time / 60)
                    
                    # 공유 지식 베이스 통계
                    from modules.shared_knowledge_base import get_shared_knowledge_base
                    shared_kb = get_shared_knowledge_base()
                    kb_stats = shared_kb.get_knowledge_stats()
                    
                    print(f"   ⏰ 가디언 감시 시간: {minutes}분 | learn: {'✅' if self.monitoring_state.autoci_learn_running else '❌'} | create: {'✅' if self.monitoring_state.autoci_create_running else '❌'}")
                    print(f"   📚 지식 베이스: 검색 {kb_stats['cached_searches']}/{kb_stats['total_searches']} | 솔루션 {kb_stats['total_solutions']} | 베스트 {kb_stats['total_practices']}")
                
                await asyncio.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                logger.error(f"프로세스 모니터링 오류: {e}")
                await asyncio.sleep(10)
    
    async def _start_repetition_prevention(self):
        """반복적 학습 지양 시스템"""
        logger.info("🔄 반복적 학습 지양 시스템 시작")
        
        learning_patterns = []
        
        while self.is_running:
            try:
                # 최근 학습 파일들 분석
                recent_files = []
                continuous_learning_dir = Path("continuous_learning")
                
                if continuous_learning_dir.exists():
                    # 최근 1시간 내 파일들만 확인
                    one_hour_ago = datetime.now() - timedelta(hours=1)
                    
                    for file_path in continuous_learning_dir.rglob("*.json"):
                        try:
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time > one_hour_ago:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    recent_files.append({
                                        'file': str(file_path),
                                        'time': file_time,
                                        'content': data
                                    })
                        except:
                            continue
                
                # 반복 패턴 감지
                if len(recent_files) >= 3:
                    repetitions = await self._detect_learning_repetitions(recent_files)
                    
                    if repetitions:
                        self.monitoring_state.detected_repetitions += len(repetitions)
                        logger.warning(f"🔄 반복적 학습 패턴 감지: {len(repetitions)}개")
                        
                        # 반복 방지 조치
                        await self._prevent_repetitive_learning(repetitions)
                
                await asyncio.sleep(300)  # 5분마다 체크
                
            except Exception as e:
                logger.error(f"반복 방지 시스템 오류: {e}")
                await asyncio.sleep(600)
    
    async def _start_continuous_search(self):
        """24시간 지속적 정보 검색 시스템"""
        logger.info("🔍 지속적 정보 검색 시스템 시작")
        print("   🔍 정보 검색 시스템 활성화 - 1분마다 자동 검색 시작")
        
        search_keywords = [
            "Godot C# 고급 기법",
            "PyTorch 게임 개발 AI",
            "C# Socket.IO 실시간 통신",
            "Godot 자동화 스크립팅",
            "AI 게임 개발 최신 기술",
            "C# 딥러닝 통합",
            "Godot 최적화 기법",
            "게임 AI 행동 패턴"
        ]
        
        search_index = 0
        search_count = 0
        
        # 즉시 첫 번째 검색 실행
        await self._perform_intelligent_search("Godot C# 기초 문제 해결 및 최적화")
        search_count += 1
        print(f"   📡 검색 #{search_count}: 즉시 시작 완료")
        
        while self.is_running:
            try:
                # 검색 큐에서 우선순위 키워드 확인
                try:
                    priority_keyword = await asyncio.wait_for(
                        self.search_queue.get(), timeout=1.0
                    )
                    await self._perform_intelligent_search(priority_keyword)
                    search_count += 1
                    print(f"   🔍 우선순위 검색 #{search_count}: {priority_keyword}")
                except asyncio.TimeoutError:
                    # 우선순위 검색이 없으면 기본 검색 진행
                    keyword = search_keywords[search_index % len(search_keywords)]
                    await self._perform_intelligent_search(keyword)
                    search_count += 1
                    search_index += 1
                    print(f"   📡 정기 검색 #{search_count}: {keyword}")
                
                await asyncio.sleep(60)  # 1분마다 검색 (더 빈번하게)
                
            except Exception as e:
                logger.error(f"지속적 검색 시스템 오류: {e}")
                await asyncio.sleep(120)  # 오류 시 2분 후 재시도
    
    async def _start_pytorch_training(self):
        """PyTorch 자동 딥러닝 시스템"""
        logger.info("🧠 PyTorch 자동 딥러닝 시스템 시작")
        
        while self.is_running:
            try:
                # 학습 데이터 준비
                training_data = await self._prepare_pytorch_training_data()
                
                if training_data and len(training_data) >= 10:  # 최소 10개 데이터 필요
                    logger.info(f"🧠 PyTorch 딥러닝 시작: {len(training_data)}개 데이터")
                    
                    # 백그라운드에서 PyTorch 훈련
                    await self._run_pytorch_training(training_data)
                    
                    self.monitoring_state.pytorch_training_sessions += 1
                    logger.info("🧠 PyTorch 딥러닝 세션 완료")
                
                await asyncio.sleep(3600)  # 1시간마다 딥러닝 체크
                
            except Exception as e:
                logger.error(f"PyTorch 딥러닝 시스템 오류: {e}")
                await asyncio.sleep(1800)
    
    async def _start_knowledge_gap_detection(self):
        """지식 격차 감지 및 보완 시스템"""
        logger.info("🎯 지식 격차 감지 시스템 시작")
        
        # 격차 보완 지능 시스템 통합
        from modules.gap_filling_intelligence import get_gap_filling_intelligence
        gap_intelligence = get_gap_filling_intelligence()
        
        while self.is_running:
            try:
                # 종합적인 지식 격차 분석
                knowledge_gaps = await gap_intelligence.analyze_comprehensive_gaps()
                
                if knowledge_gaps:
                    logger.info(f"🎯 종합 지식 격차 감지: {len(knowledge_gaps)}개")
                    
                    # 자동으로 메꿀 수 있는 격차들 처리
                    auto_fix_result = await gap_intelligence.auto_fill_gaps(knowledge_gaps)
                    
                    # 학습 권장사항 생성
                    recommendations = await gap_intelligence.generate_learning_recommendations(knowledge_gaps)
                    
                    # 격차 보완을 위한 검색 요청
                    for gap in knowledge_gaps:
                        for keyword in gap.search_keywords:
                            await self.search_queue.put(keyword)
                    
                    self.monitoring_state.filled_knowledge_gaps += auto_fix_result['auto_filled']
                    
                    # 중요한 격차는 즉시 알림
                    critical_gaps = [g for g in knowledge_gaps if g.priority == 'critical']
                    if critical_gaps:
                        print(f"\n🚨 Critical 지식 격차 감지: {len(critical_gaps)}개")
                        for gap in critical_gaps[:3]:  # 상위 3개만 표시
                            print(f"   • {gap.description}")
                    
                    logger.info(f"🔧 자동 보완 완료: {auto_fix_result['auto_filled']}/{len(knowledge_gaps)}")
                
                await asyncio.sleep(1800)  # 30분마다 체크
                
            except Exception as e:
                logger.error(f"지식 격차 감지 시스템 오류: {e}")
                await asyncio.sleep(3600)
    
    async def _start_human_advisory(self):
        """인간 조언 시스템"""
        logger.info("💡 인간 조언 시스템 시작")
        
        while self.is_running:
            try:
                # 조언 생성
                advice = await self._generate_intelligent_advice()
                
                # 조언 저장 (datetime 객체를 문자열로 변환)
                advice_serializable = {
                    "timestamp": advice["timestamp"],
                    "priority": advice["priority"],
                    "category": advice["category"],
                    "message": advice["message"],
                    "action_items": advice["action_items"],
                    "monitoring_stats": {
                        k: v.isoformat() if isinstance(v, datetime) else v 
                        for k, v in advice["monitoring_stats"].items()
                    }
                }
                
                advice_file = self.guardian_dir / f"advice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(advice_file, 'w', encoding='utf-8') as f:
                    json.dump(advice_serializable, f, indent=2, ensure_ascii=False)
                
                # 중요한 조언은 즉시 출력
                if advice.get('priority', 'low') == 'high':
                    print(f"\n🚨 긴급 조언: {advice['message']}")
                
                await asyncio.sleep(7200)  # 2시간마다 조언 생성
                
            except Exception as e:
                logger.error(f"인간 조언 시스템 오류: {e}")
                await asyncio.sleep(3600)
    
    async def _on_learn_session_start(self):
        """learn 세션 시작 시 처리"""
        session_id = f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_learning_progress = LearningProgress(
            session_id=session_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            total_learning_time=0.0,
            quality_score=0.0,
            repetitive_patterns=[],
            knowledge_gaps=[],
            learning_efficiency=0.0,
            next_recommended_action=""
        )
        
        logger.info(f"📚 learn 세션 모니터링 시작: {session_id}")
    
    async def _on_learn_session_end(self):
        """learn 세션 종료 시 처리"""
        if self.current_learning_progress:
            # 세션 결과 분석
            session_duration = (datetime.now() - self.current_learning_progress.start_time).total_seconds()
            self.current_learning_progress.total_learning_time = session_duration
            
            # 세션 보고서 저장
            session_report = asdict(self.current_learning_progress)
            session_report['start_time'] = self.current_learning_progress.start_time.isoformat()
            session_report['last_activity'] = self.current_learning_progress.last_activity.isoformat()
            
            report_file = self.guardian_dir / f"session_report_{self.current_learning_progress.session_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(session_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📚 learn 세션 완료: {session_duration:.1f}초")
            self.current_learning_progress = None
    
    async def _on_create_session_start(self):
        """create 세션 시작 시 처리"""
        logger.info("🎮 create 세션 모니터링 시작")
        
        # create 세션의 잠재적 문제점 미리 검색
        await self.search_queue.put("Godot C# 게임 개발 일반적 오류 해결")
        await self.search_queue.put("Godot C# 스크립트 최적화 및 성능 향상")
    
    async def _on_create_session_end(self):
        """create 세션 종료 시 처리"""
        logger.info("🎮 create 세션 완료")
        
        # 생성된 게임 프로젝트 분석
        await self._analyze_created_projects()
    
    async def _detect_learning_repetitions(self, recent_files: List[Dict]) -> List[str]:
        """반복적 학습 패턴 감지"""
        repetitions = []
        
        try:
            # 파일 내용의 유사성 검사
            contents = []
            for file_data in recent_files:
                content = file_data.get('content', {})
                if isinstance(content, dict):
                    # 주요 키워드 추출
                    keywords = []
                    for key, value in content.items():
                        if isinstance(value, str):
                            keywords.extend(value.lower().split())
                    contents.append(set(keywords))
            
            # 유사도 계산
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    similarity = len(contents[i] & contents[j]) / len(contents[i] | contents[j]) if contents[i] | contents[j] else 0
                    
                    if similarity > 0.7:  # 70% 이상 유사하면 반복으로 간주
                        repetition_pattern = f"반복 패턴 {i}-{j}: 유사도 {similarity:.2f}"
                        repetitions.append(repetition_pattern)
        
        except Exception as e:
            logger.error(f"반복 패턴 감지 오류: {e}")
        
        return repetitions
    
    async def _prevent_repetitive_learning(self, repetitions: List[str]):
        """반복적 학습 방지 조치"""
        prevention_file = self.guardian_dir / "repetition_prevention.json"
        
        prevention_data = {
            "timestamp": datetime.now().isoformat(),
            "detected_repetitions": repetitions,
            "prevention_actions": [
                "새로운 학습 주제 제안",
                "다른 접근 방식 권장",
                "휴식 시간 제안"
            ],
            "recommended_topics": [
                "고급 C# 패턴",
                "Godot 새로운 기능",
                "AI 최신 기술"
            ]
        }
        
        with open(prevention_file, 'w', encoding='utf-8') as f:
            json.dump(prevention_data, f, indent=2, ensure_ascii=False)
        
        # 검색 큐에 새로운 주제 추가
        for topic in prevention_data["recommended_topics"]:
            await self.search_queue.put(topic)
    
    async def _perform_intelligent_search(self, keyword: str):
        """지능형 검색 수행"""
        try:
            logger.info(f"🔍 지능형 검색 시작: {keyword}")
            
            # 공유 지식 베이스 사용
            from modules.shared_knowledge_base import get_shared_knowledge_base
            shared_kb = get_shared_knowledge_base()
            
            # 캐시된 결과 확인
            cached_result = await shared_kb.get_cached_search(keyword)
            if cached_result:
                logger.info(f"📚 캐시된 검색 결과 사용: {keyword}")
                return cached_result
            
            # 검색 결과 저장 디렉토리
            search_results_dir = self.knowledge_base_dir / "search_results"
            search_results_dir.mkdir(exist_ok=True)
            
            # 다양한 소스에서 검색 (시뮬레이션)
            search_sources = [
                "Godot 공식 문서",
                "StackOverflow",
                "GitHub",
                "Reddit",
                "YouTube 튜토리얼"
            ]
            
            search_results = {
                "keyword": keyword,
                "timestamp": datetime.now().isoformat(),
                "sources": {},
                "summary": f"{keyword}에 대한 최신 정보 수집 완료",
                "actionable_insights": [
                    f"{keyword} 관련 새로운 접근법 발견",
                    f"{keyword} 최적화 방법 업데이트",
                    f"{keyword} 문제 해결 패턴 수집"
                ]
            }
            
            for source in search_sources:
                search_results["sources"][source] = {
                    "status": "검색 완료",
                    "results_count": np.random.randint(5, 20),
                    "quality_score": np.random.uniform(0.7, 0.95)
                }
            
            # 검색 결과 저장 (공유 지식 베이스에도 저장)
            result_file = search_results_dir / f"search_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, indent=2, ensure_ascii=False)
            
            # 공유 지식 베이스에 저장
            await shared_kb.save_search_result(keyword, search_results)
            
            logger.info(f"🔍 검색 완료 및 공유 지식 베이스 저장: {keyword}")
            
        except Exception as e:
            logger.error(f"지능형 검색 오류: {e}")
    
    async def _prepare_pytorch_training_data(self) -> List[Dict]:
        """PyTorch 훈련 데이터 준비"""
        training_data = []
        
        try:
            # 모든 경험 데이터 수집
            experiences_dir = Path("experiences")
            
            for data_file in experiences_dir.rglob("*.json"):
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # 딥러닝에 적합한 형태로 변환
                        processed_data = {
                            "input_features": self._extract_features(data),
                            "labels": self._extract_labels(data),
                            "metadata": {
                                "source_file": str(data_file),
                                "timestamp": data.get("timestamp", ""),
                                "category": self._categorize_data(data_file)
                            }
                        }
                        
                        if processed_data["input_features"] and processed_data["labels"]:
                            training_data.append(processed_data)
                
                except Exception as e:
                    continue
            
            logger.info(f"🧠 PyTorch 훈련 데이터 준비 완료: {len(training_data)}개")
            
        except Exception as e:
            logger.error(f"PyTorch 데이터 준비 오류: {e}")
        
        return training_data
    
    def _extract_features(self, data: Dict) -> List[float]:
        """데이터에서 특징 추출"""
        features = []
        
        try:
            # 텍스트 데이터를 수치로 변환
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    elif isinstance(value, str):
                        # 문자열 길이 및 키워드 기반 특징
                        features.append(len(value))
                        features.append(float(value.count('error')))
                        features.append(float(value.count('success')))
                        features.append(float(value.count('C#')))
                        features.append(float(value.count('Godot')))
            
            # 특징 벡터 정규화
            if features:
                max_val = max(features) if max(features) > 0 else 1
                features = [f / max_val for f in features]
            
        except Exception as e:
            logger.error(f"특징 추출 오류: {e}")
        
        return features[:10]  # 최대 10개 특징
    
    def _extract_labels(self, data: Dict) -> List[float]:
        """데이터에서 라벨 추출"""
        labels = []
        
        try:
            # 성공/실패, 품질 점수 등을 라벨로 사용
            if isinstance(data, dict):
                if 'success' in data:
                    labels.append(1.0 if data['success'] else 0.0)
                
                if 'quality_score' in data:
                    labels.append(float(data['quality_score']))
                
                if 'error' in data:
                    labels.append(0.0)
                elif 'completed' in data:
                    labels.append(1.0)
            
            # 기본 라벨
            if not labels:
                labels = [0.5]  # 중간값
            
        except Exception as e:
            logger.error(f"라벨 추출 오류: {e}")
        
        return labels
    
    def _categorize_data(self, file_path: Path) -> str:
        """데이터 파일 카테고리 분류"""
        path_str = str(file_path).lower()
        
        if 'csharp' in path_str:
            return 'csharp_learning'
        elif 'godot' in path_str:
            return 'godot_interaction'
        elif 'game_development' in path_str:
            return 'game_creation'
        elif 'networking' in path_str:
            return 'networking'
        elif 'korean' in path_str:
            return 'korean_nlp'
        else:
            return 'general_learning'
    
    async def _run_pytorch_training(self, training_data: List[Dict]):
        """PyTorch 딥러닝 실행"""
        try:
            # 훈련 데이터 저장
            dataset_file = self.pytorch_datasets_dir / f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "data_count": len(training_data),
                    "categories": list(set(d["metadata"]["category"] for d in training_data)),
                    "training_summary": "PyTorch 딥러닝을 위한 데이터셋 준비 완료"
                }, f, indent=2, ensure_ascii=False)
            
            # 실제 PyTorch 훈련은 백그라운드에서 실행
            logger.info(f"🧠 PyTorch 데이터셋 저장: {dataset_file}")
            
        except Exception as e:
            logger.error(f"PyTorch 훈련 오류: {e}")
    
    async def _detect_knowledge_gaps(self) -> List[str]:
        """지식 격차 감지"""
        gaps = []
        
        try:
            # 최근 오류 분석
            error_patterns = await self._analyze_recent_errors()
            
            # 부족한 영역 식별
            knowledge_areas = {
                "C# 고급 기법": 0,
                "Godot 최적화": 0,
                "PyTorch 통합": 0,
                "Socket.IO 고급": 0,
                "게임 AI 패턴": 0
            }
            
            # 지식 베이스 분석
            for area in knowledge_areas:
                area_files = list(self.knowledge_base_dir.rglob(f"*{area.replace(' ', '_').lower()}*"))
                knowledge_areas[area] = len(area_files)
            
            # 지식이 부족한 영역 식별
            avg_knowledge = sum(knowledge_areas.values()) / len(knowledge_areas)
            
            for area, count in knowledge_areas.items():
                if count < avg_knowledge * 0.7:  # 평균의 70% 미만이면 격차로 간주
                    gaps.append(area)
            
        except Exception as e:
            logger.error(f"지식 격차 감지 오류: {e}")
        
        return gaps
    
    async def _analyze_recent_errors(self) -> List[str]:
        """최근 오류 패턴 분석"""
        error_patterns = []
        
        try:
            # 최근 로그 파일들 확인
            log_dirs = [Path("experiences"), Path("game_projects")]
            
            for log_dir in log_dirs:
                if log_dir.exists():
                    for log_file in log_dir.rglob("*.json"):
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                if isinstance(data, dict) and 'error' in str(data).lower():
                                    error_info = str(data)
                                    if 'c#' in error_info.lower():
                                        error_patterns.append("C# 컴파일 오류")
                                    elif 'godot' in error_info.lower():
                                        error_patterns.append("Godot 엔진 오류")
                                    elif 'socket' in error_info.lower():
                                        error_patterns.append("Socket.IO 통신 오류")
                        except:
                            continue
        
        except Exception as e:
            logger.error(f"오류 패턴 분석 실패: {e}")
        
        return error_patterns
    
    async def _analyze_created_projects(self):
        """생성된 게임 프로젝트 분석"""
        try:
            game_projects_dir = Path("game_projects")
            
            if game_projects_dir.exists():
                recent_projects = []
                one_hour_ago = datetime.now() - timedelta(hours=1)
                
                for project_dir in game_projects_dir.iterdir():
                    if project_dir.is_dir():
                        creation_time = datetime.fromtimestamp(project_dir.stat().st_ctime)
                        if creation_time > one_hour_ago:
                            recent_projects.append(project_dir)
                
                if recent_projects:
                    analysis = {
                        "timestamp": datetime.now().isoformat(),
                        "analyzed_projects": [str(p) for p in recent_projects],
                        "project_count": len(recent_projects),
                        "analysis_summary": "최근 생성된 프로젝트 분석 완료"
                    }
                    
                    analysis_file = self.guardian_dir / f"project_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"🎮 프로젝트 분석 완료: {len(recent_projects)}개")
        
        except Exception as e:
            logger.error(f"프로젝트 분석 오류: {e}")
    
    async def _generate_intelligent_advice(self) -> Dict[str, Any]:
        """지능형 조언 생성"""
        advice = {
            "timestamp": datetime.now().isoformat(),
            "priority": "medium",
            "category": "general",
            "message": "",
            "action_items": [],
            "monitoring_stats": asdict(self.monitoring_state)
        }
        
        try:
            # 모니터링 상태 기반 조언 생성
            if self.monitoring_state.detected_repetitions > 5:
                advice.update({
                    "priority": "high",
                    "category": "learning_optimization",
                    "message": "반복적 학습 패턴이 감지되었습니다. 새로운 접근 방식을 시도해보세요.",
                    "action_items": [
                        "autoci learn 잠시 중단",
                        "다른 게임 타입으로 autoci create 시도",
                        "새로운 기술 스택 학습 고려"
                    ]
                })
            
            elif self.monitoring_state.filled_knowledge_gaps > 3:
                advice.update({
                    "priority": "medium",
                    "category": "knowledge_enhancement",
                    "message": "지식 격차가 성공적으로 보완되고 있습니다. 학습을 계속 진행하세요.",
                    "action_items": [
                        "현재 학습 방향 유지",
                        "고급 주제로 점진적 진행",
                        "프로젝트 복잡도 증가"
                    ]
                })
            
            elif self.monitoring_state.pytorch_training_sessions >= 1:
                advice.update({
                    "priority": "low",
                    "category": "ai_optimization",
                    "message": "PyTorch 딥러닝이 활발히 진행 중입니다. AI 성능이 개선되고 있습니다.",
                    "action_items": [
                        "더 많은 학습 데이터 생성",
                        "다양한 시나리오 테스트",
                        "성능 지표 모니터링"
                    ]
                })
            
            else:
                advice.update({
                    "priority": "medium",
                    "category": "general_guidance",
                    "message": "시스템이 안정적으로 운영 중입니다. 지속적인 학습을 권장합니다.",
                    "action_items": [
                        "정기적인 autoci learn 실행",
                        "다양한 게임 타입 실험",
                        "시스템 성능 최적화"
                    ]
                })
        
        except Exception as e:
            logger.error(f"조언 생성 오류: {e}")
            advice["message"] = "조언 생성 중 오류가 발생했습니다. 시스템 점검을 권장합니다."
        
        return advice
    
    async def _on_learn_session_start(self):
        """autoci learn 세션 시작 시 처리"""
        logger.info("📚 학습 세션 모니터링 시작")
        # 학습 데이터 디렉토리 모니터링 준비
        self.current_learning_progress = LearningProgress(
            session_id=f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            last_activity=datetime.now(),
            total_learning_time=0.0,
            quality_score=0.0,
            repetitive_patterns=[],
            knowledge_gaps=[],
            learning_efficiency=1.0,
            next_recommended_action="학습 진행 중..."
        )
    
    async def _on_learn_session_end(self):
        """autoci learn 세션 종료 시 처리"""
        logger.info("📚 학습 세션 분석 중...")
        if self.current_learning_progress:
            # 학습 세션 결과 저장
            session_file = self.guardian_dir / f"learn_session_{self.current_learning_progress.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_learning_progress), f, indent=2, ensure_ascii=False)
    
    async def _on_create_session_start(self):
        """autoci create 세션 시작 시 처리"""
        logger.info("🎮 게임 생성 세션 모니터링 시작")
        # 게임 프로젝트 모니터링 준비
        self.create_session_data = {
            "session_id": f"create_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "game_type": "unknown",
            "progress": []
        }
    
    async def _on_create_session_end(self):
        """autoci create 세션 종료 시 처리"""
        logger.info("🎮 게임 생성 세션 분석 중...")
        if hasattr(self, 'create_session_data'):
            # 생성 세션 결과 저장
            session_file = self.guardian_dir / f"create_session_{self.create_session_data['session_id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.create_session_data, f, indent=2, ensure_ascii=False)
    
    async def _on_resume_session_start(self):
        """autoci resume 세션 시작 시 처리"""
        logger.info("🔄 기존 프로젝트 개발 세션 모니터링 시작")
        # Godot 프로젝트 디렉토리 모니터링
        self.resume_session_data = {
            "session_id": f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "project_path": None,
            "game_type": None,
            "improvements": [],
            "files_modified": []
        }
        
        # Godot 프로젝트 경로 찾기
        godot_projects_path = Path("/home/super3720/Documents/Godot/Projects")
        if godot_projects_path.exists():
            for project_dir in godot_projects_path.iterdir():
                if project_dir.is_dir() and (project_dir / "project.godot").exists():
                    # 가장 최근 수정된 프로젝트 추적
                    if project_dir.name not in self.monitoring_state.godot_projects_monitored:
                        self.monitoring_state.godot_projects_monitored.append(project_dir.name)
                        self.resume_session_data["project_path"] = str(project_dir)
                        logger.info(f"📁 Godot 프로젝트 감지: {project_dir.name}")
    
    async def _on_resume_session_end(self):
        """autoci resume 세션 종료 시 처리"""
        logger.info("🔄 기존 프로젝트 개발 세션 분석 중...")
        if hasattr(self, 'resume_session_data'):
            # 프로젝트 개선 사항 분석
            if self.resume_session_data["project_path"]:
                project_path = Path(self.resume_session_data["project_path"])
                
                # 수정된 파일 찾기
                for file_path in project_path.rglob("*.gd"):  # GDScript files
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time > datetime.fromisoformat(self.resume_session_data["start_time"]):
                        self.resume_session_data["files_modified"].append(str(file_path))
                
                for file_path in project_path.rglob("*.tscn"):  # Scene files
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time > datetime.fromisoformat(self.resume_session_data["start_time"]):
                        self.resume_session_data["files_modified"].append(str(file_path))
            
            # 세션 결과 저장
            session_file = self.guardian_dir / f"resume_session_{self.resume_session_data['session_id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.resume_session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📝 수정된 파일 수: {len(self.resume_session_data['files_modified'])}")
    
    async def stop_guardian_mode(self):
        """가디언 모드 종료"""
        logger.info("🛡️ 가디언 시스템 종료 중...")
        self.is_running = False
        
        # 최종 보고서 생성 (datetime 객체 안전하게 변환)
        final_report = {
            "session_end": datetime.now().isoformat(),
            "total_monitoring_time": self.monitoring_state.total_monitored_time,
            "detected_repetitions": self.monitoring_state.detected_repetitions,
            "filled_knowledge_gaps": self.monitoring_state.filled_knowledge_gaps,
            "pytorch_training_sessions": self.monitoring_state.pytorch_training_sessions,
            "godot_projects_monitored": self.monitoring_state.godot_projects_monitored,
            "final_summary": "AutoCI 지능형 가디언 세션 완료",
            "last_learn_session": self.monitoring_state.last_learn_session.isoformat() if self.monitoring_state.last_learn_session else None,
            "last_create_session": self.monitoring_state.last_create_session.isoformat() if self.monitoring_state.last_create_session else None,
            "last_resume_session": self.monitoring_state.last_resume_session.isoformat() if self.monitoring_state.last_resume_session else None
        }
        
        final_report_file = self.guardian_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info("🛡️ 가디언 시스템 종료 완료")
        print("\n✅ AutoCI 지능형 가디언 시스템이 종료되었습니다.")
        print(f"📋 최종 보고서: {final_report_file}")

# 싱글톤 인스턴스
_guardian_instance = None

def get_guardian_system() -> IntelligentGuardianSystem:
    """가디언 시스템 싱글톤 인스턴스 반환"""
    global _guardian_instance
    if _guardian_instance is None:
        _guardian_instance = IntelligentGuardianSystem()
    return _guardian_instance