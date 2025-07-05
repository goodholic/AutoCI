"""
AutoCI 공유 지식 베이스 시스템
autoci fix가 수집한 정보를 autoci learn과 autoci create가 활용할 수 있도록 하는 통합 시스템
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class SharedKnowledgeBase:
    """공유 지식 베이스 - 모든 AutoCI 컴포넌트가 지식을 공유"""
    
    def __init__(self):
        # 지식 베이스 디렉토리
        self.knowledge_base_dir = Path("experiences/knowledge_base")
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 검색 결과 캐시
        self.search_cache_dir = self.knowledge_base_dir / "search_results"
        self.search_cache_dir.mkdir(exist_ok=True)
        
        # 솔루션 데이터베이스
        self.solutions_dir = self.knowledge_base_dir / "solutions"
        self.solutions_dir.mkdir(exist_ok=True)
        
        # 베스트 프랙티스
        self.best_practices_dir = self.knowledge_base_dir / "best_practices"
        self.best_practices_dir.mkdir(exist_ok=True)
        
        # 메모리 캐시 (빠른 접근용)
        self.memory_cache = {
            "search_results": {},
            "solutions": {},
            "best_practices": {},
            "last_update": datetime.now()
        }
        
        # 초기 로드
        self._load_existing_knowledge()
        
        logger.info("📚 공유 지식 베이스 시스템 초기화 완료")
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        try:
            # 최근 7일간의 검색 결과 로드
            cutoff_date = datetime.now() - timedelta(days=7)
            
            # 검색 결과 로드
            for search_file in self.search_cache_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(search_file.stat().st_mtime)
                    if file_time > cutoff_date:
                        with open(search_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            keyword = data.get("keyword", "")
                            if keyword:
                                self.memory_cache["search_results"][keyword] = data
                except:
                    continue
            
            logger.info(f"📚 {len(self.memory_cache['search_results'])}개의 검색 결과 로드됨")
            
        except Exception as e:
            logger.error(f"지식 로드 오류: {e}")
    
    async def get_cached_search(self, keyword: str) -> Optional[Dict[str, Any]]:
        """캐시된 검색 결과 반환"""
        # 메모리 캐시 확인
        if keyword in self.memory_cache["search_results"]:
            result = self.memory_cache["search_results"][keyword]
            # 24시간 이내 결과만 유효
            if "timestamp" in result:
                result_time = datetime.fromisoformat(result["timestamp"])
                if datetime.now() - result_time < timedelta(hours=24):
                    logger.info(f"📚 캐시된 검색 결과 사용: {keyword}")
                    return result
        
        # 파일 시스템에서 확인
        search_file = self.search_cache_dir / f"search_{keyword.replace(' ', '_')}_*.json"
        matching_files = list(self.search_cache_dir.glob(f"search_{keyword.replace(' ', '_')}_*.json"))
        
        if matching_files:
            # 가장 최근 파일 사용
            latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 메모리 캐시에 추가
                    self.memory_cache["search_results"][keyword] = data
                    return data
            except:
                pass
        
        return None
    
    async def save_search_result(self, keyword: str, result: Dict[str, Any]):
        """검색 결과 저장 (autoci fix가 사용)"""
        try:
            # 타임스탬프 추가
            result["timestamp"] = datetime.now().isoformat()
            result["keyword"] = keyword
            
            # 파일로 저장
            filename = f"search_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.search_cache_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # 메모리 캐시에도 저장
            self.memory_cache["search_results"][keyword] = result
            
            logger.info(f"📚 검색 결과 저장됨: {keyword}")
            
        except Exception as e:
            logger.error(f"검색 결과 저장 오류: {e}")
    
    async def get_solution_for_error(self, error_type: str, error_message: str) -> Optional[Dict[str, Any]]:
        """특정 오류에 대한 솔루션 반환"""
        # 솔루션 캐시 확인
        solution_key = f"{error_type}_{hash(error_message) % 10000}"
        
        if solution_key in self.memory_cache["solutions"]:
            return self.memory_cache["solutions"][solution_key]
        
        # 파일에서 확인
        solution_file = self.solutions_dir / f"{error_type}_solutions.json"
        if solution_file.exists():
            try:
                with open(solution_file, 'r', encoding='utf-8') as f:
                    solutions = json.load(f)
                    for solution in solutions:
                        if error_message in solution.get("error_patterns", []):
                            self.memory_cache["solutions"][solution_key] = solution
                            return solution
            except:
                pass
        
        return None
    
    async def save_solution(self, error_type: str, error_message: str, solution: str, success: bool = True):
        """해결 방법 저장"""
        try:
            solution_file = self.solutions_dir / f"{error_type}_solutions.json"
            
            # 기존 솔루션 로드
            solutions = []
            if solution_file.exists():
                try:
                    with open(solution_file, 'r', encoding='utf-8') as f:
                        solutions = json.load(f)
                except:
                    solutions = []
            
            # 새 솔루션 추가
            new_solution = {
                "timestamp": datetime.now().isoformat(),
                "error_patterns": [error_message],
                "solution": solution,
                "success": success,
                "usage_count": 0
            }
            
            solutions.append(new_solution)
            
            # 저장
            with open(solution_file, 'w', encoding='utf-8') as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📚 솔루션 저장됨: {error_type}")
            
        except Exception as e:
            logger.error(f"솔루션 저장 오류: {e}")
    
    async def get_best_practices(self, topic: str) -> List[Dict[str, Any]]:
        """특정 주제의 베스트 프랙티스 반환"""
        practices = []
        
        # 메모리 캐시 확인
        if topic in self.memory_cache["best_practices"]:
            return self.memory_cache["best_practices"][topic]
        
        # 파일에서 로드
        practice_file = self.best_practices_dir / f"{topic.replace(' ', '_')}_practices.json"
        if practice_file.exists():
            try:
                with open(practice_file, 'r', encoding='utf-8') as f:
                    practices = json.load(f)
                    self.memory_cache["best_practices"][topic] = practices
            except:
                pass
        
        return practices
    
    async def save_best_practice(self, topic: str, practice: Dict[str, Any]):
        """베스트 프랙티스 저장"""
        try:
            practice_file = self.best_practices_dir / f"{topic.replace(' ', '_')}_practices.json"
            
            # 기존 프랙티스 로드
            practices = []
            if practice_file.exists():
                try:
                    with open(practice_file, 'r', encoding='utf-8') as f:
                        practices = json.load(f)
                except:
                    practices = []
            
            # 새 프랙티스 추가
            practice["timestamp"] = datetime.now().isoformat()
            practice["topic"] = topic
            practices.append(practice)
            
            # 저장
            with open(practice_file, 'w', encoding='utf-8') as f:
                json.dump(practices, f, indent=2, ensure_ascii=False)
            
            # 메모리 캐시 업데이트
            self.memory_cache["best_practices"][topic] = practices
            
            logger.info(f"📚 베스트 프랙티스 저장됨: {topic}")
            
        except Exception as e:
            logger.error(f"베스트 프랙티스 저장 오류: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """지식 베이스 통계"""
        stats = {
            "total_searches": len(list(self.search_cache_dir.glob("*.json"))),
            "cached_searches": len(self.memory_cache["search_results"]),
            "total_solutions": len(list(self.solutions_dir.glob("*.json"))),
            "total_practices": len(list(self.best_practices_dir.glob("*.json"))),
            "memory_cache_size": sum(len(v) for v in self.memory_cache.values() if isinstance(v, dict))
        }
        return stats
    
    async def cleanup_old_data(self, days: int = 30):
        """오래된 데이터 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        # 모든 디렉토리의 오래된 파일 정리
        for directory in [self.search_cache_dir, self.solutions_dir, self.best_practices_dir]:
            for file_path in directory.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1
                except:
                    continue
        
        logger.info(f"🧹 {cleaned_count}개의 오래된 파일 정리됨")
        
        # 메모리 캐시도 정리
        self.memory_cache["search_results"].clear()
        self.memory_cache["solutions"].clear()
        self.memory_cache["best_practices"].clear()
        self._load_existing_knowledge()

# 싱글톤 인스턴스
_shared_kb = None

def get_shared_knowledge_base() -> SharedKnowledgeBase:
    """공유 지식 베이스 싱글톤 반환"""
    global _shared_kb
    if _shared_kb is None:
        _shared_kb = SharedKnowledgeBase()
    return _shared_kb