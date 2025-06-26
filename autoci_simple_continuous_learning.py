#!/usr/bin/env python3
"""
AutoCI 간단 24시간 연속 학습 시스템
의존성 없이 순수 Python만 사용
"""

import os
import sys
import json
import time
import threading
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import re

class SimpleCSharpCrawler:
    """간단한 C# 지식 크롤러"""
    
    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = self.data_dir / "simple_csharp_knowledge.db"
        self.init_database()
        
        # 간단한 크롤링 대상 (REST API 사용)
        self.sources = {
            "github_search": "https://api.github.com/search/repositories?q=language:csharp+stars:>100",
            "unity_releases": "https://api.github.com/repos/Unity-Technologies/UnityCsReference/releases",
            "dotnet_releases": "https://api.github.com/repos/dotnet/core/releases"
        }
        
        self.crawl_stats = {
            "total_documents": 0,
            "today_crawled": 0,
            "last_crawl": None,
            "errors": 0,
            "knowledge_updates": 0,
            "cycles_completed": 0
        }
        
        print("📚 간단 C# 크롤러 초기화됨")
        
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simple_knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    quality_score REAL DEFAULT 0.5,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hash TEXT UNIQUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT,
                    details TEXT,
                    success INTEGER DEFAULT 1
                )
            """)
    
    def crawl_github_repositories(self):
        """GitHub 리포지토리 크롤링"""
        print("💻 GitHub C# 리포지토리 크롤링...")
        
        try:
            # GitHub API 호출
            url = self.sources["github_search"]
            response = self._make_request(url)
            
            if response:
                data = json.loads(response)
                
                if 'items' in data:
                    for item in data['items'][:10]:  # 상위 10개
                        repo_name = item.get('full_name', 'Unknown')
                        description = item.get('description', '')
                        stars = item.get('stargazers_count', 0)
                        
                        content = f"""
Repository: {repo_name}
Stars: {stars}
Description: {description}
Language: C#
URL: {item.get('html_url', '')}
Created: {item.get('created_at', '')}
Updated: {item.get('updated_at', '')}
                        """.strip()
                        
                        self._save_simple_knowledge("github", repo_name, content)
                        
                print(f"✅ GitHub 크롤링 완료: {len(data.get('items', []))}개 리포지토리")
                
        except Exception as e:
            print(f"❌ GitHub 크롤링 오류: {e}")
            self.crawl_stats["errors"] += 1
    
    def generate_synthetic_knowledge(self):
        """합성 지식 생성 (학습 시뮬레이션)"""
        print("🧠 합성 C# 지식 생성...")
        
        # C# 핵심 개념들
        concepts = [
            {
                "title": "C# Async/Await 패턴",
                "content": "async/await는 C#에서 비동기 프로그래밍을 위한 핵심 패턴입니다. Task를 반환하는 메서드에 async 키워드를 사용하고, 비동기 작업을 기다릴 때 await를 사용합니다.",
                "category": "async_programming"
            },
            {
                "title": "Unity MonoBehaviour 라이프사이클",
                "content": "Unity에서 MonoBehaviour는 게임 오브젝트의 동작을 정의합니다. Start(), Update(), FixedUpdate(), LateUpdate() 등의 메서드가 특정 순서로 호출됩니다.",
                "category": "unity"
            },
            {
                "title": "C# LINQ 쿼리 표현식",
                "content": "LINQ(Language Integrated Query)는 C#에서 데이터 쿼리를 위한 강력한 기능입니다. Where, Select, OrderBy 등의 메서드를 체이닝하여 사용합니다.",
                "category": "linq"
            },
            {
                "title": "C# 성능 최적화 기법",
                "content": "C#에서 성능을 최적화하려면 불필요한 할당을 피하고, StringBuilder 사용, 컬렉션 pre-allocation, struct vs class 선택 등을 고려해야 합니다.",
                "category": "performance"
            }
        ]
        
        import random
        
        # 랜덤하게 개념 선택하여 저장
        selected_concepts = random.sample(concepts, random.randint(2, len(concepts)))
        
        for concept in selected_concepts:
            # 내용에 현재 시간 추가하여 고유하게 만들기
            enhanced_content = f"{concept['content']}\n\n생성 시간: {datetime.now().isoformat()}"
            
            self._save_simple_knowledge("synthetic", concept["title"], enhanced_content)
        
        print(f"✅ 합성 지식 생성 완료: {len(selected_concepts)}개 개념")
    
    def _make_request(self, url: str, timeout: int = 10) -> Optional[str]:
        """HTTP 요청 수행"""
        try:
            headers = {
                'User-Agent': 'AutoCI-Simple-Crawler/1.0'
            }
            
            request = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode('utf-8')
                
        except Exception as e:
            print(f"❌ HTTP 요청 실패 {url}: {e}")
            return None
    
    def _save_simple_knowledge(self, source: str, title: str, content: str):
        """간단한 지식 저장"""
        try:
            # 중복 체크용 해시
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                # 중복 체크
                existing = conn.execute("SELECT id FROM simple_knowledge_base WHERE hash = ?", (content_hash,)).fetchone()
                
                if not existing:
                    category = self._simple_classify(content)
                    quality = self._simple_quality_score(content)
                    
                    conn.execute("""
                        INSERT INTO simple_knowledge_base (source, title, content, category, quality_score, hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (source, title, content, category, quality, content_hash))
                    
                    # 학습 진행상황 기록
                    conn.execute("""
                        INSERT INTO learning_progress (action, details)
                        VALUES (?, ?)
                    """, ("knowledge_saved", f"{source}: {title[:50]}"))
                    
                    self.crawl_stats["total_documents"] += 1
                    self.crawl_stats["today_crawled"] += 1
                    self.crawl_stats["knowledge_updates"] += 1
                    
                    print(f"✅ 지식 저장: {title[:50]}...")
                    
        except Exception as e:
            print(f"❌ 지식 저장 실패: {e}")
            self.crawl_stats["errors"] += 1
    
    def _simple_classify(self, content: str) -> str:
        """간단한 내용 분류"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['unity', 'gameobject', 'transform', 'monobehaviour']):
            return 'unity'
        elif any(word in content_lower for word in ['async', 'await', 'task', 'thread']):
            return 'async'
        elif any(word in content_lower for word in ['linq', 'query', 'select', 'where']):
            return 'linq'
        elif any(word in content_lower for word in ['performance', 'optimization', 'memory']):
            return 'performance'
        elif any(word in content_lower for word in ['class', 'interface', 'inheritance']):
            return 'oop'
        else:
            return 'general'
    
    def _simple_quality_score(self, content: str) -> float:
        """간단한 품질 점수"""
        score = 0.3  # 기본 점수
        
        # 길이에 따른 점수
        if len(content) > 200:
            score += 0.2
        if len(content) > 500:
            score += 0.2
        
        # 코드 예제나 기술적 키워드 포함 시 점수 증가
        if any(keyword in content for keyword in ['class', 'public', 'private', 'void', 'async']):
            score += 0.2
        
        if any(keyword in content.lower() for keyword in ['example', 'usage', 'how to']):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.crawl_stats.copy()

class SimpleContinuousLearningAI:
    """간단한 24시간 연속 학습 AI"""
    
    def __init__(self):
        self.crawler = SimpleCSharpCrawler()
        self.learning_active = False
        self.learning_thread = None
        
        # 학습 통계
        self.learning_stats = {
            "sessions_completed": 0,
            "total_learning_time": 0,
            "start_time": None,
            "last_update": None,
            "knowledge_growth": 0,
            "learning_efficiency": 0.0
        }
        
        # 간단한 지식 베이스 (메모리)
        self.knowledge_base = {
            "unity_tips": [],
            "csharp_patterns": [],
            "performance_tricks": [],
            "recent_updates": []
        }
        
        print("🧠 간단 연속 학습 AI 초기화됨")
    
    def start_continuous_learning(self):
        """연속 학습 시작"""
        if self.learning_active:
            print("❌ 학습이 이미 진행 중입니다.")
            return False
        
        print("🚀 24시간 연속 학습 시작!")
        
        self.learning_active = True
        self.learning_stats["start_time"] = datetime.now().isoformat()
        
        # 백그라운드 스레드로 학습 시작
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        print("✅ 백그라운드 학습 시작됨")
        return True
    
    def stop_continuous_learning(self):
        """연속 학습 중지"""
        if not self.learning_active:
            print("❌ 실행 중인 학습이 없습니다.")
            return False
        
        print("🛑 연속 학습 중지...")
        self.learning_active = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        print("✅ 연속 학습 중지됨")
        return True
    
    def _learning_loop(self):
        """학습 루프"""
        cycle_count = 0
        
        while self.learning_active:
            try:
                cycle_start = time.time()
                print(f"\n🔄 학습 사이클 #{cycle_count + 1} 시작...")
                
                # 1. 크롤링 단계
                self._run_crawling_phase()
                
                # 2. 학습 단계
                self._run_learning_phase()
                
                # 3. 통계 업데이트
                cycle_time = time.time() - cycle_start
                self._update_learning_stats(cycle_time)
                
                cycle_count += 1
                self.crawler.crawl_stats["cycles_completed"] = cycle_count
                
                print(f"✅ 사이클 #{cycle_count} 완료 ({cycle_time:.1f}초)")
                
                # 10분 대기 (실제로는 더 긴 간격 사용 가능)
                if self.learning_active:
                    time.sleep(600)  # 10분
                
            except Exception as e:
                print(f"❌ 학습 루프 오류: {e}")
                if self.learning_active:
                    time.sleep(300)  # 5분 대기 후 재시도
    
    def _run_crawling_phase(self):
        """크롤링 단계"""
        print("📡 크롤링 단계 시작...")
        
        try:
            # 순차적으로 각 소스 크롤링
            self.crawler.crawl_github_repositories()
            time.sleep(2)
            
            # 합성 지식 생성
            self.crawler.generate_synthetic_knowledge()
            
            print("✅ 크롤링 단계 완료")
            
        except Exception as e:
            print(f"❌ 크롤링 오류: {e}")
    
    def _run_learning_phase(self):
        """학습 단계"""
        print("🧠 학습 단계 시작...")
        
        try:
            # 새로운 지식 로드
            new_knowledge = self._load_recent_knowledge()
            
            if new_knowledge:
                # 지식 분류 및 저장
                for knowledge in new_knowledge:
                    self._process_knowledge(knowledge)
                
                print(f"✅ 학습 완료: {len(new_knowledge)}개 항목 처리")
            else:
                print("📝 새로운 지식 없음")
            
        except Exception as e:
            print(f"❌ 학습 오류: {e}")
    
    def _load_recent_knowledge(self) -> List[Dict]:
        """최근 지식 로드"""
        try:
            with sqlite3.connect(self.crawler.db_path) as conn:
                # 최근 30분 내 추가된 지식
                thirty_min_ago = (datetime.now() - timedelta(minutes=30)).isoformat()
                
                cursor = conn.execute("""
                    SELECT title, content, category, quality_score 
                    FROM simple_knowledge_base 
                    WHERE crawled_at > ? 
                    ORDER BY quality_score DESC 
                    LIMIT 50
                """, (thirty_min_ago,))
                
                return [{"title": row[0], "content": row[1], "category": row[2], "quality": row[3]}
                       for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"❌ 지식 로드 실패: {e}")
            return []
    
    def _process_knowledge(self, knowledge: Dict):
        """지식 처리 및 학습"""
        category = knowledge["category"]
        
        # 카테고리별로 지식베이스에 추가
        if category == "unity":
            self.knowledge_base["unity_tips"].append(knowledge)
        elif category in ["async", "oop", "linq"]:
            self.knowledge_base["csharp_patterns"].append(knowledge)
        elif category == "performance":
            self.knowledge_base["performance_tricks"].append(knowledge)
        else:
            self.knowledge_base["recent_updates"].append(knowledge)
        
        # 각 카테고리별 최대 100개 항목 유지
        for category_list in self.knowledge_base.values():
            if len(category_list) > 100:
                # 품질 점수 기준으로 정렬하여 상위 100개만 유지
                category_list.sort(key=lambda x: x.get("quality", 0), reverse=True)
                category_list[:] = category_list[:100]
    
    def _update_learning_stats(self, cycle_time: float):
        """학습 통계 업데이트"""
        self.learning_stats["sessions_completed"] += 1
        self.learning_stats["total_learning_time"] += cycle_time
        self.learning_stats["last_update"] = datetime.now().isoformat()
        
        # 지식 증가량 계산
        current_knowledge = sum(len(kb) for kb in self.knowledge_base.values())
        self.learning_stats["knowledge_growth"] = current_knowledge
        
        # 학습 효율성 계산 (지식량/시간)
        if self.learning_stats["total_learning_time"] > 0:
            self.learning_stats["learning_efficiency"] = current_knowledge / (self.learning_stats["total_learning_time"] / 60)  # 분당 지식량
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        crawl_stats = self.crawler.get_stats()
        
        return {
            "learning_active": self.learning_active,
            "learning_stats": self.learning_stats,
            "crawl_stats": crawl_stats,
            "knowledge_base_size": {
                "unity_tips": len(self.knowledge_base["unity_tips"]),
                "csharp_patterns": len(self.knowledge_base["csharp_patterns"]),
                "performance_tricks": len(self.knowledge_base["performance_tricks"]),
                "recent_updates": len(self.knowledge_base["recent_updates"]),
                "total": sum(len(kb) for kb in self.knowledge_base.values())
            }
        }
    
    def query_knowledge(self, query: str) -> List[Dict]:
        """지식 검색"""
        results = []
        query_lower = query.lower()
        
        for category, knowledge_list in self.knowledge_base.items():
            for knowledge in knowledge_list:
                if (query_lower in knowledge["title"].lower() or 
                    query_lower in knowledge["content"].lower()):
                    results.append({
                        "category": category,
                        "title": knowledge["title"],
                        "content": knowledge["content"][:200] + "...",
                        "quality": knowledge.get("quality", 0)
                    })
        
        # 품질 점수 순으로 정렬
        results.sort(key=lambda x: x["quality"], reverse=True)
        return results[:10]  # 상위 10개 반환

def main():
    """메인 함수"""
    print("🚀 AutoCI 간단 24시간 연속 학습 시스템")
    print("=" * 50)
    
    # 연속 학습 AI 초기화
    learning_ai = SimpleContinuousLearningAI()
    
    # 사용법 출력
    print("\n명령어:")
    print("  start  - 연속 학습 시작")
    print("  stop   - 연속 학습 중지")
    print("  status - 상태 확인")
    print("  search [쿼리] - 지식 검색")
    print("  quit   - 프로그램 종료")
    
    try:
        while True:
            print("\n" + "="*30)
            command = input("AutoCI> ").strip().lower()
            
            if command == "start":
                learning_ai.start_continuous_learning()
                
            elif command == "stop":
                learning_ai.stop_continuous_learning()
                
            elif command == "status":
                status = learning_ai.get_learning_status()
                
                print("\n📊 학습 상태:")
                print(f"🔄 활성: {'✅ 실행중' if status['learning_active'] else '❌ 중지됨'}")
                print(f"🧠 완료된 세션: {status['learning_stats']['sessions_completed']}")
                print(f"📚 총 지식량: {status['knowledge_base_size']['total']}")
                print(f"🎮 Unity 팁: {status['knowledge_base_size']['unity_tips']}")
                print(f"⚡ C# 패턴: {status['knowledge_base_size']['csharp_patterns']}")
                print(f"🚀 성능 팁: {status['knowledge_base_size']['performance_tricks']}")
                print(f"🔄 크롤링 사이클: {status['crawl_stats']['cycles_completed']}")
                print(f"📈 학습 효율: {status['learning_stats']['learning_efficiency']:.2f} 지식/분")
                
            elif command.startswith("search "):
                query = command[7:]  # "search " 제거
                results = learning_ai.query_knowledge(query)
                
                if results:
                    print(f"\n🔍 '{query}' 검색 결과:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. [{result['category']}] {result['title']}")
                        print(f"   {result['content']}")
                        print(f"   품질: {result['quality']:.2f}")
                else:
                    print(f"❌ '{query}'에 대한 검색 결과가 없습니다.")
                    
            elif command == "quit":
                if learning_ai.learning_active:
                    learning_ai.stop_continuous_learning()
                print("👋 AutoCI 연속 학습 시스템 종료")
                break
                
            else:
                print("❌ 알 수 없는 명령어입니다.")
                
    except KeyboardInterrupt:
        print("\n\n🛑 사용자 중단")
        if learning_ai.learning_active:
            learning_ai.stop_continuous_learning()
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
    finally:
        print("👋 프로그램 종료")

if __name__ == "__main__":
    main() 