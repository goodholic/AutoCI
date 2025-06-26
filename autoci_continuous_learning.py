#!/usr/bin/env python3
"""
AutoCI 24시간 연속 학습 시스템
C# 전문 내용 크롤링 + 실시간 학습
"""

import os
import sys
import json
import time
import threading
import requests
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import schedule

class CSharpKnowledgeCrawler:
    """C# 전문 지식 크롤러"""
    
    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.db_path = self.data_dir / "csharp_knowledge.db"
        self.init_database()
        
        # 크롤링 대상
        self.sources = {
            "microsoft_docs": [
                "https://docs.microsoft.com/en-us/dotnet/csharp/",
                "https://docs.microsoft.com/en-us/dotnet/api/",
                "https://docs.microsoft.com/en-us/aspnet/core/",
                "https://docs.microsoft.com/en-us/dotnet/framework/"
            ],
            "unity_docs": [
                "https://docs.unity3d.com/ScriptReference/",
                "https://docs.unity3d.com/Manual/",
                "https://learn.unity.com/"
            ],
            "github_repos": [
                "https://api.github.com/search/repositories?q=language:csharp+stars:>1000",
                "https://api.github.com/search/code?q=extension:cs+size:>1000"
            ],
            "stackoverflow": [
                "https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=c%23",
                "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&tagged=unity3d"
            ]
        }
        
        self.crawl_stats = {
            "total_documents": 0,
            "today_crawled": 0,
            "last_crawl": None,
            "errors": 0,
            "knowledge_updates": 0
        }
        
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT,
                    difficulty REAL DEFAULT 0.5,
                    quality_score REAL DEFAULT 0.0,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hash TEXT UNIQUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP,
                    session_end TIMESTAMP,
                    documents_processed INTEGER,
                    knowledge_gained REAL,
                    model_updates INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 0.5,
                    context TEXT,
                    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def crawl_microsoft_docs(self):
        """Microsoft 문서 크롤링"""
        print("📚 Microsoft C# 문서 크롤링 시작...")
        
        async with aiohttp.ClientSession() as session:
            for url in self.sources["microsoft_docs"]:
                try:
                    await self._crawl_url(session, url, "microsoft_docs")
                except Exception as e:
                    print(f"❌ Microsoft 문서 크롤링 오류: {e}")
                    self.crawl_stats["errors"] += 1
    
    async def crawl_unity_docs(self):
        """Unity 문서 크롤링"""
        print("🎮 Unity 문서 크롤링 시작...")
        
        async with aiohttp.ClientSession() as session:
            for url in self.sources["unity_docs"]:
                try:
                    await self._crawl_url(session, url, "unity_docs")
                except Exception as e:
                    print(f"❌ Unity 문서 크롤링 오류: {e}")
                    self.crawl_stats["errors"] += 1
    
    async def crawl_github_repos(self):
        """GitHub C# 리포지토리 크롤링"""
        print("💻 GitHub C# 코드 크롤링 시작...")
        
        async with aiohttp.ClientSession() as session:
            for url in self.sources["github_repos"]:
                try:
                    await self._crawl_github_api(session, url)
                except Exception as e:
                    print(f"❌ GitHub 크롤링 오류: {e}")
                    self.crawl_stats["errors"] += 1
    
    async def crawl_stackoverflow(self):
        """StackOverflow 질문/답변 크롤링"""
        print("❓ StackOverflow C# 질문 크롤링 시작...")
        
        async with aiohttp.ClientSession() as session:
            for url in self.sources["stackoverflow"]:
                try:
                    await self._crawl_stackoverflow_api(session, url)
                except Exception as e:
                    print(f"❌ StackOverflow 크롤링 오류: {e}")
                    self.crawl_stats["errors"] += 1
    
    async def _crawl_url(self, session: aiohttp.ClientSession, url: str, source: str):
        """URL 크롤링"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # 간단한 내용 추출 (실제로는 BeautifulSoup 등 사용)
                    title = f"Document from {url}"
                    
                    # 중복 체크를 위한 해시
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # 데이터베이스에 저장
                    self._save_knowledge(source, title, content, content_hash)
                    
        except Exception as e:
            print(f"URL 크롤링 실패 {url}: {e}")
    
    async def _crawl_github_api(self, session: aiohttp.ClientSession, url: str):
        """GitHub API 크롤링"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'AutoCI-Learning-Bot'
        }
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'items' in data:
                        for item in data['items'][:10]:  # 상위 10개만
                            repo_name = item.get('full_name', 'Unknown')
                            description = item.get('description', '')
                            
                            # 코드 파일 내용 가져오기
                            if 'contents_url' in item:
                                await self._crawl_repo_contents(session, item['contents_url'], repo_name)
                            
        except Exception as e:
            print(f"GitHub API 크롤링 실패: {e}")
    
    async def _crawl_stackoverflow_api(self, session: aiohttp.ClientSession, url: str):
        """StackOverflow API 크롤링"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'items' in data:
                        for item in data['items'][:20]:  # 상위 20개 질문
                            title = item.get('title', 'Untitled')
                            body = item.get('body', '')
                            tags = item.get('tags', [])
                            
                            content = f"Title: {title}\nTags: {', '.join(tags)}\nBody: {body}"
                            content_hash = hashlib.md5(content.encode()).hexdigest()
                            
                            self._save_knowledge("stackoverflow", title, content, content_hash)
                            
        except Exception as e:
            print(f"StackOverflow API 크롤링 실패: {e}")
    
    async def _crawl_repo_contents(self, session: aiohttp.ClientSession, contents_url: str, repo_name: str):
        """리포지토리 내용 크롤링"""
        try:
            # contents_url에서 실제 파일 목록 가져오기 (간소화)
            files_url = contents_url.replace('{+path}', '')
            
            async with session.get(files_url) as response:
                if response.status == 200:
                    files = await response.json()
                    
                    for file_info in files[:5]:  # 상위 5개 파일만
                        if file_info.get('name', '').endswith('.cs'):
                            file_content = f"Repository: {repo_name}\nFile: {file_info.get('name')}\nPath: {file_info.get('path')}"
                            content_hash = hashlib.md5(file_content.encode()).hexdigest()
                            
                            self._save_knowledge("github", f"{repo_name}/{file_info.get('name')}", file_content, content_hash)
                            
        except Exception as e:
            print(f"리포지토리 내용 크롤링 실패: {e}")
    
    def _save_knowledge(self, source: str, title: str, content: str, content_hash: str):
        """지식을 데이터베이스에 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 중복 체크
                existing = conn.execute("SELECT id FROM knowledge_base WHERE hash = ?", (content_hash,)).fetchone()
                
                if not existing:
                    # 카테고리 자동 분류
                    category = self._classify_content(content)
                    quality_score = self._assess_quality(content)
                    
                    conn.execute("""
                        INSERT INTO knowledge_base (source, title, content, category, quality_score, hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (source, title, content, category, quality_score, content_hash))
                    
                    self.crawl_stats["total_documents"] += 1
                    self.crawl_stats["today_crawled"] += 1
                    self.crawl_stats["knowledge_updates"] += 1
                    
                    print(f"✅ 새로운 지식 저장: {title[:50]}...")
                    
        except sqlite3.IntegrityError:
            # 중복 문서 - 무시
            pass
        except Exception as e:
            print(f"❌ 지식 저장 실패: {e}")
            self.crawl_stats["errors"] += 1
    
    def _classify_content(self, content: str) -> str:
        """내용 카테고리 분류"""
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in ['unity', 'gameobject', 'transform', 'monobehaviour']):
            return 'unity'
        elif any(keyword in content_lower for keyword in ['async', 'await', 'task', 'thread']):
            return 'async_programming'
        elif any(keyword in content_lower for keyword in ['class', 'interface', 'inheritance', 'polymorphism']):
            return 'oop'
        elif any(keyword in content_lower for keyword in ['linq', 'query', 'select', 'where']):
            return 'linq'
        elif any(keyword in content_lower for keyword in ['performance', 'optimization', 'memory', 'gc']):
            return 'performance'
        else:
            return 'general'
    
    def _assess_quality(self, content: str) -> float:
        """내용 품질 평가"""
        score = 0.5  # 기본 점수
        
        # 길이 기반 점수
        if len(content) > 1000:
            score += 0.2
        elif len(content) > 500:
            score += 0.1
        
        # 코드 예제 포함 여부
        if any(keyword in content for keyword in ['```', 'class ', 'public ', 'private ']):
            score += 0.2
        
        # 설명 품질 (간단한 휴리스틱)
        if any(keyword in content.lower() for keyword in ['example', 'usage', 'how to', 'tutorial']):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_crawl_stats(self) -> Dict:
        """크롤링 통계 반환"""
        return self.crawl_stats.copy()

class ContinuousLearningAI:
    """24시간 연속 학습 AI"""
    
    def __init__(self):
        self.crawler = CSharpKnowledgeCrawler()
        self.learning_active = True
        self.learning_thread = None
        self.crawl_thread = None
        
        # 학습 상태
        self.learning_stats = {
            "sessions_completed": 0,
            "total_learning_time": 0,
            "knowledge_base_size": 0,
            "last_update": None,
            "learning_rate": 0.001,
            "model_accuracy": 0.0
        }
        
        # 가상의 신경망 가중치 (실제로는 PyTorch/TensorFlow 모델)
        self.model_weights = {
            "korean_language": {},
            "csharp_knowledge": {},
            "unity_expertise": {},
            "conversation_patterns": {}
        }
        
        print("🧠 24시간 연속 학습 AI 초기화됨")
    
    def start_continuous_learning(self):
        """연속 학습 시작"""
        print("🚀 24시간 연속 학습 시작!")
        
        # 크롤링 스케줄 설정
        schedule.every(1).hours.do(self._run_crawling_cycle)
        schedule.every(30).minutes.do(self._run_learning_cycle)
        schedule.every(6).hours.do(self._save_learning_progress)
        schedule.every().day.at("03:00").do(self._daily_maintenance)
        
        # 백그라운드 스레드 시작
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.crawl_thread = threading.Thread(target=self._continuous_crawling_loop, daemon=True)
        
        self.learning_thread.start()
        self.crawl_thread.start()
        
        print("✅ 백그라운드 학습 스레드 시작됨")
    
    def _continuous_learning_loop(self):
        """연속 학습 루프"""
        while self.learning_active:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 스케줄 체크
            except Exception as e:
                print(f"❌ 학습 루프 오류: {e}")
                time.sleep(300)  # 5분 대기 후 재시도
    
    def _continuous_crawling_loop(self):
        """연속 크롤링 루프"""
        while self.learning_active:
            try:
                # 비동기 크롤링 실행
                asyncio.run(self._run_async_crawling())
                time.sleep(3600)  # 1시간 대기
            except Exception as e:
                print(f"❌ 크롤링 루프 오류: {e}")
                time.sleep(1800)  # 30분 대기 후 재시도
    
    async def _run_async_crawling(self):
        """비동기 크롤링 실행"""
        tasks = [
            self.crawler.crawl_microsoft_docs(),
            self.crawler.crawl_unity_docs(),
            self.crawler.crawl_github_repos(),
            self.crawler.crawl_stackoverflow()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _run_crawling_cycle(self):
        """크롤링 사이클 실행"""
        print("🔄 정기 크롤링 사이클 시작...")
        start_time = time.time()
        
        try:
            # 비동기 크롤링 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_async_crawling())
            loop.close()
            
            elapsed = time.time() - start_time
            print(f"✅ 크롤링 완료 ({elapsed:.1f}초)")
            
        except Exception as e:
            print(f"❌ 크롤링 사이클 오류: {e}")
    
    def _run_learning_cycle(self):
        """학습 사이클 실행"""
        print("🧠 정기 학습 사이클 시작...")
        start_time = time.time()
        
        try:
            # 1. 새로운 지식 로드
            new_knowledge = self._load_new_knowledge()
            
            # 2. 모델 업데이트 (시뮬레이션)
            updates = self._update_model_weights(new_knowledge)
            
            # 3. 학습 통계 업데이트
            self.learning_stats["sessions_completed"] += 1
            self.learning_stats["total_learning_time"] += time.time() - start_time
            self.learning_stats["knowledge_base_size"] = len(new_knowledge)
            self.learning_stats["last_update"] = datetime.now().isoformat()
            
            print(f"✅ 학습 완료: {updates}개 가중치 업데이트")
            
        except Exception as e:
            print(f"❌ 학습 사이클 오류: {e}")
    
    def _load_new_knowledge(self) -> List[Dict]:
        """새로운 지식 로드"""
        try:
            with sqlite3.connect(self.crawler.db_path) as conn:
                # 최근 1시간 내 추가된 지식
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                
                cursor = conn.execute("""
                    SELECT title, content, category, quality_score 
                    FROM knowledge_base 
                    WHERE crawled_at > ? 
                    ORDER BY quality_score DESC 
                    LIMIT 100
                """, (one_hour_ago,))
                
                return [{"title": row[0], "content": row[1], "category": row[2], "quality": row[3]} 
                       for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"❌ 지식 로드 오류: {e}")
            return []
    
    def _update_model_weights(self, knowledge_list: List[Dict]) -> int:
        """모델 가중치 업데이트 (시뮬레이션)"""
        updates = 0
        
        for knowledge in knowledge_list:
            category = knowledge["category"]
            quality = knowledge["quality"]
            
            # 가중치 업데이트 시뮬레이션
            if category not in self.model_weights["csharp_knowledge"]:
                self.model_weights["csharp_knowledge"][category] = 0.0
            
            # 품질에 따른 가중치 조정
            self.model_weights["csharp_knowledge"][category] += quality * self.learning_stats["learning_rate"]
            updates += 1
            
            # 모델 정확도 향상 시뮬레이션
            self.learning_stats["model_accuracy"] = min(
                self.learning_stats["model_accuracy"] + 0.001,
                1.0
            )
        
        return updates
    
    def _save_learning_progress(self):
        """학습 진행상황 저장"""
        try:
            progress_file = self.crawler.data_dir / "learning_progress.json"
            
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "learning_stats": self.learning_stats,
                "crawl_stats": self.crawler.get_crawl_stats(),
                "model_weights": self.model_weights
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            print("💾 학습 진행상황 저장됨")
            
        except Exception as e:
            print(f"❌ 진행상황 저장 오류: {e}")
    
    def _daily_maintenance(self):
        """일일 유지보수"""
        print("🔧 일일 유지보수 시작...")
        
        try:
            # 1. 데이터베이스 정리
            with sqlite3.connect(self.crawler.db_path) as conn:
                # 오래된 저품질 데이터 삭제
                conn.execute("""
                    DELETE FROM knowledge_base 
                    WHERE quality_score < 0.3 
                    AND crawled_at < datetime('now', '-7 days')
                """)
                
                # 중복 제거
                conn.execute("""
                    DELETE FROM knowledge_base 
                    WHERE id NOT IN (
                        SELECT MIN(id) FROM knowledge_base GROUP BY hash
                    )
                """)
            
            # 2. 통계 초기화
            self.crawler.crawl_stats["today_crawled"] = 0
            self.crawler.crawl_stats["errors"] = 0
            
            print("✅ 일일 유지보수 완료")
            
        except Exception as e:
            print(f"❌ 유지보수 오류: {e}")
    
    def get_learning_status(self) -> Dict:
        """학습 상태 반환"""
        return {
            "learning_active": self.learning_active,
            "learning_stats": self.learning_stats,
            "crawl_stats": self.crawler.get_crawl_stats(),
            "knowledge_categories": list(self.model_weights["csharp_knowledge"].keys()),
            "model_size": sum(len(weights) for weights in self.model_weights.values())
        }
    
    def stop_learning(self):
        """학습 중지"""
        print("🛑 연속 학습 중지...")
        self.learning_active = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        if self.crawl_thread and self.crawl_thread.is_alive():
            self.crawl_thread.join(timeout=5)
        
        self._save_learning_progress()
        print("✅ 연속 학습 중지됨")

def main():
    """메인 함수"""
    print("🚀 AutoCI 24시간 연속 학습 시스템")
    print("=" * 50)
    
    # 연속 학습 AI 초기화
    learning_ai = ContinuousLearningAI()
    
    try:
        # 연속 학습 시작
        learning_ai.start_continuous_learning()
        
        print("\n📊 실시간 상태 모니터링 (Ctrl+C로 종료)")
        print("-" * 50)
        
        # 상태 모니터링 루프
        while True:
            status = learning_ai.get_learning_status()
            
            print(f"\r🧠 세션: {status['learning_stats']['sessions_completed']} | "
                  f"📚 지식: {status['crawl_stats']['total_documents']} | "
                  f"🎯 정확도: {status['learning_stats']['model_accuracy']:.3f} | "
                  f"🔄 활성: {'✅' if status['learning_active'] else '❌'}", end="")
            
            time.sleep(10)  # 10초마다 상태 업데이트
            
    except KeyboardInterrupt:
        print("\n\n🛑 사용자 중단 요청")
    except Exception as e:
        print(f"\n❌ 시스템 오류: {e}")
    finally:
        learning_ai.stop_learning()
        print("👋 AutoCI 연속 학습 시스템 종료")

if __name__ == "__main__":
    main() 