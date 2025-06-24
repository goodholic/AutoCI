#!/usr/bin/env python3
"""
Advanced C# Knowledge Crawler - C# 전문 지식 대폭 확장 크롤러
"""

import asyncio
import aiohttp
import json
import time
import os
import re
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

class AdvancedCSharpCrawler:
    def __init__(self):
        self.session = None
        self.data_dir = Path("expert_learning_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # C# 전문 지식 소스들
        self.knowledge_sources = {
            "microsoft_docs": [
                "https://docs.microsoft.com/en-us/dotnet/csharp/",
                "https://docs.microsoft.com/en-us/dotnet/api/",
                "https://docs.microsoft.com/en-us/aspnet/core/"
            ],
            "unity_docs": [
                "https://docs.unity3d.com/ScriptReference/",
                "https://docs.unity3d.com/Manual/"
            ],
            "github_patterns": [
                "https://api.github.com/search/repositories?q=language:csharp+stars:>1000"
            ]
        }
        
        # 고급 C# 패턴 및 개념들
        self.advanced_topics = [
            "async await patterns", "memory management", "garbage collection",
            "dependency injection", "design patterns", "SOLID principles",
            "performance optimization", "multithreading", "parallel processing",
            "reflection", "generics", "delegates", "events", "linq",
            "entity framework", "blazor", "signalr", "minimal apis",
            "microservices", "clean architecture", "domain driven design",
            "unity optimization", "unity ecs", "unity dots", "unity addressables"
        ]

    async def start_session(self):
        """HTTP 세션 시작"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = aiohttp.ClientSession(headers=headers)

    async def close_session(self):
        """HTTP 세션 종료"""
        if self.session:
            await self.session.close()

    async def crawl_microsoft_docs(self):
        """Microsoft 공식 문서 크롤링"""
        print("🔍 Microsoft 공식 문서 크롤링 중...")
        
        for url in self.knowledge_sources["microsoft_docs"]:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # 코드 예제 추출
                        code_blocks = soup.find_all('pre', class_='lang-csharp')
                        for i, block in enumerate(code_blocks):
                            if block.text.strip():
                                await self.save_knowledge({
                                    "source": "microsoft_docs",
                                    "url": url,
                                    "type": "code_example",
                                    "content": block.text.strip(),
                                    "timestamp": datetime.now().isoformat(),
                                    "category": "official_documentation"
                                })
                        
                        # 개념 설명 추출
                        content_sections = soup.find_all(['p', 'div'], class_=['content', 'description'])
                        for section in content_sections:
                            text = section.get_text().strip()
                            if len(text) > 100 and any(topic in text.lower() for topic in self.advanced_topics):
                                await self.save_knowledge({
                                    "source": "microsoft_docs",
                                    "url": url,
                                    "type": "concept_explanation",
                                    "content": text,
                                    "timestamp": datetime.now().isoformat(),
                                    "category": "advanced_concepts"
                                })
                        
                await asyncio.sleep(1)  # 요청 간격
            except Exception as e:
                print(f"❌ Microsoft Docs 크롤링 오류: {e}")

    async def crawl_github_advanced_patterns(self):
        """GitHub에서 고급 C# 패턴 크롤링"""
        print("🔍 GitHub 고급 패턴 크롤링 중...")
        
        # 고품질 C# 리포지토리 검색
        github_queries = [
            "language:csharp+topic:design-patterns+stars:>500",
            "language:csharp+topic:clean-architecture+stars:>300",
            "language:csharp+topic:ddd+stars:>200",
            "language:csharp+topic:microservices+stars:>300",
            "language:csharp+topic:unity+stars:>500",
            "language:csharp+topic:performance+stars:>200"
        ]
        
        for query in github_queries:
            try:
                url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=10"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for repo in data.get('items', []):
                            # 리포지토리 README 크롤링
                            readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
                            try:
                                async with self.session.get(readme_url) as readme_response:
                                    if readme_response.status == 200:
                                        readme_data = await readme_response.json()
                                        import base64
                                        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                                        
                                        await self.save_knowledge({
                                            "source": "github_patterns",
                                            "url": repo['html_url'],
                                            "type": "project_documentation",
                                            "content": readme_content,
                                            "stars": repo['stargazers_count'],
                                            "timestamp": datetime.now().isoformat(),
                                            "category": "advanced_patterns"
                                        })
                            except:
                                pass
                        
                await asyncio.sleep(2)  # GitHub API 제한
            except Exception as e:
                print(f"❌ GitHub 패턴 크롤링 오류: {e}")

    async def crawl_unity_advanced_docs(self):
        """Unity 고급 문서 크롤링"""
        print("🔍 Unity 고급 문서 크롤링 중...")
        
        unity_advanced_topics = [
            "https://docs.unity3d.com/Manual/BestPractice.html",
            "https://docs.unity3d.com/Manual/MobileOptimization.html",
            "https://docs.unity3d.com/Manual/OptimizingGraphicsPerformance.html",
            "https://docs.unity3d.com/Manual/ScriptingConceptsUnderstanding.html",
            "https://docs.unity3d.com/Manual/JobSystem.html",
            "https://docs.unity3d.com/Manual/dots-introduction.html"
        ]
        
        for url in unity_advanced_topics:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Unity 코드 예제
                        code_blocks = soup.find_all('pre')
                        for block in code_blocks:
                            code_text = block.get_text().strip()
                            if 'using UnityEngine' in code_text or 'MonoBehaviour' in code_text:
                                await self.save_knowledge({
                                    "source": "unity_advanced_docs",
                                    "url": url,
                                    "type": "unity_code_example",
                                    "content": code_text,
                                    "timestamp": datetime.now().isoformat(),
                                    "category": "unity_optimization"
                                })
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ Unity 고급 문서 크롤링 오류: {e}")

    async def crawl_stackoverflow_expert_answers(self):
        """StackOverflow 전문가 답변 크롤링"""
        print("🔍 StackOverflow 전문가 답변 크롤링 중...")
        
        so_queries = [
            "c%23+performance+optimization",
            "c%23+memory+management",
            "c%23+async+await+best+practices",
            "unity3d+optimization",
            "asp.net+core+performance",
            "c%23+design+patterns"
        ]
        
        for query in so_queries:
            try:
                url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={query}&site=stackoverflow&filter=withbody"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for question in data.get('items', [])[:5]:  # 상위 5개만
                            if question.get('score', 0) > 10:  # 10점 이상만
                                await self.save_knowledge({
                                    "source": "stackoverflow_expert",
                                    "url": f"https://stackoverflow.com/questions/{question['question_id']}",
                                    "type": "expert_qa",
                                    "content": f"Q: {question.get('title', '')}\n\nA: {question.get('body', '')}",
                                    "score": question.get('score', 0),
                                    "timestamp": datetime.now().isoformat(),
                                    "category": "expert_solutions"
                                })
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ StackOverflow 크롤링 오류: {e}")

    async def generate_synthetic_patterns(self):
        """고급 C# 패턴 합성 생성"""
        print("🔍 고급 C# 패턴 합성 생성 중...")
        
        design_patterns = {
            "Repository Pattern": """
public interface IRepository<T> where T : class
{
    Task<T> GetByIdAsync(int id);
    Task<IEnumerable<T>> GetAllAsync();
    Task<T> AddAsync(T entity);
    Task UpdateAsync(T entity);
    Task DeleteAsync(int id);
}

public class GenericRepository<T> : IRepository<T> where T : class
{
    private readonly DbContext _context;
    private readonly DbSet<T> _dbSet;

    public GenericRepository(DbContext context)
    {
        _context = context;
        _dbSet = context.Set<T>();
    }

    public async Task<T> GetByIdAsync(int id)
    {
        return await _dbSet.FindAsync(id);
    }
}
""",
            "Unity Object Pool Pattern": """
using UnityEngine;
using System.Collections.Generic;

public class ObjectPool<T> : MonoBehaviour where T : MonoBehaviour
{
    [SerializeField] private T prefab;
    [SerializeField] private int poolSize = 10;
    private Queue<T> pool = new Queue<T>();
    
    private void Start()
    {
        for (int i = 0; i < poolSize; i++)
        {
            T obj = Instantiate(prefab);
            obj.gameObject.SetActive(false);
            pool.Enqueue(obj);
        }
    }
    
    public T GetFromPool()
    {
        if (pool.Count > 0)
        {
            T obj = pool.Dequeue();
            obj.gameObject.SetActive(true);
            return obj;
        }
        return Instantiate(prefab);
    }
}
"""
        }
        
        for pattern_name, code in design_patterns.items():
            await self.save_knowledge({
                "source": "synthetic_patterns",
                "type": "design_pattern", 
                "pattern_name": pattern_name,
                "content": code,
                "timestamp": datetime.now().isoformat(),
                "category": "advanced_patterns"
            })

    async def save_knowledge(self, knowledge_data):
        """지식 데이터 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"knowledge_{timestamp}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=2)

    async def run_comprehensive_crawl(self):
        """종합적인 크롤링 실행"""
        print("🚀 고급 C# 지식 수집 시작!")
        print("=" * 60)
        
        await self.start_session()
        
        try:
            # 병렬로 모든 크롤링 실행
            await asyncio.gather(
                self.crawl_microsoft_docs(),
                self.crawl_github_advanced_patterns(),
                self.crawl_unity_advanced_docs(),
                self.crawl_stackoverflow_expert_answers(),
                self.generate_synthetic_patterns(),
                return_exceptions=True
            )
            
            print("✅ 고급 C# 지식 수집 완료!")
            
            # 수집된 데이터 통계
            data_files = list(self.data_dir.glob("*.json"))
            print(f"📊 총 {len(data_files)}개의 지식 파일 수집됨")
            
            # 카테고리별 통계
            categories = {}
            for file in data_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        category = data.get('category', 'unknown')
                        categories[category] = categories.get(category, 0) + 1
                except:
                    pass
            
            print("\n📈 카테고리별 수집 현황:")
            for category, count in categories.items():
                print(f"   {category}: {count}개")
                
        finally:
            await self.close_session()

async def main():
    crawler = AdvancedCSharpCrawler()
    await crawler.run_comprehensive_crawl()

if __name__ == "__main__":
    asyncio.run(main()) 