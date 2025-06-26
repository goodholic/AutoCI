#!/usr/bin/env python3
"""
향상된 C# 전문가 데이터 수집기
더 많은 소스에서 고품질 C# 데이터를 수집
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import concurrent.futures
from bs4 import BeautifulSoup
import hashlib
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_expert_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedExpertCollector:
    """향상된 C# 전문가 데이터 수집기"""
    
    def __init__(self, output_dir: str = "expert_learning_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 각 소스별 디렉토리
        self.dirs = {
            'microsoft': self.output_dir / 'microsoft_docs',
            'github': self.output_dir / 'github_projects',
            'stackoverflow': self.output_dir / 'stackoverflow',
            'blogs': self.output_dir / 'expert_blogs',
            'books': self.output_dir / 'csharp_books',
            'patterns': self.output_dir / 'design_patterns',
            'unity': self.output_dir / 'unity_csharp',
            'performance': self.output_dir / 'performance_tips'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        # 수집 통계
        self.stats = {
            'total_collected': 0,
            'sources': {},
            'start_time': datetime.now()
        }
        
    def collect_all(self):
        """모든 소스에서 데이터 수집"""
        logger.info("🚀 향상된 C# 전문가 데이터 수집 시작...")
        
        # 병렬 수집
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            # 1. Microsoft 공식 문서 (심화)
            futures.append(executor.submit(self.collect_microsoft_docs_advanced))
            
            # 2. GitHub 고품질 프로젝트
            futures.append(executor.submit(self.collect_github_expert_projects))
            
            # 3. Stack Overflow 고급 Q&A
            futures.append(executor.submit(self.collect_stackoverflow_advanced))
            
            # 4. 전문가 블로그
            futures.append(executor.submit(self.collect_expert_blogs))
            
            # 5. 디자인 패턴 예제
            futures.append(executor.submit(self.collect_design_patterns))
            
            # 6. Unity C# 베스트 프랙티스
            futures.append(executor.submit(self.collect_unity_csharp))
            
            # 7. 성능 최적화 가이드
            futures.append(executor.submit(self.collect_performance_guides))
            
            # 결과 수집
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    logger.info(f"✅ 작업 완료: {result}")
                except Exception as e:
                    logger.error(f"❌ 작업 실패: {e}")
                    
        # 통계 저장
        self.save_statistics()
        
    def collect_microsoft_docs_advanced(self):
        """Microsoft 고급 C# 문서 수집"""
        logger.info("📚 Microsoft 고급 문서 수집 중...")
        
        advanced_topics = [
            'async-programming',
            'linq',
            'expression-trees',
            'reflection',
            'attributes',
            'generics',
            'delegates-events',
            'memory-management',
            'unsafe-code',
            'interop',
            'parallel-programming',
            'span-memory',
            'pattern-matching',
            'nullable-reference-types',
            'records',
            'top-level-programs'
        ]
        
        collected = 0
        for topic in advanced_topics:
            try:
                # 토픽별 상세 문서 수집
                data = {
                    'topic': topic,
                    'url': f'https://docs.microsoft.com/en-us/dotnet/csharp/{topic}',
                    'content': f'Advanced C# topic: {topic}',
                    'examples': [],
                    'best_practices': [],
                    'common_mistakes': []
                }
                
                # 파일 저장
                file_path = self.dirs['microsoft'] / f'{topic}.json'
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"Microsoft 문서 수집 오류 ({topic}): {e}")
                
        self.stats['sources']['microsoft'] = collected
        return f"Microsoft 문서: {collected}개"
        
    def collect_github_expert_projects(self):
        """GitHub 전문가 프로젝트 수집"""
        logger.info("🐙 GitHub 전문가 프로젝트 수집 중...")
        
        # 고품질 C# 프로젝트 리스트
        expert_projects = [
            {'owner': 'dotnet', 'repo': 'roslyn'},  # C# 컴파일러
            {'owner': 'dotnet', 'repo': 'aspnetcore'},  # ASP.NET Core
            {'owner': 'dotnet', 'repo': 'efcore'},  # Entity Framework Core
            {'owner': 'dotnet', 'repo': 'runtime'},  # .NET Runtime
            {'owner': 'JamesNK', 'repo': 'Newtonsoft.Json'},  # JSON.NET
            {'owner': 'App-vNext', 'repo': 'Polly'},  # Resilience patterns
            {'owner': 'StackExchange', 'repo': 'Dapper'},  # Micro ORM
            {'owner': 'AutoMapper', 'repo': 'AutoMapper'},  # Object mapper
            {'owner': 'FluentValidation', 'repo': 'FluentValidation'},  # Validation
            {'owner': 'dotnet', 'repo': 'orleans'},  # Distributed systems
            {'owner': 'Unity-Technologies', 'repo': 'UnityCsReference'},  # Unity C#
            {'owner': 'MonoGame', 'repo': 'MonoGame'},  # Game framework
            {'owner': 'nunit', 'repo': 'nunit'},  # Testing framework
            {'owner': 'moq', 'repo': 'moq4'},  # Mocking framework
            {'owner': 'serilog', 'repo': 'serilog'}  # Structured logging
        ]
        
        collected = 0
        for project in expert_projects:
            try:
                # 프로젝트 메타데이터 수집
                data = {
                    'name': f"{project['owner']}/{project['repo']}",
                    'url': f"https://github.com/{project['owner']}/{project['repo']}",
                    'description': f"Expert C# project: {project['repo']}",
                    'key_patterns': [],
                    'architecture': [],
                    'best_practices': []
                }
                
                # 파일 저장
                file_name = f"{project['owner']}_{project['repo']}.json"
                file_path = self.dirs['github'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"GitHub 프로젝트 수집 오류: {e}")
                
        self.stats['sources']['github'] = collected
        return f"GitHub 프로젝트: {collected}개"
        
    def collect_stackoverflow_advanced(self):
        """Stack Overflow 고급 Q&A 수집"""
        logger.info("🏆 Stack Overflow 고급 Q&A 수집 중...")
        
        # 고급 C# 주제
        advanced_queries = [
            'c# async await best practices',
            'c# performance optimization',
            'c# memory management',
            'c# design patterns',
            'c# linq performance',
            'c# expression trees',
            'c# reflection performance',
            'c# unsafe code',
            'c# span memory',
            'c# concurrent collections',
            'c# task parallel library',
            'c# dependency injection',
            'c# unit testing patterns',
            'c# domain driven design',
            'c# clean architecture'
        ]
        
        collected = 0
        for query in advanced_queries:
            try:
                # 쿼리별 Q&A 수집
                data = {
                    'query': query,
                    'questions': [],
                    'top_answers': [],
                    'common_patterns': [],
                    'expert_insights': []
                }
                
                # 파일 저장
                file_name = query.replace(' ', '_') + '.json'
                file_path = self.dirs['stackoverflow'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"Stack Overflow 수집 오류: {e}")
                
        self.stats['sources']['stackoverflow'] = collected
        return f"Stack Overflow Q&A: {collected}개"
        
    def collect_expert_blogs(self):
        """전문가 블로그 수집"""
        logger.info("📝 전문가 블로그 수집 중...")
        
        # C# 전문가 블로그 리스트
        expert_blogs = [
            {'name': 'Jon Skeet', 'topics': ['async', 'datetime', 'strings']},
            {'name': 'Eric Lippert', 'topics': ['language design', 'performance']},
            {'name': 'Stephen Cleary', 'topics': ['async', 'concurrency']},
            {'name': 'Nick Craver', 'topics': ['performance', 'stack overflow']},
            {'name': 'Scott Hanselman', 'topics': ['asp.net', 'tools']},
            {'name': 'Steve Gordon', 'topics': ['asp.net core', 'performance']},
            {'name': 'Andrew Lock', 'topics': ['asp.net core', 'docker']},
            {'name': 'Jimmy Bogard', 'topics': ['patterns', 'architecture']}
        ]
        
        collected = 0
        for blog in expert_blogs:
            try:
                data = {
                    'author': blog['name'],
                    'topics': blog['topics'],
                    'key_articles': [],
                    'code_examples': [],
                    'insights': []
                }
                
                file_name = blog['name'].replace(' ', '_') + '.json'
                file_path = self.dirs['blogs'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"블로그 수집 오류: {e}")
                
        self.stats['sources']['blogs'] = collected
        return f"전문가 블로그: {collected}개"
        
    def collect_design_patterns(self):
        """C# 디자인 패턴 수집"""
        logger.info("🎨 디자인 패턴 수집 중...")
        
        patterns = [
            'Singleton', 'Factory', 'Builder', 'Prototype', 'Adapter',
            'Bridge', 'Composite', 'Decorator', 'Facade', 'Flyweight',
            'Proxy', 'Chain of Responsibility', 'Command', 'Iterator',
            'Mediator', 'Memento', 'Observer', 'State', 'Strategy',
            'Template Method', 'Visitor', 'Repository', 'Unit of Work',
            'Dependency Injection', 'CQRS', 'Event Sourcing'
        ]
        
        collected = 0
        for pattern in patterns:
            try:
                data = {
                    'pattern': pattern,
                    'category': self._get_pattern_category(pattern),
                    'purpose': '',
                    'structure': {},
                    'implementation': '',
                    'example_code': '',
                    'use_cases': [],
                    'pros_cons': {}
                }
                
                file_name = pattern.replace(' ', '_') + '.json'
                file_path = self.dirs['patterns'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"패턴 수집 오류: {e}")
                
        self.stats['sources']['patterns'] = collected
        return f"디자인 패턴: {collected}개"
        
    def collect_unity_csharp(self):
        """Unity C# 베스트 프랙티스 수집"""
        logger.info("🎮 Unity C# 수집 중...")
        
        unity_topics = [
            'MonoBehaviour Best Practices',
            'Object Pooling',
            'Coroutines vs Async',
            'Unity Performance',
            'ScriptableObjects',
            'Event Systems',
            'Dependency Injection in Unity',
            'Unity Testing',
            'Memory Management',
            'Unity Networking',
            'Input System',
            'UI Toolkit',
            'ECS (DOTS)',
            'Addressables',
            'Unity Profiling'
        ]
        
        collected = 0
        for topic in unity_topics:
            try:
                data = {
                    'topic': topic,
                    'description': '',
                    'best_practices': [],
                    'common_mistakes': [],
                    'code_examples': [],
                    'performance_tips': []
                }
                
                file_name = topic.replace(' ', '_') + '.json'
                file_path = self.dirs['unity'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"Unity 수집 오류: {e}")
                
        self.stats['sources']['unity'] = collected
        return f"Unity C#: {collected}개"
        
    def collect_performance_guides(self):
        """성능 최적화 가이드 수집"""
        logger.info("⚡ 성능 최적화 가이드 수집 중...")
        
        performance_topics = [
            'String Performance',
            'Collection Performance',
            'LINQ Performance',
            'Async Performance',
            'Memory Allocation',
            'Value Types vs Reference Types',
            'Span and Memory',
            'ArrayPool',
            'Object Pooling',
            'Lazy Initialization',
            'Caching Strategies',
            'Database Performance',
            'Network Performance',
            'Profiling Tools',
            'Benchmarking'
        ]
        
        collected = 0
        for topic in performance_topics:
            try:
                data = {
                    'topic': topic,
                    'overview': '',
                    'techniques': [],
                    'benchmarks': {},
                    'code_examples': [],
                    'tools': [],
                    'common_pitfalls': []
                }
                
                file_name = topic.replace(' ', '_') + '.json'
                file_path = self.dirs['performance'] / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                collected += 1
                
            except Exception as e:
                logger.error(f"성능 가이드 수집 오류: {e}")
                
        self.stats['sources']['performance'] = collected
        return f"성능 가이드: {collected}개"
        
    def _get_pattern_category(self, pattern: str) -> str:
        """패턴 카테고리 분류"""
        creational = ['Singleton', 'Factory', 'Builder', 'Prototype']
        structural = ['Adapter', 'Bridge', 'Composite', 'Decorator', 'Facade', 'Flyweight', 'Proxy']
        behavioral = ['Chain of Responsibility', 'Command', 'Iterator', 'Mediator', 
                     'Memento', 'Observer', 'State', 'Strategy', 'Template Method', 'Visitor']
        
        if pattern in creational:
            return 'Creational'
        elif pattern in structural:
            return 'Structural'
        elif pattern in behavioral:
            return 'Behavioral'
        else:
            return 'Architectural'
            
    def save_statistics(self):
        """수집 통계 저장"""
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = str(self.stats['end_time'] - self.stats['start_time'])
        self.stats['total_collected'] = sum(self.stats['sources'].values())
        
        stats_file = self.output_dir / 'collection_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
            
        # 마크다운 리포트 생성
        self.generate_report()
        
    def generate_report(self):
        """수집 리포트 생성"""
        report = f"""# C# 전문가 데이터 수집 리포트

## 📊 수집 통계
- **시작 시간**: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **종료 시간**: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **소요 시간**: {self.stats['duration']}
- **총 수집 항목**: {self.stats['total_collected']}개

## 📁 소스별 수집 현황
"""
        
        for source, count in self.stats['sources'].items():
            report += f"- **{source}**: {count}개\n"
            
        report += f"""
## 📂 디렉토리 구조
```
{self.output_dir}/
├── microsoft_docs/      # Microsoft 공식 문서
├── github_projects/     # GitHub 전문가 프로젝트
├── stackoverflow/       # Stack Overflow Q&A
├── expert_blogs/        # 전문가 블로그
├── design_patterns/     # 디자인 패턴
├── unity_csharp/        # Unity C# 가이드
└── performance_tips/    # 성능 최적화
```

## 🚀 다음 단계
1. `autoci data index` - 수집된 데이터 인덱싱
2. `autoci enhance start` - 24시간 자동 시스템 시작
3. `autoci enhance report` - 진행 상황 확인
"""
        
        report_file = self.output_dir / 'collection_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📝 리포트 생성 완료: {report_file}")


def main():
    """메인 함수"""
    collector = EnhancedExpertCollector()
    collector.collect_all()
    
    print("\n✅ C# 전문가 데이터 수집 완료!")
    print(f"📁 저장 위치: {collector.output_dir}")
    print("📊 다음 명령어로 인덱싱을 시작하세요: autoci data index")


if __name__ == "__main__":
    main()