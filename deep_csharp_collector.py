#!/usr/bin/env python3
"""
심층 C# 전문가 데이터 수집기
실제 웹에서 고품질 C# 데이터를 수집
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import aiofiles
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import re
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_csharp_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepCSharpCollector:
    """심층 C# 데이터 수집기"""
    
    def __init__(self, output_dir: str = "expert_learning_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 카테고리별 디렉토리
        self.categories = {
            'microsoft_docs': {
                'path': self.output_dir / 'microsoft_docs',
                'base_url': 'https://docs.microsoft.com/en-us/dotnet/csharp/',
                'topics': [
                    'programming-guide/concepts/async/',
                    'programming-guide/concepts/linq/',
                    'programming-guide/concepts/expression-trees/',
                    'programming-guide/generics/',
                    'programming-guide/delegates/',
                    'programming-guide/events/',
                    'programming-guide/indexers/',
                    'programming-guide/interfaces/',
                    'language-reference/operators/',
                    'language-reference/keywords/',
                    'language-reference/attributes/',
                    'whats-new/csharp-10',
                    'whats-new/csharp-11',
                    'tutorials/pattern-matching',
                    'tutorials/nullable-reference-types'
                ]
            },
            'github_samples': {
                'path': self.output_dir / 'github_samples',
                'repos': [
                    {'owner': 'dotnet', 'repo': 'samples'},
                    {'owner': 'dotnet', 'repo': 'docs'},
                    {'owner': 'dotnet', 'repo': 'AspNetCore.Docs'},
                    {'owner': 'microsoft', 'repo': 'dotnet-samples'},
                    {'owner': 'Azure-Samples', 'repo': 'cognitive-services-speech-sdk'},
                    {'owner': 'Unity-Technologies', 'repo': 'EntityComponentSystemSamples'},
                    {'owner': 'dotnet', 'repo': 'machinelearning-samples'},
                    {'owner': 'dotnet', 'repo': 'blazor-samples'}
                ]
            },
            'nuget_packages': {
                'path': self.output_dir / 'nuget_packages',
                'packages': [
                    'Newtonsoft.Json',
                    'Microsoft.EntityFrameworkCore',
                    'Serilog',
                    'AutoMapper',
                    'FluentValidation',
                    'Polly',
                    'MediatR',
                    'Dapper',
                    'NUnit',
                    'xunit',
                    'Moq',
                    'FluentAssertions',
                    'BenchmarkDotNet',
                    'MessagePack',
                    'StackExchange.Redis'
                ]
            },
            'stackoverflow_advanced': {
                'path': self.output_dir / 'stackoverflow_advanced',
                'tags': [
                    'c#+async-await',
                    'c#+performance',
                    'c#+memory-management',
                    'c#+linq',
                    'c#+entity-framework-core',
                    'c#+dependency-injection',
                    'c#+design-patterns',
                    'c#+unit-testing',
                    'c#+blazor',
                    'c#+grpc'
                ],
                'min_score': 50,
                'min_views': 10000
            },
            'expert_blogs': {
                'path': self.output_dir / 'expert_blogs',
                'sources': [
                    {'name': 'Jon Skeet', 'url': 'https://codeblog.jonskeet.uk/'},
                    {'name': 'Eric Lippert', 'url': 'https://ericlippert.com/'},
                    {'name': 'Stephen Cleary', 'url': 'https://blog.stephencleary.com/'},
                    {'name': 'Steve Gordon', 'url': 'https://www.stevejgordon.co.uk/'},
                    {'name': 'Andrew Lock', 'url': 'https://andrewlock.net/'},
                    {'name': 'Nick Chapsas', 'url': 'https://www.youtube.com/@nickchapsas'},
                    {'name': 'David Fowler', 'url': 'https://twitter.com/davidfowl'}
                ]
            },
            'performance_guides': {
                'path': self.output_dir / 'performance_guides',
                'sources': [
                    'https://docs.microsoft.com/en-us/dotnet/framework/performance/',
                    'https://github.com/dotnet/performance',
                    'https://devblogs.microsoft.com/dotnet/category/performance/'
                ]
            },
            'design_patterns': {
                'path': self.output_dir / 'design_patterns',
                'patterns': [
                    'Singleton', 'Factory Method', 'Abstract Factory', 'Builder', 'Prototype',
                    'Adapter', 'Bridge', 'Composite', 'Decorator', 'Facade', 'Flyweight', 'Proxy',
                    'Chain of Responsibility', 'Command', 'Iterator', 'Mediator', 'Memento',
                    'Observer', 'State', 'Strategy', 'Template Method', 'Visitor',
                    'Repository', 'Unit of Work', 'Specification', 'CQRS', 'Event Sourcing'
                ]
            },
            'unity_best_practices': {
                'path': self.output_dir / 'unity_best_practices',
                'topics': [
                    'MonoBehaviour lifecycle',
                    'Object pooling',
                    'Coroutines vs async',
                    'Performance optimization',
                    'Memory management',
                    'ScriptableObjects',
                    'Event systems',
                    'Input system',
                    'Addressables',
                    'DOTS/ECS'
                ]
            }
        }
        
        # 디렉토리 생성
        for category_info in self.categories.values():
            category_info['path'].mkdir(exist_ok=True)
            
        # 데이터베이스 초기화
        self.init_database()
        
        # 세션 설정
        self.session = None
        
        # 수집 통계
        self.stats = {
            'total_collected': 0,
            'total_size_mb': 0,
            'categories': {},
            'errors': [],
            'start_time': datetime.now()
        }
        
    def init_database(self):
        """데이터베이스 초기화"""
        db_path = self.output_dir / 'collection_index.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS collected_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            category TEXT,
            subcategory TEXT,
            title TEXT,
            content_type TEXT,
            file_path TEXT,
            size_bytes INTEGER,
            quality_score REAL,
            collected_at TIMESTAMP,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id INTEGER,
            language TEXT,
            code TEXT,
            description TEXT,
            tags TEXT,
            FOREIGN KEY (data_id) REFERENCES collected_data(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    async def collect_all_async(self):
        """비동기로 모든 데이터 수집"""
        logger.info("🚀 심층 C# 데이터 수집 시작...")
        
        # aiohttp 세션 생성
        async with aiohttp.ClientSession() as self.session:
            tasks = []
            
            # 각 카테고리별 수집 작업
            tasks.append(self.collect_microsoft_docs_async())
            tasks.append(self.collect_github_samples_async())
            tasks.append(self.collect_nuget_packages_async())
            tasks.append(self.collect_stackoverflow_async())
            tasks.append(self.collect_expert_blogs_async())
            tasks.append(self.collect_performance_guides_async())
            tasks.append(self.collect_design_patterns_async())
            tasks.append(self.collect_unity_practices_async())
            
            # 모든 작업 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"수집 작업 실패: {result}")
                    self.stats['errors'].append(str(result))
                    
        # 통계 저장
        self.save_statistics()
        
    async def collect_microsoft_docs_async(self):
        """Microsoft 문서 수집"""
        logger.info("📚 Microsoft 문서 수집 중...")
        
        category_info = self.categories['microsoft_docs']
        collected = 0
        
        for topic in category_info['topics']:
            try:
                url = urljoin(category_info['base_url'], topic)
                content = await self.fetch_url_async(url)
                
                if content:
                    # HTML 파싱
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # 제목 추출
                    title = soup.find('h1')
                    title_text = title.text.strip() if title else topic.replace('/', ' ').title()
                    
                    # 본문 추출
                    article = soup.find('article') or soup.find('main')
                    if article:
                        # 코드 샘플 추출
                        code_samples = self.extract_code_samples(article)
                        
                        # 텍스트 콘텐츠
                        text_content = article.get_text(separator='\n', strip=True)
                        
                        # 데이터 구조화
                        doc_data = {
                            'url': url,
                            'title': title_text,
                            'topic': topic,
                            'content': text_content,
                            'code_samples': code_samples,
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        # 파일 저장
                        file_name = f"{topic.replace('/', '_')}.json"
                        file_path = category_info['path'] / file_name
                        
                        await self.save_json_async(file_path, doc_data)
                        
                        # 데이터베이스 기록
                        self.save_to_database(
                            url=url,
                            category='microsoft_docs',
                            subcategory=topic.split('/')[0],
                            title=title_text,
                            content_type='documentation',
                            file_path=str(file_path),
                            size_bytes=len(json.dumps(doc_data)),
                            quality_score=0.95
                        )
                        
                        collected += 1
                        
                        # 관련 링크 수집
                        related_links = self.extract_related_links(soup, category_info['base_url'])
                        for link in related_links[:5]:  # 최대 5개
                            await self.collect_related_doc(link, category_info)
                            
            except Exception as e:
                logger.error(f"Microsoft 문서 수집 오류 ({topic}): {e}")
                
        self.stats['categories']['microsoft_docs'] = collected
        logger.info(f"✅ Microsoft 문서 {collected}개 수집 완료")
        
    def extract_code_samples(self, element) -> List[Dict]:
        """코드 샘플 추출"""
        code_samples = []
        
        # <pre><code> 태그 찾기
        for code_block in element.find_all(['pre', 'code']):
            code_text = code_block.get_text(strip=True)
            
            if len(code_text) > 50:  # 의미있는 코드만
                # 언어 감지
                language = 'csharp'
                class_attr = code_block.get('class', [])
                if isinstance(class_attr, list):
                    for cls in class_attr:
                        if 'language-' in cls:
                            language = cls.replace('language-', '')
                            break
                            
                code_samples.append({
                    'language': language,
                    'code': code_text,
                    'lines': len(code_text.split('\n'))
                })
                
        return code_samples
        
    def extract_related_links(self, soup, base_url: str) -> List[str]:
        """관련 링크 추출"""
        links = []
        
        # 관련 문서 링크 찾기
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # 상대 경로를 절대 경로로
            if href.startswith('/'):
                href = urljoin(base_url, href)
                
            # C# 관련 링크만
            if 'csharp' in href or 'dotnet' in href:
                if href not in links and base_url in href:
                    links.append(href)
                    
        return links
        
    async def collect_related_doc(self, url: str, category_info: Dict):
        """관련 문서 수집"""
        try:
            # 이미 수집했는지 확인
            if self.is_already_collected(url):
                return
                
            content = await self.fetch_url_async(url)
            if content:
                # 간단한 처리 (메인 문서와 동일한 방식)
                # ... (구현 생략)
                pass
                
        except Exception as e:
            logger.debug(f"관련 문서 수집 스킵: {e}")
            
    def is_already_collected(self, url: str) -> bool:
        """이미 수집했는지 확인"""
        db_path = self.output_dir / 'collection_index.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM collected_data WHERE url = ?', (url,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
        
    async def collect_github_samples_async(self):
        """GitHub 샘플 코드 수집"""
        logger.info("🐙 GitHub 샘플 수집 중...")
        
        category_info = self.categories['github_samples']
        collected = 0
        
        for repo_info in category_info['repos']:
            try:
                # GitHub API 사용
                api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/contents"
                
                # 재귀적으로 C# 파일 찾기
                cs_files = await self.find_github_cs_files(api_url)
                
                for file_info in cs_files[:50]:  # 리포당 최대 50개
                    try:
                        # 파일 내용 가져오기
                        file_content = await self.fetch_github_file(file_info['download_url'])
                        
                        if file_content:
                            # 파일 분석
                            analysis = self.analyze_cs_file(file_content)
                            
                            # 데이터 구조화
                            sample_data = {
                                'repo': f"{repo_info['owner']}/{repo_info['repo']}",
                                'file_path': file_info['path'],
                                'file_name': file_info['name'],
                                'content': file_content,
                                'analysis': analysis,
                                'size': file_info['size'],
                                'url': file_info['html_url'],
                                'collected_at': datetime.now().isoformat()
                            }
                            
                            # 파일 저장
                            safe_name = file_info['path'].replace('/', '_')
                            file_path = category_info['path'] / f"{safe_name}.json"
                            
                            await self.save_json_async(file_path, sample_data)
                            
                            # 데이터베이스 기록
                            self.save_to_database(
                                url=file_info['html_url'],
                                category='github_samples',
                                subcategory=repo_info['repo'],
                                title=file_info['name'],
                                content_type='source_code',
                                file_path=str(file_path),
                                size_bytes=file_info['size'],
                                quality_score=analysis['quality_score']
                            )
                            
                            collected += 1
                            
                    except Exception as e:
                        logger.debug(f"파일 수집 스킵: {e}")
                        
            except Exception as e:
                logger.error(f"GitHub 리포 수집 오류: {e}")
                
        self.stats['categories']['github_samples'] = collected
        logger.info(f"✅ GitHub 샘플 {collected}개 수집 완료")
        
    async def find_github_cs_files(self, api_url: str, path: str = '') -> List[Dict]:
        """GitHub에서 C# 파일 찾기"""
        cs_files = []
        
        try:
            headers = {
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AutoCI-Collector'
            }
            
            # GitHub 토큰이 있으면 사용
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
                
            async with self.session.get(api_url, headers=headers) as response:
                if response.status == 200:
                    items = await response.json()
                    
                    for item in items:
                        if item['type'] == 'file' and item['name'].endswith('.cs'):
                            cs_files.append(item)
                        elif item['type'] == 'dir' and len(cs_files) < 100:
                            # 하위 디렉토리 탐색
                            sub_files = await self.find_github_cs_files(item['url'])
                            cs_files.extend(sub_files)
                            
                            if len(cs_files) >= 100:
                                break
                                
        except Exception as e:
            logger.debug(f"GitHub 파일 검색 오류: {e}")
            
        return cs_files
        
    async def fetch_github_file(self, url: str) -> Optional[str]:
        """GitHub 파일 내용 가져오기"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.debug(f"GitHub 파일 가져오기 오류: {e}")
            
        return None
        
    def analyze_cs_file(self, content: str) -> Dict:
        """C# 파일 분석"""
        analysis = {
            'lines': len(content.split('\n')),
            'classes': 0,
            'methods': 0,
            'interfaces': 0,
            'usings': [],
            'namespaces': [],
            'has_async': False,
            'has_linq': False,
            'has_generics': False,
            'complexity_estimate': 0,
            'quality_score': 0.7
        }
        
        # 정규식으로 분석
        analysis['classes'] = len(re.findall(r'\bclass\s+\w+', content))
        analysis['methods'] = len(re.findall(r'(public|private|protected|internal)\s+\w+\s+\w+\s*\(', content))
        analysis['interfaces'] = len(re.findall(r'\binterface\s+\w+', content))
        
        # using 문 추출
        usings = re.findall(r'using\s+([\w.]+);', content)
        analysis['usings'] = list(set(usings))
        
        # namespace 추출
        namespaces = re.findall(r'namespace\s+([\w.]+)', content)
        analysis['namespaces'] = list(set(namespaces))
        
        # 특징 확인
        analysis['has_async'] = 'async' in content and 'await' in content
        analysis['has_linq'] = any(linq in content for linq in ['.Where(', '.Select(', '.OrderBy(', 'from '])
        analysis['has_generics'] = '<' in content and '>' in content
        
        # 복잡도 추정 (간단한 휴리스틱)
        keywords = ['if', 'else', 'for', 'foreach', 'while', 'switch', 'try', 'catch']
        for keyword in keywords:
            analysis['complexity_estimate'] += content.count(keyword)
            
        # 품질 점수 계산
        if analysis['has_async']:
            analysis['quality_score'] += 0.1
        if analysis['has_linq']:
            analysis['quality_score'] += 0.05
        if len(analysis['usings']) > 5:
            analysis['quality_score'] += 0.05
        if analysis['methods'] > 0:
            analysis['quality_score'] += 0.1
            
        analysis['quality_score'] = min(analysis['quality_score'], 1.0)
        
        return analysis
        
    async def collect_nuget_packages_async(self):
        """NuGet 패키지 정보 수집"""
        logger.info("📦 NuGet 패키지 정보 수집 중...")
        
        category_info = self.categories['nuget_packages']
        collected = 0
        
        for package_name in category_info['packages']:
            try:
                # NuGet API
                api_url = f"https://api.nuget.org/v3-flatcontainer/{package_name.lower()}/index.json"
                
                async with self.session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        versions = data.get('versions', [])
                        
                        if versions:
                            # 최신 버전 정보 가져오기
                            latest_version = versions[-1]
                            
                            # 패키지 메타데이터
                            metadata_url = f"https://api.nuget.org/v3/registration5-semver1/{package_name.lower()}/{latest_version}.json"
                            
                            async with self.session.get(metadata_url) as meta_response:
                                if meta_response.status == 200:
                                    metadata = await meta_response.json()
                                    
                                    # 카탈로그 엔트리
                                    catalog_entry = metadata.get('catalogEntry', {})
                                    
                                    package_data = {
                                        'name': package_name,
                                        'version': latest_version,
                                        'description': catalog_entry.get('description', ''),
                                        'authors': catalog_entry.get('authors', ''),
                                        'project_url': catalog_entry.get('projectUrl', ''),
                                        'repository': catalog_entry.get('repository', {}),
                                        'tags': catalog_entry.get('tags', []),
                                        'dependencies': catalog_entry.get('dependencyGroups', []),
                                        'download_count': 0,  # 별도 API 필요
                                        'collected_at': datetime.now().isoformat()
                                    }
                                    
                                    # README 가져오기
                                    readme_url = catalog_entry.get('readmeUrl')
                                    if readme_url:
                                        readme_content = await self.fetch_url_async(readme_url)
                                        if readme_content:
                                            package_data['readme'] = readme_content
                                            
                                    # 파일 저장
                                    file_path = category_info['path'] / f"{package_name}.json"
                                    await self.save_json_async(file_path, package_data)
                                    
                                    # 데이터베이스 기록
                                    self.save_to_database(
                                        url=f"https://www.nuget.org/packages/{package_name}",
                                        category='nuget_packages',
                                        subcategory='package',
                                        title=package_name,
                                        content_type='package_info',
                                        file_path=str(file_path),
                                        size_bytes=len(json.dumps(package_data)),
                                        quality_score=0.9
                                    )
                                    
                                    collected += 1
                                    
            except Exception as e:
                logger.error(f"NuGet 패키지 수집 오류 ({package_name}): {e}")
                
        self.stats['categories']['nuget_packages'] = collected
        logger.info(f"✅ NuGet 패키지 {collected}개 수집 완료")
        
    async def collect_stackoverflow_async(self):
        """Stack Overflow 고급 Q&A 수집"""
        logger.info("🏆 Stack Overflow 고급 Q&A 수집 중...")
        
        category_info = self.categories['stackoverflow_advanced']
        collected = 0
        
        # Stack Exchange API
        base_url = "https://api.stackexchange.com/2.3/questions"
        
        for tag in category_info['tags']:
            try:
                params = {
                    'order': 'desc',
                    'sort': 'votes',
                    'tagged': tag,
                    'site': 'stackoverflow',
                    'filter': '!9_bDE(fI5',  # 본문 포함
                    'pagesize': 20
                }
                
                async with self.session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        questions = data.get('items', [])
                        
                        for question in questions:
                            # 품질 필터링
                            if (question.get('score', 0) >= category_info['min_score'] and
                                question.get('view_count', 0) >= category_info['min_views']):
                                
                                # 답변 가져오기
                                answers = await self.fetch_stackoverflow_answers(question['question_id'])
                                
                                qa_data = {
                                    'question_id': question['question_id'],
                                    'title': question['title'],
                                    'body': question.get('body', ''),
                                    'tags': question['tags'],
                                    'score': question['score'],
                                    'view_count': question['view_count'],
                                    'creation_date': question['creation_date'],
                                    'link': question['link'],
                                    'answers': answers,
                                    'collected_at': datetime.now().isoformat()
                                }
                                
                                # 코드 샘플 추출
                                code_samples = self.extract_code_from_html(question.get('body', ''))
                                for answer in answers:
                                    code_samples.extend(
                                        self.extract_code_from_html(answer.get('body', ''))
                                    )
                                qa_data['code_samples'] = code_samples
                                
                                # 파일 저장
                                file_name = f"{tag.replace('+', '_')}_{question['question_id']}.json"
                                file_path = category_info['path'] / file_name
                                
                                await self.save_json_async(file_path, qa_data)
                                
                                # 데이터베이스 기록
                                self.save_to_database(
                                    url=question['link'],
                                    category='stackoverflow_advanced',
                                    subcategory=tag,
                                    title=question['title'],
                                    content_type='qa',
                                    file_path=str(file_path),
                                    size_bytes=len(json.dumps(qa_data)),
                                    quality_score=min(question['score'] / 100, 1.0)
                                )
                                
                                collected += 1
                                
                                if collected >= 100:  # 최대 100개
                                    break
                                    
            except Exception as e:
                logger.error(f"Stack Overflow 수집 오류 ({tag}): {e}")
                
        self.stats['categories']['stackoverflow_advanced'] = collected
        logger.info(f"✅ Stack Overflow Q&A {collected}개 수집 완료")
        
    async def fetch_stackoverflow_answers(self, question_id: int) -> List[Dict]:
        """Stack Overflow 답변 가져오기"""
        answers = []
        
        try:
            url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'site': 'stackoverflow',
                'filter': '!9_bDE(fI5'  # 본문 포함
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for answer in data.get('items', [])[:3]:  # 상위 3개 답변
                        answers.append({
                            'answer_id': answer['answer_id'],
                            'body': answer.get('body', ''),
                            'score': answer['score'],
                            'is_accepted': answer.get('is_accepted', False)
                        })
                        
        except Exception as e:
            logger.debug(f"답변 가져오기 오류: {e}")
            
        return answers
        
    def extract_code_from_html(self, html_content: str) -> List[Dict]:
        """HTML에서 코드 추출"""
        code_samples = []
        
        if not html_content:
            return code_samples
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # <code> 태그 찾기
        for code_tag in soup.find_all('code'):
            code_text = code_tag.get_text(strip=True)
            
            # C# 코드인지 확인 (간단한 휴리스틱)
            if any(keyword in code_text for keyword in ['class', 'public', 'private', 'using', 'namespace', 'var']):
                code_samples.append({
                    'code': code_text,
                    'type': 'inline' if len(code_text) < 100 else 'block'
                })
                
        # <pre><code> 블록
        for pre_tag in soup.find_all('pre'):
            code_tag = pre_tag.find('code')
            if code_tag:
                code_text = code_tag.get_text(strip=True)
                code_samples.append({
                    'code': code_text,
                    'type': 'block'
                })
                
        return code_samples
        
    async def collect_expert_blogs_async(self):
        """전문가 블로그 수집"""
        logger.info("📝 전문가 블로그 수집 중...")
        
        category_info = self.categories['expert_blogs']
        collected = 0
        
        # 실제 구현에서는 각 블로그의 RSS 피드나 사이트맵을 사용
        # 여기서는 메타데이터만 수집
        
        for blog in category_info['sources']:
            try:
                blog_data = {
                    'name': blog['name'],
                    'url': blog['url'],
                    'type': 'blog',
                    'topics': self.get_expert_topics(blog['name']),
                    'sample_posts': [],
                    'collected_at': datetime.now().isoformat()
                }
                
                # RSS 피드 확인
                rss_url = await self.find_rss_feed(blog['url'])
                if rss_url:
                    posts = await self.parse_rss_feed(rss_url)
                    blog_data['sample_posts'] = posts[:10]  # 최근 10개
                    
                # 파일 저장
                file_name = f"{blog['name'].replace(' ', '_')}.json"
                file_path = category_info['path'] / file_name
                
                await self.save_json_async(file_path, blog_data)
                
                # 데이터베이스 기록
                self.save_to_database(
                    url=blog['url'],
                    category='expert_blogs',
                    subcategory='blog',
                    title=blog['name'],
                    content_type='blog_info',
                    file_path=str(file_path),
                    size_bytes=len(json.dumps(blog_data)),
                    quality_score=0.95
                )
                
                collected += 1
                
            except Exception as e:
                logger.error(f"블로그 수집 오류 ({blog['name']}): {e}")
                
        self.stats['categories']['expert_blogs'] = collected
        logger.info(f"✅ 전문가 블로그 {collected}개 수집 완료")
        
    def get_expert_topics(self, expert_name: str) -> List[str]:
        """전문가별 주요 토픽"""
        topics_map = {
            'Jon Skeet': ['async/await', 'DateTime', 'strings', 'LINQ'],
            'Eric Lippert': ['language design', 'performance', 'compiler'],
            'Stephen Cleary': ['async/await', 'concurrency', 'tasks'],
            'Steve Gordon': ['ASP.NET Core', 'performance', 'HttpClient'],
            'Andrew Lock': ['ASP.NET Core', 'Docker', 'configuration'],
            'Nick Chapsas': ['performance', 'best practices', 'new features'],
            'David Fowler': ['ASP.NET Core', 'SignalR', 'performance']
        }
        
        return topics_map.get(expert_name, ['C#', '.NET'])
        
    async def find_rss_feed(self, blog_url: str) -> Optional[str]:
        """RSS 피드 URL 찾기"""
        try:
            content = await self.fetch_url_async(blog_url)
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                
                # RSS 링크 찾기
                rss_link = soup.find('link', {'type': 'application/rss+xml'})
                if rss_link and rss_link.get('href'):
                    return urljoin(blog_url, rss_link['href'])
                    
                # 일반적인 RSS 경로 시도
                common_paths = ['/rss', '/feed', '/rss.xml', '/feed.xml', '/atom.xml']
                for path in common_paths:
                    test_url = urljoin(blog_url, path)
                    if await self.url_exists(test_url):
                        return test_url
                        
        except Exception as e:
            logger.debug(f"RSS 피드 찾기 오류: {e}")
            
        return None
        
    async def url_exists(self, url: str) -> bool:
        """URL 존재 확인"""
        try:
            async with self.session.head(url) as response:
                return response.status == 200
        except:
            return False
            
    async def parse_rss_feed(self, rss_url: str) -> List[Dict]:
        """RSS 피드 파싱"""
        posts = []
        
        try:
            content = await self.fetch_url_async(rss_url)
            if content:
                root = ET.fromstring(content)
                
                # RSS 2.0
                for item in root.findall('.//item')[:10]:
                    post = {
                        'title': item.findtext('title', ''),
                        'link': item.findtext('link', ''),
                        'pubDate': item.findtext('pubDate', ''),
                        'description': item.findtext('description', '')
                    }
                    posts.append(post)
                    
        except Exception as e:
            logger.debug(f"RSS 파싱 오류: {e}")
            
        return posts
        
    async def collect_performance_guides_async(self):
        """성능 가이드 수집"""
        logger.info("⚡ 성능 가이드 수집 중...")
        
        category_info = self.categories['performance_guides']
        collected = 0
        
        performance_topics = [
            'string-performance',
            'collection-performance',
            'linq-performance',
            'async-performance',
            'memory-allocation',
            'span-memory',
            'object-pooling',
            'caching-strategies',
            'profiling-tools',
            'benchmarking'
        ]
        
        for topic in performance_topics:
            try:
                guide_data = {
                    'topic': topic,
                    'title': topic.replace('-', ' ').title(),
                    'sources': [],
                    'best_practices': [],
                    'code_examples': [],
                    'benchmarks': [],
                    'collected_at': datetime.now().isoformat()
                }
                
                # 각 소스에서 관련 정보 수집
                for source_url in category_info['sources']:
                    if 'docs.microsoft.com' in source_url:
                        # Microsoft 문서에서 성능 가이드 찾기
                        search_url = f"{source_url}{topic}"
                        content = await self.fetch_url_async(search_url)
                        
                        if content:
                            guide_data['sources'].append({
                                'url': search_url,
                                'type': 'microsoft_docs'
                            })
                            
                # 파일 저장
                file_path = category_info['path'] / f"{topic}.json"
                await self.save_json_async(file_path, guide_data)
                
                # 데이터베이스 기록
                self.save_to_database(
                    url=f"performance-guide-{topic}",
                    category='performance_guides',
                    subcategory='guide',
                    title=guide_data['title'],
                    content_type='guide',
                    file_path=str(file_path),
                    size_bytes=len(json.dumps(guide_data)),
                    quality_score=0.9
                )
                
                collected += 1
                
            except Exception as e:
                logger.error(f"성능 가이드 수집 오류 ({topic}): {e}")
                
        self.stats['categories']['performance_guides'] = collected
        logger.info(f"✅ 성능 가이드 {collected}개 수집 완료")
        
    async def collect_design_patterns_async(self):
        """디자인 패턴 수집"""
        logger.info("🎨 디자인 패턴 수집 중...")
        
        category_info = self.categories['design_patterns']
        collected = 0
        
        for pattern in category_info['patterns']:
            try:
                pattern_data = {
                    'name': pattern,
                    'category': self.get_pattern_category(pattern),
                    'intent': '',
                    'structure': {},
                    'participants': [],
                    'implementation': {
                        'csharp': ''
                    },
                    'example_usage': '',
                    'when_to_use': [],
                    'pros': [],
                    'cons': [],
                    'related_patterns': [],
                    'collected_at': datetime.now().isoformat()
                }
                
                # 패턴 정보 추가 (실제로는 외부 소스에서 수집)
                pattern_data.update(self.get_pattern_details(pattern))
                
                # 파일 저장
                file_name = f"{pattern.replace(' ', '_')}.json"
                file_path = category_info['path'] / file_name
                
                await self.save_json_async(file_path, pattern_data)
                
                # 데이터베이스 기록
                self.save_to_database(
                    url=f"pattern-{pattern}",
                    category='design_patterns',
                    subcategory=pattern_data['category'],
                    title=pattern,
                    content_type='pattern',
                    file_path=str(file_path),
                    size_bytes=len(json.dumps(pattern_data)),
                    quality_score=0.95
                )
                
                collected += 1
                
            except Exception as e:
                logger.error(f"패턴 수집 오류 ({pattern}): {e}")
                
        self.stats['categories']['design_patterns'] = collected
        logger.info(f"✅ 디자인 패턴 {collected}개 수집 완료")
        
    def get_pattern_category(self, pattern: str) -> str:
        """패턴 카테고리 분류"""
        categories = {
            'Creational': ['Singleton', 'Factory Method', 'Abstract Factory', 'Builder', 'Prototype'],
            'Structural': ['Adapter', 'Bridge', 'Composite', 'Decorator', 'Facade', 'Flyweight', 'Proxy'],
            'Behavioral': ['Chain of Responsibility', 'Command', 'Iterator', 'Mediator', 'Memento',
                          'Observer', 'State', 'Strategy', 'Template Method', 'Visitor'],
            'Architectural': ['Repository', 'Unit of Work', 'Specification', 'CQRS', 'Event Sourcing']
        }
        
        for category, patterns in categories.items():
            if pattern in patterns:
                return category
                
        return 'Other'
        
    def get_pattern_details(self, pattern: str) -> Dict:
        """패턴 상세 정보"""
        # 실제로는 외부 소스에서 수집
        # 여기서는 예시 데이터
        
        details = {
            'Singleton': {
                'intent': 'Ensure a class has only one instance and provide a global point of access to it.',
                'implementation': {
                    'csharp': '''public sealed class Singleton
{
    private static readonly Lazy<Singleton> lazy = 
        new Lazy<Singleton>(() => new Singleton());
    
    public static Singleton Instance => lazy.Value;
    
    private Singleton()
    {
    }
}'''
                },
                'when_to_use': [
                    'When exactly one instance of a class is needed',
                    'When the single instance should be extensible by subclassing'
                ],
                'pros': ['Controlled access to sole instance', 'Reduced namespace pollution'],
                'cons': ['Difficult to test', 'Violates Single Responsibility Principle']
            }
        }
        
        return details.get(pattern, {})
        
    async def collect_unity_practices_async(self):
        """Unity 베스트 프랙티스 수집"""
        logger.info("🎮 Unity 베스트 프랙티스 수집 중...")
        
        category_info = self.categories['unity_best_practices']
        collected = 0
        
        for topic in category_info['topics']:
            try:
                practice_data = {
                    'topic': topic,
                    'description': '',
                    'best_practices': [],
                    'common_mistakes': [],
                    'code_examples': [],
                    'performance_tips': [],
                    'unity_version': '2022.3 LTS',
                    'collected_at': datetime.now().isoformat()
                }
                
                # Unity 특화 정보 추가
                practice_data.update(self.get_unity_practice_details(topic))
                
                # 파일 저장
                file_name = f"{topic.replace(' ', '_').replace('/', '_')}.json"
                file_path = category_info['path'] / file_name
                
                await self.save_json_async(file_path, practice_data)
                
                # 데이터베이스 기록
                self.save_to_database(
                    url=f"unity-practice-{topic}",
                    category='unity_best_practices',
                    subcategory='practice',
                    title=topic,
                    content_type='unity_guide',
                    file_path=str(file_path),
                    size_bytes=len(json.dumps(practice_data)),
                    quality_score=0.9
                )
                
                collected += 1
                
            except Exception as e:
                logger.error(f"Unity 프랙티스 수집 오류 ({topic}): {e}")
                
        self.stats['categories']['unity_best_practices'] = collected
        logger.info(f"✅ Unity 베스트 프랙티스 {collected}개 수집 완료")
        
    def get_unity_practice_details(self, topic: str) -> Dict:
        """Unity 프랙티스 상세 정보"""
        details = {
            'Object pooling': {
                'description': 'Reuse objects instead of instantiating and destroying them repeatedly',
                'best_practices': [
                    'Pre-instantiate objects at scene load',
                    'Use generic pool implementation',
                    'Reset object state when returning to pool'
                ],
                'code_examples': [
                    {
                        'title': 'Generic Object Pool',
                        'code': '''public class ObjectPool<T> where T : MonoBehaviour
{
    private Queue<T> pool = new Queue<T>();
    private T prefab;
    private Transform parent;
    
    public ObjectPool(T prefab, int initialSize, Transform parent = null)
    {
        this.prefab = prefab;
        this.parent = parent;
        
        for (int i = 0; i < initialSize; i++)
        {
            T obj = Object.Instantiate(prefab, parent);
            obj.gameObject.SetActive(false);
            pool.Enqueue(obj);
        }
    }
    
    public T Get()
    {
        if (pool.Count > 0)
        {
            T obj = pool.Dequeue();
            obj.gameObject.SetActive(true);
            return obj;
        }
        else
        {
            return Object.Instantiate(prefab, parent);
        }
    }
    
    public void Return(T obj)
    {
        obj.gameObject.SetActive(false);
        pool.Enqueue(obj);
    }
}'''
                    }
                ]
            }
        }
        
        return details.get(topic, {})
        
    async def fetch_url_async(self, url: str) -> Optional[str]:
        """URL 내용 비동기로 가져오기"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                    
        except asyncio.TimeoutError:
            logger.warning(f"타임아웃: {url}")
        except Exception as e:
            logger.debug(f"URL 가져오기 오류 ({url}): {e}")
            
        return None
        
    async def save_json_async(self, file_path: Path, data: Dict):
        """JSON 파일 비동기 저장"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
                
            # 파일 크기 업데이트
            size_mb = file_path.stat().st_size / 1024 / 1024
            self.stats['total_size_mb'] += size_mb
            self.stats['total_collected'] += 1
            
        except Exception as e:
            logger.error(f"파일 저장 오류 ({file_path}): {e}")
            
    def save_to_database(self, **kwargs):
        """데이터베이스에 수집 정보 저장"""
        db_path = self.output_dir / 'collection_index.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO collected_data 
            (url, category, subcategory, title, content_type, 
             file_path, size_bytes, quality_score, collected_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kwargs.get('url'),
                kwargs.get('category'),
                kwargs.get('subcategory'),
                kwargs.get('title'),
                kwargs.get('content_type'),
                kwargs.get('file_path'),
                kwargs.get('size_bytes', 0),
                kwargs.get('quality_score', 0.7),
                datetime.now(),
                json.dumps(kwargs.get('metadata', {}))
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 오류: {e}")
        finally:
            conn.close()
            
    def save_statistics(self):
        """수집 통계 저장"""
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = str(self.stats['end_time'] - self.stats['start_time'])
        
        # JSON 파일로 저장
        stats_file = self.output_dir / 'deep_collection_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
            
        # 마크다운 리포트 생성
        self.generate_report()
        
    def generate_report(self):
        """수집 리포트 생성"""
        report = f"""# 심층 C# 데이터 수집 리포트

## 📊 수집 통계
- **시작 시간**: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **종료 시간**: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **소요 시간**: {self.stats['duration']}
- **총 수집 항목**: {self.stats['total_collected']}개
- **총 데이터 크기**: {self.stats['total_size_mb']:.2f}MB

## 📁 카테고리별 수집 현황
"""
        
        for category, count in self.stats['categories'].items():
            report += f"- **{category}**: {count}개\n"
            
        if self.stats['errors']:
            report += f"\n## ⚠️ 오류 발생\n"
            for error in self.stats['errors'][:10]:  # 최대 10개
                report += f"- {error}\n"
                
        report += f"""
## 🎯 수집 품질
- Microsoft 공식 문서: 고품질 (0.95)
- GitHub 전문가 프로젝트: 고품질 (0.7-0.9)
- Stack Overflow 답변: 투표 기반 품질
- NuGet 패키지: 검증된 패키지 (0.9)

## 📂 디렉토리 구조
```
{self.output_dir}/
├── microsoft_docs/          # MS 공식 문서
├── github_samples/          # GitHub 샘플 코드
├── nuget_packages/          # NuGet 패키지 정보
├── stackoverflow_advanced/  # SO 고급 Q&A
├── expert_blogs/           # 전문가 블로그
├── performance_guides/     # 성능 가이드
├── design_patterns/        # 디자인 패턴
├── unity_best_practices/   # Unity 가이드
└── collection_index.db     # 수집 인덱스 DB
```

## 🚀 다음 단계
1. `autoci data index` - 수집된 데이터 인덱싱
2. `autoci dual start` - RAG + 파인튜닝 시작
3. `autoci enhance start` - 24시간 자동 시스템 시작

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_file = self.output_dir / 'deep_collection_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📝 리포트 생성 완료: {report_file}")


def main():
    """메인 함수"""
    collector = DeepCSharpCollector()
    
    # 비동기 실행
    asyncio.run(collector.collect_all_async())
    
    print("\n✅ 심층 C# 데이터 수집 완료!")
    print(f"📁 저장 위치: {collector.output_dir}")
    print(f"📊 총 {collector.stats['total_collected']}개 항목 수집")
    print(f"💾 총 {collector.stats['total_size_mb']:.2f}MB 데이터")
    print("\n📋 다음 명령어로 인덱싱을 시작하세요: autoci data index")


if __name__ == "__main__":
    main()