#!/usr/bin/env python3
"""
C# 전문가 수준 학습 시스템
24시간 지속적으로 C#/Unity 전문 지식을 학습하고 축적
"""

import os
import sys
import json
import sqlite3
import logging
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import re
import threading
import time
import hashlib
from collections import defaultdict, Counter
import pickle
import ast
import subprocess

logger = logging.getLogger(__name__)


class CSharpExpertLearner:
    """C# 전문가 수준 학습 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.knowledge_base_path = self.base_path / "csharp_knowledge"
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # 전문 지식 카테고리
        self.knowledge_categories = {
            'language_fundamentals': LanguageFundamentals(),
            'design_patterns': DesignPatterns(),
            'unity_expertise': UnityExpertise(),
            'performance_optimization': PerformanceOptimization(),
            'advanced_features': AdvancedFeatures(),
            'best_practices': BestPractices(),
            'error_solutions': ErrorSolutions()
        }
        
        # 학습 소스
        self.learning_sources = {
            'official_docs': 'https://docs.microsoft.com/en-us/dotnet/csharp/',
            'unity_docs': 'https://docs.unity3d.com/',
            'stackoverflow': 'https://stackoverflow.com/questions/tagged/c%23',
            'github_repos': [],  # 우수 C# 프로젝트들
            'books': [],  # 전문 서적 내용
            'tutorials': []  # 고급 튜토리얼
        }
        
        # 코드 분석 엔진
        self.code_analyzer = CSharpCodeAnalyzer()
        
        # 지식 그래프
        self.knowledge_graph = KnowledgeGraph()
        
        # 학습 통계
        self.learning_stats = {
            'total_concepts_learned': 0,
            'code_patterns_analyzed': 0,
            'solutions_discovered': 0,
            'expertise_level': 0.1,
            'last_update': datetime.now()
        }
        
        # 초기화
        self._init_database()
        self._load_existing_knowledge()
        
    def _init_database(self):
        """전문 지식 데이터베이스 초기화"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        # C# 개념 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS csharp_concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_name TEXT UNIQUE,
                category TEXT,
                difficulty_level INTEGER,
                description TEXT,
                code_examples TEXT,
                related_concepts TEXT,
                usage_frequency INTEGER DEFAULT 0,
                mastery_level REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 디자인 패턴 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                pattern_type TEXT,
                problem_solved TEXT,
                implementation TEXT,
                unity_specific BOOLEAN DEFAULT FALSE,
                use_cases TEXT,
                pros_cons TEXT,
                example_code TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 성능 최적화 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_techniques (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                technique_name TEXT,
                category TEXT,
                performance_impact TEXT,
                implementation_details TEXT,
                benchmarks TEXT,
                unity_specific BOOLEAN DEFAULT FALSE,
                code_before TEXT,
                code_after TEXT,
                measured_improvement REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 에러 해결 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                error_message TEXT,
                root_cause TEXT,
                solution_steps TEXT,
                prevention_tips TEXT,
                code_example TEXT,
                success_rate REAL DEFAULT 0.0,
                times_encountered INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 코드 스니펫 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snippet_name TEXT,
                category TEXT,
                description TEXT,
                code TEXT,
                dependencies TEXT,
                performance_notes TEXT,
                usage_examples TEXT,
                rating REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Unity 특화 지식 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unity_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE,
                category TEXT,
                unity_version TEXT,
                description TEXT,
                best_practices TEXT,
                common_pitfalls TEXT,
                example_implementation TEXT,
                performance_considerations TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def learn_from_documentation(self):
        """공식 문서에서 학습"""
        logger.info("📚 공식 문서 학습 시작...")
        
        # C# 언어 스펙 학습
        csharp_topics = [
            'types-and-variables',
            'statements-and-expressions',
            'classes-and-objects',
            'inheritance',
            'interfaces',
            'delegates',
            'events',
            'linq',
            'async-programming',
            'attributes',
            'reflection',
            'generics',
            'collections',
            'exception-handling',
            'file-io',
            'networking',
            'threading',
            'memory-management'
        ]
        
        for topic in csharp_topics:
            try:
                knowledge = await self._fetch_and_parse_doc(
                    f"https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/{topic}/"
                )
                
                if knowledge:
                    self._store_concept(knowledge)
                    
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"문서 학습 오류 ({topic}): {e}")
    
    async def _fetch_and_parse_doc(self, url: str) -> Optional[Dict]:
        """문서 가져오기 및 파싱"""
        # 실제 구현에서는 웹 스크래핑 또는 API 사용
        # 여기서는 시뮬레이션
        
        # 예시 지식 구조
        knowledge = {
            'url': url,
            'topic': url.split('/')[-2],
            'content': f"Content from {url}",
            'code_examples': [],
            'key_points': [],
            'timestamp': datetime.now()
        }
        
        return knowledge
    
    def analyze_code_patterns(self, code_directory: str):
        """코드 패턴 분석 및 학습"""
        logger.info(f"📂 코드 패턴 분석: {code_directory}")
        
        code_path = Path(code_directory)
        if not code_path.exists():
            return
        
        # C# 파일 검색
        cs_files = list(code_path.rglob("*.cs"))
        
        patterns_found = []
        
        for cs_file in cs_files:
            try:
                with open(cs_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # 코드 분석
                analysis = self.code_analyzer.analyze(code_content)
                
                # 패턴 추출
                patterns = self._extract_patterns(analysis)
                patterns_found.extend(patterns)
                
                # 모범 사례 학습
                best_practices = self._learn_best_practices(analysis)
                
                # 저장
                self._store_code_patterns(patterns, cs_file)
                
            except Exception as e:
                logger.error(f"파일 분석 오류 ({cs_file}): {e}")
        
        # 통계 업데이트
        self.learning_stats['code_patterns_analyzed'] += len(patterns_found)
        
        return patterns_found
    
    def _extract_patterns(self, analysis: Dict) -> List[Dict]:
        """코드에서 패턴 추출"""
        patterns = []
        
        # 클래스 구조 패턴
        if 'classes' in analysis:
            for class_info in analysis['classes']:
                # 싱글톤 패턴 감지
                if self._is_singleton_pattern(class_info):
                    patterns.append({
                        'type': 'singleton',
                        'class_name': class_info['name'],
                        'implementation': class_info['code']
                    })
                
                # 팩토리 패턴 감지
                if self._is_factory_pattern(class_info):
                    patterns.append({
                        'type': 'factory',
                        'class_name': class_info['name'],
                        'implementation': class_info['code']
                    })
        
        # LINQ 사용 패턴
        if 'linq_queries' in analysis:
            for query in analysis['linq_queries']:
                patterns.append({
                    'type': 'linq_pattern',
                    'query': query,
                    'complexity': self._assess_linq_complexity(query)
                })
        
        # 비동기 패턴
        if 'async_methods' in analysis:
            for method in analysis['async_methods']:
                patterns.append({
                    'type': 'async_pattern',
                    'method_name': method['name'],
                    'has_cancellation': 'CancellationToken' in method['params'],
                    'error_handling': 'try' in method['body']
                })
        
        return patterns
    
    def _is_singleton_pattern(self, class_info: Dict) -> bool:
        """싱글톤 패턴 감지"""
        indicators = [
            'private static' in class_info.get('fields', ''),
            'Instance' in class_info.get('properties', ''),
            'private' in class_info.get('constructor', '')
        ]
        
        return sum(indicators) >= 2
    
    def _is_factory_pattern(self, class_info: Dict) -> bool:
        """팩토리 패턴 감지"""
        method_names = [m['name'] for m in class_info.get('methods', [])]
        
        factory_indicators = ['Create', 'Build', 'Make', 'Generate']
        
        return any(ind in name for ind in factory_indicators for name in method_names)
    
    def _assess_linq_complexity(self, query: str) -> str:
        """LINQ 쿼리 복잡도 평가"""
        operators = ['Where', 'Select', 'OrderBy', 'GroupBy', 'Join', 'Aggregate']
        operator_count = sum(1 for op in operators if op in query)
        
        if operator_count >= 4:
            return 'complex'
        elif operator_count >= 2:
            return 'medium'
        else:
            return 'simple'
    
    def _learn_best_practices(self, analysis: Dict) -> List[Dict]:
        """모범 사례 학습"""
        best_practices = []
        
        # 명명 규칙 확인
        if self._follows_naming_conventions(analysis):
            best_practices.append({
                'type': 'naming_convention',
                'quality': 'good',
                'details': 'Follows C# naming conventions'
            })
        
        # 예외 처리 확인
        if self._has_proper_exception_handling(analysis):
            best_practices.append({
                'type': 'exception_handling',
                'quality': 'good',
                'details': 'Proper exception handling implemented'
            })
        
        # 비동기 사용 확인
        if self._uses_async_properly(analysis):
            best_practices.append({
                'type': 'async_usage',
                'quality': 'good',
                'details': 'Async/await used correctly'
            })
        
        return best_practices
    
    def _follows_naming_conventions(self, analysis: Dict) -> bool:
        """명명 규칙 준수 확인"""
        # PascalCase for classes
        classes_ok = all(
            cls['name'][0].isupper() 
            for cls in analysis.get('classes', [])
            if cls.get('name')
        )
        
        # camelCase for parameters
        # 추가 검증 로직...
        
        return classes_ok
    
    def _has_proper_exception_handling(self, analysis: Dict) -> bool:
        """적절한 예외 처리 확인"""
        methods = analysis.get('methods', [])
        
        for method in methods:
            if 'throw' in method.get('body', ''):
                # throw 후 적절한 처리가 있는지 확인
                if 'try' not in method.get('callers', []):
                    return False
        
        return True
    
    def _uses_async_properly(self, analysis: Dict) -> bool:
        """비동기 올바른 사용 확인"""
        async_methods = analysis.get('async_methods', [])
        
        for method in async_methods:
            # ConfigureAwait 사용 확인
            if 'await' in method.get('body', '') and 'ConfigureAwait' not in method.get('body', ''):
                return False
        
        return True
    
    def _store_concept(self, knowledge: Dict):
        """개념 저장"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO csharp_concepts 
            (concept_name, category, description, code_examples)
            VALUES (?, ?, ?, ?)
        ''', (
            knowledge['topic'],
            'fundamental',
            knowledge.get('content', ''),
            json.dumps(knowledge.get('code_examples', []))
        ))
        
        conn.commit()
        conn.close()
        
        self.learning_stats['total_concepts_learned'] += 1
    
    def _store_code_patterns(self, patterns: List[Dict], source_file: Path):
        """코드 패턴 저장"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        for pattern in patterns:
            cursor.execute('''
                INSERT INTO code_snippets 
                (snippet_name, category, description, code)
                VALUES (?, ?, ?, ?)
            ''', (
                f"{pattern['type']}_{source_file.stem}",
                pattern['type'],
                f"Pattern from {source_file.name}",
                json.dumps(pattern)
            ))
        
        conn.commit()
        conn.close()
    
    def learn_from_errors(self, error_log: str):
        """에러 로그에서 학습"""
        logger.info("🔍 에러 패턴 학습...")
        
        # 에러 파싱
        errors = self._parse_error_log(error_log)
        
        for error in errors:
            # 기존 해결책 검색
            existing_solution = self._find_existing_solution(error)
            
            if existing_solution:
                # 해결책 개선
                self._improve_solution(error, existing_solution)
            else:
                # 새로운 해결책 연구
                new_solution = self._research_solution(error)
                self._store_error_solution(error, new_solution)
    
    def _parse_error_log(self, error_log: str) -> List[Dict]:
        """에러 로그 파싱"""
        errors = []
        
        # 에러 패턴 매칭
        error_pattern = r'(\w+Exception): (.+?)\n\s+at (.+?)\n'
        matches = re.findall(error_pattern, error_log, re.MULTILINE)
        
        for match in matches:
            errors.append({
                'type': match[0],
                'message': match[1],
                'stack_trace': match[2],
                'timestamp': datetime.now()
            })
        
        return errors
    
    def _find_existing_solution(self, error: Dict) -> Optional[Dict]:
        """기존 해결책 검색"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM error_solutions
            WHERE error_type = ? AND error_message LIKE ?
            ORDER BY success_rate DESC
            LIMIT 1
        ''', (error['type'], f"%{error['message'][:50]}%"))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'solution_steps': json.loads(result[4]),
                'success_rate': result[7]
            }
        
        return None
    
    def _improve_solution(self, error: Dict, existing_solution: Dict):
        """해결책 개선"""
        # 성공률 업데이트 등
        pass
    
    def _research_solution(self, error: Dict) -> Dict:
        """새로운 해결책 연구"""
        # 실제로는 Stack Overflow, 문서 등에서 검색
        # 여기서는 기본 해결책 생성
        
        solutions = {
            'NullReferenceException': {
                'steps': [
                    '1. null 체크 추가: if (object != null)',
                    '2. null 조건 연산자 사용: object?.Method()',
                    '3. null 병합 연산자 사용: value ?? defaultValue',
                    '4. 초기화 확인'
                ],
                'prevention': '항상 사용 전 null 체크를 하세요.'
            },
            'IndexOutOfRangeException': {
                'steps': [
                    '1. 배열/리스트 크기 확인',
                    '2. 인덱스 범위 검증: if (index >= 0 && index < array.Length)',
                    '3. LINQ의 ElementAtOrDefault 사용 고려'
                ],
                'prevention': '컬렉션 접근 시 범위 검증을 습관화하세요.'
            }
        }
        
        error_type = error['type']
        
        return solutions.get(error_type, {
            'steps': ['일반적인 디버깅 절차를 따르세요.'],
            'prevention': '로그를 추가하여 원인을 파악하세요.'
        })
    
    def _store_error_solution(self, error: Dict, solution: Dict):
        """에러 해결책 저장"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO error_solutions 
            (error_type, error_message, solution_steps, prevention_tips)
            VALUES (?, ?, ?, ?)
        ''', (
            error['type'],
            error['message'],
            json.dumps(solution.get('steps', [])),
            solution.get('prevention', '')
        ))
        
        conn.commit()
        conn.close()
        
        self.learning_stats['solutions_discovered'] += 1
    
    def synthesize_knowledge(self) -> Dict[str, Any]:
        """학습한 지식 종합"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        # 개념 통계
        cursor.execute('SELECT COUNT(*) FROM csharp_concepts')
        concept_count = cursor.fetchone()[0]
        
        # 패턴 통계
        cursor.execute('SELECT COUNT(*) FROM design_patterns')
        pattern_count = cursor.fetchone()[0]
        
        # 에러 해결 통계
        cursor.execute('SELECT COUNT(*) FROM error_solutions')
        solution_count = cursor.fetchone()[0]
        
        # 최적화 기법 통계
        cursor.execute('SELECT COUNT(*) FROM optimization_techniques')
        optimization_count = cursor.fetchone()[0]
        
        conn.close()
        
        # 전문성 수준 계산
        expertise_level = self._calculate_expertise_level(
            concept_count, pattern_count, solution_count, optimization_count
        )
        
        return {
            'total_concepts': concept_count,
            'design_patterns': pattern_count,
            'error_solutions': solution_count,
            'optimization_techniques': optimization_count,
            'expertise_level': expertise_level,
            'last_updated': self.learning_stats['last_update']
        }
    
    def _calculate_expertise_level(self, concepts: int, patterns: int, 
                                 solutions: int, optimizations: int) -> float:
        """전문성 수준 계산"""
        # 가중치 적용
        weights = {
            'concepts': 0.3,
            'patterns': 0.25,
            'solutions': 0.25,
            'optimizations': 0.2
        }
        
        # 정규화 (목표치 대비)
        targets = {
            'concepts': 1000,
            'patterns': 100,
            'solutions': 500,
            'optimizations': 200
        }
        
        scores = {
            'concepts': min(1.0, concepts / targets['concepts']),
            'patterns': min(1.0, patterns / targets['patterns']),
            'solutions': min(1.0, solutions / targets['solutions']),
            'optimizations': min(1.0, optimizations / targets['optimizations'])
        }
        
        # 가중 평균
        expertise = sum(scores[k] * weights[k] for k in weights)
        
        return expertise
    
    def get_expert_knowledge(self, topic: str) -> Dict[str, Any]:
        """특정 주제에 대한 전문 지식 제공"""
        conn = sqlite3.connect(str(self.knowledge_base_path / "expert_knowledge.db"))
        cursor = conn.cursor()
        
        # 관련 개념 검색
        cursor.execute('''
            SELECT * FROM csharp_concepts
            WHERE concept_name LIKE ? OR description LIKE ?
            ORDER BY mastery_level DESC
            LIMIT 5
        ''', (f'%{topic}%', f'%{topic}%'))
        
        concepts = cursor.fetchall()
        
        # 관련 패턴 검색
        cursor.execute('''
            SELECT * FROM design_patterns
            WHERE pattern_name LIKE ? OR problem_solved LIKE ?
            LIMIT 3
        ''', (f'%{topic}%', f'%{topic}%'))
        
        patterns = cursor.fetchall()
        
        # 관련 코드 스니펫
        cursor.execute('''
            SELECT * FROM code_snippets
            WHERE category LIKE ? OR description LIKE ?
            ORDER BY rating DESC
            LIMIT 5
        ''', (f'%{topic}%', f'%{topic}%'))
        
        snippets = cursor.fetchall()
        
        conn.close()
        
        return {
            'topic': topic,
            'concepts': [self._format_concept(c) for c in concepts],
            'patterns': [self._format_pattern(p) for p in patterns],
            'code_examples': [self._format_snippet(s) for s in snippets],
            'expertise_level': self.learning_stats['expertise_level']
        }
    
    def _format_concept(self, concept_row: tuple) -> Dict:
        """개념 포맷팅"""
        return {
            'name': concept_row[1],
            'category': concept_row[2],
            'difficulty': concept_row[3],
            'description': concept_row[4],
            'examples': json.loads(concept_row[5]) if concept_row[5] else [],
            'mastery_level': concept_row[8]
        }
    
    def _format_pattern(self, pattern_row: tuple) -> Dict:
        """패턴 포맷팅"""
        return {
            'name': pattern_row[1],
            'type': pattern_row[2],
            'problem': pattern_row[3],
            'implementation': pattern_row[4],
            'unity_specific': bool(pattern_row[5]),
            'example': pattern_row[8]
        }
    
    def _format_snippet(self, snippet_row: tuple) -> Dict:
        """코드 스니펫 포맷팅"""
        return {
            'name': snippet_row[1],
            'category': snippet_row[2],
            'description': snippet_row[3],
            'code': snippet_row[4],
            'rating': snippet_row[8]
        }
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        try:
            # 통계 업데이트
            knowledge_summary = self.synthesize_knowledge()
            self.learning_stats.update(knowledge_summary)
            
            logger.info(f"📊 기존 지식 로드 완료: {knowledge_summary}")
            
        except Exception as e:
            logger.error(f"지식 로드 오류: {e}")


class LanguageFundamentals:
    """C# 언어 기초 지식"""
    
    def __init__(self):
        self.topics = {
            'value_types': ['int', 'float', 'double', 'bool', 'char', 'struct', 'enum'],
            'reference_types': ['class', 'interface', 'delegate', 'array', 'string'],
            'generics': ['List<T>', 'Dictionary<K,V>', 'constraints', 'covariance'],
            'linq': ['query syntax', 'method syntax', 'deferred execution'],
            'async': ['async/await', 'Task', 'TaskCompletionSource', 'ConfigureAwait']
        }


class DesignPatterns:
    """디자인 패턴 지식"""
    
    def __init__(self):
        self.patterns = {
            'creational': ['Singleton', 'Factory', 'Builder', 'Prototype'],
            'structural': ['Adapter', 'Decorator', 'Facade', 'Proxy'],
            'behavioral': ['Observer', 'Strategy', 'Command', 'Iterator'],
            'unity_specific': ['Object Pool', 'Service Locator', 'Component Pattern']
        }


class UnityExpertise:
    """Unity 전문 지식"""
    
    def __init__(self):
        self.areas = {
            'core': ['GameObject', 'Transform', 'Component', 'Prefab'],
            'scripting': ['MonoBehaviour', 'Coroutine', 'Events', 'Delegates'],
            'rendering': ['Materials', 'Shaders', 'Lighting', 'Post-processing'],
            'physics': ['Rigidbody', 'Collider', 'Raycast', 'Joints'],
            'ui': ['Canvas', 'RectTransform', 'EventSystem', 'Layout Groups'],
            'optimization': ['Profiler', 'Draw Calls', 'Batching', 'LOD']
        }


class PerformanceOptimization:
    """성능 최적화 지식"""
    
    def __init__(self):
        self.techniques = {
            'memory': ['Object Pooling', 'Struct vs Class', 'GC Optimization'],
            'cpu': ['Algorithm Complexity', 'Cache Optimization', 'Parallel Processing'],
            'unity': ['Draw Call Batching', 'Occlusion Culling', 'Level of Detail'],
            'code': ['Inlining', 'Loop Unrolling', 'Branch Prediction']
        }


class AdvancedFeatures:
    """고급 기능 지식"""
    
    def __init__(self):
        self.features = {
            'reflection': ['Type', 'MethodInfo', 'Attributes', 'Dynamic'],
            'expressions': ['Expression Trees', 'Lambda', 'LINQ Providers'],
            'unsafe': ['Pointers', 'Fixed', 'Span<T>', 'Memory<T>'],
            'interop': ['P/Invoke', 'COM Interop', 'Marshaling']
        }


class BestPractices:
    """모범 사례"""
    
    def __init__(self):
        self.practices = {
            'solid': ['Single Responsibility', 'Open/Closed', 'Liskov Substitution'],
            'clean_code': ['Meaningful Names', 'Small Functions', 'DRY'],
            'testing': ['Unit Tests', 'Integration Tests', 'Mocking'],
            'documentation': ['XML Comments', 'README', 'API Documentation']
        }


class ErrorSolutions:
    """에러 해결 방법"""
    
    def __init__(self):
        self.common_errors = {
            'NullReferenceException': {
                'causes': ['Uninitialized object', 'Destroyed GameObject'],
                'solutions': ['Null checks', 'Null conditional operator', 'Initialization']
            },
            'IndexOutOfRangeException': {
                'causes': ['Invalid index', 'Empty collection'],
                'solutions': ['Bounds checking', 'LINQ safety methods']
            }
        }


class CSharpCodeAnalyzer:
    """C# 코드 분석기"""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """코드 분석"""
        analysis = {
            'classes': self._extract_classes(code),
            'methods': self._extract_methods(code),
            'properties': self._extract_properties(code),
            'linq_queries': self._extract_linq(code),
            'async_methods': self._extract_async(code),
            'complexity': self._calculate_complexity(code)
        }
        
        return analysis
    
    def _extract_classes(self, code: str) -> List[Dict]:
        """클래스 추출"""
        class_pattern = r'(public|private|internal)?\s*(static|abstract|sealed)?\s*class\s+(\w+)'
        matches = re.findall(class_pattern, code)
        
        classes = []
        for match in matches:
            classes.append({
                'visibility': match[0] or 'internal',
                'modifiers': match[1] or '',
                'name': match[2]
            })
        
        return classes
    
    def _extract_methods(self, code: str) -> List[Dict]:
        """메서드 추출"""
        method_pattern = r'(public|private|protected|internal)?\s*(static|virtual|override|abstract)?\s*(\w+)\s+(\w+)\s*\((.*?)\)'
        matches = re.findall(method_pattern, code)
        
        methods = []
        for match in matches:
            methods.append({
                'visibility': match[0] or 'private',
                'modifiers': match[1] or '',
                'return_type': match[2],
                'name': match[3],
                'parameters': match[4]
            })
        
        return methods
    
    def _extract_properties(self, code: str) -> List[Dict]:
        """속성 추출"""
        prop_pattern = r'(public|private|protected|internal)?\s*(\w+)\s+(\w+)\s*{\s*(get|set)'
        matches = re.findall(prop_pattern, code)
        
        properties = []
        for match in matches:
            properties.append({
                'visibility': match[0] or 'private',
                'type': match[1],
                'name': match[2],
                'accessors': match[3]
            })
        
        return properties
    
    def _extract_linq(self, code: str) -> List[str]:
        """LINQ 쿼리 추출"""
        # Method syntax
        method_pattern = r'\.\s*(Where|Select|OrderBy|GroupBy|Join|Any|All|First|Last)\s*\('
        method_queries = re.findall(method_pattern, code)
        
        # Query syntax
        query_pattern = r'from\s+\w+\s+in\s+\w+'
        query_queries = re.findall(query_pattern, code)
        
        return method_queries + query_queries
    
    def _extract_async(self, code: str) -> List[Dict]:
        """비동기 메서드 추출"""
        async_pattern = r'async\s+(\w+)\s+(\w+)\s*\((.*?)\)'
        matches = re.findall(async_pattern, code)
        
        async_methods = []
        for match in matches:
            async_methods.append({
                'return_type': match[0],
                'name': match[1],
                'params': match[2]
            })
        
        return async_methods
    
    def _calculate_complexity(self, code: str) -> Dict[str, int]:
        """코드 복잡도 계산"""
        lines = code.split('\n')
        
        complexity = {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
            'cyclomatic': self._cyclomatic_complexity(code),
            'nesting_depth': self._max_nesting_depth(code)
        }
        
        return complexity
    
    def _cyclomatic_complexity(self, code: str) -> int:
        """순환 복잡도 계산"""
        # 간단한 계산: 분기문 개수 + 1
        branches = ['if', 'else', 'case', 'for', 'foreach', 'while', 'catch']
        
        complexity = 1
        for branch in branches:
            complexity += code.count(f'{branch} ')
        
        return complexity
    
    def _max_nesting_depth(self, code: str) -> int:
        """최대 중첩 깊이"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        return max_depth


class KnowledgeGraph:
    """지식 그래프"""
    
    def __init__(self):
        self.nodes = {}  # concept -> properties
        self.edges = defaultdict(list)  # concept -> related concepts
    
    def add_concept(self, concept: str, properties: Dict):
        """개념 추가"""
        self.nodes[concept] = properties
    
    def add_relation(self, concept1: str, concept2: str, relation_type: str):
        """관계 추가"""
        self.edges[concept1].append({
            'target': concept2,
            'type': relation_type
        })
    
    def get_related_concepts(self, concept: str, depth: int = 2) -> List[str]:
        """관련 개념 가져오기"""
        visited = set()
        queue = [(concept, 0)]
        related = []
        
        while queue:
            current, current_depth = queue.pop(0)
            
            if current in visited or current_depth > depth:
                continue
            
            visited.add(current)
            
            if current != concept:
                related.append(current)
            
            for edge in self.edges.get(current, []):
                queue.append((edge['target'], current_depth + 1))
        
        return related


# 테스트 함수
if __name__ == "__main__":
    print("🎓 C# 전문가 학습 시스템 테스트")
    print("=" * 60)
    
    learner = CSharpExpertLearner()
    
    # 코드 분석 테스트
    test_code = '''
    public class PlayerController : MonoBehaviour
    {
        private static PlayerController instance;
        public static PlayerController Instance => instance;
        
        private void Awake()
        {
            if (instance == null)
                instance = this;
            else
                Destroy(gameObject);
        }
        
        public async Task<bool> MoveToPositionAsync(Vector3 position, CancellationToken token)
        {
            try
            {
                await transform.DOMove(position, 1f).AsyncWaitForCompletion();
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Movement failed: {e.Message}");
                return false;
            }
        }
    }
    '''
    
    # 분석 실행
    analyzer = CSharpCodeAnalyzer()
    analysis = analyzer.analyze(test_code)
    
    print("\n📊 코드 분석 결과:")
    print(f"클래스: {len(analysis['classes'])}개")
    print(f"메서드: {len(analysis['methods'])}개")
    print(f"비동기 메서드: {len(analysis['async_methods'])}개")
    print(f"복잡도: {analysis['complexity']}")
    
    # 지식 종합
    knowledge = learner.synthesize_knowledge()
    print(f"\n🧠 학습 통계:")
    print(f"총 개념: {knowledge['total_concepts']}")
    print(f"디자인 패턴: {knowledge['design_patterns']}")
    print(f"에러 해결책: {knowledge['error_solutions']}")
    print(f"전문성 수준: {knowledge['expertise_level']:.1%}")