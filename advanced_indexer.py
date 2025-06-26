#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Advanced Indexing System - 고급 데이터 인덱싱 및 학습 결과 문서화
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict

class AdvancedIndexer:
    def __init__(self, data_dir: str = "expert_learning_data", db_path: str = "expert_index.db"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.learning_results_dir = Path("learning_results")
        self.learning_results_dir.mkdir(exist_ok=True)
        
        # 인덱스 통계
        self.stats = {
            'total_files': 0,
            'total_patterns': 0,
            'total_categories': 0,
            'processing_time': 0,
            'index_size': 0
        }
        
        # 패턴 분석기
        self.pattern_analyzers = {
            'architecture': self.analyze_architecture_patterns,
            'performance': self.analyze_performance_patterns,
            'unity': self.analyze_unity_patterns,
            'async': self.analyze_async_patterns,
            'testing': self.analyze_testing_patterns
        }
        
        self.init_database()
    
    def init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 메인 코드 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_entries (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                category TEXT,
                template_name TEXT,
                description TEXT,
                code_hash TEXT,
                quality_score INTEGER,
                complexity INTEGER,
                created_at TIMESTAMP,
                indexed_at TIMESTAMP
            )
        ''')
        
        # 패턴 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT,
                pattern_type TEXT,
                code_id TEXT,
                confidence REAL,
                FOREIGN KEY (code_id) REFERENCES code_entries(id)
            )
        ''')
        
        # 키워드 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT,
                code_id TEXT,
                frequency INTEGER,
                FOREIGN KEY (code_id) REFERENCES code_entries(id)
            )
        ''')
        
        # 학습 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP,
                patterns_learned INTEGER,
                improvements TEXT,
                metrics TEXT
            )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON code_entries(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_template ON code_entries(template_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern ON patterns(pattern_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword ON keywords(keyword)')
        
        conn.commit()
        conn.close()
    
    def index_all_data(self):
        """모든 데이터 인덱싱"""
        start_time = time.time()
        print("🚀 고급 데이터 인덱싱 시작...")
        
        if not self.data_dir.exists():
            print(f"❌ 데이터 디렉토리가 없습니다: {self.data_dir}")
            return
        
        json_files = list(self.data_dir.glob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.index_single_file, f): f for f in json_files}
            
            processed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    processed += 1
                    if processed % 50 == 0:
                        print(f"  진행중... {processed}/{len(json_files)} 파일 처리됨")
        
        # 통계 업데이트
        self.stats['processing_time'] = time.time() - start_time
        self.update_statistics()
        
        # 학습 결과 생성
        self.generate_learning_report()
        
        print(f"✅ 인덱싱 완료! 처리 시간: {self.stats['processing_time']:.2f}초")
        print(f"📊 통계: {self.stats['total_files']}개 파일, {self.stats['total_patterns']}개 패턴")
    
    def index_single_file(self, file_path: Path) -> bool:
        """단일 파일 인덱싱"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            code = data.get('code', '').strip()
            if len(code) < 100:
                return False
            
            # 기본 정보 추출
            entry_id = file_path.stem
            category = data.get('category', 'general')
            template_name = data.get('template_name', '')
            quality_score = data.get('quality_score', 80)
            
            # 고급 분석
            description = self.extract_description(data, code)
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            complexity = self.calculate_complexity(code)
            patterns = self.extract_all_patterns(code, category)
            keywords = self.extract_keywords(code)
            
            # 데이터베이스에 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 코드 엔트리 저장
            cursor.execute('''
                INSERT OR REPLACE INTO code_entries 
                (id, file_name, category, template_name, description, code_hash, 
                 quality_score, complexity, created_at, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (entry_id, file_path.name, category, template_name, description, 
                  code_hash, quality_score, complexity, 
                  datetime.now(), datetime.now()))
            
            # 패턴 저장
            for pattern_type, pattern_list in patterns.items():
                for pattern_name, confidence in pattern_list:
                    cursor.execute('''
                        INSERT INTO patterns (pattern_name, pattern_type, code_id, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (pattern_name, pattern_type, entry_id, confidence))
                    self.stats['total_patterns'] += 1
            
            # 키워드 저장
            for keyword, frequency in keywords.items():
                cursor.execute('''
                    INSERT INTO keywords (keyword, code_id, frequency)
                    VALUES (?, ?, ?)
                ''', (keyword, entry_id, frequency))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"⚠️ 파일 인덱싱 오류 {file_path}: {e}")
            return False
    
    def extract_description(self, data: Dict, code: str) -> str:
        """설명 추출"""
        if 'description' in data:
            return data['description']
        
        # 코드에서 추출
        lines = code.split('\n')
        for line in lines[:10]:
            if '///' in line or '/*' in line:
                return line.strip('/*/ ').strip()
        
        return "C# 코드 패턴"
    
    def calculate_complexity(self, code: str) -> int:
        """복잡도 계산"""
        complexity = 0
        
        # 순환 복잡도 간단 계산
        complexity += len(re.findall(r'\b(if|else|for|while|switch|case|catch)\b', code))
        
        # 중첩 깊이
        lines = code.split('\n')
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = (len(line) - len(line.lstrip())) // 4
                max_indent = max(max_indent, indent)
        complexity += max_indent
        
        # 메서드 수
        complexity += len(re.findall(r'(?:public|private|protected).*?\(', code)) // 2
        
        return min(complexity, 100)
    
    def extract_all_patterns(self, code: str, category: str) -> Dict[str, List[Tuple[str, float]]]:
        """모든 패턴 추출"""
        all_patterns = {}
        
        # 카테고리별 분석
        if category in self.pattern_analyzers:
            all_patterns[category] = self.pattern_analyzers[category](code)
        
        # 일반 패턴 분석
        all_patterns['general'] = self.analyze_general_patterns(code)
        
        return all_patterns
    
    def analyze_general_patterns(self, code: str) -> List[Tuple[str, float]]:
        """일반 디자인 패턴 분석"""
        patterns = []
        
        pattern_indicators = {
            'Singleton': [r'private static .* _instance', r'public static .* Instance'],
            'Factory': [r'Create\w+', r'Factory', r'Build\w+'],
            'Repository': [r'IRepository', r'Repository<', r'DbContext'],
            'Observer': [r'INotify', r'EventHandler', r'event .*EventArgs'],
            'Command': [r'ICommand', r'Execute\(', r'CanExecute'],
            'Strategy': [r'IStrategy', r'Strategy', r'Algorithm'],
            'Decorator': [r'IDecorator', r'Wrapper', r'decorator'],
            'Adapter': [r'IAdapter', r'Adapter', r'Convert'],
            'Mediator': [r'IMediator', r'Mediator', r'IRequest'],
            'Builder': [r'Builder', r'With\w+', r'Build\(']
        }
        
        for pattern_name, indicators in pattern_indicators.items():
            confidence = 0
            for indicator in indicators:
                if re.search(indicator, code):
                    confidence += 0.33
            
            if confidence > 0.3:
                patterns.append((pattern_name, min(confidence, 1.0)))
        
        return patterns
    
    def analyze_architecture_patterns(self, code: str) -> List[Tuple[str, float]]:
        """아키텍처 패턴 분석"""
        patterns = []
        
        # Clean Architecture 패턴
        if all(term in code for term in ['Domain', 'Application', 'Infrastructure']):
            patterns.append(('CleanArchitecture', 0.9))
        
        # DDD 패턴
        if any(term in code for term in ['AggregateRoot', 'ValueObject', 'DomainEvent']):
            patterns.append(('DomainDrivenDesign', 0.8))
        
        # CQRS 패턴
        if all(term in code for term in ['Command', 'Query', 'Handler']):
            patterns.append(('CQRS', 0.85))
        
        # Event Sourcing
        if all(term in code for term in ['Event', 'EventStore', 'Aggregate']):
            patterns.append(('EventSourcing', 0.75))
        
        return patterns
    
    def analyze_performance_patterns(self, code: str) -> List[Tuple[str, float]]:
        """성능 패턴 분석"""
        patterns = []
        
        # Memory 최적화
        if 'Span<' in code or 'Memory<' in code:
            patterns.append(('MemoryOptimization', 0.9))
        
        # Object Pooling
        if 'ObjectPool' in code or 'Pool<' in code:
            patterns.append(('ObjectPooling', 0.85))
        
        # Caching
        if 'IMemoryCache' in code or 'Cache' in code:
            patterns.append(('Caching', 0.7))
        
        # Async 최적화
        if 'ValueTask' in code:
            patterns.append(('AsyncOptimization', 0.8))
        
        return patterns
    
    def analyze_unity_patterns(self, code: str) -> List[Tuple[str, float]]:
        """Unity 패턴 분석"""
        patterns = []
        
        # DOTS/ECS
        if any(term in code for term in ['IComponentData', 'Entity', 'SystemBase']):
            patterns.append(('Unity_DOTS', 0.9))
        
        # Job System
        if 'IJob' in code or 'JobHandle' in code:
            patterns.append(('Unity_JobSystem', 0.85))
        
        # Object Pooling
        if 'ObjectPool' in code and 'GameObject' in code:
            patterns.append(('Unity_ObjectPool', 0.8))
        
        # Coroutines
        if 'IEnumerator' in code and 'yield return' in code:
            patterns.append(('Unity_Coroutine', 0.75))
        
        return patterns
    
    def analyze_async_patterns(self, code: str) -> List[Tuple[str, float]]:
        """비동기 패턴 분석"""
        patterns = []
        
        # Async/Await
        if 'async' in code and 'await' in code:
            patterns.append(('AsyncAwait', 0.9))
        
        # Channels
        if 'Channel<' in code:
            patterns.append(('Channels', 0.85))
        
        # DataFlow
        if 'DataflowBlock' in code or 'ActionBlock' in code:
            patterns.append(('DataFlow', 0.8))
        
        # Reactive
        if 'IObservable' in code or 'Observable.' in code:
            patterns.append(('ReactiveExtensions', 0.75))
        
        return patterns
    
    def analyze_testing_patterns(self, code: str) -> List[Tuple[str, float]]:
        """테스팅 패턴 분석"""
        patterns = []
        
        # Unit Testing
        if any(term in code for term in ['[Test]', '[Fact]', '[TestMethod]']):
            patterns.append(('UnitTesting', 0.9))
        
        # Mocking
        if 'Mock<' in code or 'Substitute.' in code:
            patterns.append(('Mocking', 0.85))
        
        # BDD
        if any(term in code for term in ['Given', 'When', 'Then']):
            patterns.append(('BDD', 0.7))
        
        return patterns
    
    def extract_keywords(self, code: str) -> Dict[str, int]:
        """키워드 추출 및 빈도 계산"""
        keywords = defaultdict(int)
        
        # C# 키워드
        csharp_keywords = [
            'async', 'await', 'Task', 'public', 'private', 'protected',
            'class', 'interface', 'struct', 'enum', 'delegate',
            'override', 'virtual', 'abstract', 'sealed',
            'using', 'namespace', 'static', 'const', 'readonly'
        ]
        
        # Unity 키워드
        unity_keywords = [
            'GameObject', 'Transform', 'MonoBehaviour', 'Component',
            'Rigidbody', 'Collider', 'Coroutine', 'Prefab'
        ]
        
        # 패턴 키워드
        pattern_keywords = [
            'Repository', 'Factory', 'Service', 'Manager', 'Controller',
            'Handler', 'Provider', 'Builder', 'Strategy', 'Observer'
        ]
        
        all_keywords = csharp_keywords + unity_keywords + pattern_keywords
        
        for keyword in all_keywords:
            count = len(re.findall(rf'\b{keyword}\b', code))
            if count > 0:
                keywords[keyword.lower()] = count
        
        return dict(keywords)
    
    def update_statistics(self):
        """통계 업데이트"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 카테고리 수
        cursor.execute('SELECT COUNT(DISTINCT category) FROM code_entries')
        self.stats['total_categories'] = cursor.fetchone()[0]
        
        # 데이터베이스 크기
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        self.stats['index_size'] = cursor.fetchone()[0]
        
        conn.close()
    
    def generate_learning_report(self):
        """학습 결과 보고서 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.learning_results_dir / f"indexing_report_{timestamp}.md"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 보고서 내용 생성
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🔍 AutoCI 고급 인덱싱 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**처리 시간**: {self.stats['processing_time']:.2f}초\n\n")
            
            f.write("## 📊 전체 통계\n\n")
            f.write(f"- **총 파일 수**: {self.stats['total_files']}개\n")
            f.write(f"- **총 패턴 수**: {self.stats['total_patterns']}개\n")
            f.write(f"- **카테고리 수**: {self.stats['total_categories']}개\n")
            f.write(f"- **인덱스 크기**: {self.stats['index_size'] / 1024 / 1024:.2f} MB\n\n")
            
            # 카테고리별 분포
            f.write("## 📂 카테고리별 분포\n\n")
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM code_entries 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            for category, count in cursor.fetchall():
                f.write(f"- **{category}**: {count}개\n")
            f.write("\n")
            
            # 주요 패턴
            f.write("## 🎯 주요 디자인 패턴\n\n")
            cursor.execute('''
                SELECT pattern_name, pattern_type, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM patterns 
                GROUP BY pattern_name, pattern_type
                ORDER BY count DESC
                LIMIT 20
            ''')
            for pattern, ptype, count, confidence in cursor.fetchall():
                f.write(f"- **{pattern}** ({ptype}): {count}개 사용 (신뢰도: {confidence:.2f})\n")
            f.write("\n")
            
            # 인기 키워드
            f.write("## 🔑 인기 키워드 Top 20\n\n")
            cursor.execute('''
                SELECT keyword, SUM(frequency) as total_freq
                FROM keywords
                GROUP BY keyword
                ORDER BY total_freq DESC
                LIMIT 20
            ''')
            for keyword, freq in cursor.fetchall():
                f.write(f"- **{keyword}**: {freq}회\n")
            f.write("\n")
            
            # 복잡도 분석
            f.write("## 📈 복잡도 분석\n\n")
            cursor.execute('''
                SELECT 
                    AVG(complexity) as avg_complexity,
                    MIN(complexity) as min_complexity,
                    MAX(complexity) as max_complexity
                FROM code_entries
            ''')
            avg_comp, min_comp, max_comp = cursor.fetchone()
            f.write(f"- **평균 복잡도**: {avg_comp:.2f}\n")
            f.write(f"- **최소 복잡도**: {min_comp}\n")
            f.write(f"- **최대 복잡도**: {max_comp}\n\n")
            
            # 품질 점수 분포
            f.write("## ⭐ 품질 점수 분포\n\n")
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN quality_score >= 90 THEN '90-100 (우수)'
                        WHEN quality_score >= 80 THEN '80-89 (양호)'
                        WHEN quality_score >= 70 THEN '70-79 (보통)'
                        ELSE '70 미만 (개선 필요)'
                    END as range,
                    COUNT(*) as count
                FROM code_entries
                GROUP BY range
                ORDER BY quality_score DESC
            ''')
            for range_name, count in cursor.fetchall():
                f.write(f"- **{range_name}**: {count}개\n")
            f.write("\n")
            
            # 학습 권장사항
            f.write("## 💡 학습 권장사항\n\n")
            
            # 부족한 패턴 찾기
            cursor.execute('''
                SELECT pattern_type, COUNT(DISTINCT pattern_name) as variety
                FROM patterns
                GROUP BY pattern_type
                ORDER BY variety ASC
                LIMIT 5
            ''')
            weak_patterns = cursor.fetchall()
            
            if weak_patterns:
                f.write("### 보강이 필요한 패턴 영역:\n\n")
                for ptype, variety in weak_patterns:
                    f.write(f"- **{ptype}**: {variety}개 패턴만 발견됨\n")
                f.write("\n")
            
            f.write("### 추천 학습 소스:\n\n")
            f.write("- **Architecture**: dotnet/aspnetcore, ardalis/CleanArchitecture\n")
            f.write("- **Performance**: dotnet/runtime (Span/Memory 예제)\n")
            f.write("- **Unity Advanced**: Unity-Technologies/DOTS-Samples\n")
            f.write("- **Testing**: xunit/xunit, nunit/nunit\n")
        
        # 학습 결과 DB에 저장
        session_id = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        cursor.execute('''
            INSERT INTO learning_results 
            (session_id, timestamp, patterns_learned, improvements, metrics)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, datetime.now(), self.stats['total_patterns'], 
              json.dumps({"weak_patterns": weak_patterns}),
              json.dumps(self.stats)))
        
        conn.commit()
        conn.close()
        
        print(f"📄 학습 보고서 생성됨: {report_file}")
    
    def query_patterns(self, pattern_type: str = None, min_confidence: float = 0.5) -> List[Dict]:
        """패턴 쿼리"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if pattern_type:
            cursor.execute('''
                SELECT p.pattern_name, p.confidence, c.template_name, c.category
                FROM patterns p
                JOIN code_entries c ON p.code_id = c.id
                WHERE p.pattern_type = ? AND p.confidence >= ?
                ORDER BY p.confidence DESC
            ''', (pattern_type, min_confidence))
        else:
            cursor.execute('''
                SELECT p.pattern_name, p.pattern_type, p.confidence, c.template_name
                FROM patterns p
                JOIN code_entries c ON p.code_id = c.id
                WHERE p.confidence >= ?
                ORDER BY p.confidence DESC
                LIMIT 50
            ''', (min_confidence,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'pattern': row[0],
                'type': row[1] if not pattern_type else pattern_type,
                'confidence': row[2],
                'template': row[3]
            })
        
        conn.close()
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Indexing System")
    parser.add_argument('--index', action='store_true', help='전체 데이터 인덱싱')
    parser.add_argument('--query', type=str, help='패턴 쿼리')
    parser.add_argument('--stats', action='store_true', help='통계 표시')
    
    args = parser.parse_args()
    
    indexer = AdvancedIndexer()
    
    if args.index:
        indexer.index_all_data()
    
    elif args.query:
        results = indexer.query_patterns(pattern_type=args.query)
        print(f"\n🔍 '{args.query}' 패턴 검색 결과:")
        for r in results[:10]:
            print(f"  - {r['pattern']} (신뢰도: {r['confidence']:.2f}) - {r['template']}")
    
    elif args.stats:
        indexer.update_statistics()
        print("\n📊 인덱싱 시스템 통계:")
        for key, value in indexer.stats.items():
            print(f"  - {key}: {value}")
    
    else:
        print("사용법: python advanced_indexer.py [--index|--query PATTERN|--stats]")

if __name__ == "__main__":
    main()