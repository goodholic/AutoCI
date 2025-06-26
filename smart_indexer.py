#!/usr/bin/env python3
"""
스마트 인덱서 - 고급 데이터를 모델에 효과적으로 적용
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import hashlib
import pickle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartIndexer:
    """스마트 데이터 인덱싱 시스템"""
    
    def __init__(self, data_dir: str = "expert_learning_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "smart_index.db"
        self.vector_store_path = self.data_dir / "vectors"
        self.vector_store_path.mkdir(exist_ok=True)
        
        # 데이터베이스 초기화
        self.init_database()
        
        # 인덱싱 통계
        self.stats = {
            'total_files': 0,
            'total_entries': 0,
            'categories': {},
            'quality_scores': []
        }
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 메인 인덱스 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS smart_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            content_type TEXT,
            title TEXT,
            description TEXT,
            content TEXT,
            code_snippets TEXT,
            quality_score REAL,
            complexity_level INTEGER,
            keywords TEXT,
            vector_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 코드 예제 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER,
            language TEXT,
            purpose TEXT,
            code TEXT,
            explanation TEXT,
            performance_notes TEXT,
            FOREIGN KEY (index_id) REFERENCES smart_index(id)
        )
        ''')
        
        # 학습 인사이트 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_id INTEGER,
            insight_type TEXT,
            content TEXT,
            importance_score REAL,
            FOREIGN KEY (index_id) REFERENCES smart_index(id)
        )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON smart_index(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON smart_index(quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_complexity ON smart_index(complexity_level)')
        
        conn.commit()
        conn.close()
        
    def index_all_data(self):
        """모든 데이터 인덱싱"""
        logger.info("🚀 스마트 인덱싱 시작...")
        
        # 카테고리별 처리
        categories = [
            ('microsoft_docs', self.index_microsoft_docs),
            ('github_projects', self.index_github_projects),
            ('stackoverflow', self.index_stackoverflow),
            ('expert_blogs', self.index_expert_blogs),
            ('design_patterns', self.index_design_patterns),
            ('unity_csharp', self.index_unity_csharp),
            ('performance_tips', self.index_performance_tips)
        ]
        
        for category, indexer_func in categories:
            category_path = self.data_dir / category
            if category_path.exists():
                logger.info(f"📁 {category} 인덱싱 중...")
                indexed = indexer_func(category_path)
                self.stats['categories'][category] = indexed
                
        # 통계 저장
        self.save_statistics()
        
        # 벡터 인덱스 생성
        self.create_vector_indices()
        
        # 품질 분석
        self.analyze_quality()
        
    def index_microsoft_docs(self, path: Path) -> int:
        """Microsoft 문서 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 품질 점수 계산
                quality_score = self.calculate_quality_score(data)
                
                # 복잡도 레벨 결정
                complexity = self.determine_complexity(data.get('topic', ''))
                
                # 데이터베이스에 저장
                self.save_to_index(
                    file_path=str(file_path),
                    category='microsoft_docs',
                    subcategory=data.get('topic', 'general'),
                    content_type='documentation',
                    title=data.get('topic', '').replace('-', ' ').title(),
                    description=f"Microsoft official documentation on {data.get('topic', '')}",
                    content=json.dumps(data),
                    quality_score=quality_score,
                    complexity_level=complexity
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"인덱싱 오류 ({file_path}): {e}")
                
        return indexed
        
    def index_github_projects(self, path: Path) -> int:
        """GitHub 프로젝트 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                quality_score = 0.9  # GitHub 전문가 프로젝트는 높은 품질
                
                self.save_to_index(
                    file_path=str(file_path),
                    category='github_projects',
                    subcategory='expert_code',
                    content_type='source_code',
                    title=data.get('name', ''),
                    description=data.get('description', ''),
                    content=json.dumps(data),
                    quality_score=quality_score,
                    complexity_level=3
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"GitHub 인덱싱 오류: {e}")
                
        return indexed
        
    def index_stackoverflow(self, path: Path) -> int:
        """Stack Overflow Q&A 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.save_to_index(
                    file_path=str(file_path),
                    category='stackoverflow',
                    subcategory='qa',
                    content_type='qa',
                    title=data.get('query', ''),
                    description='Expert Q&A from Stack Overflow',
                    content=json.dumps(data),
                    quality_score=0.85,
                    complexity_level=2
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"Stack Overflow 인덱싱 오류: {e}")
                
        return indexed
        
    def index_expert_blogs(self, path: Path) -> int:
        """전문가 블로그 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.save_to_index(
                    file_path=str(file_path),
                    category='expert_blogs',
                    subcategory='articles',
                    content_type='article',
                    title=f"{data.get('author', '')} - Expert Insights",
                    description=f"Expert articles on {', '.join(data.get('topics', []))}",
                    content=json.dumps(data),
                    quality_score=0.88,
                    complexity_level=3
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"블로그 인덱싱 오류: {e}")
                
        return indexed
        
    def index_design_patterns(self, path: Path) -> int:
        """디자인 패턴 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.save_to_index(
                    file_path=str(file_path),
                    category='design_patterns',
                    subcategory=data.get('category', 'general'),
                    content_type='pattern',
                    title=data.get('pattern', ''),
                    description=f"Design pattern: {data.get('pattern', '')}",
                    content=json.dumps(data),
                    quality_score=0.95,
                    complexity_level=3
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"패턴 인덱싱 오류: {e}")
                
        return indexed
        
    def index_unity_csharp(self, path: Path) -> int:
        """Unity C# 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.save_to_index(
                    file_path=str(file_path),
                    category='unity_csharp',
                    subcategory='unity_specific',
                    content_type='tutorial',
                    title=data.get('topic', ''),
                    description='Unity-specific C# best practices',
                    content=json.dumps(data),
                    quality_score=0.87,
                    complexity_level=2
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"Unity 인덱싱 오류: {e}")
                
        return indexed
        
    def index_performance_tips(self, path: Path) -> int:
        """성능 팁 인덱싱"""
        indexed = 0
        
        for file_path in path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.save_to_index(
                    file_path=str(file_path),
                    category='performance_tips',
                    subcategory='optimization',
                    content_type='guide',
                    title=data.get('topic', ''),
                    description='Performance optimization guide',
                    content=json.dumps(data),
                    quality_score=0.92,
                    complexity_level=3
                )
                
                indexed += 1
                
            except Exception as e:
                logger.error(f"성능 팁 인덱싱 오류: {e}")
                
        return indexed
        
    def save_to_index(self, **kwargs):
        """인덱스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 키워드 추출
        keywords = self.extract_keywords(kwargs.get('content', ''))
        kwargs['keywords'] = json.dumps(keywords)
        
        # 벡터 ID 생성
        vector_id = hashlib.md5(kwargs['file_path'].encode()).hexdigest()
        kwargs['vector_id'] = vector_id
        
        # 데이터 삽입
        columns = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?' for _ in kwargs])
        query = f"INSERT INTO smart_index ({columns}) VALUES ({placeholders})"
        
        cursor.execute(query, list(kwargs.values()))
        conn.commit()
        conn.close()
        
        self.stats['total_entries'] += 1
        self.stats['quality_scores'].append(kwargs.get('quality_score', 0))
        
    def calculate_quality_score(self, data: Dict) -> float:
        """품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 콘텐츠 존재 여부
        if data.get('content'): score += 0.1
        if data.get('examples'): score += 0.15
        if data.get('best_practices'): score += 0.15
        if data.get('code_examples'): score += 0.1
        
        return min(score, 1.0)
        
    def determine_complexity(self, topic: str) -> int:
        """복잡도 레벨 결정 (1-5)"""
        advanced_topics = ['async', 'parallel', 'unsafe', 'expression', 'reflection']
        intermediate_topics = ['linq', 'generics', 'delegates', 'events']
        
        topic_lower = topic.lower()
        
        if any(adv in topic_lower for adv in advanced_topics):
            return 4
        elif any(inter in topic_lower for inter in intermediate_topics):
            return 3
        else:
            return 2
            
    def extract_keywords(self, content: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 복잡한 NLP 사용)
        keywords = []
        important_terms = [
            'async', 'await', 'task', 'linq', 'generic', 'interface',
            'abstract', 'virtual', 'override', 'delegate', 'event',
            'lambda', 'expression', 'pattern', 'performance', 'memory'
        ]
        
        content_lower = content.lower()
        for term in important_terms:
            if term in content_lower:
                keywords.append(term)
                
        return keywords
        
    def create_vector_indices(self):
        """벡터 인덱스 생성"""
        logger.info("🔢 벡터 인덱스 생성 중...")
        
        # 여기서는 간단한 구현
        # 실제로는 sentence transformers 등을 사용
        
    def analyze_quality(self):
        """품질 분석"""
        if self.stats['quality_scores']:
            avg_quality = np.mean(self.stats['quality_scores'])
            logger.info(f"📊 평균 품질 점수: {avg_quality:.2f}")
            
    def save_statistics(self):
        """통계 저장"""
        self.stats['total_files'] = sum(self.stats['categories'].values())
        
        # JSON 저장
        stats_file = self.data_dir / 'indexing_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
            
        # 마크다운 리포트
        self.generate_report()
        
    def generate_report(self):
        """인덱싱 리포트 생성"""
        report = f"""# 스마트 인덱싱 리포트

## 📊 인덱싱 통계
- **총 파일 수**: {self.stats['total_files']}
- **총 엔트리 수**: {self.stats['total_entries']}
- **평균 품질 점수**: {np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0:.2f}

## 📁 카테고리별 현황
"""
        
        for category, count in self.stats['categories'].items():
            report += f"- **{category}**: {count}개\n"
            
        report += f"""
## 🎯 품질 분포
- 고품질 (0.9+): {sum(1 for s in self.stats['quality_scores'] if s >= 0.9)}개
- 중상품질 (0.8-0.9): {sum(1 for s in self.stats['quality_scores'] if 0.8 <= s < 0.9)}개
- 중품질 (0.7-0.8): {sum(1 for s in self.stats['quality_scores'] if 0.7 <= s < 0.8)}개
- 기타: {sum(1 for s in self.stats['quality_scores'] if s < 0.7)}개

## 💡 활용 방법
1. 고품질 데이터 우선 학습
2. 복잡도별 단계적 학습
3. 카테고리별 특화 학습

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_file = self.data_dir / 'indexing_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📝 리포트 생성 완료: {report_file}")
        
    def query_index(self, query: str, category: Optional[str] = None, 
                   min_quality: float = 0.7, limit: int = 10) -> List[Dict]:
        """인덱스 쿼리"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        base_query = """
        SELECT * FROM smart_index 
        WHERE quality_score >= ?
        """
        
        params = [min_quality]
        
        if category:
            base_query += " AND category = ?"
            params.append(category)
            
        if query:
            base_query += " AND (title LIKE ? OR description LIKE ? OR keywords LIKE ?)"
            query_param = f"%{query}%"
            params.extend([query_param, query_param, query_param])
            
        base_query += " ORDER BY quality_score DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(base_query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
            
        conn.close()
        return results


def main():
    """메인 함수"""
    indexer = SmartIndexer()
    
    # 전체 인덱싱
    indexer.index_all_data()
    
    print("\n✅ 스마트 인덱싱 완료!")
    print(f"📊 총 {indexer.stats['total_entries']}개 항목 인덱싱")
    print(f"📁 데이터베이스: {indexer.db_path}")
    
    # 샘플 쿼리
    print("\n🔍 샘플 쿼리 테스트...")
    results = indexer.query_index("async", min_quality=0.8, limit=5)
    print(f"'async' 검색 결과: {len(results)}개")


if __name__ == "__main__":
    main()