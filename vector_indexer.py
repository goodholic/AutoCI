#!/usr/bin/env python3
"""
벡터 임베딩 기반 고급 인덱싱 시스템
수집된 C# 데이터를 의미론적으로 검색 가능하게 만듦
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import pickle
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import faiss
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import re
from collections import defaultdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """코드 청크 데이터 클래스"""
    id: str
    file_path: str
    content: str
    chunk_type: str  # 'code', 'documentation', 'qa', 'pattern'
    category: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.7


class VectorIndexer:
    """벡터 임베딩 기반 인덱서"""
    
    def __init__(self, data_dir: str = "expert_learning_data"):
        self.data_dir = Path(data_dir)
        self.index_dir = self.data_dir / "vector_index"
        self.index_dir.mkdir(exist_ok=True)
        
        # 모델 초기화
        self.model_name = 'microsoft/codebert-base'
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FAISS 인덱스
        self.dimension = 768  # CodeBERT dimension
        self.index = None
        self.chunk_map = {}  # chunk_id -> CodeChunk
        
        # 데이터베이스
        self.db_path = self.index_dir / "vector_index.db"
        self.init_database()
        
        # 통계
        self.stats = {
            'total_chunks': 0,
            'total_files': 0,
            'categories': defaultdict(int),
            'avg_quality': 0.0
        }
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            category TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            quality_score REAL,
            metadata TEXT,
            embedding_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS semantic_clusters (
            cluster_id INTEGER PRIMARY KEY,
            cluster_name TEXT,
            cluster_description TEXT,
            centroid_embedding BLOB,
            chunk_count INTEGER,
            avg_quality REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id1 TEXT,
            chunk_id2 TEXT,
            relationship_type TEXT,
            similarity_score REAL,
            FOREIGN KEY (chunk_id1) REFERENCES vector_chunks(chunk_id),
            FOREIGN KEY (chunk_id2) REFERENCES vector_chunks(chunk_id)
        )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_type ON vector_chunks(chunk_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON vector_chunks(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON vector_chunks(quality_score)')
        
        conn.commit()
        conn.close()
        
    def load_model(self):
        """임베딩 모델 로드"""
        logger.info(f"🤖 모델 로딩: {self.model_name}")
        
        try:
            # CodeBERT 또는 다른 코드 특화 모델 사용
            self.model = SentenceTransformer('microsoft/codebert-base')
            self.model.to(self.device)
            
            logger.info(f"✅ 모델 로드 완료 (device: {self.device})")
            
        except Exception as e:
            logger.warning(f"CodeBERT 로드 실패, 대체 모델 사용: {e}")
            # 대체 모델
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384  # MiniLM dimension
            
    def index_all_data(self):
        """모든 데이터 인덱싱"""
        logger.info("🚀 벡터 인덱싱 시작...")
        
        # 모델 로드
        self.load_model()
        
        # FAISS 인덱스 초기화
        self.init_faiss_index()
        
        # 카테고리별 처리
        categories = [
            'microsoft_docs',
            'github_samples', 
            'nuget_packages',
            'stackoverflow_advanced',
            'expert_blogs',
            'performance_guides',
            'design_patterns',
            'unity_best_practices'
        ]
        
        for category in categories:
            category_path = self.data_dir / category
            if category_path.exists():
                logger.info(f"📁 {category} 인덱싱 중...")
                self.index_category(category, category_path)
                
        # 인덱스 저장
        self.save_index()
        
        # 클러스터링
        self.perform_clustering()
        
        # 관계 분석
        self.analyze_relationships()
        
        # 통계 저장
        self.save_statistics()
        
    def init_faiss_index(self):
        """FAISS 인덱스 초기화"""
        # IVF (Inverted File) 인덱스 사용
        nlist = 100  # 클러스터 수
        
        # 양자화기
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # IVF 인덱스
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        logger.info(f"📊 FAISS 인덱스 초기화 (dimension: {self.dimension}, nlist: {nlist})")
        
    def index_category(self, category: str, category_path: Path):
        """카테고리별 인덱싱"""
        chunks_to_index = []
        
        for file_path in category_path.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 데이터를 청크로 분할
                chunks = self.extract_chunks(data, category, file_path)
                
                for chunk in chunks:
                    # 품질 점수 계산
                    chunk.quality_score = self.calculate_chunk_quality(chunk)
                    
                    # 임베딩 생성
                    chunk.embedding = self.create_embedding(chunk.content)
                    
                    # 청크 저장
                    self.save_chunk(chunk)
                    chunks_to_index.append(chunk)
                    
                    self.stats['total_chunks'] += 1
                    self.stats['categories'][category] += 1
                    
            except Exception as e:
                logger.error(f"파일 인덱싱 오류 ({file_path}): {e}")
                
        # FAISS에 추가
        if chunks_to_index:
            self.add_to_faiss(chunks_to_index)
            
        self.stats['total_files'] += len(list(category_path.glob('*.json')))
        
    def extract_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """데이터에서 청크 추출"""
        chunks = []
        
        if category == 'microsoft_docs':
            chunks.extend(self.extract_doc_chunks(data, category, file_path))
            
        elif category == 'github_samples':
            chunks.extend(self.extract_code_chunks(data, category, file_path))
            
        elif category == 'stackoverflow_advanced':
            chunks.extend(self.extract_qa_chunks(data, category, file_path))
            
        elif category == 'design_patterns':
            chunks.extend(self.extract_pattern_chunks(data, category, file_path))
            
        else:
            # 기본 청크 추출
            chunks.extend(self.extract_default_chunks(data, category, file_path))
            
        return chunks
        
    def extract_doc_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """문서에서 청크 추출"""
        chunks = []
        
        # 제목과 설명
        if 'title' in data and 'content' in data:
            doc_chunk = CodeChunk(
                id=self.generate_chunk_id(file_path, 'doc', 0),
                file_path=str(file_path),
                content=f"{data['title']}\n\n{data.get('content', '')[:1000]}",
                chunk_type='documentation',
                category=category,
                metadata={'topic': data.get('topic', '')}
            )
            chunks.append(doc_chunk)
            
        # 코드 샘플
        for i, sample in enumerate(data.get('code_samples', [])):
            if 'code' in sample:
                code_chunk = CodeChunk(
                    id=self.generate_chunk_id(file_path, 'code', i),
                    file_path=str(file_path),
                    content=sample['code'],
                    chunk_type='code',
                    category=category,
                    metadata={
                        'language': sample.get('language', 'csharp'),
                        'topic': data.get('topic', '')
                    }
                )
                chunks.append(code_chunk)
                
        return chunks
        
    def extract_code_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """코드 파일에서 청크 추출"""
        chunks = []
        
        if 'content' in data:
            # 코드를 의미있는 단위로 분할
            code_content = data['content']
            code_blocks = self.split_code_into_blocks(code_content)
            
            for i, block in enumerate(code_blocks):
                if len(block['code']) > 50:  # 의미있는 크기
                    chunk = CodeChunk(
                        id=self.generate_chunk_id(file_path, 'code', i),
                        file_path=str(file_path),
                        content=block['code'],
                        chunk_type='code',
                        category=category,
                        metadata={
                            'block_type': block['type'],
                            'file_name': data.get('file_name', ''),
                            'repo': data.get('repo', ''),
                            'start_line': block.get('start_line', 0),
                            'end_line': block.get('end_line', 0)
                        }
                    )
                    chunks.append(chunk)
                    
        return chunks
        
    def split_code_into_blocks(self, code: str) -> List[Dict]:
        """코드를 의미있는 블록으로 분할"""
        blocks = []
        lines = code.split('\n')
        
        current_block = {
            'type': 'unknown',
            'code': '',
            'start_line': 0,
            'end_line': 0
        }
        
        in_class = False
        in_method = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            # 클래스 시작
            if re.search(r'\bclass\s+\w+', line):
                if current_block['code']:
                    blocks.append(current_block)
                current_block = {
                    'type': 'class',
                    'code': line + '\n',
                    'start_line': i,
                    'end_line': i
                }
                in_class = True
                brace_count = 0
                
            # 메서드 시작
            elif re.search(r'(public|private|protected|internal)\s+\w+\s+\w+\s*\(', line):
                if current_block['code'] and current_block['type'] == 'method':
                    blocks.append(current_block)
                    
                current_block = {
                    'type': 'method',
                    'code': line + '\n',
                    'start_line': i,
                    'end_line': i
                }
                in_method = True
                brace_count = 0
                
            else:
                current_block['code'] += line + '\n'
                current_block['end_line'] = i
                
            # 중괄호 카운트
            brace_count += line.count('{') - line.count('}')
            
            # 블록 종료
            if brace_count == 0 and (in_class or in_method) and '{' in current_block['code']:
                blocks.append(current_block)
                current_block = {
                    'type': 'unknown',
                    'code': '',
                    'start_line': i + 1,
                    'end_line': i + 1
                }
                in_class = False
                in_method = False
                
        # 마지막 블록
        if current_block['code']:
            blocks.append(current_block)
            
        return blocks
        
    def extract_qa_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """Q&A에서 청크 추출"""
        chunks = []
        
        # 질문
        if 'title' in data and 'body' in data:
            question_chunk = CodeChunk(
                id=self.generate_chunk_id(file_path, 'question', 0),
                file_path=str(file_path),
                content=f"Q: {data['title']}\n\n{data.get('body', '')[:500]}",
                chunk_type='qa',
                category=category,
                metadata={
                    'tags': data.get('tags', []),
                    'score': data.get('score', 0)
                }
            )
            chunks.append(question_chunk)
            
        # 답변
        for i, answer in enumerate(data.get('answers', [])):
            if 'body' in answer:
                answer_chunk = CodeChunk(
                    id=self.generate_chunk_id(file_path, 'answer', i),
                    file_path=str(file_path),
                    content=f"A: {answer['body'][:1000]}",
                    chunk_type='qa',
                    category=category,
                    metadata={
                        'score': answer.get('score', 0),
                        'is_accepted': answer.get('is_accepted', False)
                    }
                )
                chunks.append(answer_chunk)
                
        # 코드 샘플
        for i, sample in enumerate(data.get('code_samples', [])):
            if 'code' in sample:
                code_chunk = CodeChunk(
                    id=self.generate_chunk_id(file_path, 'qa_code', i),
                    file_path=str(file_path),
                    content=sample['code'],
                    chunk_type='code',
                    category=category,
                    metadata={'from_qa': True}
                )
                chunks.append(code_chunk)
                
        return chunks
        
    def extract_pattern_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """디자인 패턴에서 청크 추출"""
        chunks = []
        
        # 패턴 설명
        if 'name' in data:
            pattern_chunk = CodeChunk(
                id=self.generate_chunk_id(file_path, 'pattern', 0),
                file_path=str(file_path),
                content=f"Pattern: {data['name']}\n\nIntent: {data.get('intent', '')}\n\nWhen to use: {data.get('when_to_use', [])}",
                chunk_type='pattern',
                category=category,
                metadata={
                    'pattern_name': data['name'],
                    'pattern_category': data.get('category', '')
                }
            )
            chunks.append(pattern_chunk)
            
        # 구현 코드
        if 'implementation' in data and 'csharp' in data['implementation']:
            impl_chunk = CodeChunk(
                id=self.generate_chunk_id(file_path, 'pattern_code', 0),
                file_path=str(file_path),
                content=data['implementation']['csharp'],
                chunk_type='code',
                category=category,
                metadata={
                    'pattern_name': data.get('name', ''),
                    'is_pattern_implementation': True
                }
            )
            chunks.append(impl_chunk)
            
        return chunks
        
    def extract_default_chunks(self, data: Dict, category: str, file_path: Path) -> List[CodeChunk]:
        """기본 청크 추출"""
        chunks = []
        
        # JSON 전체를 텍스트로
        content = json.dumps(data, indent=2)
        
        # 크기 제한으로 분할
        max_chunk_size = 1000
        for i in range(0, len(content), max_chunk_size):
            chunk_content = content[i:i+max_chunk_size]
            
            chunk = CodeChunk(
                id=self.generate_chunk_id(file_path, 'default', i // max_chunk_size),
                file_path=str(file_path),
                content=chunk_content,
                chunk_type='documentation',
                category=category,
                metadata={}
            )
            chunks.append(chunk)
            
        return chunks
        
    def generate_chunk_id(self, file_path: Path, chunk_type: str, index: int) -> str:
        """청크 ID 생성"""
        base = f"{file_path.stem}_{chunk_type}_{index}"
        return hashlib.md5(base.encode()).hexdigest()
        
    def calculate_chunk_quality(self, chunk: CodeChunk) -> float:
        """청크 품질 점수 계산"""
        quality = 0.5  # 기본 점수
        
        # 청크 타입별 가중치
        type_weights = {
            'code': 0.8,
            'pattern': 0.9,
            'qa': 0.7,
            'documentation': 0.6
        }
        quality = type_weights.get(chunk.chunk_type, 0.5)
        
        # 콘텐츠 길이
        if len(chunk.content) > 200:
            quality += 0.1
            
        # 메타데이터 품질
        if chunk.metadata.get('score', 0) > 50:  # Stack Overflow 점수
            quality += 0.1
        if chunk.metadata.get('is_accepted', False):
            quality += 0.1
        if chunk.metadata.get('pattern_name'):  # 디자인 패턴
            quality += 0.15
            
        # 코드 품질 지표
        if chunk.chunk_type == 'code':
            # 주석 존재
            if '//' in chunk.content or '/*' in chunk.content:
                quality += 0.05
            # async/await 사용
            if 'async' in chunk.content and 'await' in chunk.content:
                quality += 0.05
            # LINQ 사용
            if any(linq in chunk.content for linq in ['.Where(', '.Select(', '.OrderBy(']):
                quality += 0.05
                
        return min(quality, 1.0)
        
    def create_embedding(self, text: str) -> np.ndarray:
        """텍스트 임베딩 생성"""
        try:
            # 코드 전처리
            processed_text = self.preprocess_code(text)
            
            # 임베딩 생성
            with torch.no_grad():
                embedding = self.model.encode(processed_text, convert_to_numpy=True)
                
            # 정규화
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            # 랜덤 임베딩 반환 (fallback)
            return np.random.randn(self.dimension).astype('float32')
            
    def preprocess_code(self, text: str) -> str:
        """코드 전처리"""
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 너무 긴 텍스트 잘라내기
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text.strip()
        
    def save_chunk(self, chunk: CodeChunk):
        """청크를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO vector_chunks
            (chunk_id, file_path, content, chunk_type, category,
             quality_score, metadata, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.id,
                chunk.file_path,
                chunk.content,
                chunk.chunk_type,
                chunk.category,
                chunk.quality_score,
                json.dumps(chunk.metadata),
                -1  # 임시 ID, FAISS 추가 후 업데이트
            ))
            
            conn.commit()
            
            # 메모리에도 저장
            self.chunk_map[chunk.id] = chunk
            
        except Exception as e:
            logger.error(f"청크 저장 오류: {e}")
        finally:
            conn.close()
            
    def add_to_faiss(self, chunks: List[CodeChunk]):
        """FAISS 인덱스에 추가"""
        if not chunks:
            return
            
        # 임베딩 추출
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # 인덱스가 비어있으면 학습 필요
        if not self.index.is_trained:
            logger.info("🎯 FAISS 인덱스 학습 중...")
            self.index.train(embeddings)
            
        # 인덱스에 추가
        start_id = len(self.chunk_map) - len(chunks)
        self.index.add(embeddings)
        
        # 청크 ID 업데이트
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            embedding_id = start_id + i
            cursor.execute('''
            UPDATE vector_chunks SET embedding_id = ? WHERE chunk_id = ?
            ''', (embedding_id, chunk.id))
            
        conn.commit()
        conn.close()
        
    def save_index(self):
        """인덱스 저장"""
        # FAISS 인덱스 저장
        faiss_path = self.index_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))
        
        # 청크 맵 저장
        chunk_map_path = self.index_dir / "chunk_map.pkl"
        with open(chunk_map_path, 'wb') as f:
            pickle.dump(self.chunk_map, f)
            
        logger.info(f"💾 인덱스 저장 완료: {self.index_dir}")
        
    def load_index(self):
        """저장된 인덱스 로드"""
        # FAISS 인덱스 로드
        faiss_path = self.index_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            
        # 청크 맵 로드
        chunk_map_path = self.index_dir / "chunk_map.pkl"
        if chunk_map_path.exists():
            with open(chunk_map_path, 'rb') as f:
                self.chunk_map = pickle.load(f)
                
        # 모델 로드
        if self.model is None:
            self.load_model()
            
    def search(self, query: str, k: int = 10, filters: Optional[Dict] = None) -> List[Tuple[CodeChunk, float]]:
        """의미론적 검색"""
        # 인덱스 로드
        if self.index is None:
            self.load_index()
            
        # 쿼리 임베딩
        query_embedding = self.create_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding, k * 2)  # 필터링을 위해 더 많이 검색
        
        # 결과 필터링 및 정렬
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:  # 유효하지 않은 인덱스
                continue
                
            # 청크 찾기
            chunk = None
            for chunk_id, c in self.chunk_map.items():
                if self.get_embedding_id(chunk_id) == idx:
                    chunk = c
                    break
                    
            if chunk is None:
                continue
                
            # 필터 적용
            if filters:
                if 'category' in filters and chunk.category != filters['category']:
                    continue
                if 'chunk_type' in filters and chunk.chunk_type != filters['chunk_type']:
                    continue
                if 'min_quality' in filters and chunk.quality_score < filters['min_quality']:
                    continue
                    
            # 유사도 점수 (거리를 유사도로 변환)
            similarity = 1 / (1 + distance)
            
            results.append((chunk, similarity))
            
            if len(results) >= k:
                break
                
        return results
        
    def get_embedding_id(self, chunk_id: str) -> int:
        """청크 ID로 임베딩 ID 가져오기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT embedding_id FROM vector_chunks WHERE chunk_id = ?', (chunk_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] if result else -1
        
    def perform_clustering(self):
        """의미론적 클러스터링"""
        logger.info("🔮 의미론적 클러스터링 수행 중...")
        
        if len(self.chunk_map) < 100:
            logger.info("청크가 너무 적어 클러스터링 건너뜀")
            return
            
        # 모든 임베딩 추출
        embeddings = []
        chunk_ids = []
        
        for chunk_id, chunk in self.chunk_map.items():
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                chunk_ids.append(chunk_id)
                
        embeddings = np.array(embeddings).astype('float32')
        
        # K-means 클러스터링
        n_clusters = min(50, len(embeddings) // 20)
        kmeans = faiss.Kmeans(self.dimension, n_clusters, niter=20, verbose=True)
        kmeans.train(embeddings)
        
        # 클러스터 할당
        distances, assignments = kmeans.index.search(embeddings, 1)
        
        # 클러스터 정보 저장
        clusters = defaultdict(list)
        for i, (chunk_id, cluster_id) in enumerate(zip(chunk_ids, assignments[:, 0])):
            clusters[cluster_id].append({
                'chunk_id': chunk_id,
                'distance': distances[i, 0]
            })
            
        # 데이터베이스에 저장
        self.save_clusters(clusters, kmeans.centroids)
        
    def save_clusters(self, clusters: Dict, centroids: np.ndarray):
        """클러스터 정보 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for cluster_id, chunk_list in clusters.items():
            # 클러스터 통계
            chunk_count = len(chunk_list)
            avg_quality = np.mean([
                self.chunk_map[item['chunk_id']].quality_score 
                for item in chunk_list
            ])
            
            # 클러스터 이름 생성 (가장 많은 카테고리)
            categories = defaultdict(int)
            for item in chunk_list:
                chunk = self.chunk_map[item['chunk_id']]
                categories[chunk.category] += 1
                
            cluster_name = max(categories.items(), key=lambda x: x[1])[0]
            
            # 클러스터 설명 (샘플 콘텐츠 기반)
            sample_chunks = [self.chunk_map[item['chunk_id']] for item in chunk_list[:3]]
            cluster_description = self.generate_cluster_description(sample_chunks)
            
            # 저장
            cursor.execute('''
            INSERT OR REPLACE INTO semantic_clusters
            (cluster_id, cluster_name, cluster_description, 
             centroid_embedding, chunk_count, avg_quality)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                int(cluster_id),
                cluster_name,
                cluster_description,
                centroids[cluster_id].tobytes(),
                chunk_count,
                avg_quality
            ))
            
        conn.commit()
        conn.close()
        
    def generate_cluster_description(self, sample_chunks: List[CodeChunk]) -> str:
        """클러스터 설명 생성"""
        # 청크 타입
        types = [chunk.chunk_type for chunk in sample_chunks]
        type_str = ', '.join(set(types))
        
        # 주요 키워드 추출 (간단한 방법)
        all_content = ' '.join([chunk.content[:100] for chunk in sample_chunks])
        keywords = re.findall(r'\b[A-Z][a-zA-Z]+\b', all_content)  # 대문자로 시작하는 단어
        keyword_str = ', '.join(set(keywords[:5]))
        
        return f"Types: {type_str}. Keywords: {keyword_str}"
        
    def analyze_relationships(self):
        """청크 간 관계 분석"""
        logger.info("🔗 청크 관계 분석 중...")
        
        # 샘플링 (모든 쌍은 너무 많음)
        sample_size = min(1000, len(self.chunk_map))
        sample_chunks = list(self.chunk_map.values())[:sample_size]
        
        relationships = []
        
        # 유사도 기반 관계
        for i in range(len(sample_chunks)):
            for j in range(i + 1, len(sample_chunks)):
                chunk1, chunk2 = sample_chunks[i], sample_chunks[j]
                
                # 임베딩 유사도
                similarity = np.dot(chunk1.embedding, chunk2.embedding)
                
                if similarity > 0.8:  # 높은 유사도
                    relationships.append({
                        'chunk_id1': chunk1.id,
                        'chunk_id2': chunk2.id,
                        'type': 'similar',
                        'score': float(similarity)
                    })
                    
                # 같은 파일의 다른 청크
                if chunk1.file_path == chunk2.file_path:
                    relationships.append({
                        'chunk_id1': chunk1.id,
                        'chunk_id2': chunk2.id,
                        'type': 'same_file',
                        'score': 1.0
                    })
                    
        # 관계 저장
        self.save_relationships(relationships)
        
    def save_relationships(self, relationships: List[Dict]):
        """관계 정보 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for rel in relationships:
            cursor.execute('''
            INSERT OR IGNORE INTO chunk_relationships
            (chunk_id1, chunk_id2, relationship_type, similarity_score)
            VALUES (?, ?, ?, ?)
            ''', (rel['chunk_id1'], rel['chunk_id2'], rel['type'], rel['score']))
            
        conn.commit()
        conn.close()
        
    def save_statistics(self):
        """통계 저장"""
        # 평균 품질 계산
        all_qualities = [chunk.quality_score for chunk in self.chunk_map.values()]
        self.stats['avg_quality'] = np.mean(all_qualities) if all_qualities else 0
        
        # JSON 저장
        stats_file = self.index_dir / 'vector_index_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
            
        # 마크다운 리포트
        self.generate_report()
        
    def generate_report(self):
        """인덱싱 리포트 생성"""
        # 클러스터 정보 로드
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM semantic_clusters')
        cluster_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM chunk_relationships')
        relationship_count = cursor.fetchone()[0]
        
        conn.close()
        
        report = f"""# 벡터 인덱싱 리포트

## 📊 인덱싱 통계
- **총 청크 수**: {self.stats['total_chunks']:,}
- **총 파일 수**: {self.stats['total_files']:,}
- **평균 품질 점수**: {self.stats['avg_quality']:.3f}
- **인덱스 차원**: {self.dimension}
- **클러스터 수**: {cluster_count}
- **관계 수**: {relationship_count:,}

## 📁 카테고리별 청크 분포
"""
        
        for category, count in sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.stats['total_chunks'] * 100) if self.stats['total_chunks'] > 0 else 0
            report += f"- **{category}**: {count:,} ({percentage:.1f}%)\n"
            
        report += f"""
## 🔍 검색 기능
- 의미론적 검색 지원
- 카테고리/타입별 필터링
- 품질 점수 기반 랭킹
- 유사 코드 찾기

## 💡 활용 예시
```python
# 검색 예시
indexer = VectorIndexer()
results = indexer.search(
    "async await pattern", 
    k=5,
    filters={{'category': 'github_samples', 'min_quality': 0.8}}
)

for chunk, similarity in results:
    print(f"유사도: {{similarity:.3f}}")
    print(chunk.content[:200])
```

## 🚀 다음 단계
1. `autoci dual start` - RAG + 파인튜닝 시작
2. `autoci enhance start /path` - 24시간 자동 시스템 시작

---
*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        report_file = self.index_dir / 'vector_index_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"📝 리포트 생성 완료: {report_file}")


def main():
    """메인 함수"""
    indexer = VectorIndexer()
    
    # 전체 인덱싱
    indexer.index_all_data()
    
    print("\n✅ 벡터 인덱싱 완료!")
    print(f"📊 총 {indexer.stats['total_chunks']:,}개 청크 인덱싱")
    print(f"📁 인덱스 위치: {indexer.index_dir}")
    print(f"🎯 평균 품질: {indexer.stats['avg_quality']:.3f}")
    
    # 검색 테스트
    print("\n🔍 검색 테스트...")
    results = indexer.search("async await best practices", k=3)
    print(f"검색 결과: {len(results)}개")
    
    for i, (chunk, similarity) in enumerate(results):
        print(f"\n[{i+1}] 유사도: {similarity:.3f}")
        print(f"카테고리: {chunk.category}, 타입: {chunk.chunk_type}")
        print(f"내용: {chunk.content[:100]}...")


if __name__ == "__main__":
    main()