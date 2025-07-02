#!/usr/bin/env python3
"""
고급 학습 데이터 저장 및 검색 시스템

대규모 학습 데이터를 효율적으로 저장하고 빠르게 검색할 수 있는
상용 수준의 데이터베이스 시스템
"""

import os
import json
import time
import hashlib
import pickle
import sqlite3
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import lmdb
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import msgpack
import zstandard as zstd


@dataclass
class LearningData:
    """학습 데이터"""
    id: str
    type: str  # code, error, solution, pattern, insight
    category: str
    content: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    quality_score: float = 0.5
    usage_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    related_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class AdvancedLearningStorage:
    """고급 학습 데이터 저장소"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/mnt/d/AutoCI/AutoCI/advanced_storage")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 저장소 경로
        self.sqlite_path = self.base_path / "metadata.db"
        self.lmdb_path = self.base_path / "content_store"
        self.faiss_path = self.base_path / "vector_index"
        self.cache_path = self.base_path / "cache"
        
        # LMDB 설정 (대용량 바이너리 데이터)
        self.lmdb_env = lmdb.open(
            str(self.lmdb_path),
            map_size=10 * 1024 * 1024 * 1024,  # 10GB
            max_dbs=10
        )
        
        # 압축기
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()
        
        # 벡터화
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.embedding_dim = 768  # BERT 차원
        
        # FAISS 인덱스 (벡터 검색)
        self.faiss_index = None
        self.index_to_id = {}
        self.id_to_index = {}
        self._initialize_faiss()
        
        # 캐시
        self.memory_cache = {}
        self.cache_size = 1000
        self.access_queue = deque(maxlen=self.cache_size)
        
        # 통계
        self.stats = {
            "total_items": 0,
            "total_searches": 0,
            "cache_hits": 0,
            "average_search_time": 0.0,
            "storage_size_mb": 0.0,
            "index_size": 0
        }
        
        # 인덱싱 큐
        self.indexing_queue = deque()
        self.indexing_thread = threading.Thread(target=self._background_indexing)
        self.indexing_thread.daemon = True
        self.is_indexing = True
        
        # 초기화
        self._initialize_database()
        self._load_statistics()
        
        # 백그라운드 프로세스 시작
        self.indexing_thread.start()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # 메타데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_metadata (
                id TEXT PRIMARY KEY,
                type TEXT,
                category TEXT,
                timestamp TEXT,
                version INTEGER,
                quality_score REAL,
                usage_count INTEGER,
                last_accessed TEXT,
                tags TEXT,
                related_ids TEXT,
                metadata TEXT,
                embedding_id INTEGER
            )
        """)
        
        # 검색 인덱스
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_index (
                term TEXT,
                data_id TEXT,
                frequency REAL,
                PRIMARY KEY (term, data_id)
            )
        """)
        
        # 관계 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                strength REAL,
                PRIMARY KEY (source_id, target_id)
            )
        """)
        
        # 통계 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated TEXT
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON learning_metadata(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON learning_metadata(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON learning_metadata(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON learning_metadata(quality_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage ON learning_metadata(usage_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_term ON search_index(term)")
        
        conn.commit()
        conn.close()
    
    def _initialize_faiss(self):
        """FAISS 벡터 인덱스 초기화"""
        index_file = self.faiss_path / "index.faiss"
        mapping_file = self.faiss_path / "mapping.pkl"
        
        if index_file.exists() and mapping_file.exists():
            # 기존 인덱스 로드
            self.faiss_index = faiss.read_index(str(index_file))
            with open(mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
                self.index_to_id = mapping_data['index_to_id']
                self.id_to_index = mapping_data['id_to_index']
        else:
            # 새 인덱스 생성
            self.faiss_path.mkdir(exist_ok=True)
            # IndexFlatL2: 정확한 L2 거리 계산
            # 대규모일 경우 IndexIVFPQ 등 사용
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            
            # GPU 사용 가능 시
            if faiss.get_num_gpus() > 0:
                self.faiss_index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.faiss_index
                )
    
    def store(self, data: LearningData) -> str:
        """데이터 저장"""
        start_time = time.time()
        
        # ID 생성 (없는 경우)
        if not data.id:
            data.id = self._generate_id(data)
        
        # 컨텐츠 저장 (LMDB)
        content_data = {
            'content': data.content,
            'embeddings': data.embeddings.tolist() if data.embeddings is not None else None
        }
        compressed_content = self.compressor.compress(
            msgpack.packb(content_data, use_bin_type=True)
        )
        
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(data.id.encode(), compressed_content)
        
        # 메타데이터 저장 (SQLite)
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        embedding_id = None
        if data.embeddings is not None:
            # 벡터 인덱스에 추가
            embedding_id = len(self.index_to_id)
            self.faiss_index.add(data.embeddings.reshape(1, -1))
            self.index_to_id[embedding_id] = data.id
            self.id_to_index[data.id] = embedding_id
        
        cursor.execute("""
            INSERT OR REPLACE INTO learning_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.id,
            data.type,
            data.category,
            data.timestamp,
            data.version,
            data.quality_score,
            data.usage_count,
            data.last_accessed,
            json.dumps(data.tags),
            json.dumps(data.related_ids),
            json.dumps(data.metadata),
            embedding_id
        ))
        
        # 검색 인덱스 업데이트
        self._update_search_index(cursor, data)
        
        conn.commit()
        conn.close()
        
        # 캐시 업데이트
        self._update_cache(data.id, data)
        
        # 통계 업데이트
        self.stats["total_items"] += 1
        self._update_storage_size()
        
        # 인덱싱 큐에 추가
        self.indexing_queue.append(data.id)
        
        elapsed = time.time() - start_time
        return data.id
    
    def retrieve(self, data_id: str) -> Optional[LearningData]:
        """데이터 검색"""
        # 캐시 확인
        if data_id in self.memory_cache:
            self.stats["cache_hits"] += 1
            self._update_access(data_id)
            return self.memory_cache[data_id]
        
        # 메타데이터 조회
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM learning_metadata WHERE id = ?
        """, (data_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        # 컨텐츠 조회 (LMDB)
        with self.lmdb_env.begin() as txn:
            compressed_content = txn.get(data_id.encode())
            if not compressed_content:
                conn.close()
                return None
        
        # 압축 해제
        content_data = msgpack.unpackb(
            self.decompressor.decompress(compressed_content),
            raw=False
        )
        
        # LearningData 객체 생성
        data = LearningData(
            id=row[0],
            type=row[1],
            category=row[2],
            content=content_data['content'],
            embeddings=np.array(content_data['embeddings']) if content_data['embeddings'] else None,
            metadata=json.loads(row[10]),
            timestamp=row[3],
            version=row[4],
            quality_score=row[5],
            usage_count=row[6],
            last_accessed=row[7],
            related_ids=json.loads(row[9]),
            tags=json.loads(row[8])
        )
        
        # 사용 횟수 증가
        cursor.execute("""
            UPDATE learning_metadata 
            SET usage_count = usage_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), data_id))
        
        conn.commit()
        conn.close()
        
        # 캐시 업데이트
        self._update_cache(data_id, data)
        self._update_access(data_id)
        
        return data
    
    def search(
        self,
        query: Union[str, np.ndarray],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Tuple[LearningData, float]]:
        """고급 검색"""
        start_time = time.time()
        self.stats["total_searches"] += 1
        
        results = []
        
        if isinstance(query, str):
            # 텍스트 검색
            results = self._text_search(query, filters, limit * 2)
        elif isinstance(query, np.ndarray):
            # 벡터 검색
            results = self._vector_search(query, filters, limit * 2)
        
        # 필터 적용
        if filters:
            results = self._apply_filters(results, filters)
        
        # 정렬 및 제한
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]
        
        # 통계 업데이트
        elapsed = time.time() - start_time
        self.stats["average_search_time"] = (
            self.stats["average_search_time"] * 0.9 + elapsed * 0.1
        )
        
        return results
    
    def _text_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Tuple[LearningData, float]]:
        """텍스트 기반 검색"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # 검색어 토큰화
        terms = query.lower().split()
        
        # 검색 쿼리 구성
        sql = """
            SELECT DISTINCT lm.id, SUM(si.frequency) as relevance
            FROM learning_metadata lm
            JOIN search_index si ON lm.id = si.data_id
            WHERE si.term IN ({})
        """.format(','.join('?' * len(terms)))
        
        params = terms
        
        # 기본 필터
        if filters:
            if 'type' in filters:
                sql += " AND lm.type = ?"
                params.append(filters['type'])
            if 'category' in filters:
                sql += " AND lm.category = ?"
                params.append(filters['category'])
            if 'min_quality' in filters:
                sql += " AND lm.quality_score >= ?"
                params.append(filters['min_quality'])
        
        sql += " GROUP BY lm.id ORDER BY relevance DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            data_id = row[0]
            relevance = row[1]
            
            # 데이터 조회
            data = self.retrieve(data_id)
            if data:
                results.append((data, relevance))
        
        conn.close()
        return results
    
    def _vector_search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Tuple[LearningData, float]]:
        """벡터 기반 검색"""
        if self.faiss_index.ntotal == 0:
            return []
        
        # FAISS 검색
        k = min(limit, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(
            query_vector.reshape(1, -1), k
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # 무효한 인덱스
                continue
            
            data_id = self.index_to_id.get(idx)
            if data_id:
                data = self.retrieve(data_id)
                if data:
                    # 거리를 유사도로 변환 (코사인 유사도)
                    similarity = 1.0 / (1.0 + dist)
                    results.append((data, similarity))
        
        return results
    
    def _apply_filters(
        self,
        results: List[Tuple[LearningData, float]],
        filters: Dict[str, Any]
    ) -> List[Tuple[LearningData, float]]:
        """필터 적용"""
        filtered = []
        
        for data, score in results:
            # 타입 필터
            if 'type' in filters and data.type != filters['type']:
                continue
            
            # 카테고리 필터
            if 'category' in filters and data.category != filters['category']:
                continue
            
            # 품질 필터
            if 'min_quality' in filters and data.quality_score < filters['min_quality']:
                continue
            
            # 시간 필터
            if 'since' in filters:
                data_time = datetime.fromisoformat(data.timestamp)
                if data_time < filters['since']:
                    continue
            
            # 태그 필터
            if 'tags' in filters:
                required_tags = set(filters['tags'])
                if not required_tags.intersection(set(data.tags)):
                    continue
            
            filtered.append((data, score))
        
        return filtered
    
    def update_embeddings(self, data_id: str, embeddings: np.ndarray):
        """임베딩 업데이트"""
        data = self.retrieve(data_id)
        if not data:
            return
        
        data.embeddings = embeddings
        
        # FAISS 인덱스 업데이트
        if data_id in self.id_to_index:
            # 기존 인덱스 제거하고 새로 추가
            old_index = self.id_to_index[data_id]
            # FAISS는 직접 업데이트 불가, 재구축 필요
            # 실제로는 주기적으로 재구축하는 것이 좋음
        else:
            # 새로 추가
            embedding_id = len(self.index_to_id)
            self.faiss_index.add(embeddings.reshape(1, -1))
            self.index_to_id[embedding_id] = data_id
            self.id_to_index[data_id] = embedding_id
        
        # 저장
        self.store(data)
    
    def find_similar(
        self,
        data_id: str,
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Tuple[LearningData, float]]:
        """유사한 데이터 찾기"""
        source_data = self.retrieve(data_id)
        if not source_data:
            return []
        
        similar = []
        
        # 임베딩 기반 유사도
        if source_data.embeddings is not None:
            similar_by_vector = self._vector_search(
                source_data.embeddings,
                filters={'type': source_data.type},
                limit=limit * 2
            )
            similar.extend(similar_by_vector)
        
        # 태그 기반 유사도
        if source_data.tags:
            tag_query = ' '.join(source_data.tags)
            similar_by_tags = self._text_search(
                tag_query,
                filters={'type': source_data.type},
                limit=limit
            )
            
            # 점수 조정
            for data, score in similar_by_tags:
                if data.id != data_id:
                    tag_similarity = len(set(data.tags) & set(source_data.tags)) / len(set(data.tags) | set(source_data.tags))
                    adjusted_score = score * 0.5 + tag_similarity * 0.5
                    similar.append((data, adjusted_score))
        
        # 중복 제거 및 정렬
        seen = {data_id}
        unique_similar = []
        for data, score in similar:
            if data.id not in seen and score >= min_similarity:
                seen.add(data.id)
                unique_similar.append((data, score))
        
        unique_similar.sort(key=lambda x: x[1], reverse=True)
        return unique_similar[:limit]
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0
    ):
        """관계 생성"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO relationships VALUES (?, ?, ?, ?)
        """, (source_id, target_id, relationship_type, strength))
        
        conn.commit()
        conn.close()
        
        # 양방향 관계 업데이트
        source_data = self.retrieve(source_id)
        target_data = self.retrieve(target_id)
        
        if source_data and target_id not in source_data.related_ids:
            source_data.related_ids.append(target_id)
            self.store(source_data)
        
        if target_data and source_id not in target_data.related_ids:
            target_data.related_ids.append(source_id)
            self.store(target_data)
    
    def get_relationships(self, data_id: str) -> List[Dict[str, Any]]:
        """관계 조회"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT target_id, relationship_type, strength
            FROM relationships
            WHERE source_id = ?
            ORDER BY strength DESC
        """, (data_id,))
        
        relationships = []
        for row in cursor.fetchall():
            relationships.append({
                'target_id': row[0],
                'type': row[1],
                'strength': row[2]
            })
        
        conn.close()
        return relationships
    
    def _update_search_index(self, cursor, data: LearningData):
        """검색 인덱스 업데이트"""
        # 기존 인덱스 삭제
        cursor.execute("DELETE FROM search_index WHERE data_id = ?", (data.id,))
        
        # 텍스트 추출
        text_content = []
        
        # 컨텐츠에서 텍스트 추출
        def extract_text(obj):
            if isinstance(obj, str):
                text_content.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item)
        
        extract_text(data.content)
        text_content.extend(data.tags)
        
        # 토큰화 및 빈도 계산
        all_text = ' '.join(text_content).lower()
        tokens = all_text.split()
        token_freq = defaultdict(int)
        
        for token in tokens:
            if len(token) > 2:  # 2글자 이상
                token_freq[token] += 1
        
        # 인덱스 추가
        for token, freq in token_freq.items():
            cursor.execute("""
                INSERT INTO search_index (term, data_id, frequency)
                VALUES (?, ?, ?)
            """, (token, data.id, freq / len(tokens)))
    
    def _background_indexing(self):
        """백그라운드 인덱싱"""
        while self.is_indexing:
            try:
                if self.indexing_queue:
                    data_id = self.indexing_queue.popleft()
                    
                    # 관련 데이터 찾기
                    data = self.retrieve(data_id)
                    if data:
                        # 유사한 데이터 찾기
                        similar = self.find_similar(data_id, limit=5)
                        
                        # 관계 생성
                        for similar_data, similarity in similar:
                            if similarity > 0.8:
                                self.create_relationship(
                                    data_id,
                                    similar_data.id,
                                    "similar",
                                    similarity
                                )
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"인덱싱 오류: {e}")
                time.sleep(1)
    
    def _update_cache(self, data_id: str, data: LearningData):
        """캐시 업데이트"""
        # LRU 캐시
        if len(self.memory_cache) >= self.cache_size:
            # 가장 오래된 항목 제거
            if self.access_queue:
                old_id = self.access_queue[0]
                if old_id in self.memory_cache and old_id != data_id:
                    del self.memory_cache[old_id]
        
        self.memory_cache[data_id] = data
    
    def _update_access(self, data_id: str):
        """접근 기록 업데이트"""
        # 기존 위치에서 제거
        try:
            self.access_queue.remove(data_id)
        except ValueError:
            pass
        
        # 끝에 추가
        self.access_queue.append(data_id)
    
    def _generate_id(self, data: LearningData) -> str:
        """ID 생성"""
        content = f"{data.type}_{data.category}_{json.dumps(data.content, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _update_storage_size(self):
        """저장소 크기 업데이트"""
        total_size = 0
        
        # SQLite 크기
        if self.sqlite_path.exists():
            total_size += self.sqlite_path.stat().st_size
        
        # LMDB 크기
        lmdb_data = self.lmdb_path / "data.mdb"
        if lmdb_data.exists():
            total_size += lmdb_data.stat().st_size
        
        # FAISS 크기
        faiss_index = self.faiss_path / "index.faiss"
        if faiss_index.exists():
            total_size += faiss_index.stat().st_size
        
        self.stats["storage_size_mb"] = total_size / (1024 * 1024)
    
    def _load_statistics(self):
        """통계 로드"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM learning_metadata")
        self.stats["total_items"] = cursor.fetchone()[0]
        
        if self.faiss_index:
            self.stats["index_size"] = self.faiss_index.ntotal
        
        conn.close()
        self._update_storage_size()
    
    def optimize(self):
        """저장소 최적화"""
        print("저장소 최적화 시작...")
        
        # SQLite VACUUM
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("VACUUM")
        conn.close()
        
        # LMDB 압축
        self.lmdb_env.sync()
        
        # FAISS 인덱스 재구축
        if self.faiss_index.ntotal > 10000:
            print("FAISS 인덱스 재구축 중...")
            # 클러스터링 기반 인덱스로 전환
            new_index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.embedding_dim),
                self.embedding_dim,
                min(self.faiss_index.ntotal // 10, 1000)
            )
            
            # 모든 벡터 추출 및 재추가
            vectors = []
            for i in range(self.faiss_index.ntotal):
                vectors.append(self.faiss_index.reconstruct(i))
            
            vectors = np.array(vectors)
            new_index.train(vectors)
            new_index.add(vectors)
            
            self.faiss_index = new_index
            
            # 저장
            faiss.write_index(self.faiss_index, str(self.faiss_path / "index.faiss"))
        
        # 캐시 정리
        self.memory_cache.clear()
        self.access_queue.clear()
        
        print("최적화 완료!")
    
    def export_data(
        self,
        output_path: str,
        filters: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> int:
        """데이터 내보내기"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # 쿼리 구성
        sql = "SELECT id FROM learning_metadata WHERE 1=1"
        params = []
        
        if filters:
            if 'type' in filters:
                sql += " AND type = ?"
                params.append(filters['type'])
            if 'category' in filters:
                sql += " AND category = ?"
                params.append(filters['category'])
            if 'min_quality' in filters:
                sql += " AND quality_score >= ?"
                params.append(filters['min_quality'])
        
        cursor.execute(sql, params)
        
        exported_data = []
        for row in cursor.fetchall():
            data = self.retrieve(row[0])
            if data:
                if format == "json":
                    exported_data.append(asdict(data))
                else:  # msgpack
                    exported_data.append(data)
        
        conn.close()
        
        # 저장
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, ensure_ascii=False, indent=2)
        else:  # msgpack
            with open(output_path, 'wb') as f:
                f.write(msgpack.packb(exported_data, use_bin_type=True))
        
        return len(exported_data)
    
    def import_data(self, input_path: str, format: str = "json") -> int:
        """데이터 가져오기"""
        if format == "json":
            with open(input_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        else:  # msgpack
            with open(input_path, 'rb') as f:
                data_list = msgpack.unpackb(f.read(), raw=False)
        
        imported = 0
        for data_dict in data_list:
            # numpy array 복원
            if 'embeddings' in data_dict and data_dict['embeddings']:
                data_dict['embeddings'] = np.array(data_dict['embeddings'])
            
            data = LearningData(**data_dict)
            self.store(data)
            imported += 1
        
        return imported
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 반환"""
        self._load_statistics()
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # 타입별 분포
        cursor.execute("""
            SELECT type, COUNT(*) FROM learning_metadata
            GROUP BY type
        """)
        type_distribution = dict(cursor.fetchall())
        
        # 카테고리별 분포
        cursor.execute("""
            SELECT category, COUNT(*) FROM learning_metadata
            GROUP BY category
        """)
        category_distribution = dict(cursor.fetchall())
        
        # 품질 분포
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN quality_score >= 0.9 THEN 'excellent'
                    WHEN quality_score >= 0.7 THEN 'good'
                    WHEN quality_score >= 0.5 THEN 'fair'
                    ELSE 'poor'
                END as quality_level,
                COUNT(*) as count
            FROM learning_metadata
            GROUP BY quality_level
        """)
        quality_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            **self.stats,
            "type_distribution": type_distribution,
            "category_distribution": category_distribution,
            "quality_distribution": quality_distribution,
            "cache_size": len(self.memory_cache),
            "indexing_queue_size": len(self.indexing_queue)
        }
    
    def close(self):
        """저장소 닫기"""
        self.is_indexing = False
        
        # 인덱싱 스레드 종료
        if self.indexing_thread:
            self.indexing_thread.join(timeout=5)
        
        # FAISS 인덱스 저장
        if self.faiss_index and self.faiss_index.ntotal > 0:
            faiss.write_index(self.faiss_index, str(self.faiss_path / "index.faiss"))
            with open(self.faiss_path / "mapping.pkl", 'wb') as f:
                pickle.dump({
                    'index_to_id': self.index_to_id,
                    'id_to_index': self.id_to_index
                }, f)
        
        # LMDB 닫기
        self.lmdb_env.close()
        
        # 통계 저장
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        for key, value in self.stats.items():
            cursor.execute("""
                INSERT OR REPLACE INTO statistics VALUES (?, ?, ?)
            """, (key, str(value), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()


def demo():
    """데모 실행"""
    print("고급 학습 데이터 저장소 데모")
    print("-" * 50)
    
    # 저장소 초기화
    storage = AdvancedLearningStorage()
    
    # 샘플 데이터 저장
    print("\n1. 데이터 저장:")
    
    # 코드 패턴
    code_data = LearningData(
        id="",
        type="pattern",
        category="player_movement",
        content={
            "pattern_name": "smooth_movement",
            "code": """
func _physics_process(delta):
    var input_vector = Vector2()
    input_vector.x = Input.get_action_strength("ui_right") - Input.get_action_strength("ui_left")
    input_vector.y = Input.get_action_strength("ui_down") - Input.get_action_strength("ui_up")
    input_vector = input_vector.normalized()
    
    if input_vector != Vector2.ZERO:
        velocity = velocity.move_toward(input_vector * max_speed, acceleration * delta)
    else:
        velocity = velocity.move_toward(Vector2.ZERO, friction * delta)
    
    move_and_slide()
""",
            "description": "부드러운 플레이어 이동 구현"
        },
        tags=["movement", "physics", "player"],
        quality_score=0.9,
        embeddings=np.random.rand(768)  # 실제로는 BERT 등으로 생성
    )
    
    data_id = storage.store(code_data)
    print(f"  저장됨: {data_id}")
    
    # 오류 해결책
    error_data = LearningData(
        id="",
        type="solution",
        category="collision_error",
        content={
            "error": "CollisionShape2D not found",
            "cause": "CollisionShape2D가 Area2D의 자식이 아님",
            "solution": "Area2D 노드 아래에 CollisionShape2D 추가",
            "code_fix": """
# Scene structure:
Area2D
  └── CollisionShape2D  # 이 구조가 필요
  └── Sprite2D
"""
        },
        tags=["collision", "error", "area2d"],
        quality_score=0.85
    )
    
    error_id = storage.store(error_data)
    print(f"  저장됨: {error_id}")
    
    # 인사이트
    insight_data = LearningData(
        id="",
        type="insight",
        category="optimization",
        content={
            "title": "get_node 캐싱으로 성능 향상",
            "insight": "onready var를 사용하여 노드 참조를 캐싱하면 성능이 크게 향상됨",
            "before": "get_node('UI/HealthBar').value = health",
            "after": "@onready var health_bar = $UI/HealthBar\nhealth_bar.value = health",
            "performance_gain": "30-50% in _process"
        },
        tags=["optimization", "performance", "nodes"],
        quality_score=0.95
    )
    
    insight_id = storage.store(insight_data)
    
    # 관계 생성
    storage.create_relationship(code_data.id, insight_id, "optimizes", 0.8)
    
    # 2. 검색
    print("\n2. 텍스트 검색 - 'movement':")
    results = storage.search("movement", limit=5)
    for data, score in results:
        print(f"  - {data.type}: {data.content.get('pattern_name', data.content.get('title', 'N/A'))} (점수: {score:.2f})")
    
    # 3. 유사 데이터 찾기
    print(f"\n3. '{code_data.id}'와 유사한 데이터:")
    similar = storage.find_similar(code_data.id, limit=3)
    for data, similarity in similar:
        print(f"  - {data.type}: {data.category} (유사도: {similarity:.2f})")
    
    # 4. 필터링 검색
    print("\n4. 고품질 패턴 검색:")
    high_quality = storage.search(
        "godot",
        filters={'type': 'pattern', 'min_quality': 0.8},
        limit=5
    )
    for data, score in high_quality:
        print(f"  - {data.content.get('pattern_name', 'N/A')} (품질: {data.quality_score})")
    
    # 5. 통계
    print("\n5. 저장소 통계:")
    stats = storage.get_statistics()
    print(f"  총 항목: {stats['total_items']}")
    print(f"  저장소 크기: {stats['storage_size_mb']:.2f} MB")
    print(f"  평균 검색 시간: {stats['average_search_time']*1000:.1f} ms")
    print(f"  캐시 히트: {stats['cache_hits']}")
    
    # 정리
    storage.close()


if __name__ == "__main__":
    # 필요한 패키지 설치 확인
    try:
        import lmdb
        import faiss
        import msgpack
        import zstandard
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        print(f"필요한 패키지를 설치하세요: {e}")
        print("pip install lmdb faiss-cpu msgpack zstandard scikit-learn")
        exit(1)
    
    demo()