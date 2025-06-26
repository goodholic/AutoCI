#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean Data Pipeline for AutoCI
대형 언어 모델 방식의 한국어 데이터 수집 및 전처리 파이프라인

ChatGPT/Gemini/Claude 방식:
1. 대규모 웹 크롤링
2. 다양한 도메인 데이터 수집  
3. 품질 필터링
4. 토크나이징 최적화
5. 배치 처리
"""

import os
import json
import asyncio
import aiohttp
import aiofiles
import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import time
from urllib.parse import urljoin, urlparse
import feedparser
from konlpy.tag import Okt, Mecab, Hannanum
import kss  # Korean Sentence Splitter
from transformers import AutoTokenizer
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KoreanWebCrawler:
    """한국어 웹 크롤링 - ChatGPT 방식"""
    
    def __init__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 한국어 사이트 소스들
        self.korean_sources = {
            'news': {
                'naver_news': 'https://news.naver.com',
                'daum_news': 'https://news.daum.net',
                'ytn': 'https://www.ytn.co.kr',
                'kbs': 'https://news.kbs.co.kr',
                'mbc': 'https://imnews.imbc.com',
                'sbs': 'https://news.sbs.co.kr'
            },
            'blogs': {
                'naver_blog': 'https://blog.naver.com',
                'tistory': 'https://tistory.com',
                'brunch': 'https://brunch.co.kr'
            },
            'forums': {
                'clien': 'https://www.clien.net',
                'dcinside': 'https://www.dcinside.com',
                'bobaedream': 'https://www.bobaedream.co.kr',
                'mlbpark': 'https://mlbpark.donga.com'
            },
            'government': {
                'korea_kr': 'https://www.korea.kr',
                'mois': 'https://www.mois.go.kr',
                'moef': 'https://www.moef.go.kr',
                'me': 'https://www.moe.go.kr'
            },
            'education': {
                'kocw': 'http://www.kocw.net',
                'kmooc': 'http://www.kmooc.kr',
                'coursera_ko': 'https://www.coursera.org/browse?language=ko'
            },
            'literature': {
                'yes24': 'http://www.yes24.com',
                'kyobobook': 'http://www.kyobobook.co.kr',
                'aladin': 'https://www.aladin.co.kr'
            }
        }
        
        self.visited_urls = set()
        self.crawl_stats = {
            'total_pages': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'korean_content_found': 0,
            'duplicate_content': 0
        }
    
    async def crawl_korean_web(self, max_pages: int = 1000) -> List[Dict[str, Any]]:
        """한국어 웹 크롤링 실행"""
        logger.info(f"🕷️ 한국어 웹 크롤링 시작 - 최대 {max_pages}페이지")
        
        all_content = []
        semaphore = asyncio.Semaphore(10)  # 동시 요청 제한
        
        tasks = []
        page_count = 0
        
        for category, sources in self.korean_sources.items():
            for source_name, base_url in sources.items():
                if page_count >= max_pages:
                    break
                
                # 각 소스별로 크롤링 태스크 생성
                task = self._crawl_source(semaphore, category, source_name, base_url)
                tasks.append(task)
                page_count += 1
        
        # 모든 크롤링 태스크 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 수집
        for result in results:
            if isinstance(result, list):
                all_content.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"크롤링 오류: {result}")
        
        logger.info(f"✅ 크롤링 완료 - 총 {len(all_content)}개 컨텐츠 수집")
        logger.info(f"📊 통계: {self.crawl_stats}")
        
        return all_content
    
    async def _crawl_source(self, semaphore: asyncio.Semaphore, category: str, source_name: str, base_url: str) -> List[Dict[str, Any]]:
        """개별 소스 크롤링"""
        async with semaphore:
            try:
                logger.info(f"🔍 크롤링 중: {category}/{source_name}")
                
                content_list = []
                
                # RSS 피드 확인
                rss_content = await self._try_rss_crawl(base_url)
                if rss_content:
                    content_list.extend(rss_content)
                
                # 일반 웹 페이지 크롤링
                web_content = await self._crawl_web_pages(base_url, category)
                if web_content:
                    content_list.extend(web_content)
                
                self.crawl_stats['successful_crawls'] += 1
                return content_list
                
            except Exception as e:
                logger.error(f"소스 크롤링 실패 {source_name}: {e}")
                self.crawl_stats['failed_crawls'] += 1
                return []
    
    async def _try_rss_crawl(self, base_url: str) -> List[Dict[str, Any]]:
        """RSS 피드 크롤링 시도"""
        rss_paths = ['/rss', '/rss.xml', '/feed', '/feed.xml', '/atom.xml']
        content_list = []
        
        for rss_path in rss_paths:
            try:
                rss_url = urljoin(base_url, rss_path)
                
                async with self.session.get(rss_url, headers=self.headers) as response:
                    if response.status == 200:
                        rss_content = await response.text()
                        feed = feedparser.parse(rss_content)
                        
                        for entry in feed.entries[:10]:  # 최신 10개
                            if hasattr(entry, 'title') and hasattr(entry, 'summary'):
                                content = {
                                    'source_type': 'rss',
                                    'source_url': getattr(entry, 'link', rss_url),
                                    'title': entry.title,
                                    'content': entry.summary,
                                    'published_date': getattr(entry, 'published', ''),
                                    'collected_date': datetime.now().isoformat(),
                                    'language': 'korean',
                                    'domain': urlparse(base_url).netloc
                                }
                                
                                if self._is_korean_content(content['title'] + content['content']):
                                    content_list.append(content)
                                    self.crawl_stats['korean_content_found'] += 1
                        break
                        
            except Exception as e:
                continue  # 다음 RSS 경로 시도
        
        return content_list
    
    async def _crawl_web_pages(self, base_url: str, category: str) -> List[Dict[str, Any]]:
        """일반 웹 페이지 크롤링"""
        try:
            async with self.session.get(base_url, headers=self.headers) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 텍스트 추출
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ''
                
                # 본문 추출 (다양한 태그에서)
                content_tags = soup.find_all(['p', 'div', 'article', 'section'], 
                                           class_=re.compile(r'content|article|post|text', re.I))
                
                content_text = ''
                for tag in content_tags:
                    text = tag.get_text().strip()
                    if len(text) > 50:  # 의미있는 길이의 텍스트만
                        content_text += text + '\n'
                
                if not content_text:
                    # 대안: 모든 p 태그에서 텍스트 추출
                    paragraphs = soup.find_all('p')
                    content_text = '\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
                
                # 한국어 컨텐츠 확인
                full_text = title_text + content_text
                if self._is_korean_content(full_text) and len(content_text) > 100:
                    self.crawl_stats['korean_content_found'] += 1
                    self.crawl_stats['total_pages'] += 1
                    
                    return [{
                        'source_type': 'web',
                        'source_url': base_url,
                        'title': title_text,
                        'content': content_text,
                        'collected_date': datetime.now().isoformat(),
                        'language': 'korean',
                        'domain': urlparse(base_url).netloc,
                        'category': category,
                        'content_hash': hashlib.md5(content_text.encode()).hexdigest()
                    }]
                
                return []
                
        except Exception as e:
            logger.error(f"웹 페이지 크롤링 오류 {base_url}: {e}")
            return []
    
    def _is_korean_content(self, text: str) -> bool:
        """한국어 컨텐츠 여부 확인"""
        if not text or len(text) < 10:
            return False
        
        # 한글 문자 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣a-zA-Z0-9]', text))
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio >= 0.3  # 30% 이상 한글
    
    async def close(self):
        """세션 종료"""
        await self.session.close()

class KoreanTextProcessor:
    """한국어 텍스트 전처리 - 대형 모델 방식"""
    
    def __init__(self):
        self.okt = Okt()
        try:
            self.mecab = Mecab()
        except:
            self.mecab = None
            logger.warning("Mecab 사용 불가, Okt만 사용")
        
        self.hannanum = Hannanum()
        
        # 전처리 규칙
        self.cleaning_rules = {
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': True,
            'normalize_whitespace': True,
            'remove_special_chars': False,  # 한국어 특성상 유지
            'min_length': 10,
            'max_length': 10000
        }
        
        # 품질 필터 규칙
        self.quality_filters = {
            'min_korean_ratio': 0.3,
            'max_repetition_ratio': 0.3,
            'min_sentence_count': 2,
            'forbidden_patterns': [
                r'(.)\1{10,}',  # 같은 문자 10번 이상 반복
                r'[^\w\s가-힣.,!?;:()]{5,}',  # 특수문자 5개 이상 연속
            ]
        }
    
    def process_text_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """텍스트 배치 처리"""
        logger.info(f"📝 텍스트 배치 처리 시작 - {len(texts)}개")
        
        processed_texts = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_single_text, text) for text in texts]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        processed_texts.append(result)
                        
                    if (i + 1) % 100 == 0:
                        logger.info(f"진행률: {i+1}/{len(texts)}")
                        
                except Exception as e:
                    logger.error(f"텍스트 처리 오류: {e}")
        
        logger.info(f"✅ 배치 처리 완료 - {len(processed_texts)}개 성공")
        return processed_texts
    
    def process_single_text(self, text: str) -> Optional[Dict[str, Any]]:
        """단일 텍스트 처리"""
        try:
            # 1. 기본 정리
            cleaned_text = self._clean_text(text)
            
            # 2. 품질 검사
            if not self._quality_check(cleaned_text):
                return None
            
            # 3. 문장 분리
            sentences = kss.split_sentences(cleaned_text)
            
            # 4. 형태소 분석
            morphological_analysis = self._analyze_morphology(cleaned_text)
            
            # 5. 언어학적 특성 추출
            linguistic_features = self._extract_linguistic_features(cleaned_text, sentences)
            
            # 6. 토큰화
            tokens = self._tokenize_korean(cleaned_text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'sentences': sentences,
                'sentence_count': len(sentences),
                'morphology': morphological_analysis,
                'features': linguistic_features,
                'tokens': tokens,
                'token_count': len(tokens),
                'processing_date': datetime.now().isoformat(),
                'quality_score': self._calculate_quality_score(cleaned_text, sentences, morphological_analysis)
            }
            
        except Exception as e:
            logger.error(f"텍스트 처리 오류: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        cleaned = text
        
        if self.cleaning_rules['remove_html']:
            cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        if self.cleaning_rules['remove_urls']:
            cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        if self.cleaning_rules['remove_emails']:
            cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
        
        if self.cleaning_rules['remove_phone_numbers']:
            cleaned = re.sub(r'(\d{2,3}[-.]?\d{3,4}[-.]?\d{4})', '', cleaned)
        
        if self.cleaning_rules['normalize_whitespace']:
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
        
        return cleaned
    
    def _quality_check(self, text: str) -> bool:
        """품질 검사"""
        if not text:
            return False
        
        # 길이 검사
        if len(text) < self.quality_filters['min_length']:
            return False
        if len(text) > self.quality_filters['max_length']:
            return False
        
        # 한국어 비율 검사
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣a-zA-Z0-9]', text))
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        if korean_ratio < self.quality_filters['min_korean_ratio']:
            return False
        
        # 반복 패턴 검사
        for pattern in self.quality_filters['forbidden_patterns']:
            if re.search(pattern, text):
                return False
        
        # 문장 수 검사
        sentences = kss.split_sentences(text)
        if len(sentences) < self.quality_filters['min_sentence_count']:
            return False
        
        return True
    
    def _analyze_morphology(self, text: str) -> Dict[str, Any]:
        """형태소 분석"""
        try:
            # OKT 분석
            okt_morphs = self.okt.morphs(text)
            okt_pos = self.okt.pos(text)
            okt_nouns = self.okt.nouns(text)
            
            analysis = {
                'morphemes': okt_morphs[:100],  # 처음 100개만 저장
                'pos_tags': okt_pos[:100],
                'nouns': okt_nouns[:50],
                'total_morphemes': len(okt_morphs),
                'unique_morphemes': len(set(okt_morphs)),
                'vocabulary_richness': len(set(okt_morphs)) / len(okt_morphs) if okt_morphs else 0
            }
            
            # Mecab 사용 가능하면 추가 분석
            if self.mecab:
                try:
                    mecab_morphs = self.mecab.morphs(text)
                    analysis['mecab_morphemes'] = mecab_morphs[:100]
                    analysis['mecab_total'] = len(mecab_morphs)
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"형태소 분석 오류: {e}")
            return {'error': str(e)}
    
    def _extract_linguistic_features(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """언어학적 특성 추출"""
        features = {}
        
        # 기본 통계
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 문장 유형 분석
        question_sentences = [s for s in sentences if s.strip().endswith('?')]
        exclamation_sentences = [s for s in sentences if s.strip().endswith('!')]
        
        features['question_ratio'] = len(question_sentences) / len(sentences) if sentences else 0
        features['exclamation_ratio'] = len(exclamation_sentences) / len(sentences) if sentences else 0
        
        # 격식성 분석
        formal_endings = ['습니다', '입니다', '하십시오', '하겠습니다']
        informal_endings = ['해', '해요', '이야', '야', '네']
        
        formal_count = sum(text.count(ending) for ending in formal_endings)
        informal_count = sum(text.count(ending) for ending in informal_endings)
        
        if formal_count + informal_count > 0:
            features['formality_score'] = formal_count / (formal_count + informal_count)
        else:
            features['formality_score'] = 0.5
        
        # 어휘 다양성
        words = text.split()
        unique_words = set(words)
        features['lexical_diversity'] = len(unique_words) / len(words) if words else 0
        
        # 복잡성 점수
        features['complexity_score'] = self._calculate_complexity(text, sentences)
        
        return features
    
    def _calculate_complexity(self, text: str, sentences: List[str]) -> float:
        """텍스트 복잡성 점수 계산"""
        if not sentences:
            return 0.0
        
        # 평균 문장 길이
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # 어려운 단어 비율 (3음절 이상)
        words = text.split()
        complex_words = [w for w in words if len(w) >= 3]
        complex_word_ratio = len(complex_words) / len(words) if words else 0
        
        # 복잡성 점수 (0-1)
        complexity = min((avg_sentence_length / 20) * 0.5 + complex_word_ratio * 0.5, 1.0)
        
        return complexity
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """한국어 토큰화"""
        try:
            # 여러 토크나이저 조합 사용
            tokens = []
            
            # 1. 형태소 단위
            morphemes = self.okt.morphs(text)
            tokens.extend(morphemes)
            
            # 2. 어절 단위  
            words = text.split()
            tokens.extend(words)
            
            # 3. 문자 단위 (한글만)
            korean_chars = re.findall(r'[가-힣]', text)
            
            # 중복 제거 및 정리
            unique_tokens = list(set(tokens))
            
            return unique_tokens[:500]  # 최대 500개 토큰
            
        except Exception as e:
            logger.error(f"토큰화 오류: {e}")
            return text.split()[:100]
    
    def _calculate_quality_score(self, text: str, sentences: List[str], morphology: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        score = 0.0
        
        # 길이 점수 (0.3)
        ideal_length = 500
        length_score = min(len(text) / ideal_length, 1.0) * 0.3
        score += length_score
        
        # 문장 구조 점수 (0.2)
        if sentences and len(sentences) >= 2:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            structure_score = min(avg_sentence_length / 50, 1.0) * 0.2
            score += structure_score
        
        # 어휘 다양성 점수 (0.3)
        if 'vocabulary_richness' in morphology:
            vocab_score = morphology['vocabulary_richness'] * 0.3
            score += vocab_score
        
        # 문법적 완성도 점수 (0.2)
        grammar_patterns = ['은', '는', '이', '가', '을', '를', '에', '에서']
        pattern_count = sum(text.count(pattern) for pattern in grammar_patterns)
        grammar_score = min(pattern_count / 10, 1.0) * 0.2
        score += grammar_score
        
        return min(score, 1.0)

class KoreanDataPipeline:
    """한국어 데이터 파이프라인 통합 관리"""
    
    def __init__(self):
        self.crawler = KoreanWebCrawler()
        self.processor = KoreanTextProcessor()
        self.db_path = "korean_training_data.db"
        self.setup_database()
        
        self.pipeline_stats = {
            'total_crawled': 0,
            'total_processed': 0,
            'high_quality_texts': 0,
            'training_ready_texts': 0,
            'last_run': None
        }
    
    def setup_database(self):
        """데이터베이스 설정"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_korean_data (
                id INTEGER PRIMARY KEY,
                source_type TEXT,
                source_url TEXT,
                title TEXT,
                content TEXT,
                collected_date TIMESTAMP,
                domain TEXT,
                category TEXT,
                language TEXT,
                content_hash TEXT UNIQUE,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_korean_data (
                id INTEGER PRIMARY KEY,
                raw_data_id INTEGER,
                cleaned_text TEXT,
                sentences TEXT,  -- JSON
                morphology TEXT,  -- JSON
                features TEXT,  -- JSON
                tokens TEXT,  -- JSON
                quality_score REAL,
                token_count INTEGER,
                sentence_count INTEGER,
                processing_date TIMESTAMP,
                training_ready BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (raw_data_id) REFERENCES raw_korean_data (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_dataset (
                id INTEGER PRIMARY KEY,
                text TEXT,
                metadata TEXT,  -- JSON
                quality_score REAL,
                tokens INTEGER,
                created_date TIMESTAMP,
                dataset_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def run_full_pipeline(self, max_pages: int = 500):
        """전체 파이프라인 실행"""
        logger.info("🚀 한국어 데이터 파이프라인 시작!")
        logger.info("ChatGPT/Gemini/Claude 방식으로 대규모 한국어 데이터 수집 및 처리")
        
        try:
            # 1단계: 웹 크롤링
            logger.info("\n📥 1단계: 한국어 웹 데이터 수집")
            crawled_data = await self.crawler.crawl_korean_web(max_pages)
            self._save_raw_data(crawled_data)
            self.pipeline_stats['total_crawled'] = len(crawled_data)
            
            # 2단계: 텍스트 전처리
            logger.info("\n🔧 2단계: 텍스트 전처리 및 품질 필터링")
            processed_data = await self._process_crawled_data()
            self.pipeline_stats['total_processed'] = len(processed_data)
            
            # 3단계: 훈련 데이터셋 생성
            logger.info("\n📚 3단계: 훈련 데이터셋 생성")
            training_dataset = self._create_training_dataset(processed_data)
            self.pipeline_stats['training_ready_texts'] = len(training_dataset)
            
            # 4단계: 데이터 품질 리포트
            logger.info("\n📊 4단계: 데이터 품질 리포트 생성")
            quality_report = self._generate_quality_report()
            
            self.pipeline_stats['last_run'] = datetime.now()
            
            # 결과 출력
            self._print_pipeline_results(quality_report)
            
            return {
                'crawled_data': crawled_data,
                'processed_data': processed_data,
                'training_dataset': training_dataset,
                'quality_report': quality_report,
                'stats': self.pipeline_stats
            }
            
        except Exception as e:
            logger.error(f"파이프라인 실행 오류: {e}")
            raise
        finally:
            await self.crawler.close()
    
    def _save_raw_data(self, data_list: List[Dict[str, Any]]):
        """원본 데이터 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        duplicate_count = 0
        
        for data in data_list:
            try:
                cursor.execute('''
                    INSERT INTO raw_korean_data 
                    (source_type, source_url, title, content, collected_date, domain, category, language, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('source_type', ''),
                    data.get('source_url', ''),
                    data.get('title', ''),
                    data.get('content', ''),
                    data.get('collected_date', datetime.now().isoformat()),
                    data.get('domain', ''),
                    data.get('category', ''),
                    data.get('language', 'korean'),
                    data.get('content_hash', '')
                ))
                saved_count += 1
                
            except sqlite3.IntegrityError:
                duplicate_count += 1
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 원본 데이터 저장: {saved_count}개 저장, {duplicate_count}개 중복")
    
    async def _process_crawled_data(self) -> List[Dict[str, Any]]:
        """크롤링된 데이터 전처리"""
        # 미처리 데이터 가져오기
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM raw_korean_data WHERE processed = FALSE")
        raw_data = cursor.fetchall()
        conn.close()
        
        if not raw_data:
            logger.info("처리할 새로운 데이터가 없습니다.")
            return []
        
        logger.info(f"📝 {len(raw_data)}개 텍스트 처리 시작")
        
        # 배치 처리
        texts = [content for _, content in raw_data]
        processed_results = self.processor.process_text_batch(texts)
        
        # 처리 결과 저장
        self._save_processed_data(raw_data, processed_results)
        
        return processed_results
    
    def _save_processed_data(self, raw_data: List[tuple], processed_results: List[Dict[str, Any]]):
        """처리된 데이터 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, result in enumerate(processed_results):
            if i < len(raw_data):
                raw_id = raw_data[i][0]
                
                # 처리된 데이터 저장
                cursor.execute('''
                    INSERT INTO processed_korean_data 
                    (raw_data_id, cleaned_text, sentences, morphology, features, tokens, 
                     quality_score, token_count, sentence_count, processing_date, training_ready)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    raw_id,
                    result.get('cleaned_text', ''),
                    json.dumps(result.get('sentences', []), ensure_ascii=False),
                    json.dumps(result.get('morphology', {}), ensure_ascii=False),
                    json.dumps(result.get('features', {}), ensure_ascii=False),
                    json.dumps(result.get('tokens', []), ensure_ascii=False),
                    result.get('quality_score', 0.0),
                    result.get('token_count', 0),
                    result.get('sentence_count', 0),
                    result.get('processing_date', datetime.now().isoformat()),
                    result.get('quality_score', 0) >= 0.7  # 0.7 이상이면 훈련 가능
                ))
                
                # 원본 데이터 처리 완료 표시
                cursor.execute(
                    "UPDATE raw_korean_data SET processed = TRUE WHERE id = ?",
                    (raw_id,)
                )
        
        conn.commit()
        conn.close()
        
        logger.info(f"💾 처리된 데이터 저장 완료: {len(processed_results)}개")
    
    def _create_training_dataset(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """훈련 데이터셋 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 고품질 데이터만 선별 (quality_score >= 0.7)
        cursor.execute('''
            SELECT cleaned_text, features, quality_score, token_count, sentence_count
            FROM processed_korean_data 
            WHERE training_ready = TRUE AND quality_score >= 0.7
            ORDER BY quality_score DESC
        ''')
        
        high_quality_data = cursor.fetchall()
        conn.close()
        
        training_dataset = []
        dataset_version = f"korean_v{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        for text, features_json, quality_score, token_count, sentence_count in high_quality_data:
            features = json.loads(features_json) if features_json else {}
            
            training_sample = {
                'text': text,
                'metadata': {
                    'quality_score': quality_score,
                    'token_count': token_count,
                    'sentence_count': sentence_count,
                    'features': features,
                    'dataset_version': dataset_version,
                    'created_date': datetime.now().isoformat()
                }
            }
            
            training_dataset.append(training_sample)
        
        # 훈련 데이터셋 저장
        self._save_training_dataset(training_dataset, dataset_version)
        
        logger.info(f"📚 훈련 데이터셋 생성 완료: {len(training_dataset)}개 샘플")
        return training_dataset
    
    def _save_training_dataset(self, dataset: List[Dict[str, Any]], version: str):
        """훈련 데이터셋 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sample in dataset:
            cursor.execute('''
                INSERT INTO training_dataset 
                (text, metadata, quality_score, tokens, created_date, dataset_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                sample['text'],
                json.dumps(sample['metadata'], ensure_ascii=False),
                sample['metadata']['quality_score'],
                sample['metadata']['token_count'],
                sample['metadata']['created_date'],
                version
            ))
        
        conn.commit()
        conn.close()
    
    def _generate_quality_report(self) -> Dict[str, Any]:
        """품질 리포트 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        cursor.execute("SELECT COUNT(*) FROM raw_korean_data")
        total_raw = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processed_korean_data")
        total_processed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM processed_korean_data WHERE training_ready = TRUE")
        training_ready = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(quality_score) FROM processed_korean_data")
        avg_quality = cursor.fetchone()[0] or 0
        
        # 품질 분포
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN quality_score >= 0.9 THEN 'excellent'
                    WHEN quality_score >= 0.7 THEN 'good'
                    WHEN quality_score >= 0.5 THEN 'fair'
                    ELSE 'poor'
                END as quality_level,
                COUNT(*) as count
            FROM processed_korean_data
            GROUP BY quality_level
        ''')
        quality_distribution = dict(cursor.fetchall())
        
        # 도메인별 통계
        cursor.execute('''
            SELECT r.domain, COUNT(*) as count, AVG(p.quality_score) as avg_quality
            FROM raw_korean_data r
            JOIN processed_korean_data p ON r.id = p.raw_data_id
            GROUP BY r.domain
            ORDER BY count DESC
        ''')
        domain_stats = cursor.fetchall()
        
        conn.close()
        
        report = {
            'total_statistics': {
                'raw_data_count': total_raw,
                'processed_count': total_processed,
                'training_ready_count': training_ready,
                'average_quality_score': round(avg_quality, 3),
                'processing_rate': round(total_processed / total_raw * 100, 1) if total_raw > 0 else 0
            },
            'quality_distribution': quality_distribution,
            'domain_statistics': [
                {'domain': domain, 'count': count, 'avg_quality': round(avg_quality, 3)}
                for domain, count, avg_quality in domain_stats
            ],
            'generated_date': datetime.now().isoformat()
        }
        
        self.pipeline_stats['high_quality_texts'] = quality_distribution.get('excellent', 0) + quality_distribution.get('good', 0)
        
        return report
    
    def _print_pipeline_results(self, quality_report: Dict[str, Any]):
        """파이프라인 결과 출력"""
        print(f"""
        🎉 한국어 데이터 파이프라인 완료!
        ================================
        
        📊 전체 통계:
        - 수집된 원본 데이터: {quality_report['total_statistics']['raw_data_count']:,}개
        - 처리된 데이터: {quality_report['total_statistics']['processed_count']:,}개  
        - 훈련 가능한 데이터: {quality_report['total_statistics']['training_ready_count']:,}개
        - 평균 품질 점수: {quality_report['total_statistics']['average_quality_score']}
        - 처리 성공률: {quality_report['total_statistics']['processing_rate']}%
        
        🏆 품질 분포:
        - 우수 (0.9+): {quality_report['quality_distribution'].get('excellent', 0):,}개
        - 양호 (0.7+): {quality_report['quality_distribution'].get('good', 0):,}개  
        - 보통 (0.5+): {quality_report['quality_distribution'].get('fair', 0):,}개
        - 낮음 (<0.5): {quality_report['quality_distribution'].get('poor', 0):,}개
        
        🌐 도메인별 상위 5개:
        """)
        
        for i, domain_stat in enumerate(quality_report['domain_statistics'][:5]):
            print(f"        {i+1}. {domain_stat['domain']}: {domain_stat['count']:,}개 (품질: {domain_stat['avg_quality']})")
        
        print(f"""
        ✅ 이제 AutoCI가 ChatGPT처럼 고품질 한국어 데이터로 학습할 준비가 완료되었습니다!
        """)

async def main():
    """메인 실행 함수"""
    pipeline = KoreanDataPipeline()
    
    try:
        print("""
        🇰🇷 AutoCI Korean Data Pipeline
        ===============================
        
        ChatGPT/Gemini/Claude 방식의 대규모 한국어 데이터 수집 및 처리
        
        처리 단계:
        1️⃣ 한국어 웹사이트 크롤링 (뉴스, 블로그, 포럼, 정부사이트 등)
        2️⃣ 텍스트 정리 및 품질 필터링
        3️⃣ 형태소 분석 및 언어학적 특성 추출
        4️⃣ 토큰화 및 훈련 데이터셋 생성
        5️⃣ 품질 평가 및 리포트 생성
        
        🚀 시작합니다...
        """)
        
        # 파이프라인 실행
        results = await pipeline.run_full_pipeline(max_pages=100)  # 테스트용 100페이지
        
        # 샘플 데이터 출력
        if results['training_dataset']:
            print(f"\n📝 훈련 데이터 샘플:")
            sample = results['training_dataset'][0]
            print(f"텍스트: {sample['text'][:200]}...")
            print(f"품질 점수: {sample['metadata']['quality_score']}")
            print(f"토큰 수: {sample['metadata']['token_count']}")
        
    except KeyboardInterrupt:
        print(f"\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"메인 실행 오류: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 