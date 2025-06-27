#!/usr/bin/env python3
"""
24시간 연속 C# 신경망 학습 시스템
실제로 신경망 가중치를 업데이트하며 학습하는 시스템
"""

import os
import sys
import json
import time
import logging
import threading
import sqlite3
import asyncio
import aiohttp
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np

# PyTorch 임포트
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. 제한된 기능으로 실행됩니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('neural_continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CSharpDataset(Dataset):
    """C# 학습 데이터셋"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.data = []
        self.vocab = self._build_vocab()
        self._load_data()
        
    def _build_vocab(self) -> Dict[str, int]:
        """어휘 사전 구축"""
        # C# 키워드와 Unity 관련 용어
        keywords = [
            'public', 'private', 'protected', 'class', 'interface', 'namespace',
            'using', 'void', 'int', 'float', 'string', 'bool', 'var', 'const',
            'static', 'async', 'await', 'Task', 'return', 'if', 'else', 'for',
            'foreach', 'while', 'switch', 'case', 'break', 'continue', 'try',
            'catch', 'finally', 'throw', 'new', 'this', 'base', 'override',
            'virtual', 'abstract', 'sealed', 'partial', 'get', 'set', 'value',
            'GameObject', 'Transform', 'Vector3', 'Quaternion', 'Rigidbody',
            'Collider', 'MonoBehaviour', 'Start', 'Update', 'FixedUpdate',
            'OnCollisionEnter', 'OnTriggerEnter', 'Instantiate', 'Destroy'
        ]
        
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        for i, word in enumerate(keywords):
            vocab[word] = i + 4
            
        return vocab
        
    def _load_data(self):
        """데이터 로드"""
        # 실제 구현시 데이터베이스나 파일에서 로드
        self.data = []
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


class CSharpNeuralNetwork(nn.Module):
    """C# 전문 신경망 모델"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = self._create_positional_encoding()
        
        # 트랜스포머 레이어
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        # 출력 레이어
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    def _create_positional_encoding(self, max_len: int = 5000):
        """위치 인코딩 생성"""
        pe = torch.zeros(max_len, self.hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * 
                           (-np.log(10000.0) / self.hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def forward(self, x, mask=None):
        # 임베딩
        seq_len = x.size(1)
        x = self.embedding(x) * np.sqrt(self.hidden_size)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # 트랜스포머
        x = x.transpose(0, 1)  # (batch, seq, hidden) -> (seq, batch, hidden)
        x = self.transformer(x, mask=mask)
        x = x.transpose(0, 1)  # (seq, batch, hidden) -> (batch, seq, hidden)
        
        # 출력
        return self.output_projection(x)


class NeuralContinuousLearner:
    """24시간 연속 신경망 학습 시스템"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / "neural_learning_data"
        self.model_path = self.base_path / "neural_models"
        self.data_path.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)
        
        # 학습 설정
        self.learning_config = {
            'batch_size': 32,
            'learning_rate': 0.0001,
            'max_epochs': 1000,
            'save_interval': 100,
            'eval_interval': 50,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0
        }
        
        # 데이터 소스
        self.data_sources = {
            'github': 'https://api.github.com/search/code',
            'stackoverflow': 'https://api.stackexchange.com/2.3/search',
            'unity_docs': 'https://docs.unity3d.com/ScriptReference/',
            'ms_docs': 'https://docs.microsoft.com/en-us/dotnet/csharp/'
        }
        
        # 학습 상태
        self.learning_state = {
            'is_running': False,
            'total_steps': 0,
            'total_samples': 0,
            'current_loss': 0.0,
            'best_loss': float('inf'),
            'learning_history': deque(maxlen=1000),
            'last_checkpoint': None
        }
        
        # 신경망 초기화
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.optimizer = None
            self.scheduler = None
            self._init_neural_network()
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 스케줄러 설정
        self._setup_scheduler()
        
    def _init_database(self):
        """학습 데이터베이스 초기화"""
        db_path = self.data_path / "neural_learning.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                code_snippet TEXT,
                description TEXT,
                quality_score REAL,
                learned_at TIMESTAMP,
                loss_value REAL,
                is_validated BOOLEAN DEFAULT 0
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                total_samples INTEGER,
                avg_loss REAL,
                learning_rate REAL,
                model_version TEXT,
                metrics TEXT
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS code_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_code TEXT,
                frequency INTEGER,
                quality_score REAL,
                last_seen TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def _init_neural_network(self):
        """신경망 초기화"""
        if not TORCH_AVAILABLE:
            return
            
        logger.info(f"🧠 신경망 초기화 (Device: {self.device})")
        
        # 모델 생성
        vocab_size = 50000  # C# 어휘 크기
        self.model = CSharpNeuralNetwork(
            vocab_size=vocab_size,
            hidden_size=512,
            num_layers=6
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 체크포인트 로드
        self._load_checkpoint()
        
    def _setup_scheduler(self):
        """24시간 학습 스케줄 설정"""
        # 매 시간마다 실행할 작업들
        schedule.every().hour.do(self._hourly_learning)
        
        # 매 30분마다 데이터 수집
        schedule.every(30).minutes.do(self._collect_data)
        
        # 매 10분마다 학습
        schedule.every(10).minutes.do(self._train_batch)
        
        # 매 2시간마다 평가
        schedule.every(2).hours.do(self._evaluate_model)
        
        # 매일 자정에 모델 백업
        schedule.every().day.at("00:00").do(self._backup_model)
        
        # 매일 새벽 3시에 최적화
        schedule.every().day.at("03:00").do(self._optimize_model)
        
    async def _collect_data(self):
        """데이터 수집 (비동기)"""
        logger.info("📥 C# 코드 데이터 수집 시작...")
        
        collected = 0
        
        # GitHub에서 C# 코드 수집
        try:
            async with aiohttp.ClientSession() as session:
                # Unity 관련 C# 코드 검색
                queries = [
                    'Unity MonoBehaviour language:csharp',
                    'Unity Coroutine async await',
                    'Unity GameObject Transform',
                    'Unity Physics Rigidbody',
                    'Unity UI Canvas',
                    'C# LINQ performance',
                    'C# async Task pattern',
                    'C# design patterns'
                ]
                
                for query in queries:
                    url = f"{self.data_sources['github']}?q={query}&per_page=10"
                    
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for item in data.get('items', []):
                                    # 코드 품질 평가
                                    quality = self._evaluate_code_quality(item)
                                    
                                    if quality > 0.7:  # 고품질 코드만 저장
                                        self._save_code_sample(
                                            source='github',
                                            code=item.get('content', ''),
                                            description=item.get('name', ''),
                                            quality=quality
                                        )
                                        collected += 1
                                        
                    except Exception as e:
                        logger.error(f"데이터 수집 오류: {e}")
                        
                    await asyncio.sleep(2)  # API 제한 고려
                    
        except Exception as e:
            logger.error(f"수집 중 오류: {e}")
            
        logger.info(f"✅ {collected}개 코드 샘플 수집 완료")
        
    def _evaluate_code_quality(self, code_item: Dict) -> float:
        """코드 품질 평가"""
        quality_score = 0.5  # 기본 점수
        
        code = code_item.get('content', '')
        
        # 품질 지표들
        indicators = {
            'has_comments': '///' in code or '/*' in code,
            'uses_async': 'async' in code and 'await' in code,
            'has_error_handling': 'try' in code and 'catch' in code,
            'uses_linq': 'using System.Linq' in code,
            'proper_naming': not any(bad in code.lower() for bad in ['temp', 'test', 'foo', 'bar']),
            'has_unity_patterns': any(pattern in code for pattern in ['MonoBehaviour', 'Update', 'Start'])
        }
        
        # 점수 계산
        for indicator, present in indicators.items():
            if present:
                quality_score += 0.1
                
        # 코드 길이 고려
        lines = code.count('\n')
        if 10 < lines < 500:
            quality_score += 0.1
            
        return min(quality_score, 1.0)
        
    def _save_code_sample(self, source: str, code: str, description: str, quality: float):
        """코드 샘플 저장"""
        self.conn.execute('''
            INSERT INTO learning_samples (source, code_snippet, description, quality_score, learned_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (source, code, description, quality, datetime.now()))
        self.conn.commit()
        
    def _train_batch(self):
        """배치 학습"""
        if not TORCH_AVAILABLE or not self.model:
            logger.warning("⚠️ 신경망을 사용할 수 없습니다")
            return
            
        logger.info("🧠 신경망 배치 학습 시작...")
        
        # 학습 데이터 로드
        samples = self.conn.execute('''
            SELECT code_snippet, description 
            FROM learning_samples 
            WHERE is_validated = 0 
            ORDER BY quality_score DESC 
            LIMIT ?
        ''', (self.learning_config['batch_size'],)).fetchall()
        
        if not samples:
            logger.info("학습할 새로운 데이터가 없습니다")
            return
            
        # 학습 모드
        self.model.train()
        total_loss = 0
        
        for code, description in samples:
            try:
                # 코드를 토큰화 (간단한 예시)
                tokens = self._tokenize_code(code)
                
                if len(tokens) < 5:
                    continue
                    
                # 텐서 변환
                input_ids = torch.tensor([tokens[:-1]], device=self.device)
                target_ids = torch.tensor([tokens[1:]], device=self.device)
                
                # 순전파
                outputs = self.model(input_ids)
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    target_ids.reshape(-1)
                )
                
                # 역전파
                loss.backward()
                
                # 그래디언트 누적
                if self.learning_state['total_steps'] % self.learning_config['gradient_accumulation_steps'] == 0:
                    # 그래디언트 클리핑
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.learning_config['max_grad_norm']
                    )
                    
                    # 옵티마이저 스텝
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                total_loss += loss.item()
                self.learning_state['total_steps'] += 1
                
            except Exception as e:
                logger.error(f"학습 중 오류: {e}")
                continue
                
        # 학습 상태 업데이트
        avg_loss = total_loss / len(samples) if samples else 0
        self.learning_state['current_loss'] = avg_loss
        self.learning_state['total_samples'] += len(samples)
        self.learning_state['learning_history'].append({
            'timestamp': datetime.now(),
            'loss': avg_loss,
            'samples': len(samples)
        })
        
        # 검증 완료 표시
        sample_ids = [s[0] for s in samples]
        self.conn.execute(f'''
            UPDATE learning_samples 
            SET is_validated = 1, loss_value = ? 
            WHERE code_snippet IN ({','.join(['?']*len(sample_ids))})
        ''', [avg_loss] + sample_ids)
        self.conn.commit()
        
        logger.info(f"✅ 배치 학습 완료 - Loss: {avg_loss:.4f}, Samples: {len(samples)}")
        
        # 체크포인트 저장
        if self.learning_state['total_steps'] % self.learning_config['save_interval'] == 0:
            self._save_checkpoint()
            
    def _tokenize_code(self, code: str) -> List[int]:
        """코드 토큰화 (간단한 구현)"""
        # 실제로는 더 정교한 토크나이저 필요
        tokens = []
        words = code.split()
        
        for word in words[:100]:  # 최대 100 토큰
            if word in self.model.embedding.weight:
                tokens.append(self.model.embedding.weight[word])
            else:
                tokens.append(1)  # <UNK>
                
        return tokens
        
    def _hourly_learning(self):
        """시간별 종합 학습"""
        logger.info("⏰ 시간별 종합 학습 시작...")
        
        # 수집된 패턴 분석
        patterns = self.conn.execute('''
            SELECT pattern_type, COUNT(*) as count 
            FROM code_patterns 
            GROUP BY pattern_type 
            ORDER BY count DESC 
            LIMIT 10
        ''').fetchall()
        
        logger.info("📊 주요 학습 패턴:")
        for pattern_type, count in patterns:
            logger.info(f"  - {pattern_type}: {count}개")
            
        # 학습 진행 상황 저장
        metrics = {
            'patterns': dict(patterns),
            'total_steps': self.learning_state['total_steps'],
            'avg_loss': self.learning_state['current_loss']
        }
        
        self.conn.execute('''
            INSERT INTO learning_progress 
            (timestamp, total_samples, avg_loss, learning_rate, model_version, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            self.learning_state['total_samples'],
            self.learning_state['current_loss'],
            self.scheduler.get_last_lr()[0] if self.scheduler else 0.0001,
            'v1.0',
            json.dumps(metrics)
        ))
        self.conn.commit()
        
    def _evaluate_model(self):
        """모델 평가"""
        if not TORCH_AVAILABLE or not self.model:
            return
            
        logger.info("🔍 모델 평가 시작...")
        
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        # 평가 데이터 로드
        eval_samples = self.conn.execute('''
            SELECT code_snippet, description 
            FROM learning_samples 
            WHERE quality_score > 0.8 
            ORDER BY RANDOM() 
            LIMIT 100
        ''').fetchall()
        
        with torch.no_grad():
            for code, description in eval_samples:
                try:
                    tokens = self._tokenize_code(code)
                    if len(tokens) < 5:
                        continue
                        
                    input_ids = torch.tensor([tokens[:-1]], device=self.device)
                    target_ids = tokens[1:]
                    
                    outputs = self.model(input_ids)
                    predictions = outputs.argmax(dim=-1).cpu().numpy()[0]
                    
                    # 정확도 계산
                    correct = sum(p == t for p, t in zip(predictions, target_ids))
                    total_correct += correct
                    total_samples += len(target_ids)
                    
                except Exception as e:
                    continue
                    
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        logger.info(f"✅ 평가 완료 - 정확도: {accuracy:.2%}")
        
        # 최고 성능 모델 저장
        if self.learning_state['current_loss'] < self.learning_state['best_loss']:
            self.learning_state['best_loss'] = self.learning_state['current_loss']
            self._save_checkpoint(best=True)
            
    def _save_checkpoint(self, best=False):
        """체크포인트 저장"""
        if not TORCH_AVAILABLE or not self.model:
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learning_state': self.learning_state,
            'timestamp': datetime.now().isoformat()
        }
        
        if best:
            path = self.model_path / "best_model.pt"
        else:
            path = self.model_path / f"checkpoint_{self.learning_state['total_steps']}.pt"
            
        torch.save(checkpoint, path)
        logger.info(f"💾 체크포인트 저장: {path}")
        
        self.learning_state['last_checkpoint'] = str(path)
        
    def _load_checkpoint(self):
        """체크포인트 로드"""
        if not TORCH_AVAILABLE:
            return
            
        best_model_path = self.model_path / "best_model.pt"
        
        if best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.learning_state.update(checkpoint['learning_state'])
                logger.info(f"✅ 체크포인트 로드 완료: {best_model_path}")
            except Exception as e:
                logger.error(f"체크포인트 로드 실패: {e}")
                
    def _backup_model(self):
        """일일 모델 백업"""
        logger.info("💾 일일 모델 백업 시작...")
        
        backup_path = self.model_path / f"backup_{datetime.now().strftime('%Y%m%d')}.pt"
        
        if self.learning_state['last_checkpoint']:
            import shutil
            shutil.copy(self.learning_state['last_checkpoint'], backup_path)
            logger.info(f"✅ 백업 완료: {backup_path}")
            
    def _optimize_model(self):
        """모델 최적화"""
        logger.info("🔧 모델 최적화 시작...")
        
        # 오래된 데이터 정리
        cutoff_date = datetime.now() - timedelta(days=7)
        self.conn.execute('''
            DELETE FROM learning_samples 
            WHERE learned_at < ? AND quality_score < 0.5
        ''', (cutoff_date,))
        
        # 패턴 통계 업데이트
        self.conn.execute('''
            INSERT OR REPLACE INTO code_patterns (pattern_type, pattern_code, frequency, quality_score, last_seen)
            SELECT 
                'unity_pattern' as pattern_type,
                code_snippet as pattern_code,
                COUNT(*) as frequency,
                AVG(quality_score) as quality_score,
                MAX(learned_at) as last_seen
            FROM learning_samples
            WHERE code_snippet LIKE '%MonoBehaviour%'
            GROUP BY code_snippet
            HAVING COUNT(*) > 3
        ''')
        
        self.conn.commit()
        logger.info("✅ 최적화 완료")
        
    def start(self):
        """24시간 연속 학습 시작"""
        self.learning_state['is_running'] = True
        
        logger.info("🚀 24시간 C# 신경망 학습 시작!")
        logger.info(f"🖥️ Device: {self.device if TORCH_AVAILABLE else 'CPU only'}")
        logger.info(f"📊 현재 학습 샘플: {self.learning_state['total_samples']}")
        
        # 초기 데이터 수집
        asyncio.run(self._collect_data())
        
        # 학습 루프
        while self.learning_state['is_running']:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
                
                # 상태 표시
                if self.learning_state['total_steps'] % 100 == 0:
                    self._print_status()
                    
            except KeyboardInterrupt:
                logger.info("🛑 사용자 중단...")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}")
                time.sleep(300)  # 5분 후 재시도
                
    def stop(self):
        """학습 중지"""
        self.learning_state['is_running'] = False
        self._save_checkpoint()
        logger.info("🛑 학습이 중지되었습니다")
        
    def _print_status(self):
        """현재 상태 출력"""
        print("\n" + "="*60)
        print("🧠 C# 신경망 학습 상태")
        print("="*60)
        print(f"총 학습 단계: {self.learning_state['total_steps']:,}")
        print(f"총 학습 샘플: {self.learning_state['total_samples']:,}")
        print(f"현재 Loss: {self.learning_state['current_loss']:.4f}")
        print(f"최고 Loss: {self.learning_state['best_loss']:.4f}")
        
        if self.scheduler:
            print(f"학습률: {self.scheduler.get_last_lr()[0]:.6f}")
            
        # 최근 학습 패턴
        recent_patterns = self.conn.execute('''
            SELECT pattern_type, frequency 
            FROM code_patterns 
            ORDER BY last_seen DESC 
            LIMIT 5
        ''').fetchall()
        
        if recent_patterns:
            print("\n최근 학습 패턴:")
            for pattern, freq in recent_patterns:
                print(f"  - {pattern}: {freq}회")
                
        print("="*60 + "\n")
        
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            'is_running': self.learning_state['is_running'],
            'total_steps': self.learning_state['total_steps'],
            'total_samples': self.learning_state['total_samples'],
            'current_loss': self.learning_state['current_loss'],
            'best_loss': self.learning_state['best_loss'],
            'device': str(self.device) if TORCH_AVAILABLE else 'CPU',
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
        
        # 최근 학습 기록
        recent_history = list(self.learning_state['learning_history'])[-10:]
        stats['recent_history'] = [
            {
                'time': h['timestamp'].strftime('%H:%M'),
                'loss': h['loss'],
                'samples': h['samples']
            }
            for h in recent_history
        ]
        
        return stats


def main():
    """메인 함수"""
    learner = NeuralContinuousLearner()
    
    print("🧠 24시간 C# 신경망 학습 시스템")
    print("="*60)
    print("이 시스템은 실제로 신경망을 학습시켜 C# 코딩 능력을 향상시킵니다.")
    print("GitHub, StackOverflow 등에서 고품질 C# 코드를 수집하고 학습합니다.")
    print("="*60)
    
    try:
        learner.start()
    except KeyboardInterrupt:
        print("\n\n학습을 중지합니다...")
        learner.stop()
        
        
if __name__ == "__main__":
    main()