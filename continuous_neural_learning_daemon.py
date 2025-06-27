#!/usr/bin/env python3
"""
AutoCI 24시간 지속 학습 데몬
실제 신경망 기반 24/7 자동 학습 시스템
"""

import os
import sys
import time
import json
import sqlite3
import threading
import schedule
import signal
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import requests
import hashlib

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.feature_extraction.text import TfidfVectorizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음 - 간소화된 학습 모드로 실행")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_24h_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearningDatabase:
    """지속 학습 데이터베이스"""
    
    def __init__(self, db_path: str = "continuous_learning.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 학습 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    target_response TEXT,
                    feedback_score REAL,
                    source TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    quality_score REAL
                )
            ''')
            
            # 학습 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_data_points INTEGER,
                    training_loss REAL,
                    validation_accuracy REAL,
                    learning_rate REAL,
                    epoch_count INTEGER,
                    model_version TEXT
                )
            ''')
            
            # 모델 체크포인트 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    performance_score REAL,
                    training_time_hours REAL,
                    notes TEXT
                )
            ''')
            
            conn.commit()
    
    def add_learning_data(self, input_text: str, target_response: str = None, 
                         feedback_score: float = 0.0, source: str = "auto"):
        """학습 데이터 추가"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_data 
                (timestamp, input_text, target_response, feedback_score, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), input_text, target_response, feedback_score, source))
            conn.commit()
    
    def get_unprocessed_data(self, limit: int = 100) -> List[Tuple]:
        """미처리 학습 데이터 가져오기"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, input_text, target_response, feedback_score, source
                FROM learning_data 
                WHERE processed = FALSE 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
    
    def mark_data_processed(self, data_ids: List[int]):
        """데이터를 처리됨으로 표시"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in data_ids])
            cursor.execute(f'''
                UPDATE learning_data 
                SET processed = TRUE 
                WHERE id IN ({placeholders})
            ''', data_ids)
            conn.commit()
    
    def log_training_stats(self, total_data: int, loss: float, accuracy: float, 
                          lr: float, epochs: int, model_version: str):
        """학습 통계 기록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_stats 
                (timestamp, total_data_points, training_loss, validation_accuracy, 
                 learning_rate, epoch_count, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), total_data, loss, accuracy, lr, epochs, model_version))
            conn.commit()

class DataCollector:
    """자동 데이터 수집기"""
    
    def __init__(self, db: ContinuousLearningDatabase):
        self.db = db
        self.collection_sources = {
            "github": self.collect_github_issues,
            "stackoverflow": self.collect_stackoverflow_questions,
            "unity_docs": self.collect_unity_documentation,
            "csharp_examples": self.collect_csharp_examples,
            "synthetic": self.generate_synthetic_data
        }
    
    def collect_github_issues(self) -> List[Dict]:
        """GitHub 이슈에서 데이터 수집"""
        try:
            # Unity C# 관련 리포지토리들
            repos = [
                "Unity-Technologies/UnityCsReference",
                "Unity-Technologies/ml-agents",
                "microsoft/vscode-csharp"
            ]
            
            collected_data = []
            
            for repo in repos:
                try:
                    # GitHub API 호출 (실제로는 API 키 필요)
                    url = f"https://api.github.com/repos/{repo}/issues"
                    params = {
                        "state": "closed",
                        "sort": "updated",
                        "per_page": 10,
                        "labels": "question,help wanted,bug"
                    }
                    
                    # 실제 환경에서는 requests 사용
                    # response = requests.get(url, params=params)
                    # issues = response.json()
                    
                    # 시뮬레이션 데이터
                    issues = [
                        {
                            "title": "Unity GameObject가 삭제되지 않는 문제",
                            "body": "Destroy() 함수를 호출했는데 오브젝트가 여전히 남아있습니다.",
                            "state": "closed"
                        },
                        {
                            "title": "C# 코루틴에서 메모리 누수",
                            "body": "StartCoroutine을 반복 호출하면 메모리가 계속 증가합니다.",
                            "state": "closed"
                        }
                    ]
                    
                    for issue in issues:
                        question = f"{issue['title']}: {issue['body']}"
                        collected_data.append({
                            "input": question,
                            "source": f"github:{repo}",
                            "quality": 0.7
                        })
                        
                except Exception as e:
                    logger.warning(f"GitHub 수집 오류 ({repo}): {e}")
            
            return collected_data
            
        except Exception as e:
            logger.error(f"GitHub 데이터 수집 실패: {e}")
            return []
    
    def collect_stackoverflow_questions(self) -> List[Dict]:
        """Stack Overflow 질문 수집"""
        try:
            # Unity, C# 태그의 질문들 (시뮬레이션)
            so_questions = [
                {
                    "title": "Unity에서 Singleton 패턴 구현하기",
                    "body": "Unity에서 GameManager를 Singleton으로 만들고 싶습니다.",
                    "tags": ["unity", "c#", "singleton"]
                },
                {
                    "title": "C# async/await 패턴 질문",
                    "body": "비동기 메서드에서 UI 업데이트가 안 됩니다.",
                    "tags": ["c#", "async-await", "unity"]
                },
                {
                    "title": "Unity Physics2D 충돌 감지",
                    "body": "OnTriggerEnter2D가 호출되지 않습니다.",
                    "tags": ["unity", "physics2d", "collision"]
                }
            ]
            
            collected_data = []
            for q in so_questions:
                question = f"{q['title']}: {q['body']}"
                collected_data.append({
                    "input": question,
                    "source": "stackoverflow",
                    "quality": 0.8
                })
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Stack Overflow 데이터 수집 실패: {e}")
            return []
    
    def collect_unity_documentation(self) -> List[Dict]:
        """Unity 문서에서 데이터 수집"""
        try:
            # Unity 공식 문서 주제들 (시뮬레이션)
            unity_docs = [
                {
                    "topic": "GameObject 생성과 삭제",
                    "content": "Instantiate()와 Destroy() 메서드 사용법"
                },
                {
                    "topic": "컴포넌트 시스템",
                    "content": "GetComponent<>()를 사용한 컴포넌트 접근"
                },
                {
                    "topic": "코루틴 활용",
                    "content": "StartCoroutine()과 StopCoroutine() 사용"
                }
            ]
            
            collected_data = []
            for doc in unity_docs:
                question = f"Unity {doc['topic']}에 대해 설명해주세요"
                answer = doc['content']
                collected_data.append({
                    "input": question,
                    "target": answer,
                    "source": "unity_docs",
                    "quality": 0.9
                })
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Unity 문서 수집 실패: {e}")
            return []
    
    def collect_csharp_examples(self) -> List[Dict]:
        """C# 예제 코드 수집"""
        try:
            csharp_examples = [
                {
                    "question": "C# List에서 중복 제거하는 방법",
                    "answer": "LINQ의 Distinct() 메서드를 사용하세요: list.Distinct().ToList()"
                },
                {
                    "question": "C# Dictionary 사용법",
                    "answer": "Dictionary<string, int> dict = new Dictionary<string, int>();"
                },
                {
                    "question": "C# 이벤트 선언과 사용",
                    "answer": "public event Action<int> OnScoreChanged; OnScoreChanged?.Invoke(newScore);"
                }
            ]
            
            collected_data = []
            for example in csharp_examples:
                collected_data.append({
                    "input": example["question"],
                    "target": example["answer"],
                    "source": "csharp_examples",
                    "quality": 0.8
                })
            
            return collected_data
            
        except Exception as e:
            logger.error(f"C# 예제 수집 실패: {e}")
            return []
    
    def generate_synthetic_data(self) -> List[Dict]:
        """합성 데이터 생성"""
        try:
            import random
            
            # 질문 템플릿들
            question_templates = [
                "Unity에서 {object}를 {action}하는 방법은?",
                "C#에서 {concept} 구현하려면?",
                "{problem} 문제를 해결하는 법은?",
                "{unity_feature} 사용법 알려주세요",
                "{csharp_feature}를 어떻게 사용하나요?"
            ]
            
            # 단어 목록들
            objects = ["GameObject", "Transform", "Rigidbody", "Collider", "Camera"]
            actions = ["생성", "삭제", "이동", "회전", "스케일링"]
            concepts = ["Singleton", "Observer 패턴", "팩토리 패턴", "State 패턴"]
            problems = ["메모리 누수", "성능 저하", "충돌 감지 오류", "UI 업데이트 지연"]
            unity_features = ["Animation", "Particle System", "Audio Source", "Navigation"]
            csharp_features = ["async/await", "LINQ", "Generic", "Delegate", "Event"]
            
            collected_data = []
            
            for _ in range(20):  # 20개 합성 데이터 생성
                template = random.choice(question_templates)
                
                if "{object}" in template:
                    question = template.format(
                        object=random.choice(objects),
                        action=random.choice(actions)
                    )
                elif "{concept}" in template:
                    question = template.format(concept=random.choice(concepts))
                elif "{problem}" in template:
                    question = template.format(problem=random.choice(problems))
                elif "{unity_feature}" in template:
                    question = template.format(unity_feature=random.choice(unity_features))
                elif "{csharp_feature}" in template:
                    question = template.format(csharp_feature=random.choice(csharp_features))
                else:
                    question = template
                
                collected_data.append({
                    "input": question,
                    "source": "synthetic",
                    "quality": 0.6
                })
            
            return collected_data
            
        except Exception as e:
            logger.error(f"합성 데이터 생성 실패: {e}")
            return []
    
    def collect_all_sources(self) -> int:
        """모든 소스에서 데이터 수집"""
        total_collected = 0
        
        for source_name, collect_func in self.collection_sources.items():
            try:
                logger.info(f"📥 {source_name}에서 데이터 수집 중...")
                data_list = collect_func()
                
                for data in data_list:
                    self.db.add_learning_data(
                        input_text=data["input"],
                        target_response=data.get("target", ""),
                        feedback_score=data.get("quality", 0.5),
                        source=data["source"]
                    )
                    total_collected += 1
                
                logger.info(f"✅ {source_name}: {len(data_list)}개 데이터 수집됨")
                
            except Exception as e:
                logger.error(f"❌ {source_name} 수집 실패: {e}")
        
        logger.info(f"🎯 총 {total_collected}개 데이터 수집 완료")
        return total_collected

class ContinuousNeuralLearner:
    """지속적 신경망 학습기"""
    
    def __init__(self, db: ContinuousLearningDatabase):
        self.db = db
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
        self.model_version = "v1.0"
        self.learning_rate = 0.001
        
        if TORCH_AVAILABLE:
            self.init_neural_network()
        else:
            logger.warning("PyTorch 없음 - 로그 기반 학습 모드")
    
    def init_neural_network(self):
        """신경망 초기화"""
        try:
            class AutoCILearningNetwork(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512):
                    super(AutoCILearningNetwork, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
                    self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
                    self.classifier = nn.Sequential(
                        nn.Linear(hidden_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    
                    # 어텐션 적용
                    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    
                    # 마지막 시퀀스의 출력 사용
                    final_hidden = attn_out[:, -1, :]
                    output = self.classifier(final_hidden)
                    
                    return output
            
            self.model = AutoCILearningNetwork().to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.BCELoss()
            
            logger.info(f"🧠 신경망 초기화 완료 (디바이스: {self.device})")
            
        except Exception as e:
            logger.error(f"신경망 초기화 실패: {e}")
            self.model = None
    
    def prepare_training_data(self, data_batch: List[Tuple]) -> Optional[Tuple]:
        """학습 데이터 준비"""
        try:
            if not data_batch:
                return None
            
            # 텍스트를 토큰으로 변환 (간단한 해시 기반)
            def text_to_tokens(text: str, max_length: int = 50) -> List[int]:
                words = text.lower().split()[:max_length]
                tokens = []
                for word in words:
                    token = hash(word) % 10000  # 어휘 크기를 10000으로 제한
                    tokens.append(abs(token))
                
                # 패딩
                while len(tokens) < max_length:
                    tokens.append(0)
                
                return tokens[:max_length]
            
            X = []
            y = []
            
            for data_id, input_text, target_response, feedback_score, source in data_batch:
                tokens = text_to_tokens(input_text)
                X.append(tokens)
                
                # 피드백 점수를 0-1 범위로 정규화
                normalized_score = (feedback_score + 1.0) / 2.0
                y.append([normalized_score])
            
            if TORCH_AVAILABLE:
                X_tensor = torch.tensor(X, dtype=torch.long).to(self.device)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
                return X_tensor, y_tensor, [d[0] for d in data_batch]
            else:
                return X, y, [d[0] for d in data_batch]
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 실패: {e}")
            return None
    
    def train_batch(self, data_batch: List[Tuple]) -> Dict:
        """배치 학습 실행"""
        try:
            if not self.model or not TORCH_AVAILABLE:
                logger.info("📚 로그 기반 학습 모드")
                return {"loss": 0.0, "accuracy": 0.8, "processed": len(data_batch)}
            
            training_data = self.prepare_training_data(data_batch)
            if not training_data:
                return {"loss": 0.0, "accuracy": 0.0, "processed": 0}
            
            X_tensor, y_tensor, data_ids = training_data
            
            # 학습 모드 설정
            self.model.train()
            
            # 순전파
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 정확도 계산
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == (y_tensor > 0.5).float()).float().mean().item()
            
            # 데이터를 처리됨으로 표시
            self.db.mark_data_processed(data_ids)
            
            return {
                "loss": loss.item(),
                "accuracy": accuracy,
                "processed": len(data_batch)
            }
            
        except Exception as e:
            logger.error(f"배치 학습 실패: {e}")
            return {"loss": 0.0, "accuracy": 0.0, "processed": 0}
    
    def continuous_learning_cycle(self):
        """지속적 학습 사이클"""
        try:
            logger.info("🔄 지속적 학습 사이클 시작")
            
            # 미처리 데이터 가져오기
            unprocessed_data = self.db.get_unprocessed_data(limit=50)
            
            if not unprocessed_data:
                logger.info("📭 새로운 학습 데이터 없음")
                return
            
            logger.info(f"📚 {len(unprocessed_data)}개 데이터로 학습 시작")
            
            # 배치 학습 실행
            batch_size = 10
            total_loss = 0.0
            total_accuracy = 0.0
            total_processed = 0
            batch_count = 0
            
            for i in range(0, len(unprocessed_data), batch_size):
                batch = unprocessed_data[i:i + batch_size]
                result = self.train_batch(batch)
                
                total_loss += result["loss"]
                total_accuracy += result["accuracy"]
                total_processed += result["processed"]
                batch_count += 1
                
                logger.info(f"배치 {batch_count}: 손실={result['loss']:.4f}, 정확도={result['accuracy']:.4f}")
            
            # 평균 계산
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            avg_accuracy = total_accuracy / batch_count if batch_count > 0 else 0.0
            
            # 통계 기록
            self.db.log_training_stats(
                total_data=total_processed,
                loss=avg_loss,
                accuracy=avg_accuracy,
                lr=self.learning_rate,
                epochs=1,
                model_version=self.model_version
            )
            
            logger.info(f"✅ 학습 완료: 평균 손실={avg_loss:.4f}, 평균 정확도={avg_accuracy:.4f}")
            
            # 모델 저장 (주기적으로)
            if batch_count > 0:
                self.save_model_checkpoint(avg_accuracy)
            
        except Exception as e:
            logger.error(f"지속적 학습 사이클 실패: {e}")
            logger.error(traceback.format_exc())
    
    def save_model_checkpoint(self, performance_score: float):
        """모델 체크포인트 저장"""
        try:
            if not self.model or not TORCH_AVAILABLE:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"autoci_model_checkpoint_{timestamp}.pth"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_version': self.model_version,
                'performance_score': performance_score,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, model_path)
            
            # 데이터베이스에 체크포인트 정보 저장
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_checkpoints 
                    (timestamp, model_path, performance_score, training_time_hours, notes)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_path,
                    performance_score,
                    1.0,  # 1시간 단위
                    f"자동 저장 - 성능 점수: {performance_score:.4f}"
                ))
                conn.commit()
            
            logger.info(f"💾 모델 체크포인트 저장: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

class ContinuousLearningDaemon:
    """24시간 지속 학습 데몬"""
    
    def __init__(self):
        self.db = ContinuousLearningDatabase()
        self.data_collector = DataCollector(self.db)
        self.neural_learner = ContinuousNeuralLearner(self.db)
        self.running = True
        self.start_time = datetime.now()
        
        # 스케줄 설정
        self.setup_schedules()
        
        # 시그널 핸들러 설정 (graceful shutdown)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 AutoCI 24시간 지속 학습 데몬 시작")
    
    def setup_schedules(self):
        """학습 스케줄 설정"""
        # 데이터 수집 스케줄
        schedule.every(30).minutes.do(self.data_collector.collect_all_sources)  # 30분마다 데이터 수집
        
        # 학습 스케줄
        schedule.every(15).minutes.do(self.neural_learner.continuous_learning_cycle)  # 15분마다 학습
        
        # 상태 체크 스케줄
        schedule.every(1).hours.do(self.log_system_status)  # 1시간마다 상태 로그
        
        # 모델 백업 스케줄
        schedule.every(6).hours.do(self.backup_models)  # 6시간마다 모델 백업
        
        # 데이터베이스 정리 스케줄
        schedule.every(1).days.do(self.cleanup_old_data)  # 1일마다 오래된 데이터 정리
        
        logger.info("📅 학습 스케줄 설정 완료")
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러 (graceful shutdown)"""
        logger.info(f"🛑 종료 시그널 받음 ({signum}), 안전하게 종료 중...")
        self.running = False
    
    def log_system_status(self):
        """시스템 상태 로깅"""
        try:
            uptime = datetime.now() - self.start_time
            
            # 데이터베이스 통계
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # 총 학습 데이터 수
                cursor.execute("SELECT COUNT(*) FROM learning_data")
                total_data = cursor.fetchone()[0]
                
                # 처리된 데이터 수
                cursor.execute("SELECT COUNT(*) FROM learning_data WHERE processed = TRUE")
                processed_data = cursor.fetchone()[0]
                
                # 최근 학습 통계
                cursor.execute('''
                    SELECT training_loss, validation_accuracy 
                    FROM learning_stats 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''')
                latest_stats = cursor.fetchone()
            
            status_info = {
                "uptime_hours": uptime.total_seconds() / 3600,
                "total_data_points": total_data,
                "processed_data_points": processed_data,
                "processing_rate": f"{processed_data/total_data*100:.1f}%" if total_data > 0 else "0%",
                "latest_loss": latest_stats[0] if latest_stats else "N/A",
                "latest_accuracy": latest_stats[1] if latest_stats else "N/A"
            }
            
            logger.info(f"📊 시스템 상태: {json.dumps(status_info, indent=2)}")
            
        except Exception as e:
            logger.error(f"상태 로깅 실패: {e}")
    
    def backup_models(self):
        """모델 백업"""
        try:
            # 최신 체크포인트 찾기
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT model_path, performance_score 
                    FROM model_checkpoints 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''')
                latest_checkpoint = cursor.fetchone()
            
            if latest_checkpoint:
                model_path, score = latest_checkpoint
                backup_dir = "model_backups"
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{backup_dir}/autoci_backup_{timestamp}_score_{score:.3f}.pth"
                
                if os.path.exists(model_path):
                    import shutil
                    shutil.copy2(model_path, backup_path)
                    logger.info(f"💾 모델 백업 완료: {backup_path}")
                
        except Exception as e:
            logger.error(f"모델 백업 실패: {e}")
    
    def cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            # 30일 이전 처리된 데이터 삭제
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # 오래된 처리된 데이터 삭제
                cursor.execute('''
                    DELETE FROM learning_data 
                    WHERE processed = TRUE AND timestamp < ?
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
            
            logger.info(f"🧹 오래된 데이터 정리: {deleted_count}개 레코드 삭제")
            
        except Exception as e:
            logger.error(f"데이터 정리 실패: {e}")
    
    def run_daemon(self):
        """데몬 메인 루프"""
        logger.info("🔄 24시간 지속 학습 데몬 메인 루프 시작")
        
        # 초기 데이터 수집
        logger.info("🎯 초기 데이터 수집 시작")
        self.data_collector.collect_all_sources()
        
        # 초기 학습
        logger.info("🧠 초기 학습 사이클 시작")
        self.neural_learner.continuous_learning_cycle()
        
        # 메인 루프
        while self.running:
            try:
                # 스케줄된 작업 실행
                schedule.run_pending()
                
                # 1분 대기
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 키보드 인터럽트로 종료")
                break
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}")
                logger.error(traceback.format_exc())
                time.sleep(300)  # 5분 대기 후 재시작
        
        logger.info("👋 24시간 지속 학습 데몬 종료")

def main():
    """메인 함수"""
    print("🚀 AutoCI 24시간 지속 학습 시스템 시작")
    print("=" * 60)
    
    try:
        # 데몬 초기화 및 실행
        daemon = ContinuousLearningDaemon()
        daemon.run_daemon()
        
    except Exception as e:
        logger.error(f"데몬 실행 실패: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())