#!/usr/bin/env python3
"""
AutoCI PyTorch 기반 딥러닝 모듈
AutoCI의 학습을 강화하고 지속적인 개선을 위한 신경망 기반 학습 시스템
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import pickle
from collections import deque
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoCIKnowledgeDataset(Dataset):
    """AutoCI 학습 데이터셋"""
    def __init__(self, data_path: str, max_seq_length: int = 512):
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.data = []
        self.load_data()
        
    def load_data(self):
        """학습 데이터 로드"""
        # continuous_learning/answers 디렉토리에서 데이터 로드
        answers_dir = self.data_path / "continuous_learning" / "answers"
        if answers_dir.exists():
            for date_dir in answers_dir.iterdir():
                if date_dir.is_dir():
                    for json_file in date_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                self.data.append({
                                    'question': data.get('question', ''),
                                    'answer': data.get('answer', ''),
                                    'topic': data.get('topic', ''),
                                    'quality_score': data.get('quality_score', 0.5)
                                })
                        except Exception as e:
                            logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'answer': item['answer'],
            'topic': item['topic'],
            'quality_score': torch.tensor(item['quality_score'], dtype=torch.float32)
        }

class AutoCIDeepLearningNetwork(nn.Module):
    """AutoCI 심층 학습 네트워크"""
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.3):
        super(AutoCIDeepLearningNetwork, self).__init__()
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM 레이어 (양방향)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # 지식 추출 레이어
        self.knowledge_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 품질 평가 레이어
        self.quality_assessor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 주제 분류 레이어
        self.topic_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5가지 핵심 주제
        )
        
    def forward(self, input_ids, attention_mask=None):
        # 임베딩
        embedded = self.embedding(input_ids)
        
        # LSTM 처리
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        attn_out = attn_out.transpose(0, 1)
        
        # 평균 풀링
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(attn_out.size()).float()
            sum_embeddings = torch.sum(attn_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(attn_out, dim=1)
        
        # 지식 추출
        knowledge_features = self.knowledge_extractor(pooled)
        
        # 품질 평가
        quality_score = self.quality_assessor(knowledge_features)
        
        # 주제 분류
        topic_logits = self.topic_classifier(knowledge_features)
        
        return {
            'knowledge_features': knowledge_features,
            'quality_score': quality_score,
            'topic_logits': topic_logits
        }

class ReinforcementLearningAgent:
    """강화학습 에이전트 - AutoCI의 지속적인 개선을 위한 RL 에이전트"""
    def __init__(self, state_dim: int = 128, action_dim: int = 10, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # DQN 네트워크
        self.q_network = self._build_dqn()
        self.target_network = self._build_dqn()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 경험 리플레이 버퍼
        self.memory = deque(maxlen=10000)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def _build_dqn(self) -> nn.Module:
        """DQN 네트워크 구축"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """행동 선택 (ε-greedy)"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size: int = 32):
        """경험 리플레이를 통한 학습"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AutoCIPyTorchLearningSystem:
    """AutoCI PyTorch 기반 통합 학습 시스템"""
    def __init__(self, base_path: str = "./"):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models" / "pytorch_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 모델 초기화
        self.deep_learning_network = AutoCIDeepLearningNetwork().to(self.device)
        self.rl_agent = ReinforcementLearningAgent()
        
        # 옵티마이저
        self.dl_optimizer = optim.Adam(self.deep_learning_network.parameters(), lr=0.001)
        
        # 손실 함수
        self.quality_criterion = nn.MSELoss()
        self.topic_criterion = nn.CrossEntropyLoss()
        
        # 학습 통계
        self.training_stats = {
            'total_epochs': 0,
            'total_samples': 0,
            'average_quality_score': 0.0,
            'topic_accuracy': 0.0,
            'learning_history': []
        }
        
    def train_on_experience(self, experiences: List[Dict[str, Any]], epochs: int = 10):
        """경험 데이터로 학습"""
        logger.info(f"Starting training on {len(experiences)} experiences for {epochs} epochs")
        
        # 데이터 준비
        dataset = self._prepare_dataset(experiences)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.deep_learning_network.train()
        
        for epoch in range(epochs):
            total_loss = 0
            quality_losses = []
            topic_losses = []
            
            for batch in dataloader:
                # 입력 데이터 준비
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                quality_targets = batch['quality_score'].to(self.device)
                topic_targets = batch['topic_label'].to(self.device)
                
                # 순전파
                outputs = self.deep_learning_network(input_ids, attention_mask)
                
                # 손실 계산
                quality_loss = self.quality_criterion(
                    outputs['quality_score'].squeeze(), 
                    quality_targets
                )
                topic_loss = self.topic_criterion(
                    outputs['topic_logits'], 
                    topic_targets
                )
                
                total_loss = quality_loss + topic_loss
                
                # 역전파
                self.dl_optimizer.zero_grad()
                total_loss.backward()
                self.dl_optimizer.step()
                
                quality_losses.append(quality_loss.item())
                topic_losses.append(topic_loss.item())
            
            # 에폭 통계
            avg_quality_loss = np.mean(quality_losses)
            avg_topic_loss = np.mean(topic_losses)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Quality Loss: {avg_quality_loss:.4f}, "
                       f"Topic Loss: {avg_topic_loss:.4f}")
            
            # 통계 업데이트
            self.training_stats['total_epochs'] += 1
            self.training_stats['learning_history'].append({
                'epoch': self.training_stats['total_epochs'],
                'quality_loss': avg_quality_loss,
                'topic_loss': avg_topic_loss,
                'timestamp': datetime.now().isoformat()
            })
        
        # 모델 저장
        self.save_models()
        
    def reinforcement_learning_step(self, state: Dict[str, Any], 
                                  action: str, 
                                  reward: float, 
                                  next_state: Dict[str, Any]):
        """강화학습 단계 실행"""
        # 상태를 벡터로 변환
        state_vector = self._state_to_vector(state)
        next_state_vector = self._state_to_vector(next_state)
        
        # 행동을 인덱스로 변환
        action_idx = self._action_to_index(action)
        
        # 경험 저장
        done = reward > 0.9  # 높은 보상을 받으면 에피소드 종료
        self.rl_agent.remember(state_vector, action_idx, reward, next_state_vector, done)
        
        # 학습
        if len(self.rl_agent.memory) > 32:
            self.rl_agent.replay()
    
    def get_knowledge_embedding(self, text: str) -> np.ndarray:
        """텍스트에서 지식 임베딩 추출"""
        self.deep_learning_network.eval()
        
        with torch.no_grad():
            # 간단한 토크나이징 (실제로는 더 정교한 토크나이저 필요)
            tokens = self._simple_tokenize(text)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            outputs = self.deep_learning_network(input_ids)
            knowledge_features = outputs['knowledge_features'].cpu().numpy()
            
        return knowledge_features[0]
    
    def assess_quality(self, text: str) -> float:
        """텍스트 품질 평가"""
        self.deep_learning_network.eval()
        
        with torch.no_grad():
            tokens = self._simple_tokenize(text)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            outputs = self.deep_learning_network(input_ids)
            quality_score = outputs['quality_score'].item()
            
        return quality_score
    
    def classify_topic(self, text: str) -> str:
        """주제 분류"""
        self.deep_learning_network.eval()
        
        topic_names = ["C# 프로그래밍", "한글 용어", "Godot 엔진", "네트워킹", "AI 최적화"]
        
        with torch.no_grad():
            tokens = self._simple_tokenize(text)
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            outputs = self.deep_learning_network(input_ids)
            topic_idx = torch.argmax(outputs['topic_logits'], dim=1).item()
            
        return topic_names[topic_idx]
    
    def save_models(self):
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 딥러닝 모델 저장
        dl_path = self.models_dir / f"deep_learning_model_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.deep_learning_network.state_dict(),
            'optimizer_state_dict': self.dl_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, dl_path)
        
        # RL 모델 저장
        rl_path = self.models_dir / f"rl_agent_{timestamp}.pth"
        torch.save({
            'q_network_state_dict': self.rl_agent.q_network.state_dict(),
            'target_network_state_dict': self.rl_agent.target_network.state_dict(),
            'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
            'epsilon': self.rl_agent.epsilon
        }, rl_path)
        
        logger.info(f"Models saved to {self.models_dir}")
    
    def load_models(self, dl_path: str = None, rl_path: str = None):
        """모델 로드"""
        if dl_path and Path(dl_path).exists():
            checkpoint = torch.load(dl_path, map_location=self.device)
            self.deep_learning_network.load_state_dict(checkpoint['model_state_dict'])
            self.dl_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            logger.info(f"Deep learning model loaded from {dl_path}")
        
        if rl_path and Path(rl_path).exists():
            checkpoint = torch.load(rl_path, map_location=self.device)
            self.rl_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.rl_agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.rl_agent.epsilon = checkpoint.get('epsilon', self.rl_agent.epsilon)
            logger.info(f"RL agent loaded from {rl_path}")
    
    def _prepare_dataset(self, experiences: List[Dict[str, Any]]) -> Dataset:
        """경험 데이터를 데이터셋으로 변환"""
        # 간단한 구현 - 실제로는 더 정교한 처리 필요
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                # 간단한 토크나이징
                tokens = self._tokenize(item['question'] + " " + item['answer'])
                
                # 패딩 처리 - 모든 시퀀스를 동일한 길이로 만들기
                max_length = 512
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    attention_mask = [1] * max_length
                else:
                    padding_length = max_length - len(tokens)
                    attention_mask = [1] * len(tokens) + [0] * padding_length
                    tokens = tokens + [0] * padding_length
                
                return {
                    'input_ids': torch.tensor(tokens, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'quality_score': torch.tensor(item.get('quality_score', 0.5), dtype=torch.float),
                    'topic_label': torch.tensor(self._get_topic_label(item.get('topic', '')), dtype=torch.long)
                }
            
            def _tokenize(self, text):
                # 매우 간단한 토크나이징 (실제로는 BPE 등 사용)
                return [ord(c) % 50000 for c in text]
            
            def _get_topic_label(self, topic):
                topic_map = {
                    "C#": 0, "한글": 1, "Godot": 2, "네트워킹": 3, "AI": 4
                }
                for key, val in topic_map.items():
                    if key in topic:
                        return val
                return 0
        
        return SimpleDataset(experiences)
    
    def _simple_tokenize(self, text: str, max_length: int = 512) -> List[int]:
        """간단한 토크나이징"""
        tokens = [ord(c) % 50000 for c in text[:max_length]]
        # 패딩
        while len(tokens) < max_length:
            tokens.append(0)
        return tokens
    
    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """상태를 벡터로 변환"""
        # 간단한 구현
        vector = np.zeros(128)
        
        # 상태 정보 인코딩
        if 'quality_score' in state:
            vector[0] = state['quality_score']
        if 'topic' in state:
            vector[1:6] = self._encode_topic(state['topic'])
        if 'complexity' in state:
            vector[6] = state['complexity']
            
        return vector
    
    def _action_to_index(self, action: str) -> int:
        """행동을 인덱스로 변환"""
        actions = [
            "generate_question", "improve_answer", "add_example",
            "simplify", "add_detail", "fix_error", "optimize",
            "translate", "integrate", "review"
        ]
        return actions.index(action) if action in actions else 0
    
    def _encode_topic(self, topic: str) -> np.ndarray:
        """주제를 원-핫 벡터로 인코딩"""
        topics = ["C# 프로그래밍", "한글 용어", "Godot 엔진", "네트워킹", "AI 최적화"]
        vector = np.zeros(5)
        
        for i, t in enumerate(topics):
            if t in topic:
                vector[i] = 1.0
                break
                
        return vector


class PyTorchGameAI(nn.Module):
    """게임 개발 의사결정을 위한 PyTorch 기반 AI 모델"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 128):
        super(PyTorchGameAI, self).__init__()
        
        # 게임 개발 의사결정 네트워크
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 태스크별 헤드
        self.game_design_head = nn.Linear(hidden_dim, 64)  # 게임 디자인 결정
        self.code_generation_head = nn.Linear(hidden_dim, 64)  # 코드 생성 파라미터
        self.optimization_head = nn.Linear(hidden_dim, 32)  # 최적화 결정
        self.creativity_head = nn.Linear(hidden_dim, 32)  # 창의성 파라미터
        
        # 최종 출력 레이어
        self.output_layer = nn.Linear(192, output_dim)  # 64+64+32+32 = 192
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 인코딩
        encoded = self.encoder(x)
        
        # 태스크별 특징 추출
        game_design = self.game_design_head(encoded)
        code_gen = self.code_generation_head(encoded)
        optimization = self.optimization_head(encoded)
        creativity = self.creativity_head(encoded)
        
        # 특징 결합
        combined = torch.cat([game_design, code_gen, optimization, creativity], dim=-1)
        
        # 최종 출력
        output = self.output_layer(combined)
        
        return output
    
    def make_game_decision(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """게임 개발 의사결정"""
        # 상태를 텐서로 변환
        state_vector = self._game_state_to_vector(game_state)
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 모델 예측
        with torch.no_grad():
            output = self.forward(state_tensor)
            
        # 출력을 의사결정으로 변환
        decision = self._output_to_decision(output.squeeze().cpu().numpy())
        
        return decision
    
    def _game_state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """게임 상태를 벡터로 변환"""
        vector = np.zeros(256)
        
        # 게임 타입 인코딩
        game_types = ["platformer", "racing", "rpg", "puzzle", "shooter"]
        if "game_type" in state:
            for i, gt in enumerate(game_types):
                if gt == state["game_type"]:
                    vector[i] = 1.0
        
        # 진행 상태 인코딩
        if "progress" in state:
            vector[10] = state["progress"] / 100.0
        
        # 품질 메트릭 인코딩
        if "quality_score" in state:
            vector[11] = state["quality_score"] / 100.0
            
        # 기능 상태 인코딩
        if "features" in state:
            for i, feature in enumerate(state["features"][:20]):
                vector[20 + i] = 1.0 if feature.get("completed", False) else 0.0
                
        # 에러 카운트 인코딩
        if "error_count" in state:
            vector[40] = min(state["error_count"] / 10.0, 1.0)
            
        # 시간 정보 인코딩
        if "elapsed_hours" in state:
            vector[41] = min(state["elapsed_hours"] / 24.0, 1.0)
            
        return vector
    
    def _output_to_decision(self, output: np.ndarray) -> Dict[str, Any]:
        """모델 출력을 의사결정으로 변환"""
        decision = {
            "next_action": self._get_next_action(output[:32]),
            "priority_features": self._get_priority_features(output[32:64]),
            "optimization_level": float(output[64:68].mean()),
            "creativity_factor": float(output[68:72].mean()),
            "estimated_time": float(np.exp(output[72]) * 60),  # 분 단위
            "confidence": float(torch.sigmoid(torch.tensor(output[73]))),
            "suggested_improvements": self._get_improvements(output[74:])
        }
        
        return decision
    
    def _get_next_action(self, action_vector: np.ndarray) -> str:
        """다음 액션 결정"""
        actions = [
            "implement_core_mechanics",
            "add_visual_effects", 
            "optimize_performance",
            "implement_ui",
            "add_sound_effects",
            "create_levels",
            "implement_ai",
            "add_multiplayer",
            "polish_gameplay",
            "fix_bugs"
        ]
        
        # Softmax를 통한 확률 계산
        probs = torch.softmax(torch.tensor(action_vector[:len(actions)]), dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        
        return actions[action_idx]
    
    def _get_priority_features(self, feature_vector: np.ndarray) -> List[str]:
        """우선순위 기능 결정"""
        features = [
            "player_movement", "collision_detection", "scoring_system",
            "level_progression", "save_system", "menu_system",
            "particle_effects", "sound_manager", "ai_enemies",
            "power_ups", "achievements", "leaderboard"
        ]
        
        # 상위 3개 기능 선택
        feature_scores = feature_vector[:len(features)]
        top_indices = np.argsort(feature_scores)[-3:][::-1]
        
        return [features[i] for i in top_indices]
    
    def _get_improvements(self, improvement_vector: np.ndarray) -> List[str]:
        """개선 사항 제안"""
        improvements = []
        
        if improvement_vector[0] > 0.5:
            improvements.append("코드 리팩토링 필요")
        if improvement_vector[1] > 0.5:
            improvements.append("성능 최적화 권장")
        if improvement_vector[2] > 0.5:
            improvements.append("UI/UX 개선 필요")
        if improvement_vector[3] > 0.5:
            improvements.append("더 많은 테스트 필요")
            
        return improvements


# 사용 예시
if __name__ == "__main__":
    # 시스템 초기화
    pytorch_system = AutoCIPyTorchLearningSystem()
    
    # 샘플 경험 데이터
    sample_experiences = [
        {
            'question': 'C#에서 async/await를 사용하는 방법은?',
            'answer': 'async/await는 비동기 프로그래밍을 위한 키워드입니다...',
            'quality_score': 0.85,
            'topic': 'C# 프로그래밍'
        },
        {
            'question': 'Godot에서 노드를 생성하는 방법은?',
            'answer': 'Godot에서 노드는 add_child() 메서드를 사용하여...',
            'quality_score': 0.90,
            'topic': 'Godot 엔진'
        }
    ]
    
    # 학습 실행
    pytorch_system.train_on_experience(sample_experiences, epochs=5)
    
    # 품질 평가
    quality = pytorch_system.assess_quality("이것은 테스트 답변입니다.")
    print(f"품질 점수: {quality:.2f}")
    
    # 주제 분류
    topic = pytorch_system.classify_topic("Godot 엔진에서 시그널을 사용하는 방법")
    print(f"분류된 주제: {topic}")