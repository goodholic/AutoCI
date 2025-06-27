#!/usr/bin/env python3
"""
Neural GPT AutoCI - 완전한 신경망 기반 AutoCI
규칙 기반/패턴 매칭 완전 제거, 순수 신경망만 사용
ChatGPT 수준의 수십억 파라미터 트랜스포머 모델
"""

import os
import sys
import time
import json
import math
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pickle
import threading
import queue

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        GPT2LMHeadModel, GPT2Config,
        get_linear_schedule_with_warmup,
        DataCollatorForLanguageModeling
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch/Transformers 필수! 설치: pip install torch transformers")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """신경망 모델 설정"""
    vocab_size: int = 50000          # 어휘 크기
    hidden_size: int = 4096          # 은닉층 크기 (GPT-3: 12288)
    num_layers: int = 32             # 레이어 수 (GPT-3: 96)
    num_heads: int = 32              # 어텐션 헤드 수 (GPT-3: 96)
    intermediate_size: int = 16384   # FFN 중간층 크기 (GPT-3: 49152)
    max_position_embeddings: int = 2048  # 최대 시퀀스 길이
    dropout: float = 0.1             # 드롭아웃 비율
    layer_norm_epsilon: float = 1e-5
    
    # 학습 설정
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    max_steps: int = 1000000
    batch_size: int = 8              # GPU 메모리에 따라 조정
    gradient_accumulation_steps: int = 32
    
    @property
    def total_parameters(self) -> int:
        """총 파라미터 수 계산"""
        # 임베딩: vocab_size * hidden_size + max_pos * hidden_size
        embeddings = self.vocab_size * self.hidden_size + self.max_position_embeddings * self.hidden_size
        
        # 트랜스포머 블록: 각 블록당 약 12 * hidden_size^2
        # (attention: 4 * hidden_size^2, ffn: 8 * hidden_size^2)
        transformer_blocks = self.num_layers * 12 * self.hidden_size * self.hidden_size
        
        # 출력 레이어: hidden_size * vocab_size
        output_layer = self.hidden_size * self.vocab_size
        
        total = embeddings + transformer_blocks + output_layer
        return total

class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 (ChatGPT 스타일)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.head_dim * config.num_heads == config.hidden_size
        
        # Query, Key, Value 프로젝션
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # 인과적 마스크 (GPT 스타일)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
            .view(1, 1, config.max_position_embeddings, config.max_position_embeddings)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Q, K, V 계산
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 멀티헤드로 reshape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 어텐션 점수 계산
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # 인과적 마스크 적용 (GPT 스타일)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
        
        # 어텐션 가중치 계산
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 어텐션 적용
        context = torch.matmul(attention_weights, value)
        
        # 헤드 결합
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 출력 프로젝션
        output = self.out_proj(context)
        
        return output

class FeedForward(nn.Module):
    """피드포워드 네트워크 (ChatGPT 스타일)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU 활성화 함수 (GPT 스타일)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """트랜스포머 블록 (ChatGPT 스타일)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm (GPT 스타일)
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class NeuralGPTAutoCI(nn.Module):
    """완전한 신경망 기반 AutoCI (규칙 없음)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 트랜스포머 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # 최종 레이어 정규화
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # 출력 헤드 (다음 토큰 예측)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
        # 총 파라미터 수 출력
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"🧠 신경망 초기화: {total_params:,} 파라미터 ({total_params/1e9:.1f}B)")
    
    def _init_weights(self, module):
        """가중치 초기화 (GPT 스타일)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        # 위치 ID 생성
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 임베딩
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # 트랜스포머 블록들 통과
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states)
        
        # 최종 정규화
        hidden_states = self.final_layer_norm(hidden_states)
        
        # 다음 토큰 예측 로짓
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate(self, tokenizer, prompt: str, max_length: int = 256, 
                temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        """순수 신경망 기반 텍스트 생성 (규칙 없음)"""
        
        self.eval()
        device = next(self.parameters()).device
        
        # 프롬프트 토큰화
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # 모델 예측
                logits = self.forward(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k 필터링
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Top-p (nucleus) 샘플링
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # top_p 임계값 이후 토큰들 제거
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 다음 토큰 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 시퀀스에 추가
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # 종료 토큰 확인
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # 최대 길이 제한
                if input_ids.size(1) >= self.config.max_position_embeddings:
                    break
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = generated_text[len(prompt):].strip()
        
        return response

class LargeScaleDataset(Dataset):
    """대규모 학습 데이터셋"""
    
    def __init__(self, data_files: List[str], tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info("📚 대규모 데이터셋 로딩 중...")
        
        for data_file in data_files:
            self._load_data_file(data_file)
        
        logger.info(f"✅ 총 {len(self.examples):,} 개 학습 예제 로드됨")
    
    def _load_data_file(self, file_path: str):
        """데이터 파일 로드"""
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ 데이터 파일 없음: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num % 10000 == 0:
                        logger.info(f"📖 {file_path}: {line_num:,} 줄 처리됨")
                    
                    try:
                        data = json.loads(line.strip())
                        if self._is_valid_conversation(data):
                            text = self._format_conversation(data)
                            if text:
                                self.examples.append(text)
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        if line_num % 1000 == 0:  # 가끔만 로그
                            logger.warning(f"데이터 처리 오류 (줄 {line_num}): {e}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ 파일 로드 실패 {file_path}: {e}")
    
    def _is_valid_conversation(self, data: Dict) -> bool:
        """유효한 대화 데이터인지 확인"""
        return (
            isinstance(data, dict) and
            'user_message' in data and
            'ai_response' in data and
            isinstance(data['user_message'], str) and
            isinstance(data['ai_response'], str) and
            len(data['user_message'].strip()) > 0 and
            len(data['ai_response'].strip()) > 0 and
            len(data['user_message']) < 1000 and  # 너무 긴 텍스트 제외
            len(data['ai_response']) < 2000
        )
    
    def _format_conversation(self, data: Dict) -> Optional[str]:
        """대화를 학습용 텍스트로 포맷"""
        user_msg = data['user_message'].strip()
        ai_msg = data['ai_response'].strip()
        
        # ChatGPT 스타일 포맷
        formatted = f"<|user|>{user_msg}<|assistant|>{ai_msg}<|endoftext|>"
        
        return formatted
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        # 레이블은 input_ids와 동일 (언어 모델링)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class NeuralTrainer:
    """순수 신경망 학습기 (규칙 없음)"""
    
    def __init__(self, model: NeuralGPTAutoCI, config: ModelConfig, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델을 GPU로 이동
        self.model = self.model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # 학습 통계
        self.train_stats = {
            "step": 0,
            "epoch": 0,
            "total_loss": 0.0,
            "best_loss": float('inf'),
            "learning_rate": config.learning_rate
        }
        
        logger.info(f"🎯 신경망 학습기 초기화 (디바이스: {self.device})")
    
    def create_dataloader(self, dataset: LargeScaleDataset, batch_size: int = None) -> DataLoader:
        """데이터로더 생성"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """단일 학습 스텝"""
        self.model.train()
        
        # 배치를 GPU로 이동
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 순전파
        logits = self.model(input_ids, attention_mask)
        
        # 손실 계산 (다음 토큰 예측)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 옵티마이저 스텝
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """전체 에포크 학습"""
        total_loss = 0.0
        num_batches = len(dataloader)
        
        logger.info(f"🚀 에포크 {self.train_stats['epoch']} 학습 시작 ({num_batches:,} 배치)")
        
        for batch_idx, batch in enumerate(dataloader):
            # 학습 스텝
            loss = self.train_step(batch)
            total_loss += loss
            
            self.train_stats["step"] += 1
            self.train_stats["total_loss"] += loss
            
            # 로그 출력
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"배치 {batch_idx:,}/{num_batches:,}: 손실={loss:.4f}, 평균={avg_loss:.4f}")
            
            # 주기적 모델 저장
            if self.train_stats["step"] % 1000 == 0:
                self.save_checkpoint()
            
            # 최대 스텝 체크
            if self.train_stats["step"] >= self.config.max_steps:
                logger.info(f"✅ 최대 스텝 ({self.config.max_steps:,}) 도달")
                break
        
        avg_epoch_loss = total_loss / num_batches
        self.train_stats["epoch"] += 1
        
        # 최고 성능 업데이트
        if avg_epoch_loss < self.train_stats["best_loss"]:
            self.train_stats["best_loss"] = avg_epoch_loss
            self.save_best_model()
        
        logger.info(f"✅ 에포크 완료: 평균 손실={avg_epoch_loss:.4f}")
        
        return avg_epoch_loss
    
    def save_checkpoint(self, checkpoint_path: str = None):
        """체크포인트 저장"""
        if checkpoint_path is None:
            checkpoint_path = f"neural_autoci_checkpoint_step_{self.train_stats['step']}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "train_stats": self.train_stats,
            "step": self.train_stats["step"]
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def save_best_model(self):
        """최고 성능 모델 저장"""
        best_model_path = f"neural_autoci_best_model.pt"
        self.save_checkpoint(best_model_path)
        logger.info(f"🏆 최고 성능 모델 저장: {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"⚠️ 체크포인트 파일 없음: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_stats = checkpoint["train_stats"]
            
            logger.info(f"✅ 체크포인트 로드: {checkpoint_path}")
            logger.info(f"   스텝: {self.train_stats['step']:,}")
            logger.info(f"   최고 손실: {self.train_stats['best_loss']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 로드 실패: {e}")
            return False

class PureNeuralAutoCI:
    """완전한 순수 신경망 AutoCI (규칙 완전 제거)"""
    
    def __init__(self, model_path: str = None):
        self.config = ModelConfig()
        
        # 토크나이저 초기화
        self.tokenizer = self._create_tokenizer()
        
        # 신경망 모델 초기화
        self.model = NeuralGPTAutoCI(self.config)
        
        # 학습기 초기화
        self.trainer = NeuralTrainer(self.model, self.config, self.tokenizer)
        
        # 모델 로드 (있는 경우)
        if model_path and os.path.exists(model_path):
            self.trainer.load_checkpoint(model_path)
        
        # 대화 기록
        self.conversation_history = []
        
        logger.info("🤖 순수 신경망 AutoCI 초기화 완료")
        logger.info(f"📊 모델 크기: {self.config.total_parameters:,} 파라미터")
    
    def _create_tokenizer(self):
        """토크나이저 생성"""
        try:
            # GPT-2 토크나이저 사용 (한국어 지원)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # 특수 토큰 추가
            special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|unk|>",
                "additional_special_tokens": ["<|user|>", "<|assistant|>"]
            }
            
            tokenizer.add_special_tokens(special_tokens)
            
            # 어휘 크기 업데이트
            self.config.vocab_size = len(tokenizer)
            
            logger.info(f"✅ 토크나이저 초기화: 어휘 크기 {len(tokenizer):,}")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"❌ 토크나이저 초기화 실패: {e}")
            # 간단한 폴백 토크나이저
            return self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """간단한 폴백 토크나이저"""
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {"<|pad|>": 0, "<|unk|>": 1, "<|endoftext|>": 2, "<|user|>": 3, "<|assistant|>": 4}
                self.pad_token_id = 0
                self.eos_token_id = 2
                
            def encode(self, text, return_tensors=None):
                # 간단한 문자 기반 인코딩
                tokens = [ord(c) % 1000 + 5 for c in text[:100]]  # 최대 100자
                if return_tensors == "pt":
                    return torch.tensor([tokens])
                return tokens
            
            def decode(self, tokens, skip_special_tokens=False):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                # 간단한 디코딩
                try:
                    text = ''.join(chr(max(32, t-5)) for t in tokens if t >= 5)
                    return text[:200]  # 최대 200자
                except:
                    return "생성된 응답"
            
            def __call__(self, text, truncation=True, max_length=512, padding="max_length", return_tensors=None):
                tokens = self.encode(text)[:max_length]
                
                if padding == "max_length":
                    while len(tokens) < max_length:
                        tokens.append(self.pad_token_id)
                
                attention_mask = [1 if t != self.pad_token_id else 0 for t in tokens]
                
                result = {
                    "input_ids": torch.tensor([tokens]) if return_tensors == "pt" else tokens,
                    "attention_mask": torch.tensor([attention_mask]) if return_tensors == "pt" else attention_mask
                }
                
                return result
            
            def __len__(self):
                return 50000
        
        logger.info("⚠️ 폴백 토크나이저 사용")
        return SimpleTokenizer()
    
    def chat(self, user_input: str) -> str:
        """순수 신경망 기반 대화 (규칙 없음)"""
        
        # 대화 프롬프트 구성
        prompt = f"<|user|>{user_input}<|assistant|>"
        
        try:
            # 순수 신경망으로 응답 생성
            response = self.model.generate(
                self.tokenizer,
                prompt,
                max_length=200,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            # 대화 기록 저장
            self.conversation_history.append({
                "user_input": user_input,
                "ai_response": response,
                "timestamp": datetime.now().isoformat(),
                "generation_method": "pure_neural"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 신경망 응답 생성 실패: {e}")
            # 긴급 폴백 (매우 간단)
            return "신경망 처리 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def train_on_data(self, data_files: List[str], epochs: int = 1):
        """대규모 데이터로 신경망 학습"""
        logger.info(f"🚀 대규모 신경망 학습 시작: {epochs} 에포크")
        
        # 데이터셋 생성
        dataset = LargeScaleDataset(data_files, self.tokenizer)
        
        if len(dataset) == 0:
            logger.error("❌ 학습 데이터가 없습니다!")
            return
        
        # 데이터로더 생성
        dataloader = self.trainer.create_dataloader(dataset)
        
        # 학습 실행
        for epoch in range(epochs):
            logger.info(f"📚 에포크 {epoch+1}/{epochs} 시작")
            
            avg_loss = self.trainer.train_epoch(dataloader)
            
            logger.info(f"✅ 에포크 {epoch+1} 완료: 평균 손실 {avg_loss:.4f}")
            
            # 중간 저장
            if (epoch + 1) % 5 == 0:
                self.trainer.save_checkpoint(f"neural_autoci_epoch_{epoch+1}.pt")
        
        # 최종 모델 저장
        self.trainer.save_best_model()
        logger.info("🎉 신경망 학습 완료!")
    
    def interactive_chat(self):
        """대화형 인터페이스"""
        print("\n🤖 순수 신경망 AutoCI (규칙 없음)")
        print("=" * 60)
        print("완전한 신경망 기반 AI와 대화하세요!")
        print("종료: 'quit', 학습: 'train', 상태: 'status'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n💬 당신: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                elif user_input.lower() == 'status':
                    print(f"\n📊 모델 상태:")
                    print(f"  파라미터: {self.config.total_parameters:,}")
                    print(f"  레이어: {self.config.num_layers}")
                    print(f"  헤드: {self.config.num_heads}")
                    print(f"  은닉 크기: {self.config.hidden_size}")
                    print(f"  대화 기록: {len(self.conversation_history)}")
                    continue
                
                elif user_input.lower() == 'train':
                    print("🚀 신경망 학습을 시작하려면 데이터 파일이 필요합니다.")
                    print("   예: data_files = ['conversations.jsonl', 'korean_qa.jsonl']")
                    continue
                
                # 순수 신경망으로 응답 생성
                print("🧠 신경망이 응답을 생성 중...")
                response = self.chat(user_input)
                print(f"\n🤖 Neural AutoCI: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
        
        print(f"\n👋 순수 신경망 AutoCI를 이용해주셔서 감사합니다!")

def create_training_data():
    """학습용 데이터 생성 예시"""
    sample_data = [
        {
            "user_message": "Unity에서 GameObject를 생성하는 방법을 알려주세요.",
            "ai_response": "Unity에서 GameObject를 생성하는 방법은 여러 가지가 있습니다. 가장 일반적인 방법은 Instantiate 메서드를 사용하는 것입니다. GameObject prefab이 있다면 Instantiate(prefab)로 생성할 수 있고, 새로운 빈 GameObject를 만들려면 new GameObject(\"이름\")을 사용하면 됩니다."
        },
        {
            "user_message": "C#에서 리스트 사용법을 알려주세요.",
            "ai_response": "C#에서 List<T>는 동적 배열입니다. List<int> numbers = new List<int>();로 생성하고, Add()로 요소 추가, Remove()로 제거, Count로 개수 확인할 수 있습니다. foreach나 for 루프로 순회할 수 있습니다."
        }
    ]
    
    with open("sample_training_data.jsonl", "w", encoding="utf-8") as f:
        for data in sample_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    logger.info("📝 샘플 학습 데이터 생성됨: sample_training_data.jsonl")

def main():
    """메인 함수"""
    print("🚀 순수 신경망 AutoCI (규칙 완전 제거)")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch가 필요합니다!")
        print("설치: pip install torch transformers")
        return 1
    
    try:
        # 샘플 데이터 생성
        create_training_data()
        
        # 순수 신경망 AutoCI 초기화
        autoci = PureNeuralAutoCI()
        
        # 대화형 인터페이스 시작
        autoci.interactive_chat()
        
    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())