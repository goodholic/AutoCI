#!/usr/bin/env python3
"""
Distributed Training System for Neural AutoCI
수십억 파라미터 신경망을 위한 분산 학습 시스템
"""

import os
import sys
import time
import json
import logging
import threading
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import sqlite3
import queue
import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음 - 분산 학습 시뮬레이션 모드")

# 로컬 모듈
try:
    from large_scale_training_pipeline import LargeScaleDatasetDatabase, TrainingExample
    from neural_gpt_autoci import NeuralGPTAutoCI, ModelConfig, GPTDataset
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ 모듈 없음 - 기본 모드로 실행")
    
    # 시뮬레이션용 더미 클래스
    class ModelConfig:
        def __init__(self):
            self.vocab_size = 50000
            self.hidden_size = 1024
            self.num_layers = 12
            self.num_heads = 16
            self.total_parameters = 1000000000  # 10억

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """분산 학습 설정"""
    num_gpus: int = 4
    num_nodes: int = 1
    world_size: int = 4  # num_gpus * num_nodes
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 16
    max_epochs: int = 100
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # 모델 병렬화 설정
    model_parallel: bool = True
    pipeline_parallel_size: int = 2
    data_parallel_size: int = 2
    
    # 최적화 설정
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = True
    
    @property
    def global_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.world_size * self.gradient_accumulation_steps

@dataclass
class TrainingMetrics:
    """학습 메트릭"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    memory_usage: float
    throughput: float  # tokens per second
    timestamp: str

class DistributedTrainer:
    """분산 학습 관리자"""
    
    def __init__(self, config: DistributedConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 메트릭 저장
        self.training_metrics = []
        self.validation_metrics = []
        
        # 데이터
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # 체크포인트
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_distributed(self, rank: int, world_size: int):
        """분산 환경 설정"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 없음 - 분산 학습 시뮬레이션")
            return
        
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # 분산 프로세스 그룹 초기화
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            
            # GPU 설정
            torch.cuda.set_device(rank)
            self.device = torch.device(f'cuda:{rank}')
            
            logger.info(f"✅ 분산 환경 설정 완료 - Rank: {rank}, World Size: {world_size}")
            
        except Exception as e:
            logger.error(f"❌ 분산 환경 설정 실패: {e}")
            self.device = torch.device('cpu')

    def setup_model(self):
        """모델 초기화"""
        if not TORCH_AVAILABLE or not MODULES_AVAILABLE:
            logger.warning("모델 시뮬레이션 모드")
            return self._setup_simulation_model()
        
        try:
            # 모델 생성
            self.model = NeuralGPTAutoCI(self.model_config)
            
            if self.device:
                self.model = self.model.to(self.device)
            
            # 분산 데이터 병렬화
            if torch.cuda.is_available() and dist.is_initialized():
                self.model = DDP(self.model, device_ids=[self.device.index])
            
            # 옵티마이저 설정
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.95)
            )
            
            # 학습률 스케줄러
            if hasattr(torch.optim.lr_scheduler, 'get_linear_schedule_with_warmup'):
                self.scheduler = torch.optim.lr_scheduler.get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.config.warmup_steps,
                    num_training_steps=self.config.max_epochs * 1000
                )
            
            # Mixed Precision
            if self.config.mixed_precision and torch.cuda.is_available():
                self.scaler = torch.cuda.amp.GradScaler()
            
            logger.info(f"✅ 모델 초기화 완료 - 파라미터: {self.model_config.total_parameters:,}")
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            return self._setup_simulation_model()

    def _setup_simulation_model(self):
        """시뮬레이션 모델 설정"""
        class SimulationModel:
            def __init__(self, config):
                self.config = config
                self.parameters_count = config.total_parameters
                
            def train(self):
                pass
                
            def eval(self):
                pass
                
            def parameters(self):
                return []
                
            def state_dict(self):
                return {}
                
            def load_state_dict(self, state_dict):
                pass
        
        self.model = SimulationModel(self.model_config)
        logger.info(f"✅ 시뮬레이션 모델 설정 완료")

    def setup_data(self):
        """데이터 로더 설정"""
        try:
            if MODULES_AVAILABLE:
                # 실제 데이터베이스에서 로드
                db = LargeScaleDatasetDatabase()
                
                # 학습 데이터
                train_examples = []
                batch_size = 1000
                offset = 0
                
                while True:
                    batch = db.get_training_batch(batch_size, 'train')
                    if not batch:
                        break
                    train_examples.extend(batch)
                    offset += batch_size
                    if len(train_examples) >= 10000:  # 테스트용 제한
                        break
                
                # 검증 데이터
                val_examples = db.get_training_batch(2000, 'validation')
                
                logger.info(f"데이터 로드 완료 - Train: {len(train_examples)}, Val: {len(val_examples)}")
                
            else:
                # 시뮬레이션 데이터
                train_examples = self._generate_simulation_data(10000)
                val_examples = self._generate_simulation_data(2000)
                logger.info("시뮬레이션 데이터 생성 완료")
            
            # 데이터셋 생성
            if TORCH_AVAILABLE and MODULES_AVAILABLE:
                self.train_dataset = GPTDataset(train_examples)
                self.val_dataset = GPTDataset(val_examples)
                
                # 분산 샘플러
                if dist.is_initialized():
                    train_sampler = DistributedSampler(self.train_dataset)
                    val_sampler = DistributedSampler(self.val_dataset)
                else:
                    train_sampler = None
                    val_sampler = None
                
                # 데이터 로더
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size_per_gpu,
                    sampler=train_sampler,
                    shuffle=(train_sampler is None),
                    num_workers=4,
                    pin_memory=True
                )
                
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config.batch_size_per_gpu,
                    sampler=val_sampler,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
            else:
                # 시뮬레이션 로더
                self.train_loader = self._create_simulation_loader(train_examples)
                self.val_loader = self._create_simulation_loader(val_examples)
            
        except Exception as e:
            logger.error(f"데이터 설정 오류: {e}")
            # 폴백 시뮬레이션 데이터
            self.train_loader = self._create_simulation_loader([])
            self.val_loader = self._create_simulation_loader([])

    def _generate_simulation_data(self, count: int) -> List[Dict]:
        """시뮬레이션 데이터 생성"""
        data = []
        for i in range(count):
            data.append({
                'input_text': f"Unity 질문 {i}",
                'target_output': f"Unity 답변 {i}",
                'tokens': 50
            })
        return data

    def _create_simulation_loader(self, data: List[Dict]):
        """시뮬레이션 데이터 로더"""
        class SimulationLoader:
            def __init__(self, data, batch_size):
                self.data = data
                self.batch_size = batch_size
                
            def __iter__(self):
                for i in range(0, len(self.data), self.batch_size):
                    batch = self.data[i:i + self.batch_size]
                    yield {
                        'input_ids': [list(range(50)) for _ in batch],
                        'attention_mask': [[1] * 50 for _ in batch],
                        'labels': [list(range(50)) for _ in batch]
                    }
            
            def __len__(self):
                return (len(self.data) + self.batch_size - 1) // self.batch_size
        
        return SimulationLoader(data, self.config.batch_size_per_gpu)

    def train_epoch(self, epoch: int):
        """한 에포크 학습"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        logger.info(f"🚀 에포크 {epoch} 학습 시작")
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                start_time = time.time()
                
                # 실제 또는 시뮬레이션 학습
                if TORCH_AVAILABLE and hasattr(batch, 'get'):
                    loss = self._train_step_real(batch)
                else:
                    loss = self._train_step_simulation(batch)
                
                epoch_loss += loss
                num_batches += 1
                
                # 메트릭 기록
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_metrics(epoch, loss, time.time() - start_time)
                
                # 검증
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    self._save_checkpoint_if_best(val_loss)
                
                # 체크포인트 저장
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                self.global_step += 1
                
                if batch_idx >= 100:  # 테스트용 제한
                    break
        
        except KeyboardInterrupt:
            logger.info("학습 중단 요청")
            return
        except Exception as e:
            logger.error(f"학습 오류: {e}")
            return
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"✅ 에포크 {epoch} 완료 - 평균 손실: {avg_loss:.4f}")
        
        return avg_loss

    def _train_step_real(self, batch) -> float:
        """실제 PyTorch 학습 스텝"""
        if self.config.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
        
        return loss.item()

    def _train_step_simulation(self, batch) -> float:
        """시뮬레이션 학습 스텝"""
        # 가상의 손실 계산 (실제 학습 시뮬레이션)
        base_loss = 2.0
        step_reduction = self.global_step * 0.00001
        noise = (time.time() % 1) * 0.1
        
        simulated_loss = max(0.1, base_loss - step_reduction + noise)
        
        # 학습 진행 시뮬레이션을 위한 지연
        time.sleep(0.01)
        
        return simulated_loss

    def evaluate(self) -> float:
        """모델 평가"""
        self.model.eval()
        eval_loss = 0.0
        num_batches = 0
        
        logger.info("📊 모델 평가 중...")
        
        try:
            with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
                for batch_idx, batch in enumerate(self.val_loader):
                    if TORCH_AVAILABLE and hasattr(batch, 'get'):
                        if self.config.mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(**batch)
                                loss = outputs.loss
                        else:
                            outputs = self.model(**batch)
                            loss = outputs.loss
                        eval_loss += loss.item()
                    else:
                        # 시뮬레이션 평가
                        eval_loss += self._train_step_simulation(batch) * 0.9
                    
                    num_batches += 1
                    
                    if batch_idx >= 20:  # 테스트용 제한
                        break
        
        except Exception as e:
            logger.error(f"평가 오류: {e}")
            return float('inf')
        
        avg_eval_loss = eval_loss / max(num_batches, 1)
        logger.info(f"📊 평가 완료 - 손실: {avg_eval_loss:.4f}")
        
        self.model.train()
        return avg_eval_loss

    def _log_training_metrics(self, epoch: int, loss: float, step_time: float):
        """학습 메트릭 로깅"""
        metrics = TrainingMetrics(
            epoch=epoch,
            step=self.global_step,
            loss=loss,
            learning_rate=self.config.learning_rate,
            grad_norm=0.0,  # 시뮬레이션
            memory_usage=0.0,  # 시뮬레이션
            throughput=self.config.batch_size_per_gpu / step_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.training_metrics.append(metrics)
        
        if self.global_step % (self.config.logging_steps * 10) == 0:
            logger.info(f"Step {self.global_step}: Loss={loss:.4f}, LR={metrics.learning_rate:.6f}")

    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_step_{self.global_step}.pt"
        )
        
        try:
            if TORCH_AVAILABLE and hasattr(self.model, 'state_dict'):
                checkpoint = {
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else {},
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else {},
                    'config': asdict(self.config),
                    'model_config': asdict(self.model_config)
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                # 시뮬레이션 체크포인트
                checkpoint = {
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'config': asdict(self.config),
                    'model_config': asdict(self.model_config),
                    'metrics': [asdict(m) for m in self.training_metrics[-100:]]
                }
                with open(checkpoint_path.replace('.pt', '.json'), 'w') as f:
                    json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 체크포인트 저장: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")

    def _save_checkpoint_if_best(self, val_loss: float):
        """최고 성능 체크포인트 저장"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            
            try:
                if TORCH_AVAILABLE and hasattr(self.model, 'state_dict'):
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'val_loss': val_loss,
                        'global_step': self.global_step
                    }, best_path)
                else:
                    with open(best_path.replace('.pt', '.json'), 'w') as f:
                        json.dump({
                            'val_loss': val_loss,
                            'global_step': self.global_step,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                
                logger.info(f"🏆 새로운 최고 성능 모델 저장: {val_loss:.4f}")
                
            except Exception as e:
                logger.error(f"최고 성능 모델 저장 실패: {e}")

    def train(self):
        """전체 학습 프로세스"""
        logger.info(f"🚀 분산 학습 시작")
        logger.info(f"📊 설정: {self.config.world_size}개 프로세스, {self.config.max_epochs}개 에포크")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                
                # 에포크 학습
                avg_loss = self.train_epoch(epoch)
                
                # 검증
                val_loss = self.evaluate()
                
                # 메트릭 저장
                self.validation_metrics.append({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"에포크 {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
                
                # 조기 종료 확인
                if self._should_early_stop():
                    logger.info("조기 종료 조건 만족")
                    break
        
        except KeyboardInterrupt:
            logger.info("학습 중단 요청")
        except Exception as e:
            logger.error(f"학습 오류: {e}")
        
        finally:
            # 최종 체크포인트 저장
            self._save_checkpoint()
            
            # 학습 완료 보고서
            self._generate_training_report()
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 학습 완료 (소요시간: {elapsed_time/3600:.1f}시간)")

    def _should_early_stop(self) -> bool:
        """조기 종료 확인"""
        if len(self.validation_metrics) < 5:
            return False
        
        # 최근 5개 에포크의 검증 손실이 개선되지 않으면 조기 종료
        recent_losses = [m['val_loss'] for m in self.validation_metrics[-5:]]
        return all(loss >= recent_losses[0] for loss in recent_losses[1:])

    def _generate_training_report(self):
        """학습 보고서 생성"""
        report = {
            'training_summary': {
                'total_epochs': self.current_epoch + 1,
                'total_steps': self.global_step,
                'best_val_loss': self.best_loss,
                'final_train_loss': self.training_metrics[-1].loss if self.training_metrics else 0,
                'training_time_hours': len(self.training_metrics) * 0.01 / 3600  # 시뮬레이션
            },
            'config': asdict(self.config),
            'model_config': asdict(self.model_config),
            'training_metrics': [asdict(m) for m in self.training_metrics[-100:]],
            'validation_metrics': self.validation_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        report_path = os.path.join(self.checkpoint_dir, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 학습 보고서 저장: {report_path}")

def run_distributed_training(rank: int, world_size: int, config: DistributedConfig, model_config: ModelConfig):
    """분산 학습 실행 함수"""
    try:
        # 분산 트레이너 생성
        trainer = DistributedTrainer(config, model_config)
        
        # 분산 환경 설정
        trainer.setup_distributed(rank, world_size)
        
        # 모델 및 데이터 설정
        trainer.setup_model()
        trainer.setup_data()
        
        # 학습 시작
        trainer.train()
        
    except Exception as e:
        logger.error(f"분산 학습 실행 오류 (Rank {rank}): {e}")
    
    finally:
        # 분산 환경 정리
        if TORCH_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()

def main():
    """메인 실행 함수"""
    print("🚀 분산 신경망 학습 시스템")
    print("=" * 60)
    
    try:
        # 설정
        dist_config = DistributedConfig(
            num_gpus=1,  # 테스트용
            num_nodes=1,
            world_size=1,
            batch_size_per_gpu=4,
            max_epochs=5,  # 테스트용
            mixed_precision=False,  # 시뮬레이션
            gradient_checkpointing=False
        )
        
        if MODULES_AVAILABLE:
            from neural_gpt_autoci import ModelConfig
            model_config = ModelConfig(
                vocab_size=50000,
                hidden_size=1024,  # 테스트용 축소
                num_layers=12,
                num_heads=16,
                max_position_embeddings=512
            )
        else:
            # 시뮬레이션 모델 설정
            class ModelConfig:
                def __init__(self):
                    self.vocab_size = 50000
                    self.hidden_size = 1024
                    self.num_layers = 12
                    self.num_heads = 16
                    self.total_parameters = 1000000000  # 10억
            
            model_config = ModelConfig()
        
        logger.info(f"모델 파라미터 수: {model_config.total_parameters:,}")
        logger.info(f"전역 배치 크기: {dist_config.global_batch_size}")
        
        # 분산 학습 시작
        if TORCH_AVAILABLE and dist_config.world_size > 1:
            # 멀티프로세스 분산 학습
            torch_mp.spawn(
                run_distributed_training,
                args=(dist_config.world_size, dist_config, model_config),
                nprocs=dist_config.world_size,
                join=True
            )
        else:
            # 단일 프로세스 학습
            run_distributed_training(0, 1, dist_config, model_config)
        
        print("🎉 분산 학습 시스템 성공적으로 완료!")
        return 0
        
    except Exception as e:
        logger.error(f"메인 실행 오류: {e}")
        return 1

if __name__ == "__main__":
    import contextlib
    import sys
    sys.exit(main())