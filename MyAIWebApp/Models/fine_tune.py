#!/usr/bin/env python3
"""
Code Llama Fine-tuning Script
C# 전문가 모델 학습을 위한 파인튜닝 스크립트
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정"""
    model_path: str = "../../CodeLlama-7b-Instruct-hf"
    output_dir: str = "./fine_tuned_model"
    data_path: str = "../../expert_training_data/training_dataset.json"
    
    # 학습 파라미터
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # LoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # 기타 설정
    max_length: int = 2048
    use_8bit: bool = True
    use_wandb: bool = False
    seed: int = 42

class CSharpDataset:
    """C# 학습 데이터셋"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """데이터 로드"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ {len(data)}개의 학습 데이터 로드 완료")
        return data
    
    def prepare_dataset(self) -> Dataset:
        """데이터셋 준비"""
        formatted_data = []
        
        for item in tqdm(self.data, desc="데이터 전처리"):
            # 프롬프트 포맷팅
            if "instruction" in item and "output" in item:
                text = self._format_prompt(
                    instruction=item["instruction"],
                    input_text=item.get("input", ""),
                    output=item["output"]
                )
            elif "code" in item:
                # 코드 설명 생성
                text = self._format_code_explanation(item["code"], item.get("patterns", []))
            else:
                continue
            
            # 토큰화
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False
            )
            
            formatted_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy()
            })
        
        return Dataset.from_list(formatted_data)
    
    def _format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """프롬프트 포맷팅"""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    def _format_code_explanation(self, code: str, patterns: List[str]) -> str:
        """코드 설명 포맷팅"""
        patterns_str = ", ".join(patterns) if patterns else "일반적인 C# 코드"
        
        return f"""### Instruction:
다음 C# 코드를 분석하고 설명해주세요.

### Input:
```csharp
{code[:1000]}  # 처음 1000자만
```

### Response:
이 코드는 {patterns_str} 패턴을 사용하는 C# 코드입니다. 
주요 특징:
- 최신 C# 기능 활용
- SOLID 원칙 준수
- 적절한 에러 처리
- 명확한 코드 구조"""

class ProgressCallback(TrainerCallback):
    """학습 진행 상황 콜백"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 진행 상황 로깅
            if "loss" in logs:
                logger.info(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
            
            # 학습률 로깅
            if "learning_rate" in logs:
                logger.info(f"Learning Rate: {logs['learning_rate']:.2e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"✅ Epoch {state.epoch} 완료")
    
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("🎉 학습 완료!")
        logger.info(f"총 학습 스텝: {state.global_step}")

class CodeLlamaFineTuner:
    """Code Llama 파인튜닝 클래스"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_wandb(self):
        """Weights & Biases 설정"""
        if self.config.use_wandb:
            wandb.init(
                project="autoci-csharp-expert",
                name=f"fine-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config.__dict__
            )
            logger.info("✅ Weights & Biases 초기화 완료")
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        logger.info("🚀 모델 로드 중...")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 모델 로드 설정
        if self.config.use_8bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 8-bit 학습 준비
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("✅ 8-bit 양자화 모델 로드 완료")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✅ 모델 로드 완료")
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_data(self) -> tuple:
        """데이터 준비"""
        logger.info("📚 데이터셋 준비 중...")
        
        # 데이터셋 로드
        dataset = CSharpDataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_length
        )
        
        # 전체 데이터셋 준비
        full_dataset = dataset.prepare_dataset()
        
        # 학습/검증 분할 (90/10)
        train_size = int(0.9 * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        
        train_dataset = full_dataset.select(range(train_size))
        eval_dataset = full_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"✅ 학습 데이터: {len(train_dataset)}개")
        logger.info(f"✅ 검증 데이터: {len(eval_dataset)}개")
        
        return train_dataset, eval_dataset
    
    def setup_training(self, train_dataset, eval_dataset):
        """학습 설정"""
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=self.config.save_total_limit,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            remove_unused_columns=False,
            label_names=["labels"]
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 콜백 설정
        callbacks = [ProgressCallback()]
        if eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )
            )
        
        # Trainer 설정
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        logger.info("✅ 학습 설정 완료")
    
    def train(self):
        """모델 학습"""
        logger.info("🏃 학습 시작...")
        
        # 학습 시작 시간
        start_time = datetime.now()
        
        try:
            # 학습 실행
            train_result = self.trainer.train()
            
            # 학습 시간 계산
            training_time = datetime.now() - start_time
            hours = training_time.total_seconds() / 3600
            
            logger.info(f"✅ 학습 완료! (소요 시간: {hours:.1f}시간)")
            logger.info(f"최종 Loss: {train_result.training_loss:.4f}")
            
            # 모델 저장
            self.save_model()
            
            # 학습 통계 저장
            self.save_training_stats(train_result, hours)
            
        except Exception as e:
            logger.error(f"❌ 학습 중 오류 발생: {str(e)}")
            raise
    
    def save_model(self):
        """모델 저장"""
        logger.info("💾 모델 저장 중...")
        
        # 최종 모델 저장
        self.trainer.save_model(self.config.output_dir)
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # LoRA 가중치만 저장
        lora_path = Path(self.config.output_dir) / "adapter_model"
        self.model.save_pretrained(str(lora_path))
        
        logger.info(f"✅ 모델 저장 완료: {self.config.output_dir}")
    
    def save_training_stats(self, train_result, hours: float):
        """학습 통계 저장"""
        stats = {
            "training_completed": datetime.now().isoformat(),
            "total_hours": hours,
            "final_loss": float(train_result.training_loss),
            "total_steps": train_result.global_step,
            "model_path": self.config.model_path,
            "output_dir": self.config.output_dir,
            "config": self.config.__dict__
        }
        
        stats_path = Path(self.config.output_dir) / "training_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 학습 통계 저장: {stats_path}")
    
    def run(self):
        """전체 파인튜닝 프로세스 실행"""
        try:
            # W&B 설정
            self.setup_wandb()
            
            # 모델 로드
            self.load_model_and_tokenizer()
            
            # 데이터 준비
            train_dataset, eval_dataset = self.prepare_data()
            
            # 학습 설정
            self.setup_training(train_dataset, eval_dataset)
            
            # 학습 실행
            self.train()
            
            logger.info("🎉 파인튜닝 프로세스 완료!")
            
        except Exception as e:
            logger.error(f"❌ 파인튜닝 실패: {str(e)}")
            raise
        finally:
            if self.config.use_wandb:
                wandb.finish()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Llama C# Expert 파인튜닝")
    parser.add_argument("--data", type=str, help="학습 데이터 경로")
    parser.add_argument("--model", type=str, help="기본 모델 경로")
    parser.add_argument("--output", type=str, help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="학습률")
    parser.add_argument("--use-wandb", action="store_true", help="W&B 사용")
    
    args = parser.parse_args()
    
    # 설정 생성
    config = ModelConfig()
    
    if args.data:
        config.data_path = args.data
    if args.model:
        config.model_path = args.model
    if args.output:
        config.output_dir = args.output
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.use_wandb:
        config.use_wandb = True
    
    # 파인튜너 실행
    fine_tuner = CodeLlamaFineTuner(config)
    fine_tuner.run()

if __name__ == "__main__":
    main()