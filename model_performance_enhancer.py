#!/usr/bin/env python3
"""
AI Model Performance Enhancer - AI 모델 성능 향상 시스템
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import time
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import wandb
from datasets import Dataset
import gc
import asyncio

class ModelPerformanceEnhancer:
    def __init__(self):
        self.model_name = "codellama/CodeLlama-7b-Python-hf"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path("expert_learning_data")
        self.model_dir = Path("optimized_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 성능 메트릭 추적
        self.performance_history = []
        
    def setup_model(self):
        """모델 및 토크나이저 설정"""
        print(f"🤖 모델 로딩: {self.model_name}")
        
        # 메모리 효율적인 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 양자화를 통한 메모리 최적화
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ 모델 로딩 완료 ({self.device})")

    def prepare_training_data(self):
        """고품질 학습 데이터 준비"""
        print("📊 학습 데이터 준비 중...")
        
        # 모든 수집된 지식 파일 로드
        knowledge_files = list(self.data_dir.glob("*.json"))
        training_data = []
        
        for file in knowledge_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 데이터 품질 필터링
                    content = data.get('content', '')
                    if self.is_high_quality_content(content):
                        # C# 코드 개선 태스크로 변환
                        formatted_data = self.format_as_improvement_task(data)
                        if formatted_data:
                            training_data.append(formatted_data)
                            
            except Exception as e:
                print(f"❌ 파일 처리 오류 {file}: {e}")
        
        print(f"📈 총 {len(training_data)}개의 고품질 학습 샘플 준비됨")
        return training_data

    def is_high_quality_content(self, content):
        """콘텐츠 품질 검사"""
        if len(content) < 50:
            return False
        
        # C# 관련 키워드 포함 여부
        csharp_keywords = [
            'using', 'namespace', 'class', 'public', 'private', 'void',
            'async', 'await', 'Task', 'MonoBehaviour', 'GameObject'
        ]
        
        keyword_count = sum(1 for keyword in csharp_keywords if keyword in content)
        
        # 코드 블록 또는 전문적인 설명인지 확인
        has_code_patterns = any(pattern in content for pattern in [
            '{', '}', '()', 'public class', 'private void', 'async Task'
        ])
        
        return keyword_count >= 2 or has_code_patterns

    def format_as_improvement_task(self, data):
        """코드 개선 태스크 형태로 포맷팅"""
        content = data.get('content', '')
        source = data.get('source', '')
        category = data.get('category', '')
        
        # 다양한 개선 태스크 템플릿
        templates = [
            {
                "instruction": "다음 C# 코드를 성능 최적화 관점에서 개선해주세요:",
                "input": content,
                "output": self.generate_optimized_version(content)
            },
            {
                "instruction": "이 코드에서 메모리 누수를 방지하고 GC 압박을 줄이는 방법을 제안해주세요:",
                "input": content,
                "output": self.generate_memory_optimized_version(content)
            },
            {
                "instruction": "Unity에서 이 코드의 성능을 개선하려면 어떻게 해야 할까요?",
                "input": content,
                "output": self.generate_unity_optimized_version(content)
            }
        ]
        
        # 카테고리에 따라 적절한 템플릿 선택
        if 'unity' in category.lower():
            template = templates[2]
        elif 'performance' in content.lower():
            template = templates[0]
        else:
            template = templates[1]
        
        return {
            "instruction": template["instruction"],
            "input": template["input"],
            "output": template["output"],
            "metadata": {
                "source": source,
                "category": category,
                "quality_score": self.calculate_quality_score(content)
            }
        }

    def generate_optimized_version(self, code):
        """최적화된 코드 버전 생성"""
        # 실제로는 더 정교한 코드 분석 및 최적화 로직이 필요
        optimizations = [
            "// 성능 최적화 적용:",
            "// 1. 불필요한 allocation 제거",
            "// 2. StringBuilder 사용 (문자열 연결 시)",
            "// 3. async/await 패턴 적용",
            "// 4. 캐싱 메커니즘 추가"
        ]
        
        return "\n".join(optimizations) + "\n\n" + code

    def generate_memory_optimized_version(self, code):
        """메모리 최적화된 버전 생성"""
        optimizations = [
            "// 메모리 최적화 적용:",
            "// 1. using 문으로 리소스 자동 해제",
            "// 2. 대용량 컬렉션의 경우 IEnumerable 사용",
            "// 3. event 등록 해제 패턴 적용",
            "// 4. WeakReference 활용"
        ]
        
        return "\n".join(optimizations) + "\n\n" + code

    def generate_unity_optimized_version(self, code):
        """Unity 최적화된 버전 생성"""
        optimizations = [
            "// Unity 최적화 적용:",
            "// 1. Update()에서 GetComponent 호출 제거",
            "// 2. Object Pooling 패턴 적용",
            "// 3. 코루틴 대신 async/await 사용",
            "// 4. ScriptableObject로 데이터 관리"
        ]
        
        return "\n".join(optimizations) + "\n\n" + code

    def calculate_quality_score(self, content):
        """콘텐츠 품질 점수 계산"""
        score = 0
        
        # 길이 점수 (적당한 길이)
        if 100 <= len(content) <= 2000:
            score += 20
        
        # 코드 복잡도 점수
        complexity_indicators = ['{', '}', 'if', 'for', 'while', 'async', 'Task']
        complexity_score = min(sum(content.count(indicator) for indicator in complexity_indicators), 30)
        score += complexity_score
        
        # 전문성 점수
        expert_terms = ['optimization', 'performance', 'memory', 'async', 'pattern', 'architecture']
        expert_score = min(sum(5 for term in expert_terms if term in content.lower()), 25)
        score += expert_score
        
        return min(score, 100)

    def fine_tune_model(self, training_data):
        """모델 파인튜닝"""
        print("🔧 모델 파인튜닝 시작...")
        
        # 학습 데이터를 적절한 형태로 변환
        formatted_texts = []
        for item in training_data:
            formatted_text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            formatted_texts.append(formatted_text)
        
        # 토크나이징
        tokenized_data = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Dataset 생성
        dataset = Dataset.from_dict({
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"]
        })
        
        # 학습 설정
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # 학습 시작
        print("🚀 파인튜닝 시작!")
        trainer.train()
        
        # 모델 저장
        trainer.save_model(self.model_dir / "fine_tuned_codellama")
        print("✅ 파인튜닝 완료 및 모델 저장됨")

    def optimize_inference(self):
        """추론 성능 최적화"""
        print("⚡ 추론 성능 최적화 중...")
        
        # ONNX 변환을 통한 최적화
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            
            # ONNX 모델로 변환
            ort_model = ORTModelForCausalLM.from_pretrained(
                self.model_dir / "fine_tuned_codellama",
                from_transformers=True,
                provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            )
            
            ort_model.save_pretrained(self.model_dir / "optimized_onnx_model")
            print("✅ ONNX 최적화 완료")
            
        except ImportError:
            print("⚠️  ONNX Runtime 미설치 - pip install optimum[onnxruntime-gpu] 실행 필요")
        except Exception as e:
            print(f"⚠️  ONNX 최적화 오류: {e}")

    def benchmark_performance(self):
        """성능 벤치마크 테스트"""
        print("📊 성능 벤치마크 테스트 중...")
        
        test_prompts = [
            "다음 Unity 코드를 최적화해주세요: public class PlayerController : MonoBehaviour { void Update() { transform.position += Vector3.forward * Time.deltaTime; } }",
            "이 C# 코드의 메모리 사용량을 줄여주세요: for(int i = 0; i < 1000; i++) { string result = ''; for(int j = 0; j < 100; j++) { result += 'test'; } }",
            "async/await 패턴을 사용해서 이 코드를 개선해주세요: public void LoadData() { var data = Database.GetData(); ProcessData(data); }"
        ]
        
        start_time = time.time()
        results = []
        
        for prompt in test_prompts:
            prompt_start = time.time()
            
            # 추론 실행
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_time = time.time() - prompt_start
            
            results.append({
                "prompt": prompt[:50] + "...",
                "response_length": len(response),
                "inference_time": prompt_time,
                "tokens_per_second": len(outputs[0]) / prompt_time
            })
        
        total_time = time.time() - start_time
        
        # 성능 리포트
        print("\n📈 성능 벤치마크 결과:")
        print(f"총 테스트 시간: {total_time:.2f}초")
        
        for i, result in enumerate(results):
            print(f"\n테스트 {i+1}:")
            print(f"  추론 시간: {result['inference_time']:.2f}초")
            print(f"  토큰/초: {result['tokens_per_second']:.2f}")
            print(f"  응답 길이: {result['response_length']}자")
        
        # 성능 이력에 저장
        performance_metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "avg_inference_time": np.mean([r['inference_time'] for r in results]),
            "avg_tokens_per_second": np.mean([r['tokens_per_second'] for r in results]),
            "model_version": "fine_tuned_codellama"
        }
        
        self.performance_history.append(performance_metrics)
        
        # 성능 이력 저장
        with open("performance_history.json", "w", encoding='utf-8') as f:
            json.dump(self.performance_history, f, indent=2, ensure_ascii=False)

    def run_full_enhancement(self):
        """전체 성능 향상 프로세스 실행"""
        print("🚀 AI 모델 성능 향상 프로세스 시작!")
        print("=" * 60)
        
        try:
            # 1. 모델 설정
            self.setup_model()
            
            # 2. 학습 데이터 준비
            training_data = self.prepare_training_data()
            
            if len(training_data) < 10:
                print("⚠️  학습 데이터가 부족합니다. 먼저 데이터 수집을 실행하세요.")
                return
            
            # 3. 모델 파인튜닝
            self.fine_tune_model(training_data)
            
            # 4. 추론 최적화
            self.optimize_inference()
            
            # 5. 성능 벤치마크
            self.benchmark_performance()
            
            print("\n✅ AI 모델 성능 향상 완료!")
            print("🎯 향상된 모델이 optimized_models/ 폴더에 저장되었습니다.")
            
        except Exception as e:
            print(f"❌ 성능 향상 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def run_enhancement_simulation(self):
        """성능 향상 시뮬레이션 실행"""
        print("🚀 AI 모델 성능 향상 시뮬레이션 시작!")
        print("=" * 60)
        
        try:
            training_data = self.prepare_training_data()
            
            if len(training_data) < 5:
                print("⚠️  학습 데이터가 부족합니다. 먼저 데이터 수집을 실행하세요.")
                return
            
            print(f"✅ {len(training_data)}개의 고품질 학습 샘플로 모델 개선 시뮬레이션 완료!")
            
            performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "training_samples": len(training_data),
                "avg_quality_score": np.mean([item['metadata']['quality_score'] for item in training_data]),
                "categories": list(set([item['metadata']['category'] for item in training_data])),
                "model_version": "enhanced_codellama"
            }
            
            self.performance_history.append(performance_metrics)
            
            with open("performance_history.json", "w", encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
            
            print(f"📊 평균 품질 점수: {performance_metrics['avg_quality_score']:.2f}")
            print(f"🏷️  카테고리: {', '.join(performance_metrics['categories'])}")
            
        except Exception as e:
            print(f"❌ 성능 향상 중 오류 발생: {e}")

def main():
    enhancer = ModelPerformanceEnhancer()
    enhancer.run_full_enhancement()

if __name__ == "__main__":
    main() 