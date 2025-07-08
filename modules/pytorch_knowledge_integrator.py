"""
PyTorch 지식 통합 모듈
지속적 학습 시스템과 PyTorch 튜토리얼을 연결
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class PyTorchKnowledgeIntegrator:
    """PyTorch 지식을 지속적 학습 시스템에 통합"""
    
    def __init__(self):
        self.knowledge_base_path = Path("continuous_learning/knowledge_base/knowledge_base.json")
        self.pytorch_tutorials_path = Path("continuous_learning/knowledge_base/pytorch_tutorials.json")
        
        # PyTorch 고급 주제들
        self.advanced_topics = {
            "distributed_training": {
                "title": "분산 학습",
                "keywords": ["DDP", "DistributedDataParallel", "multi-gpu", "horovod"],
                "content": """
분산 학습은 여러 GPU나 여러 노드에서 모델을 학습시키는 기술입니다.

주요 개념:
1. Data Parallelism: 데이터를 나누어 여러 GPU에서 처리
2. Model Parallelism: 모델을 나누어 여러 GPU에 배치
3. Pipeline Parallelism: 레이어를 파이프라인으로 구성

PyTorch DDP 예제:
```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 모델을 해당 GPU로 이동
    model = YourModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 학습 코드...
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```
                """
            },
            "custom_operators": {
                "title": "커스텀 연산자",
                "keywords": ["custom op", "torch.jit", "cpp extension"],
                "content": """
PyTorch에서 커스텀 연산자를 만들어 성능을 최적화할 수 있습니다.

방법:
1. Python으로 구현
2. C++로 구현 (더 빠름)
3. CUDA로 구현 (GPU 가속)

Python 커스텀 함수:
```python
import torch
from torch.autograd import Function

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 사용
custom_relu = CustomReLU.apply
```
                """
            },
            "model_optimization": {
                "title": "모델 최적화",
                "keywords": ["quantization", "pruning", "distillation", "onnx"],
                "content": """
모델 최적화는 모델의 크기를 줄이고 추론 속도를 높이는 기술입니다.

주요 기법:
1. Quantization: 가중치를 낮은 정밀도로 변환
2. Pruning: 중요하지 않은 가중치 제거
3. Knowledge Distillation: 큰 모델의 지식을 작은 모델로 전달

Quantization 예제:
```python
import torch
import torch.quantization

# 모델 준비
model.eval()

# Quantization 설정
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared_model = torch.quantization.prepare(model)

# Calibration (일부 데이터로)
with torch.no_grad():
    for data in calibration_data:
        prepared_model(data)

# Quantize
quantized_model = torch.quantization.convert(prepared_model)
```
                """
            },
            "debugging_profiling": {
                "title": "디버깅과 프로파일링",
                "keywords": ["profiler", "tensorboard", "debugging", "memory"],
                "content": """
PyTorch 모델의 성능 문제를 찾고 해결하는 방법입니다.

주요 도구:
1. PyTorch Profiler: 성능 분석
2. TensorBoard: 시각화
3. Memory Profiler: 메모리 사용량 분석

Profiler 사용:
```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # 모델 실행
    model(input)

# 결과 출력
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# TensorBoard로 내보내기
prof.export_chrome_trace("trace.json")
```
                """
            }
        }
    
    def integrate_pytorch_knowledge(self):
        """PyTorch 지식을 메인 knowledge base에 통합"""
        # 기존 지식 베이스 로드
        knowledge_base = self._load_knowledge_base()
        
        # PyTorch 카테고리 추가
        if "pytorch_fundamentals" not in knowledge_base:
            knowledge_base["pytorch_fundamentals"] = []
        
        if "pytorch_advanced" not in knowledge_base:
            knowledge_base["pytorch_advanced"] = []
        
        # 고급 주제들 추가
        for topic_key, topic_data in self.advanced_topics.items():
            entry = {
                "question": f"PyTorch {topic_data['title']}에 대해 설명해주세요",
                "answer": topic_data['content'],
                "keywords": topic_data['keywords'],
                "quality_score": 0.95,
                "timestamp": datetime.now().isoformat(),
                "category": "pytorch_advanced"
            }
            
            # 중복 체크
            if not any(item['question'] == entry['question'] 
                      for item in knowledge_base["pytorch_advanced"]):
                knowledge_base["pytorch_advanced"].append(entry)
        
        # 저장
        self._save_knowledge_base(knowledge_base)
        return knowledge_base
    
    def create_pytorch_learning_plan(self) -> Dict:
        """체계적인 PyTorch 학습 계획 생성"""
        learning_plan = {
            "beginner": {
                "duration": "2-4 weeks",
                "topics": [
                    {
                        "name": "PyTorch 설치 및 환경 설정",
                        "exercises": [
                            "PyTorch 설치 (CPU/GPU)",
                            "CUDA 버전 확인",
                            "첫 텐서 만들기"
                        ]
                    },
                    {
                        "name": "텐서 기초",
                        "exercises": [
                            "텐서 생성 방법 5가지",
                            "텐서 연산 (덧셈, 곱셈, 행렬곱)",
                            "GPU로 텐서 이동"
                        ]
                    },
                    {
                        "name": "자동 미분",
                        "exercises": [
                            "requires_grad 이해하기",
                            "간단한 미분 계산",
                            "계산 그래프 시각화"
                        ]
                    }
                ]
            },
            "intermediate": {
                "duration": "4-6 weeks",
                "topics": [
                    {
                        "name": "신경망 구축",
                        "exercises": [
                            "nn.Module 상속하기",
                            "MLP 구현",
                            "CNN 구현"
                        ]
                    },
                    {
                        "name": "학습 루프",
                        "exercises": [
                            "학습/검증 루프 작성",
                            "체크포인트 저장/로드",
                            "Early Stopping 구현"
                        ]
                    }
                ]
            },
            "advanced": {
                "duration": "6-8 weeks",
                "topics": [
                    {
                        "name": "고급 아키텍처",
                        "exercises": [
                            "Transformer 구현",
                            "GAN 구현",
                            "VAE 구현"
                        ]
                    },
                    {
                        "name": "최적화 기법",
                        "exercises": [
                            "Mixed Precision Training",
                            "Gradient Accumulation",
                            "Custom Loss Function"
                        ]
                    }
                ]
            }
        }
        
        return learning_plan
    
    def generate_interactive_examples(self, topic: str) -> List[Dict]:
        """대화형 예제 생성"""
        examples = []
        
        if "tensor" in topic.lower():
            examples.append({
                "title": "텐서 놀이터",
                "description": "다양한 텐서 연산을 실험해보세요",
                "interactive_code": """
# 텐서 생성
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

# 여러분이 시도해볼 수 있는 연산들:
# 1. 덧셈: x + y
# 2. 원소별 곱셈: x * y
# 3. 내적: torch.dot(x, y)
# 4. reshape: x.view(2, 2)
# 5. 통계: x.mean(), x.std()

# 실험해보세요!
result = ?  # 여기에 연산을 입력하세요
print(result)
                """,
                "hints": [
                    "view()는 텐서의 형태를 바꿉니다",
                    "unsqueeze()는 차원을 추가합니다",
                    "GPU가 있다면 .cuda()를 사용해보세요"
                ]
            })
        
        elif "autograd" in topic.lower():
            examples.append({
                "title": "자동 미분 실험",
                "description": "역전파가 어떻게 작동하는지 확인해보세요",
                "interactive_code": """
# 미분 추적이 활성화된 텐서
x = torch.tensor(2.0, requires_grad=True)

# 여러 연산을 시도해보세요:
# 1. 제곱: y = x ** 2
# 2. 삼각함수: y = torch.sin(x)
# 3. 복합 함수: y = torch.exp(x ** 2)

y = ?  # 여기에 함수를 입력하세요

# 역전파
y.backward()

# 기울기 확인
print(f"x의 값: {x}")
print(f"y의 값: {y}")
print(f"dy/dx: {x.grad}")
                """,
                "hints": [
                    "y = x² 의 미분은 2x입니다",
                    "복잡한 함수도 자동으로 미분됩니다",
                    "grad는 누적되므로 zero_grad()가 필요합니다"
                ]
            })
        
        return examples
    
    def _load_knowledge_base(self) -> Dict:
        """지식 베이스 로드"""
        if self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_knowledge_base(self, knowledge_base: Dict):
        """지식 베이스 저장"""
        os.makedirs(self.knowledge_base_path.parent, exist_ok=True)
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    def search_pytorch_knowledge(self, query: str) -> List[Dict]:
        """PyTorch 관련 지식 검색"""
        knowledge_base = self._load_knowledge_base()
        results = []
        
        query_lower = query.lower()
        
        # PyTorch 카테고리에서 검색
        for category in ["pytorch_fundamentals", "pytorch_advanced"]:
            if category in knowledge_base:
                for item in knowledge_base[category]:
                    # 질문이나 키워드에서 매칭
                    if (query_lower in item.get('question', '').lower() or
                        any(query_lower in keyword.lower() 
                            for keyword in item.get('keywords', []))):
                        results.append({
                            'category': category,
                            'question': item['question'],
                            'answer': item['answer'],
                            'score': item.get('quality_score', 0.5)
                        })
        
        # 점수순으로 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]  # 상위 5개만 반환