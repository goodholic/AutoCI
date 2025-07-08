"""
PyTorch 튜토리얼 및 대화형 학습 시스템
AI가 PyTorch의 기초부터 고급까지 설명하고 코드 예제를 제공
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

class PyTorchTutorialSystem:
    """PyTorch 대화형 학습 시스템"""
    
    def __init__(self, ai_model=None):
        self.ai_model = ai_model
        self.knowledge_base_path = Path("continuous_learning/knowledge_base/pytorch_tutorials.json")
        self.conversation_history = []
        self.current_topic = None
        self.tutorial_progress = {}
        
        # PyTorch 학습 주제들
        self.topics = {
            "basics": {
                "title": "PyTorch 기초",
                "subtopics": [
                    "텐서(Tensor) 이해하기",
                    "자동 미분(Autograd)",
                    "신경망 구축 (nn.Module)",
                    "데이터 로딩 (DataLoader)",
                    "옵티마이저와 손실 함수"
                ]
            },
            "intermediate": {
                "title": "PyTorch 중급",
                "subtopics": [
                    "CNN (Convolutional Neural Networks)",
                    "RNN/LSTM 구현",
                    "Transfer Learning",
                    "모델 저장 및 불러오기",
                    "GPU 활용하기"
                ]
            },
            "advanced": {
                "title": "PyTorch 고급",
                "subtopics": [
                    "Custom Dataset과 DataLoader",
                    "Mixed Precision Training",
                    "Distributed Training",
                    "모델 최적화 (Quantization, Pruning)",
                    "Production 배포"
                ]
            },
            "practical": {
                "title": "실전 프로젝트",
                "subtopics": [
                    "이미지 분류 프로젝트",
                    "자연어 처리 (NLP) 프로젝트",
                    "강화학습 구현",
                    "GAN 구현하기",
                    "시계열 예측"
                ]
            }
        }
        
        # 기본 PyTorch 지식 베이스
        self.pytorch_knowledge = {
            "텐서(Tensor) 이해하기": {
                "explanation": """
텐서(Tensor)는 PyTorch의 가장 기본적인 자료구조입니다.
NumPy의 ndarray와 비슷하지만, GPU에서 연산이 가능하고 자동 미분을 지원합니다.

주요 특징:
1. 다차원 배열 구조
2. GPU 가속 지원
3. 자동 미분 (requires_grad=True)
4. 다양한 수학 연산 지원
                """,
                "code_examples": [
                    {
                        "title": "텐서 생성하기",
                        "code": """
import torch

# 1. 직접 데이터로 텐서 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. NumPy 배열에서 텐서 생성
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)  # x_data와 같은 shape
x_rand = torch.rand_like(x_data, dtype=torch.float)

# 4. 특정 크기의 텐서 생성
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
"""
                    },
                    {
                        "title": "텐서 속성과 연산",
                        "code": """
# 텐서 속성
tensor = torch.rand(3, 4)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}")

# GPU로 이동 (가능한 경우)
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# 기본 연산
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 덧셈
add_result = tensor1 + tensor2
# 또는
add_result = torch.add(tensor1, tensor2)

# 행렬 곱셈
matmul_result = tensor1 @ tensor2
# 또는
matmul_result = torch.matmul(tensor1, tensor2)

# 원소별 곱셈
mul_result = tensor1 * tensor2
"""
                    }
                ]
            },
            "자동 미분(Autograd)": {
                "explanation": """
PyTorch의 Autograd는 자동 미분을 위한 엔진입니다.
신경망 학습에 필수적인 역전파(backpropagation)를 자동으로 계산해줍니다.

핵심 개념:
1. requires_grad=True: 미분 추적 활성화
2. backward(): 역전파 실행
3. grad: 계산된 기울기 저장
4. no_grad(): 미분 추적 비활성화 (추론 시 사용)
                """,
                "code_examples": [
                    {
                        "title": "자동 미분 기본",
                        "code": """
import torch

# requires_grad=True로 미분 추적 활성화
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 연산 수행
y = x + 2
z = y * y * 3
out = z.mean()

print(f"y: {y}")
print(f"z: {z}")
print(f"out: {out}")

# 역전파
out.backward()

# 기울기 확인
print(f"x.grad: {x.grad}")
"""
                    },
                    {
                        "title": "간단한 최적화 예제",
                        "code": """
# 간단한 선형 회귀
import torch
import torch.nn as nn

# 데이터 준비
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 모델 정의
model = nn.Linear(1, 1)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(100):
    # 순전파
    predictions = model(x)
    loss = criterion(predictions, y)
    
    # 역전파
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()        # 역전파
    optimizer.step()       # 파라미터 업데이트
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 학습된 파라미터 확인
print(f"Weight: {model.weight.item():.4f}")
print(f"Bias: {model.bias.item():.4f}")
"""
                    }
                ]
            }
        }
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """PyTorch 지식 베이스 로드"""
        if self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                loaded_knowledge = json.load(f)
                self.pytorch_knowledge.update(loaded_knowledge)
    
    def _save_knowledge_base(self):
        """지식 베이스 저장"""
        os.makedirs(self.knowledge_base_path.parent, exist_ok=True)
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(self.pytorch_knowledge, f, ensure_ascii=False, indent=2)
    
    async def start_tutorial_session(self):
        """대화형 튜토리얼 세션 시작"""
        print("\n🔥 PyTorch 대화형 학습 시스템")
        print("=" * 60)
        print("PyTorch의 기초부터 고급까지 차근차근 배워봅시다!")
        print("\n사용 가능한 명령어:")
        print("  - '주제': 학습 주제 목록 보기")
        print("  - '예제': 현재 주제의 코드 예제 보기")
        print("  - '실행': 코드 예제 실행하기")
        print("  - '질문': 현재 주제에 대해 질문하기")
        print("  - '다음': 다음 주제로 이동")
        print("  - '진도': 학습 진행률 확인")
        print("  - '종료': 학습 종료")
        print("=" * 60)
        
        await self._show_topics()
        
        while True:
            user_input = input("\n💭 무엇을 배우고 싶으신가요? > ").strip()
            
            if user_input.lower() in ['종료', 'exit', 'quit']:
                print("\n👋 학습을 종료합니다. 다음에 또 만나요!")
                break
            
            await self._process_user_input(user_input)
    
    async def _show_topics(self):
        """학습 주제 목록 표시"""
        print("\n📚 학습 주제:")
        for key, topic in self.topics.items():
            print(f"\n[{key}] {topic['title']}")
            for i, subtopic in enumerate(topic['subtopics'], 1):
                progress = self.tutorial_progress.get(subtopic, 0)
                status = "✅" if progress >= 100 else f"📊 {progress}%"
                print(f"  {i}. {subtopic} {status}")
    
    async def _process_user_input(self, user_input: str):
        """사용자 입력 처리"""
        input_lower = user_input.lower()
        
        if input_lower == '주제':
            await self._show_topics()
        
        elif input_lower == '예제' and self.current_topic:
            await self._show_examples(self.current_topic)
        
        elif input_lower == '실행' and self.current_topic:
            await self._run_example(self.current_topic)
        
        elif input_lower == '질문':
            await self._answer_question()
        
        elif input_lower == '다음':
            await self._next_topic()
        
        elif input_lower == '진도':
            await self._show_progress()
        
        else:
            # 주제 선택 또는 자유 질문
            await self._handle_topic_or_question(user_input)
    
    async def _show_examples(self, topic: str):
        """현재 주제의 예제 표시"""
        if topic in self.pytorch_knowledge:
            knowledge = self.pytorch_knowledge[topic]
            print(f"\n📖 {topic} - 설명:")
            print(knowledge['explanation'])
            
            if 'code_examples' in knowledge:
                print("\n💻 코드 예제:")
                for i, example in enumerate(knowledge['code_examples'], 1):
                    print(f"\n[예제 {i}] {example['title']}")
                    print("-" * 40)
                    print(example['code'])
    
    async def _run_example(self, topic: str):
        """예제 코드 실행"""
        if topic in self.pytorch_knowledge and 'code_examples' in self.pytorch_knowledge[topic]:
            examples = self.pytorch_knowledge[topic]['code_examples']
            
            print("\n🏃 어떤 예제를 실행하시겠습니까?")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. {example['title']}")
            
            try:
                choice = int(input("\n번호 선택: ")) - 1
                if 0 <= choice < len(examples):
                    example = examples[choice]
                    print(f"\n▶️ '{example['title']}' 실행 중...")
                    print("-" * 40)
                    
                    # 실제로는 여기서 코드를 안전하게 실행하는 로직 필요
                    # exec() 사용은 보안상 위험하므로 실제 구현 시 주의
                    print("⚠️  코드 실행 결과:")
                    print("(실제 실행은 보안을 위해 시뮬레이션됩니다)")
                    print("\n[코드가 성공적으로 실행되었습니다]")
                    
                    # 진도 업데이트
                    self.tutorial_progress[topic] = min(
                        self.tutorial_progress.get(topic, 0) + 25, 100
                    )
            except ValueError:
                print("❌ 올바른 번호를 입력해주세요.")
    
    async def _answer_question(self):
        """현재 주제에 대한 질문 답변"""
        if not self.current_topic:
            print("❓ 먼저 학습할 주제를 선택해주세요.")
            return
        
        question = input(f"\n🤔 {self.current_topic}에 대해 무엇이 궁금하신가요? > ")
        
        # AI 모델이 있다면 활용, 없다면 기본 답변
        if self.ai_model:
            context = self.pytorch_knowledge.get(self.current_topic, {})
            response = await self._generate_ai_response(question, context)
            print(f"\n🤖 답변: {response}")
        else:
            print(f"\n🤖 {self.current_topic}에 대한 일반적인 답변입니다.")
            print("더 자세한 답변을 원하시면 AI 모델을 연결해주세요.")
    
    async def _generate_ai_response(self, question: str, context: Dict) -> str:
        """AI를 활용한 답변 생성"""
        prompt = f"""
        주제: {self.current_topic}
        질문: {question}
        
        관련 지식:
        {json.dumps(context, ensure_ascii=False, indent=2)}
        
        위 정보를 바탕으로 초보자도 이해하기 쉽게 설명해주세요.
        코드 예제가 필요하다면 간단한 예제를 포함해주세요.
        """
        
        # AI 모델 호출 (실제 구현 필요)
        return "AI 답변이 여기에 표시됩니다."
    
    async def _next_topic(self):
        """다음 주제로 이동"""
        all_subtopics = []
        for topic in self.topics.values():
            all_subtopics.extend(topic['subtopics'])
        
        if self.current_topic in all_subtopics:
            current_index = all_subtopics.index(self.current_topic)
            if current_index < len(all_subtopics) - 1:
                self.current_topic = all_subtopics[current_index + 1]
                print(f"\n➡️ 다음 주제: {self.current_topic}")
                await self._show_examples(self.current_topic)
            else:
                print("\n🎉 모든 주제를 완료했습니다!")
    
    async def _show_progress(self):
        """학습 진행률 표시"""
        print("\n📊 학습 진행률:")
        total_topics = sum(len(topic['subtopics']) for topic in self.topics.values())
        completed = sum(1 for progress in self.tutorial_progress.values() if progress >= 100)
        
        print(f"전체 진행률: {completed}/{total_topics} ({completed/total_topics*100:.1f}%)")
        print("\n세부 진행률:")
        
        for key, topic in self.topics.items():
            topic_completed = sum(
                1 for subtopic in topic['subtopics'] 
                if self.tutorial_progress.get(subtopic, 0) >= 100
            )
            print(f"  {topic['title']}: {topic_completed}/{len(topic['subtopics'])}")
    
    async def _handle_topic_or_question(self, user_input: str):
        """주제 선택 또는 자유 질문 처리"""
        # 숫자로 주제 선택
        try:
            topic_num = int(user_input)
            all_subtopics = []
            for topic in self.topics.values():
                all_subtopics.extend(topic['subtopics'])
            
            if 1 <= topic_num <= len(all_subtopics):
                self.current_topic = all_subtopics[topic_num - 1]
                print(f"\n✅ '{self.current_topic}' 주제를 선택했습니다.")
                await self._show_examples(self.current_topic)
                return
        except ValueError:
            pass
        
        # 키워드로 주제 검색
        for key, topic in self.topics.items():
            if key in user_input.lower() or topic['title'] in user_input:
                print(f"\n📚 {topic['title']} 관련 주제:")
                for i, subtopic in enumerate(topic['subtopics'], 1):
                    print(f"  {i}. {subtopic}")
                return
        
        # 자유 질문으로 처리
        print(f"\n🤔 '{user_input}'에 대해 알아보겠습니다.")
        if self.ai_model:
            response = await self._generate_ai_response(user_input, self.pytorch_knowledge)
            print(f"\n🤖 답변: {response}")
        else:
            print("AI 모델이 연결되지 않아 기본 답변만 제공됩니다.")
            print("'주제'를 입력하여 학습 주제를 선택해주세요.")
    
    def add_custom_knowledge(self, topic: str, knowledge: Dict):
        """커스텀 지식 추가"""
        self.pytorch_knowledge[topic] = knowledge
        self._save_knowledge_base()
        print(f"✅ '{topic}' 지식이 추가되었습니다.")