#!/usr/bin/env python3
"""
PyTorch 튜터 시스템
PyTorch의 기초부터 고급까지 단계별로 학습을 도와주는 AI 튜터
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PyTorchTopic:
    """PyTorch 학습 주제"""
    topic_id: str
    category: str  # basics, tensors, autograd, nn, optimization, etc.
    title: str
    difficulty: str  # beginner, intermediate, advanced
    prerequisites: List[str]
    content: Dict[str, Any]

class PyTorchTutor:
    """PyTorch 학습 도우미"""
    
    def __init__(self):
        self.current_topic = None
        self.user_progress = {}
        self.learning_path = []
        
        # PyTorch 학습 커리큘럼
        self.curriculum = {
            "basics": {
                "title": "PyTorch 기초",
                "topics": [
                    {
                        "id": "pytorch_intro",
                        "title": "PyTorch란 무엇인가?",
                        "content": {
                            "설명": "PyTorch는 Facebook에서 개발한 오픈소스 머신러닝 프레임워크입니다.",
                            "특징": [
                                "동적 계산 그래프 (Dynamic Computational Graph)",
                                "Pythonic한 문법",
                                "GPU 가속 지원",
                                "자동 미분 (Autograd)"
                            ],
                            "예제": """
import torch

# PyTorch 버전 확인
print(torch.__version__)

# CUDA 사용 가능 여부 확인
print(torch.cuda.is_available())
"""
                        }
                    },
                    {
                        "id": "tensor_basics",
                        "title": "텐서(Tensor) 기초",
                        "content": {
                            "설명": "텐서는 PyTorch의 기본 데이터 구조로, 다차원 배열입니다.",
                            "주요_메서드": {
                                "생성": ["torch.tensor()", "torch.zeros()", "torch.ones()", "torch.randn()"],
                                "형태": ["shape", "size()", "reshape()", "view()"],
                                "연산": ["add()", "mul()", "matmul()", "sum()"]
                            },
                            "예제": """
import torch

# 텐서 생성
x = torch.tensor([1, 2, 3])
print(f"1D 텐서: {x}")

# 2D 텐서 생성
matrix = torch.tensor([[1, 2], [3, 4]])
print(f"2D 텐서:\\n{matrix}")

# 랜덤 텐서
random_tensor = torch.randn(3, 3)
print(f"랜덤 텐서:\\n{random_tensor}")

# 텐서 연산
y = torch.tensor([4, 5, 6])
z = x + y
print(f"덧셈 결과: {z}")
"""
                        }
                    }
                ]
            },
            "tensors": {
                "title": "텐서 심화",
                "topics": [
                    {
                        "id": "tensor_operations",
                        "title": "텐서 연산 심화",
                        "content": {
                            "설명": "텐서의 다양한 연산과 변환 방법을 학습합니다.",
                            "연산_종류": {
                                "산술연산": ["add", "sub", "mul", "div"],
                                "행렬연산": ["mm", "bmm", "matmul"],
                                "집계연산": ["sum", "mean", "max", "min"],
                                "형태변환": ["reshape", "view", "squeeze", "unsqueeze"]
                            },
                            "예제": """
import torch

# 행렬 곱셈
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.matmul(A, B)  # 또는 A @ B
print(f"행렬 곱셈 결과 shape: {C.shape}")

# Broadcasting
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1], [2], [3]])
z = x + y  # Broadcasting 적용
print(f"Broadcasting 결과:\\n{z}")

# In-place 연산
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x.add_(1)  # In-place addition
print(f"In-place 덧셈: {x}")
"""
                        }
                    },
                    {
                        "id": "tensor_indexing",
                        "title": "텐서 인덱싱과 슬라이싱",
                        "content": {
                            "설명": "텐서의 특정 요소나 부분을 선택하는 방법을 학습합니다.",
                            "기법": [
                                "기본 인덱싱",
                                "슬라이싱",
                                "마스킹",
                                "gather와 scatter"
                            ],
                            "예제": """
import torch

# 2D 텐서 생성
tensor = torch.randn(5, 5)

# 기본 인덱싱
print(f"첫 번째 행: {tensor[0]}")
print(f"(1,2) 위치 요소: {tensor[1, 2]}")

# 슬라이싱
print(f"처음 3개 행:\\n{tensor[:3]}")
print(f"2-4열:\\n{tensor[:, 1:4]}")

# 조건부 선택 (마스킹)
mask = tensor > 0
positive_values = tensor[mask]
print(f"양수 값들: {positive_values}")

# Fancy indexing
indices = torch.tensor([0, 2, 4])
selected_rows = tensor[indices]
print(f"선택된 행들:\\n{selected_rows}")
"""
                        }
                    }
                ]
            },
            "autograd": {
                "title": "자동 미분 (Autograd)",
                "topics": [
                    {
                        "id": "autograd_basics",
                        "title": "Autograd 기초",
                        "content": {
                            "설명": "PyTorch의 자동 미분 시스템을 이해하고 사용하는 방법을 학습합니다.",
                            "핵심개념": [
                                "requires_grad",
                                "backward()",
                                "grad",
                                "계산 그래프"
                            ],
                            "예제": """
import torch

# requires_grad=True로 텐서 생성
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

# 역전파
y.backward()

# 그래디언트 확인
print(f"x의 값: {x}")
print(f"y의 값: {y}")
print(f"dy/dx: {x.grad}")  # 2*x + 3 = 2*2 + 3 = 7

# 더 복잡한 예제
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y * y * 3
out = z.mean()

out.backward()
print(f"x의 그래디언트: {x.grad}")
"""
                        }
                    },
                    {
                        "id": "computational_graph",
                        "title": "계산 그래프 이해하기",
                        "content": {
                            "설명": "PyTorch가 연산을 추적하고 그래디언트를 계산하는 방법을 학습합니다.",
                            "주요개념": [
                                "동적 계산 그래프",
                                "leaf node",
                                "grad_fn",
                                "detach()"
                            ],
                            "예제": """
import torch

# 계산 그래프 생성
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x * y
w = z + x

# 그래프 정보 확인
print(f"z.grad_fn: {z.grad_fn}")
print(f"w.grad_fn: {w.grad_fn}")
print(f"x.is_leaf: {x.is_leaf}")
print(f"w.is_leaf: {w.is_leaf}")

# 역전파
w.backward()
print(f"x.grad: {x.grad}")  # dw/dx = 1 + y = 3
print(f"y.grad: {y.grad}")  # dw/dy = x = 1

# detach() 사용
x_detached = x.detach()
print(f"x_detached.requires_grad: {x_detached.requires_grad}")
"""
                        }
                    }
                ]
            },
            "nn": {
                "title": "신경망 구축 (torch.nn)",
                "topics": [
                    {
                        "id": "nn_module",
                        "title": "nn.Module 기초",
                        "content": {
                            "설명": "PyTorch에서 신경망을 구축하는 기본 방법을 학습합니다.",
                            "핵심요소": [
                                "nn.Module 상속",
                                "__init__ 메서드",
                                "forward 메서드",
                                "파라미터 관리"
                            ],
                            "예제": """
import torch
import torch.nn as nn
import torch.nn.functional as F

# 간단한 신경망 정의
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 생성
model = SimpleNet(10, 20, 5)
print(model)

# 파라미터 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 순전파
input_data = torch.randn(1, 10)
output = model(input_data)
print(f"출력 shape: {output.shape}")
"""
                        }
                    },
                    {
                        "id": "common_layers",
                        "title": "주요 레이어 이해하기",
                        "content": {
                            "설명": "자주 사용되는 신경망 레이어들을 학습합니다.",
                            "레이어_종류": {
                                "Linear": "완전 연결 레이어",
                                "Conv2d": "2D 합성곱 레이어",
                                "MaxPool2d": "2D 최대 풀링",
                                "BatchNorm2d": "배치 정규화",
                                "Dropout": "드롭아웃"
                            },
                            "예제": """
import torch
import torch.nn as nn

# CNN 예제
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 합성곱 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 정규화와 드롭아웃
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.25)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 모델 테스트
model = SimpleCNN()
input_tensor = torch.randn(1, 1, 28, 28)
output = model(input_tensor)
print(f"입력 shape: {input_tensor.shape}")
print(f"출력 shape: {output.shape}")
"""
                        }
                    }
                ]
            },
            "training": {
                "title": "모델 학습",
                "topics": [
                    {
                        "id": "loss_functions",
                        "title": "손실 함수",
                        "content": {
                            "설명": "다양한 손실 함수와 사용 시나리오를 학습합니다.",
                            "손실함수": {
                                "MSELoss": "회귀 문제",
                                "CrossEntropyLoss": "다중 분류",
                                "BCELoss": "이진 분류",
                                "L1Loss": "MAE 손실"
                            },
                            "예제": """
import torch
import torch.nn as nn

# 회귀 손실
mse_loss = nn.MSELoss()
predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item()}")

# 분류 손실
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(5, 10)  # 5개 샘플, 10개 클래스
labels = torch.randint(0, 10, (5,))  # 정답 레이블
loss = ce_loss(logits, labels)
print(f"Cross Entropy Loss: {loss.item()}")

# 커스텀 손실 함수
def custom_loss(output, target):
    return torch.mean((output - target) ** 2 + 0.01 * torch.abs(output))

output = torch.randn(10, 1, requires_grad=True)
target = torch.randn(10, 1)
loss = custom_loss(output, target)
print(f"Custom Loss: {loss.item()}")
"""
                        }
                    },
                    {
                        "id": "optimizers",
                        "title": "옵티마이저",
                        "content": {
                            "설명": "다양한 최적화 알고리즘을 이해하고 사용하는 방법을 학습합니다.",
                            "옵티마이저_종류": {
                                "SGD": "확률적 경사 하강법",
                                "Adam": "적응적 모멘트 추정",
                                "RMSprop": "Root Mean Square Propagation",
                                "AdamW": "Weight Decay가 개선된 Adam"
                            },
                            "예제": """
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델
model = nn.Linear(10, 1)

# 다양한 옵티마이저
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

# 학습 루프 예제
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    # 가상의 데이터
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 역전파 및 최적화
    optimizer.zero_grad()  # 그래디언트 초기화
    loss.backward()        # 역전파
    optimizer.step()       # 파라미터 업데이트
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(10):
    # 학습 코드...
    scheduler.step()
    print(f"Current LR: {scheduler.get_last_lr()}")
"""
                        }
                    },
                    {
                        "id": "training_loop",
                        "title": "완전한 학습 루프",
                        "content": {
                            "설명": "전체 학습 과정을 구현하는 방법을 학습합니다.",
                            "구성요소": [
                                "데이터 로딩",
                                "모델 정의",
                                "손실 함수와 옵티마이저",
                                "학습 루프",
                                "검증 루프",
                                "모델 저장/로드"
                            ],
                            "예제": """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 가상의 데이터셋
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000,))
X_val = torch.randn(200, 20)
y_val = torch.randint(0, 2, (200,))

# 데이터셋과 데이터로더
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# 모델, 손실 함수, 옵티마이저
model = BinaryClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (output > 0.5).float()
        correct += pred.eq(target.float()).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)

# 검증 함수
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            output = model(data).squeeze()
            loss = criterion(output, target.float())
            
            total_loss += loss.item()
            pred = (output > 0.5).float()
            correct += pred.eq(target.float()).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)

# 학습 루프
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")

# 모델 로드
model.load_state_dict(torch.load('best_model.pth'))
print("Model loaded!")
"""
                        }
                    }
                ]
            }
        }
        
        # 대화형 학습 프롬프트
        self.interactive_prompts = {
            "tensor_practice": [
                "torch.tensor([1, 2, 3])을 생성해보세요",
                "3x3 크기의 랜덤 텐서를 만들어보세요",
                "두 텐서를 더하는 코드를 작성해보세요"
            ],
            "autograd_practice": [
                "requires_grad=True인 텐서를 만들어보세요",
                "간단한 함수의 미분을 계산해보세요",
                "backward()를 호출하고 grad를 확인해보세요"
            ],
            "nn_practice": [
                "nn.Linear 레이어를 만들어보세요",
                "간단한 nn.Module 클래스를 정의해보세요",
                "forward 메서드를 구현해보세요"
            ]
        }
        
    def get_topic_explanation(self, topic_id: str) -> Dict[str, Any]:
        """특정 주제에 대한 설명을 반환"""
        for category, content in self.curriculum.items():
            for topic in content.get("topics", []):
                if topic["id"] == topic_id:
                    return {
                        "category": category,
                        "title": topic["title"],
                        "content": topic["content"],
                        "prerequisites": topic.get("prerequisites", [])
                    }
        return None
    
    def suggest_next_topic(self, current_topic: str) -> List[str]:
        """다음에 학습할 주제 추천"""
        suggestions = []
        
        # 현재 주제의 카테고리 찾기
        current_category = None
        current_index = None
        
        for category, content in self.curriculum.items():
            topics = content.get("topics", [])
            for i, topic in enumerate(topics):
                if topic["id"] == current_topic:
                    current_category = category
                    current_index = i
                    break
        
        if current_category and current_index is not None:
            topics = self.curriculum[current_category]["topics"]
            
            # 같은 카테고리의 다음 주제
            if current_index + 1 < len(topics):
                suggestions.append(topics[current_index + 1]["id"])
            
            # 다음 카테고리의 첫 주제
            categories = list(self.curriculum.keys())
            current_cat_index = categories.index(current_category)
            if current_cat_index + 1 < len(categories):
                next_category = categories[current_cat_index + 1]
                if self.curriculum[next_category]["topics"]:
                    suggestions.append(self.curriculum[next_category]["topics"][0]["id"])
        
        return suggestions
    
    def generate_practice_code(self, topic_id: str) -> str:
        """주제에 맞는 실습 코드 생성"""
        topic_info = self.get_topic_explanation(topic_id)
        if not topic_info:
            return None
            
        practice_template = f"""# {topic_info['title']} 실습

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 아래 코드를 완성해보세요

"""
        
        # 주제별 실습 코드 템플릿 추가
        if topic_id == "tensor_basics":
            practice_template += """# 1. 다양한 방법으로 텐서 생성하기
# TODO: 크기가 (3, 4)인 0으로 채워진 텐서를 생성하세요
zeros_tensor = # ???

# TODO: 크기가 (2, 3)인 1으로 채워진 텐서를 생성하세요
ones_tensor = # ???

# TODO: 0부터 9까지의 숫자로 텐서를 생성하세요
range_tensor = # ???

# 2. 텐서 연산
# TODO: 두 텐서를 element-wise로 곱하세요
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result = # ???

print(f"Zeros: {zeros_tensor}")
print(f"Ones: {ones_tensor}")
print(f"Range: {range_tensor}")
print(f"Multiplication: {result}")
"""
        elif topic_id == "autograd_basics":
            practice_template += """# 1. 자동 미분 연습
# TODO: x에 대해 requires_grad=True로 설정하세요
x = torch.tensor(3.0)  # ???

# TODO: y = x^2 + 2x + 1 함수를 정의하세요
y = # ???

# TODO: 역전파를 수행하세요
# ???

# TODO: x의 그래디언트를 출력하세요
print(f"dy/dx at x=3: {???}")

# 2. 더 복잡한 함수
# TODO: 두 변수 함수 z = x^2 + y^2의 그래디언트를 계산하세요
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = # ???

# ???

print(f"dz/dx: {???}")
print(f"dz/dy: {???}")
"""
        elif topic_id == "nn_module":
            practice_template += """# 1. 간단한 신경망 만들기
# TODO: 2층 신경망을 완성하세요
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        # TODO: 첫 번째 Linear 레이어를 정의하세요
        self.layer1 = # ???
        # TODO: 두 번째 Linear 레이어를 정의하세요
        self.layer2 = # ???
        
    def forward(self, x):
        # TODO: 첫 번째 레이어 통과 후 ReLU 적용
        x = # ???
        # TODO: 두 번째 레이어 통과
        x = # ???
        return x

# TODO: 입력 크기 10, 은닉층 크기 20, 출력 크기 5인 모델을 생성하세요
model = # ???

# 모델 테스트
test_input = torch.randn(1, 10)
output = model(test_input)
print(f"Model output shape: {output.shape}")
"""
        
        return practice_template
    
    def check_understanding(self, topic_id: str, user_code: str) -> Dict[str, Any]:
        """사용자의 이해도를 체크하고 피드백 제공"""
        feedback = {
            "correct": False,
            "hints": [],
            "suggestions": []
        }
        
        # 기본적인 구문 체크
        required_elements = {
            "tensor_basics": ["torch.zeros", "torch.ones", "torch.tensor"],
            "autograd_basics": ["requires_grad=True", "backward()"],
            "nn_module": ["nn.Module", "super()", "forward"]
        }
        
        if topic_id in required_elements:
            for element in required_elements[topic_id]:
                if element in user_code:
                    feedback["correct"] = True
                else:
                    feedback["hints"].append(f"'{element}'를 사용해보세요")
        
        # 주제별 추가 피드백
        if topic_id == "tensor_basics":
            if "torch.zeros((3, 4))" in user_code or "torch.zeros(3, 4)" in user_code:
                feedback["suggestions"].append("좋습니다! torch.zeros를 올바르게 사용했습니다.")
            
        elif topic_id == "autograd_basics":
            if "x.grad" in user_code:
                feedback["suggestions"].append("그래디언트를 확인하는 것 잘했습니다!")
            if "zero_grad()" not in user_code:
                feedback["hints"].append("실제 학습에서는 optimizer.zero_grad()도 중요합니다")
                
        return feedback
    
    def get_learning_path(self, skill_level: str = "beginner") -> List[str]:
        """사용자 수준에 맞는 학습 경로 제공"""
        if skill_level == "beginner":
            return [
                "pytorch_intro",
                "tensor_basics",
                "tensor_operations",
                "autograd_basics",
                "nn_module",
                "loss_functions",
                "optimizers",
                "training_loop"
            ]
        elif skill_level == "intermediate":
            return [
                "tensor_indexing",
                "computational_graph",
                "common_layers",
                "training_loop"
            ]
        else:  # advanced
            return [
                "computational_graph",
                "common_layers",
                "training_loop"
            ]
    
    def search_topic(self, query: str) -> List[Dict[str, Any]]:
        """키워드로 관련 주제 검색"""
        results = []
        query_lower = query.lower()
        
        for category, content in self.curriculum.items():
            for topic in content.get("topics", []):
                # 제목, 설명, 내용에서 검색
                if (query_lower in topic["title"].lower() or
                    query_lower in topic["content"].get("설명", "").lower() or
                    any(query_lower in str(v).lower() for v in topic["content"].values())):
                    
                    results.append({
                        "topic_id": topic["id"],
                        "title": topic["title"],
                        "category": category,
                        "category_title": content["title"]
                    })
        
        return results
    
    def format_response(self, topic_id: str, style: str = "detailed") -> str:
        """주제를 사용자 친화적인 형식으로 포맷팅"""
        topic_info = self.get_topic_explanation(topic_id)
        if not topic_info:
            return "해당 주제를 찾을 수 없습니다."
        
        response = f"📚 **{topic_info['title']}**\n\n"
        
        if style == "detailed":
            content = topic_info["content"]
            
            # 설명
            if "설명" in content:
                response += f"**📝 설명**\n{content['설명']}\n\n"
            
            # 특징이나 주요 개념
            for key in ["특징", "핵심개념", "주요개념", "연산_종류", "레이어_종류", "손실함수", "옵티마이저_종류"]:
                if key in content:
                    response += f"**✨ {key.replace('_', ' ').title()}**\n"
                    if isinstance(content[key], list):
                        for item in content[key]:
                            response += f"  • {item}\n"
                    elif isinstance(content[key], dict):
                        for k, v in content[key].items():
                            response += f"  • {k}: {v}\n"
                    response += "\n"
            
            # 예제 코드
            if "예제" in content:
                response += f"**💻 예제 코드**\n```python\n{content['예제'].strip()}\n```\n\n"
            
            # 다음 학습 주제 추천
            next_topics = self.suggest_next_topic(topic_id)
            if next_topics:
                response += "**🎯 다음 학습 추천**\n"
                for next_id in next_topics:
                    next_info = self.get_topic_explanation(next_id)
                    if next_info:
                        response += f"  • {next_info['title']} (`{next_id}`)\n"
        
        elif style == "summary":
            response += topic_info["content"].get("설명", "")
            
        return response