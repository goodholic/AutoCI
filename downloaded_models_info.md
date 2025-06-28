# AutoCI 다운로드된 모델 정보

## 다운로드 완료: 1/3

### 사용 가능한 모델들:
- ✅ Llama 3.1 70B Instruct - ./Llama-3.1-70B-Instruct
- ✅ Qwen2.5 72B Instruct - ./Qwen2.5-72B-Instruct
- ✅ DeepSeek V2.5 - ./DeepSeek-V2.5

### 사용법:
```python
# 모델 로드 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 3.1 70B (예시)
tokenizer = AutoTokenizer.from_pretrained("./Llama-3.1-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "./Llama-3.1-70B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # 메모리 절약
)
```

### 메모리 사용량:
- 4-bit 양자화 사용 시 모델당 ~25-30GB RAM
- 32GB RAM에서 1개 모델 동시 실행 권장
- GPU 메모리: 8GB+ 권장

### 성능 최적화:
- `load_in_4bit=True` 사용
- `device_map="auto"` 사용
- 필요시 CPU 오프로딩 활용
