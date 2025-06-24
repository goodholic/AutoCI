import os
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers 라이브러리가 설치되지 않았습니다.")

model = None
tokenizer = None
last_model_update = 0

def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers 라이브러리가 필요합니다.")
        return
    
    try:
        model_path = "CodeLlama-7b-Instruct-hf"
        if os.path.exists(model_path):
            print(f"모델 로딩 중: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("✅ 모델 로딩 완료!")
        else:
            print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")

def check_model_update():
    """모델 업데이트 확인"""
    global model, tokenizer, last_model_update
    
    while True:
        if os.path.exists("model_updated.signal"):
            signal_time = os.path.getmtime("model_updated.signal")
            if signal_time > last_model_update:
                print("🔄 새로운 모델 감지! 리로딩...")
                load_model()
                last_model_update = signal_time
                os.remove("model_updated.signal")
        
        time.sleep(60)  # 1분마다 체크

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작
    load_model()
    
    # 모델 업데이트 체크 스레드 시작
    update_thread = threading.Thread(target=check_model_update, daemon=True)
    update_thread.start()
    
    yield
    # 종료

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_code(request: dict):
    """코드 생성 API"""
    global model, tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "transformers 라이브러리가 설치되지 않았습니다."}
    
    if model is None or tokenizer is None:
        return {"error": "모델이 로드되지 않았습니다."}
    
    try:
        prompt = request.get("prompt", "")
        
        # 입력 토큰화
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # 코드 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 원본 프롬프트 제거
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return {"generated_text": generated_text}
        
    except Exception as e:
        return {"error": f"코드 생성 중 오류 발생: {str(e)}"}

@app.get("/status")
async def get_status():
    """서버 상태 확인"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "transformers_available": TRANSFORMERS_AVAILABLE
    }