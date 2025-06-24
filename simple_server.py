from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers 라이브러리가 설치되지 않았습니다.")

app = FastAPI()

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    generated_text: str
    error: Optional[str] = None

def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers 라이브러리가 필요합니다. pip install transformers torch")
        return False
    
    try:
        model_path = "CodeLlama-7b-Instruct-hf"
        
        if not os.path.exists(model_path):
            print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
            print("먼저 모델을 다운로드해주세요:")
            print("python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='codellama/CodeLlama-7b-Instruct-hf', local_dir='./CodeLlama-7b-Instruct-hf')\"")
            return False
        
        print(f"🔄 모델 로딩 중: {model_path}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 모델 로드 (메모리 효율을 위한 설정)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print("✅ 모델 로딩 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Code Llama AI Server",
        "endpoints": {
            "/generate": "POST - 코드 생성",
            "/status": "GET - 서버 상태 확인",
            "/health": "GET - 헬스 체크"
        }
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    """코드 생성 API"""
    global model, tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise HTTPException(status_code=500, detail="transformers 라이브러리가 설치되지 않았습니다.")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    
    try:
        # C# 코드 생성을 위한 프롬프트 포맷팅
        formatted_prompt = f"### Instruction:\n{request.prompt}\n\n### Response:\n"
        
        # 토큰화
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 코드 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + request.max_length,
                temperature=request.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 원본 프롬프트 제거
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return GenerateResponse(generated_text=generated_text)
        
    except Exception as e:
        return GenerateResponse(
            generated_text="",
            error=f"코드 생성 중 오류 발생: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """서버 상태 확인"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "cuda_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
        "device": "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    if model is not None and tokenizer is not None:
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=503, detail="Service unavailable")

# 개발 중 사용할 수 있는 테스트 엔드포인트
@app.post("/test")
async def test_generation():
    """간단한 테스트 생성"""
    test_prompt = "Create a simple Unity player controller script in C#"
    request = GenerateRequest(prompt=test_prompt)
    return await generate_code(request)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Code Llama AI Server 시작...")
    print("📍 API 문서: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)