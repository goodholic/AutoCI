from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    generated_text: str

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    model_path = "/mnt/c/Users/super/Desktop/Unity Project(25년도 제작)/26.AutoCI/AutoCI/CodeLlama-7b-Instruct-hf"
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    return {"message": "C# Code Generator API", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt for Code Llama Instruct
        system_prompt = "You are an expert C# developer. Generate clean, efficient, and well-commented C# code."
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{request.prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after [/INST])
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[-1].strip()
        
        return GenerateResponse(generated_text=generated_text)
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)