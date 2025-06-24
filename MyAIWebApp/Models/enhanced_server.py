#!/usr/bin/env python3
"""
향상된 AI 모델 서버
Code Llama 7B-Instruct 기반 C# 코드 생성 및 개선 API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
import logging
import json
import asyncio
from datetime import datetime
from pathlib import Path
import re
from collections import deque
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="AutoCI AI Model Server",
    description="Code Llama 7B-Instruct 기반 C# 전문가 AI 서비스",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
tokenizer = None
device = None
model_config = {
    "model_path": "../../CodeLlama-7b-Instruct-hf",
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1
}

# 요청 히스토리 (간단한 캐싱)
request_history = deque(maxlen=100)

# Pydantic 모델
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="코드 생성을 위한 프롬프트")
    max_tokens: Optional[int] = Field(512, description="생성할 최대 토큰 수")
    temperature: Optional[float] = Field(0.7, description="생성 온도 (0.0-1.0)")
    language: Optional[str] = Field("csharp", description="프로그래밍 언어")
    context: Optional[str] = Field(None, description="추가 컨텍스트")

class GenerateResponse(BaseModel):
    generated_code: str
    tokens_used: int
    generation_time: float
    language: str
    suggestions: Optional[List[str]] = None

class ImproveRequest(BaseModel):
    code: str = Field(..., description="개선할 코드")
    language: str = Field("csharp", description="프로그래밍 언어")
    context: Optional[str] = Field(None, description="코드 컨텍스트")
    improvement_type: Optional[str] = Field("general", description="개선 유형")

class ImproveResponse(BaseModel):
    improved_code: str
    suggestions: List[str]
    quality_score: float
    improvements_made: List[str]

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="분석할 코드")
    language: str = Field("csharp", description="프로그래밍 언어")

class AnalyzeResponse(BaseModel):
    complexity: str
    patterns_found: List[str]
    potential_issues: List[str]
    best_practices: List[str]
    quality_score: float

# 커스텀 Stopping Criteria
class CSharpStoppingCriteria(StoppingCriteria):
    """C# 코드 생성을 위한 중단 기준"""
    
    def __init__(self, tokenizer, stop_sequences=["\n\n\n", "```\n", "### Instruction:"]):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        
    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][-50:], skip_special_tokens=True)
        return any(stop_seq in generated_text for stop_seq in self.stop_sequences)

# 모델 로드 함수
async def load_model():
    """모델과 토크나이저 로드"""
    global model, tokenizer, device
    
    logger.info("🚀 Code Llama 7B-Instruct 모델 로드 중...")
    
    try:
        # 디바이스 설정
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("⚠️  CPU 사용 (느릴 수 있습니다)")
        
        # 모델 경로 확인
        model_path = Path(model_config["model_path"])
        if not model_path.exists():
            # 상대 경로 시도
            model_path = Path(__file__).parent / model_config["model_path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        tokenizer.pad_token = tokenizer.eos_token
        
        # 모델 로드 (8-bit quantization 옵션)
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True
        }
        
        # 8-bit 로드 시도
        try:
            model_kwargs["load_in_8bit"] = True
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
            logger.info("✅ 8-bit 양자화로 모델 로드 완료")
        except:
            # 일반 로드
            model_kwargs.pop("load_in_8bit", None)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
            logger.info("✅ 모델 로드 완료")
        
        if device.type == "cpu":
            model = model.to(device)
        
        # 모델 정보
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"📊 모델 파라미터: {param_count:.1f}B")
        
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {str(e)}")
        raise

# 시작 시 모델 로드
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    await load_model()
    logger.info("✅ AI 모델 서버 준비 완료!")

# API 엔드포인트
@app.get("/")
async def root():
    """API 루트"""
    return {
        "service": "AutoCI AI Model Server",
        "version": "2.0.0",
        "model": "Code Llama 7B-Instruct",
        "status": "ready" if model is not None else "loading",
        "endpoints": {
            "generate": "/generate",
            "improve": "/improve",
            "analyze": "/analyze",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    """코드 생성 API"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")
    
    start_time = time.time()
    
    try:
        # C# 특화 프롬프트 구성
        if request.language.lower() == "csharp":
            system_prompt = """You are an expert C# developer with deep knowledge of:
- .NET Core/5/6/7/8
- ASP.NET Core
- Entity Framework Core
- LINQ
- Async/Await patterns
- SOLID principles
- Design patterns
- Best practices and modern C# features

Generate clean, efficient, and well-documented C# code."""
        else:
            system_prompt = f"You are an expert {request.language} developer."
        
        # 전체 프롬프트 구성
        full_prompt = f"""### System:
{system_prompt}

### Instruction:
{request.prompt}

{f'### Context: {request.context}' if request.context else ''}

### Response:
```{request.language}
"""
        
        # 토큰화
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device:
            inputs = inputs.to(device)
        
        # Stopping criteria 설정
        stopping_criteria = StoppingCriteriaList([
            CSharpStoppingCriteria(tokenizer)
        ])
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=model_config["top_p"],
                top_k=model_config["top_k"],
                repetition_penalty=model_config["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                do_sample=True
            )
        
        # 생성된 텍스트 추출
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거하고 생성된 코드만 추출
        code_start = generated.find(f"```{request.language}")
        if code_start != -1:
            code_start += len(f"```{request.language}") + 1
            code_end = generated.find("```", code_start)
            if code_end != -1:
                generated_code = generated[code_start:code_end].strip()
            else:
                generated_code = generated[code_start:].strip()
        else:
            # 프롬프트 이후 부분 추출
            response_start = generated.find("### Response:")
            if response_start != -1:
                generated_code = generated[response_start + 13:].strip()
            else:
                generated_code = generated[len(full_prompt):].strip()
        
        # 코드 정리
        generated_code = clean_generated_code(generated_code, request.language)
        
        # 제안사항 생성
        suggestions = generate_suggestions(generated_code, request.language)
        
        # 응답
        generation_time = time.time() - start_time
        tokens_used = len(outputs[0]) - len(inputs.input_ids[0])
        
        # 히스토리에 추가
        request_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": request.prompt,
            "language": request.language,
            "tokens": tokens_used,
            "time": generation_time
        })
        
        return GenerateResponse(
            generated_code=generated_code,
            tokens_used=tokens_used,
            generation_time=round(generation_time, 2),
            language=request.language,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"코드 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improve", response_model=ImproveResponse)
async def improve_code(request: ImproveRequest):
    """코드 개선 API"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")
    
    try:
        # 코드 분석
        quality_score, issues = analyze_code_quality(request.code, request.language)
        
        # 개선 프롬프트 구성
        improvement_prompt = f"""### Instruction:
Improve the following {request.language} code by:
1. Following best practices and design patterns
2. Improving error handling
3. Optimizing performance
4. Adding proper documentation
5. Ensuring SOLID principles
6. Using modern language features

Original code:
```{request.language}
{request.code}
```

{f'Context: {request.context}' if request.context else ''}

Provide the improved version with explanations.

### Response:
Improved code:
```{request.language}
"""
        
        # 생성
        response = await generate_code(GenerateRequest(
            prompt=improvement_prompt,
            max_tokens=1024,
            temperature=0.5,  # 더 보수적인 생성
            language=request.language
        ))
        
        # 개선사항 추출
        improvements_made = extract_improvements(request.code, response.generated_code)
        
        # 제안사항 생성
        suggestions = [
            "Consider using dependency injection for better testability",
            "Add XML documentation comments for public APIs",
            "Implement proper logging for production use",
            "Consider async/await for I/O operations",
            "Add unit tests for critical business logic"
        ]
        
        # 품질 점수 재계산
        new_quality_score, _ = analyze_code_quality(response.generated_code, request.language)
        
        return ImproveResponse(
            improved_code=response.generated_code,
            suggestions=suggestions[:5],  # 상위 5개만
            quality_score=new_quality_score,
            improvements_made=improvements_made
        )
        
    except Exception as e:
        logger.error(f"코드 개선 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code(request: AnalyzeRequest):
    """코드 분석 API"""
    try:
        # 코드 복잡도 분석
        complexity = analyze_complexity(request.code)
        
        # 패턴 감지
        patterns = detect_patterns(request.code, request.language)
        
        # 잠재적 이슈
        potential_issues = detect_issues(request.code, request.language)
        
        # 베스트 프랙티스
        best_practices = check_best_practices(request.code, request.language)
        
        # 품질 점수
        quality_score, _ = analyze_code_quality(request.code, request.language)
        
        return AnalyzeResponse(
            complexity=complexity,
            patterns_found=patterns,
            potential_issues=potential_issues,
            best_practices=best_practices,
            quality_score=quality_score
        )
        
    except Exception as e:
        logger.error(f"코드 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 헬퍼 함수들
def clean_generated_code(code: str, language: str) -> str:
    """생성된 코드 정리"""
    # 불필요한 마크다운 제거
    code = re.sub(r'^```\w*\n?', '', code)
    code = re.sub(r'\n?```$', '', code)
    
    # 중복된 빈 줄 제거
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    # 언어별 특수 처리
    if language.lower() == "csharp":
        # using 문 정리
        lines = code.split('\n')
        using_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith('using ') and line.strip().endswith(';'):
                using_lines.append(line.strip())
            else:
                other_lines.append(line)
        
        # using 문 정렬
        using_lines = sorted(list(set(using_lines)))
        
        if using_lines:
            code = '\n'.join(using_lines) + '\n\n' + '\n'.join(other_lines)
    
    return code.strip()

def generate_suggestions(code: str, language: str) -> List[str]:
    """코드에 대한 제안사항 생성"""
    suggestions = []
    
    if language.lower() == "csharp":
        # null 체크 제안
        if 'null' not in code and ('string' in code or 'object' in code):
            suggestions.append("Consider adding null checks for reference types")
        
        # async/await 제안
        if 'Task' in code and 'async' not in code:
            suggestions.append("Consider using async/await for asynchronous operations")
        
        # LINQ 제안
        if 'for' in code and 'List' in code:
            suggestions.append("Consider using LINQ for collection operations")
        
        # 예외 처리 제안
        if 'try' not in code and ('File' in code or 'Database' in code or 'Http' in code):
            suggestions.append("Add try-catch blocks for potential exceptions")
        
        # 인터페이스 제안
        if 'class' in code and 'interface' not in code:
            suggestions.append("Consider defining interfaces for better abstraction")
    
    return suggestions

def analyze_code_quality(code: str, language: str) -> tuple[float, List[str]]:
    """코드 품질 분석"""
    score = 1.0
    issues = []
    
    if language.lower() == "csharp":
        # 주석 확인
        if '///' not in code and 'public' in code:
            score -= 0.1
            issues.append("Missing XML documentation comments")
        
        # 에러 처리
        if 'try' not in code and ('Exception' in code or 'throw' in code):
            score -= 0.15
            issues.append("Missing proper error handling")
        
        # 네이밍 컨벤션
        if re.search(r'[a-z][A-Z]', code):  # camelCase in wrong places
            score -= 0.05
            issues.append("Potential naming convention issues")
        
        # 매직 넘버
        if re.search(r'[^0-9][0-9]{2,}[^0-9]', code):
            score -= 0.05
            issues.append("Magic numbers detected")
        
        # 긴 메서드
        lines = code.split('\n')
        method_lines = 0
        in_method = False
        for line in lines:
            if '{' in line and ('public' in line or 'private' in line or 'protected' in line):
                in_method = True
                method_lines = 0
            elif in_method:
                method_lines += 1
                if method_lines > 50:
                    score -= 0.1
                    issues.append("Methods too long")
                    break
            elif '}' in line and in_method:
                in_method = False
    
    return max(0.0, score), issues

def analyze_complexity(code: str) -> str:
    """코드 복잡도 분석"""
    lines = code.split('\n')
    
    # 기본 메트릭
    loc = len([l for l in lines if l.strip()])
    
    # 사이클로매틱 복잡도 간단 추정
    complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'catch']
    complexity_count = sum(1 for line in lines for keyword in complexity_keywords if keyword in line)
    
    if complexity_count < 5:
        return "Low"
    elif complexity_count < 10:
        return "Medium"
    else:
        return "High"

def detect_patterns(code: str, language: str) -> List[str]:
    """디자인 패턴 감지"""
    patterns = []
    
    if language.lower() == "csharp":
        # Singleton
        if 'private static' in code and 'Instance' in code:
            patterns.append("Singleton Pattern")
        
        # Factory
        if 'Create' in code and 'return new' in code:
            patterns.append("Factory Pattern")
        
        # Repository
        if 'Repository' in code or ('Add' in code and 'Get' in code and 'Delete' in code):
            patterns.append("Repository Pattern")
        
        # Observer
        if 'event' in code or 'EventHandler' in code:
            patterns.append("Observer Pattern")
        
        # Dependency Injection
        if 'interface' in code and 'constructor' in code:
            patterns.append("Dependency Injection")
    
    return patterns

def detect_issues(code: str, language: str) -> List[str]:
    """잠재적 이슈 감지"""
    issues = []
    
    if language.lower() == "csharp":
        # SQL Injection 위험
        if 'SELECT' in code and '+' in code:
            issues.append("Potential SQL injection vulnerability")
        
        # 하드코딩된 비밀
        if re.search(r'(password|key|secret)\s*=\s*"[^"]+"', code, re.IGNORECASE):
            issues.append("Hardcoded secrets detected")
        
        # 메모리 누수 가능성
        if 'IDisposable' in code and 'using' not in code:
            issues.append("Potential memory leak - IDisposable not properly handled")
        
        # 무한 루프 가능성
        if 'while (true)' in code and 'break' not in code:
            issues.append("Potential infinite loop")
    
    return issues

def check_best_practices(code: str, language: str) -> List[str]:
    """베스트 프랙티스 체크"""
    practices = []
    
    if language.lower() == "csharp":
        # 좋은 practices 확인
        if '///' in code:
            practices.append("✓ XML documentation comments used")
        
        if 'async' in code and 'await' in code:
            practices.append("✓ Async/await pattern properly used")
        
        if 'try' in code and 'catch' in code:
            practices.append("✓ Proper error handling implemented")
        
        if 'interface' in code:
            practices.append("✓ Interface-based design")
        
        if 'readonly' in code:
            practices.append("✓ Immutability considered")
        
        # 개선 필요
        if 'var' not in code and language == "csharp":
            practices.append("⚠ Consider using 'var' for type inference")
        
        if 'string.Format' in code:
            practices.append("⚠ Consider using string interpolation ($\"\")")
    
    return practices

def extract_improvements(original: str, improved: str) -> List[str]:
    """개선사항 추출"""
    improvements = []
    
    # 간단한 비교
    if 'async' in improved and 'async' not in original:
        improvements.append("Added async/await for asynchronous operations")
    
    if 'try' in improved and 'try' not in original:
        improvements.append("Added error handling")
    
    if '///' in improved and '///' not in original:
        improvements.append("Added XML documentation comments")
    
    if 'interface' in improved and 'interface' not in original:
        improvements.append("Introduced interface-based design")
    
    if improved.count('\n') > original.count('\n') * 1.2:
        improvements.append("Improved code structure and readability")
    
    return improvements

@app.get("/stats")
async def get_stats():
    """서버 통계"""
    return {
        "total_requests": len(request_history),
        "recent_requests": list(request_history)[-10:],
        "model_info": {
            "name": "Code Llama 7B-Instruct",
            "loaded": model is not None,
            "device": str(device) if device else "not set"
        },
        "server_uptime": "N/A"  # 구현 필요
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)