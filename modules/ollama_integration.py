#!/usr/bin/env python3
"""
Ollama Integration Module for AutoCI
Ollama를 통한 로컬 LLM 실행 및 관리
"""

import os
import json
import subprocess
import requests
from typing import Optional, Dict, List, Any
import asyncio
import aiohttp
from dataclasses import dataclass
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OllamaModel:
    """Ollama 모델 정보"""
    name: str
    size: str
    digest: str
    modified_at: str

class OllamaManager:
    """Ollama 서비스 관리 및 모델 실행"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def check_ollama_installed(self) -> bool:
        """Ollama 설치 확인"""
        try:
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def install_ollama(self) -> bool:
        """Ollama 자동 설치 (WSL 지원)"""
        try:
            logger.info("Ollama 설치 중...")
            cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Ollama 설치 완료")
                # Ollama 서비스 시작
                subprocess.run(['ollama', 'serve'], capture_output=True, detach=True)
                return True
            else:
                logger.error(f"Ollama 설치 실패: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ollama 설치 중 오류: {e}")
            return False
    
    def start_service(self) -> bool:
        """Ollama 서비스 시작"""
        try:
            # 이미 실행 중인지 확인
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info("Ollama 서비스가 이미 실행 중입니다.")
                return True
        except:
            pass
        
        try:
            # 백그라운드에서 Ollama 서비스 시작
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            logger.info("Ollama 서비스 시작 중...")
            
            # 서비스 시작 대기
            for _ in range(30):
                try:
                    response = requests.get(f"{self.base_url}/api/tags", timeout=1)
                    if response.status_code == 200:
                        logger.info("Ollama 서비스 시작 완료")
                        return True
                except:
                    pass
                asyncio.sleep(1)
            
            return False
        except Exception as e:
            logger.error(f"Ollama 서비스 시작 실패: {e}")
            return False
    
    def list_models(self) -> List[OllamaModel]:
        """설치된 모델 목록"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model_data in data.get('models', []):
                    model = OllamaModel(
                        name=model_data['name'],
                        size=model_data.get('size', 'Unknown'),
                        digest=model_data.get('digest', ''),
                        modified_at=model_data.get('modified_at', '')
                    )
                    models.append(model)
                return models
            return []
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """모델 다운로드"""
        try:
            logger.info(f"{model_name} 모델 다운로드 중...")
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"{model_name} 모델 다운로드 완료")
                return True
            else:
                logger.error(f"모델 다운로드 실패: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"모델 다운로드 중 오류: {e}")
            return False
    
    async def generate(self, 
                      model: str, 
                      prompt: str, 
                      system: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 2048) -> str:
        """비동기 텍스트 생성"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/generate", 
                                      json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        error = await response.text()
                        logger.error(f"생성 실패: {error}")
                        return ""
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {e}")
            return ""
    
    def generate_sync(self,
                     model: str,
                     prompt: str,
                     system: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: int = 2048) -> str:
        """동기식 텍스트 생성"""
        return asyncio.run(self.generate(model, prompt, system, temperature, max_tokens))

class AutoCILlamaInterface:
    """AutoCI와 Ollama/Llama 통합 인터페이스"""
    
    def __init__(self):
        self.ollama = OllamaManager()
        self.models = {
            'code': 'codellama:7b-instruct',
            'general': 'llama2:7b',
            'python': 'codellama:7b-python'
        }
        self.setup()
    
    def setup(self):
        """초기 설정"""
        # Ollama 설치 확인 및 설치
        if not self.ollama.check_ollama_installed():
            logger.info("Ollama가 설치되어 있지 않습니다. 설치를 시작합니다...")
            if not self.ollama.install_ollama():
                raise Exception("Ollama 설치 실패")
        
        # 서비스 시작
        if not self.ollama.start_service():
            raise Exception("Ollama 서비스 시작 실패")
        
        # 필수 모델 다운로드
        installed_models = [m.name for m in self.ollama.list_models()]
        for model_type, model_name in self.models.items():
            if model_name not in installed_models:
                logger.info(f"{model_name} 모델이 없습니다. 다운로드를 시작합니다...")
                self.ollama.pull_model(model_name)
    
    def generate_code(self, 
                     prompt: str, 
                     language: str = "python",
                     context: Optional[str] = None) -> str:
        """코드 생성"""
        system_prompt = f"""You are an expert {language} programmer. 
Generate clean, efficient, and well-commented code.
Follow best practices and design patterns."""
        
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        model = self.models.get('python' if language == 'python' else 'code')
        return self.ollama.generate_sync(
            model=model,
            prompt=prompt,
            system=system_prompt,
            temperature=0.3  # 코드 생성은 낮은 temperature
        )
    
    def analyze_code(self, code: str, task: str = "analyze") -> str:
        """코드 분석"""
        prompt = f"""Analyze the following code and {task}:

```
{code}
```

Provide detailed analysis including:
1. Code quality
2. Potential issues
3. Optimization suggestions
4. Best practices"""
        
        return self.ollama.generate_sync(
            model=self.models['code'],
            prompt=prompt,
            temperature=0.5
        )
    
    def game_design_assistant(self, prompt: str) -> str:
        """게임 디자인 어시스턴트"""
        system_prompt = """You are an experienced game designer and developer.
Help design game mechanics, balance systems, and create engaging gameplay.
Consider technical feasibility and player experience."""
        
        return self.ollama.generate_sync(
            model=self.models['general'],
            prompt=prompt,
            system=system_prompt,
            temperature=0.8  # 창의적인 답변을 위해 높은 temperature
        )
    
    def create_godot_script(self, 
                           description: str,
                           node_type: str = "Node2D") -> str:
        """Godot GDScript 생성"""
        prompt = f"""Create a Godot GDScript for a {node_type} that:
{description}

Include:
- Proper extends declaration
- Export variables for editor configuration
- Signal definitions if needed
- Well-structured functions
- Comments explaining the logic"""
        
        code = self.generate_code(prompt, language="gdscript")
        return code
    
    def optimize_for_autoci(self, code: str, target: str = "performance") -> str:
        """AutoCI 환경에 최적화된 코드 생성"""
        prompt = f"""Optimize the following code for {target} in the AutoCI environment:

```
{code}
```

Consider:
- WSL environment constraints
- 24-hour continuous operation
- Memory efficiency
- Error handling and recovery
- Logging and monitoring"""
        
        return self.ollama.generate_sync(
            model=self.models['code'],
            prompt=prompt,
            temperature=0.4
        )

# 전역 인스턴스
_llama_interface = None

def get_llama_interface() -> AutoCILlamaInterface:
    """싱글톤 인터페이스 반환"""
    global _llama_interface
    if _llama_interface is None:
        _llama_interface = AutoCILlamaInterface()
    return _llama_interface

# 사용 예제
if __name__ == "__main__":
    # 인터페이스 초기화
    llama = get_llama_interface()
    
    # 코드 생성 예제
    code = llama.generate_code(
        "Create a function to calculate fibonacci numbers with memoization",
        language="python"
    )
    print("Generated Code:")
    print(code)
    
    # Godot 스크립트 생성 예제
    godot_script = llama.create_godot_script(
        "A player character that can move with arrow keys and jump with space",
        node_type="CharacterBody2D"
    )
    print("\nGodot Script:")
    print(godot_script)
    
    # 게임 디자인 어시스턴트 예제
    game_idea = llama.game_design_assistant(
        "Design a unique puzzle mechanic for a 2D platformer"
    )
    print("\nGame Design Idea:")
    print(game_idea)