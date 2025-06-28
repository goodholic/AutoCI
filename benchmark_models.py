#!/usr/bin/env python3
"""
AutoCI v3.0 - AI 모델 성능 벤치마크
다운로드된 모델들의 성능을 테스트하고 비교합니다.
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime
from pathlib import Path

print("📦 필요한 패키지 확인 중...")

try:
    import torch
    print("✅ PyTorch 로드 완료")
except ImportError:
    print("❌ PyTorch가 설치되어 있지 않습니다.")
    print("📦 설치: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Transformers 로드 완료")
except ImportError:
    print("❌ Transformers가 설치되어 있지 않습니다.")
    print("📦 설치: pip install transformers")
    sys.exit(1)

try:
    import nvidia_ml_py3 as nvml
    gpu_support = True
    print("✅ NVIDIA GPU 지원 가능")
except ImportError:
    gpu_support = False
    print("⚠️ NVIDIA GPU 모니터링 불가 (nvidia-ml-py3 미설치)")

class ModelBenchmark:
    """AI 모델 벤치마크 클래스"""
    
    def __init__(self):
        self.results = {}
        self.test_prompts = [
            "간단한 2D 플랫포머 게임을 만들어줘",
            "C# 스크립트로 플레이어 컨트롤러를 작성해줘", 
            "게임에 점프 메커니즘을 추가하는 방법을 알려줘",
            "유니티에서 적 AI를 구현하는 코드를 보여줘",
            "한국어로 게임 개발 팁을 설명해줘"
        ]
        
        # GPU 정보 초기화
        self.gpu_available = False
        self.gpu_count = 0
        
        if gpu_support:
            try:
                nvml.nvmlInit()
                self.gpu_available = True
                self.gpu_count = nvml.nvmlDeviceGetCount()
            except:
                self.gpu_available = False
    
    def get_system_info(self):
        """시스템 정보 수집"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
            "memory_available": psutil.virtual_memory().available // (1024**3),  # GB
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available()
        }
        
        if self.gpu_available and gpu_support:
            gpu_info = []
            for i in range(self.gpu_count):
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode()
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.append({
                        "name": name,
                        "memory_total": memory.total // (1024**2),  # MB
                        "memory_free": memory.free // (1024**2)   # MB
                    })
                except Exception as e:
                    gpu_info.append({"error": str(e)})
            info["gpu_info"] = gpu_info
        
        return info
    
    def find_available_models(self):
        """사용 가능한 모델 찾기"""
        models = {}
        
        # 기본 모델 확인
        if os.path.exists("CodeLlama-7b-Instruct-hf"):
            models["CodeLlama-7B"] = {
                "path": "CodeLlama-7b-Instruct-hf",
                "size": "7B",
                "type": "코딩 전문"
            }
        
        # 고급 모델들 확인
        model_paths = {
            "Llama-3.1-70B": {
                "path": "models/Llama-3.1-70B-Instruct",
                "size": "70B", 
                "type": "고급 추론"
            },
            "Qwen2.5-72B": {
                "path": "models/Qwen2.5-72B-Instruct",
                "size": "72B",
                "type": "한국어 최적화"
            },
            "DeepSeek-V2.5": {
                "path": "models/DeepSeek-V2.5", 
                "size": "236B",
                "type": "코딩 전문"
            }
        }
        
        for name, info in model_paths.items():
            if os.path.exists(info["path"]):
                models[name] = info
        
        return models
    
    def benchmark_model(self, model_name, model_path):
        """개별 모델 벤치마크"""
        print(f"\n🧠 {model_name} 벤치마크 시작...")
        
        result = {
            "model_name": model_name,
            "model_path": model_path,
            "tests": [],
            "errors": []
        }
        
        try:
            # 모델 로딩 시간 측정
            load_start = time.time()
            
            print(f"📥 토크나이저 로딩 중...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print(f"📥 모델 로딩 중 (4-bit 양자화)...")
            
            # 모델 크기에 따라 다른 로딩 전략 사용
            if "7B" in model_name:
                # 작은 모델은 일반 로딩
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # 큰 모델은 4-bit 양자화
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True
                )
            
            load_time = time.time() - load_start
            result["load_time"] = load_time
            
            print(f"✅ 모델 로딩 완료 ({load_time:.2f}초)")
            
            # 메모리 사용량 체크
            memory_used = psutil.virtual_memory().used // (1024**3)  # GB
            result["memory_used_gb"] = memory_used
            
            # 간단한 테스트만 실행 (시간 절약)
            test_prompt = self.test_prompts[0]  # 첫 번째 프롬프트만 사용
            print(f"📝 테스트: {test_prompt[:30]}...")
            
            test_result = self.run_single_test(model, tokenizer, test_prompt)
            result["tests"].append(test_result)
            result["average_response_time"] = test_result["response_time"]
            result["average_tokens_per_second"] = test_result["tokens_per_second"]
            
            print(f"   응답 시간: {test_result['response_time']:.2f}초")
            print(f"   토큰 속도: {test_result['tokens_per_second']:.1f} tokens/sec")
            print(f"   메모리 사용: {memory_used}GB")
            
            # 메모리 정리
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = f"❌ {model_name} 벤치마크 실패: {str(e)}"
            print(error_msg)
            result["errors"].append(error_msg)
        
        return result
    
    def run_single_test(self, model, tokenizer, prompt):
        """단일 테스트 실행"""
        start_time = time.time()
        
        try:
            # 토큰화
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(model.device)
            
            input_length = inputs.shape[1]
            
            # 생성 (짧게)
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=input_length + 20,  # 20개 토큰만 생성 (빠른 테스트)
                    do_sample=False,  # 결정적 생성
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 응답 시간 계산
            response_time = time.time() - start_time
            
            # 생성된 토큰 수 계산
            generated_length = outputs.shape[1] - input_length
            tokens_per_second = generated_length / response_time if response_time > 0 else 0
            
            # 생성된 텍스트 디코딩
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return {
                "prompt": prompt,
                "response_time": response_time,
                "generated_tokens": generated_length,
                "tokens_per_second": tokens_per_second,
                "generated_text": generated_text[:50] + "..." if len(generated_text) > 50 else generated_text
            }
            
        except Exception as e:
            return {
                "prompt": prompt,
                "response_time": 999.0,
                "generated_tokens": 0,
                "tokens_per_second": 0.0,
                "error": str(e)
            }
    
    def run_benchmark(self):
        """전체 벤치마크 실행"""
        print("🚀 AutoCI v3.0 - AI 모델 성능 벤치마크")
        print("=" * 50)
        print()
        
        # 시스템 정보 출력
        system_info = self.get_system_info()
        print("💻 시스템 정보:")
        print(f"   CPU: {system_info['cpu_count']}코어")
        print(f"   메모리: {system_info['memory_total']}GB (사용 가능: {system_info['memory_available']}GB)")
        print(f"   CUDA: {'사용 가능' if system_info['cuda_available'] else '사용 불가'}")
        
        if system_info['gpu_available'] and 'gpu_info' in system_info:
            for i, gpu in enumerate(system_info['gpu_info']):
                if 'error' not in gpu:
                    print(f"   GPU {i}: {gpu['name']} ({gpu['memory_total']}MB)")
        print()
        
        # 사용 가능한 모델 찾기
        available_models = self.find_available_models()
        
        if not available_models:
            print("❌ 사용 가능한 모델이 없습니다.")
            print("💡 먼저 다음 명령어로 모델을 다운로드하세요:")
            print("   ./download_advanced_models.sh")
            print("   또는 기본 모델: git clone https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf")
            return
        
        print(f"📋 발견된 모델: {len(available_models)}개")
        for name, info in available_models.items():
            print(f"   • {name} ({info['size']}) - {info['type']}")
        print()
        
        # 각 모델 벤치마크 실행
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "model_results": []
        }
        
        for model_name, model_info in available_models.items():
            result = self.benchmark_model(model_name, model_info["path"])
            result.update(model_info)
            benchmark_results["model_results"].append(result)
        
        # 결과 저장
        self.save_results(benchmark_results)
        
        # 결과 요약 출력
        self.print_summary(benchmark_results)
    
    def save_results(self, results):
        """결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 벤치마크 결과 저장: {filename}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("📊 벤치마크 결과 요약")
        print("=" * 60)
        
        model_results = results["model_results"]
        
        if not model_results:
            print("❌ 성공한 벤치마크가 없습니다.")
            return
        
        # 성능 순으로 정렬 (토큰/초 기준)
        successful_results = [r for r in model_results if "average_tokens_per_second" in r and r["average_tokens_per_second"] > 0]
        if successful_results:
            successful_results.sort(key=lambda x: x["average_tokens_per_second"], reverse=True)
        
        if successful_results:
            print("\n🏆 성능 순위 (토큰/초 기준):")
            for i, result in enumerate(successful_results, 1):
                print(f"{i}. {result['model_name']} ({result['size']})")
                print(f"   • 토큰 속도: {result['average_tokens_per_second']:.1f} tokens/sec")
                print(f"   • 응답 시간: {result['average_response_time']:.2f}초")
                print(f"   • 메모리 사용: {result['memory_used_gb']}GB")
                print(f"   • 로딩 시간: {result['load_time']:.2f}초")
                print(f"   • 특화 분야: {result['type']}")
                print()
        
        # 실패한 모델들
        failed_results = [r for r in model_results if r.get("errors")]
        if failed_results:
            print("❌ 실패한 모델들:")
            for result in failed_results:
                print(f"   • {result['model_name']}: {result['errors'][0] if result['errors'] else 'Unknown error'}")
            print()
        
        # 권장사항
        print("💡 권장사항:")
        memory_total = results["system_info"]["memory_total"]
        
        if successful_results:
            best_model = successful_results[0]
            print(f"   • 최고 성능: {best_model['model_name']} ({best_model['average_tokens_per_second']:.1f} tokens/sec)")
        
        if memory_total >= 32:
            print("   • 32GB+ RAM: 모든 고급 모델 사용 가능")
        elif memory_total >= 16:
            print("   • 16GB RAM: 기본 모델(CodeLlama-7B) 또는 4-bit 양자화 모델 권장")
        else:
            print("   • 16GB 미만 RAM: 더 많은 메모리 필요")
        
        print("\n🚀 AutoCI 시작 명령어:")
        if successful_results:
            print("   source autoci_env/bin/activate")
            print("   python start_autoci_agent.py --advanced-models")
        else:
            print("   먼저 모델을 다운로드하세요: ./download_advanced_models.sh")

def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("AutoCI v3.0 - AI 모델 성능 벤치마크")
        print("사용법: python benchmark_models.py")
        print("")
        print("이 스크립트는 다운로드된 AI 모델들의 성능을 테스트합니다.")
        print("테스트 항목:")
        print("  • 모델 로딩 시간")
        print("  • 응답 생성 속도")
        print("  • 메모리 사용량")
        print("  • 토큰 생성 속도")
        print("")
        print("⚠️ 주의: 큰 모델들은 많은 메모리를 사용하므로 시스템 사양을 확인하세요.")
        return
    
    benchmark = ModelBenchmark()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main() 