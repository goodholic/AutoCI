#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Quick RAG Enhancer
수집된 578개 C# 지식 데이터를 즉시 AI 응답에 반영하는 간단한 RAG 시스템
"""

import json
import os
import sys
import re
from typing import List, Dict
import requests

class QuickRAG:
    """간단하고 빠른 RAG 시스템"""
    
    def __init__(self, data_dir: str = "expert_learning_data"):
        self.data_dir = data_dir
        self.knowledge_base = []
        self.load_knowledge()
    
    def load_knowledge(self):
        """지식 베이스 로드"""
        print("🔍 C# 지식 베이스 로딩 중...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 데이터 디렉토리가 없습니다: {self.data_dir}")
            return
        
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        valid_count = 0
        
        for file_name in json_files:
            try:
                with open(os.path.join(self.data_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 유효한 코드가 있는 경우만 추가
                code = data.get('code', '').strip()
                description = data.get('description', '').strip()
                
                if len(code) > 50 and description:
                    entry = {
                        'description': description,
                        'code': code,
                        'category': data.get('category', 'general'),
                        'keywords': data.get('keywords', [])
                    }
                    self.knowledge_base.append(entry)
                    valid_count += 1
                    
            except Exception as e:
                print(f"❌ 파일 로드 실패 {file_name}: {e}")
        
        print(f"✅ 지식 베이스 로드 완료: {valid_count}/{len(json_files)}개 유효한 항목")
    
    def search_relevant_code(self, query: str, max_results: int = 3) -> List[Dict]:
        """키워드 기반 관련 코드 검색"""
        if not self.knowledge_base:
            return []
        
        query_lower = query.lower()
        relevant_codes = []
        
        for entry in self.knowledge_base:
            score = 0
            
            # 설명에서 키워드 매칭
            if any(word in entry['description'].lower() for word in query_lower.split()):
                score += 3
            
            # 코드에서 키워드 매칭
            if any(word in entry['code'].lower() for word in query_lower.split()):
                score += 2
            
            # 카테고리 매칭
            if any(word in entry['category'].lower() for word in query_lower.split()):
                score += 2
            
            # 키워드 매칭
            if any(keyword.lower() in query_lower for keyword in entry['keywords']):
                score += 4
            
            # Unity 관련 추가 점수
            if 'unity' in query_lower:
                if 'unity' in entry['description'].lower() or 'unity' in entry['code'].lower():
                    score += 5
            
            if score > 0:
                entry_copy = entry.copy()
                entry_copy['relevance_score'] = score
                relevant_codes.append(entry_copy)
        
        # 점수순으로 정렬하고 상위 결과만 반환
        relevant_codes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_codes[:max_results]
    
    def enhance_prompt(self, user_query: str) -> str:
        """RAG로 프롬프트 향상"""
        relevant_codes = self.search_relevant_code(user_query, max_results=3)
        
        if not relevant_codes:
            return user_query
        
        enhanced_prompt = f"""사용자 요청: {user_query}

관련 코드 예제들 (수집된 C# 지식 베이스에서 검색):

"""
        
        for i, code_entry in enumerate(relevant_codes, 1):
            enhanced_prompt += f"""
--- 예제 {i} (카테고리: {code_entry['category']}, 점수: {code_entry['relevance_score']}) ---
설명: {code_entry['description'][:150]}...
코드 샘플:
{code_entry['code'][:300]}...
키워드: {', '.join(code_entry['keywords'][:5])}

"""
        
        enhanced_prompt += f"""
위의 관련 예제들을 참고하여 사용자 요청에 맞는 고품질 C# 코드를 생성해주세요.
Unity 최적화와 모던 C# 기법을 적극 활용하세요."""

        return enhanced_prompt

def test_rag_system():
    """RAG 시스템 테스트"""
    rag = QuickRAG()
    
    if not rag.knowledge_base:
        print("❌ 지식 베이스가 비어있습니다. 먼저 데이터를 수집해주세요.")
        return
    
    print(f"\n📊 지식 베이스 상태: {len(rag.knowledge_base)}개 C# 코드 예제")
    
    test_queries = [
        "Unity 플레이어 컨트롤러",
        "Object Pool 패턴",
        "async await 비동기",
        "메모리 최적화",
        "ECS 시스템"
    ]
    
    print("\n🧪 RAG 검색 테스트:")
    for query in test_queries:
        results = rag.search_relevant_code(query, max_results=2)
        print(f"\n🔍 '{query}' -> {len(results)}개 관련 코드")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['category']} (점수: {result['relevance_score']})")
            print(f"     {result['description'][:80]}...")

def enhance_ai_response(user_query: str) -> str:
    """AI 응답을 RAG로 향상"""
    rag = QuickRAG()
    
    if not rag.knowledge_base:
        print("⚠️ RAG 지식 베이스가 비어있습니다. 기본 쿼리를 사용합니다.")
        return user_query
    
    # 1. RAG로 프롬프트 향상
    enhanced_prompt = rag.enhance_prompt(user_query)
    print(f"✨ RAG 향상 완료: 기본 {len(user_query)}자 -> 향상된 {len(enhanced_prompt)}자")
    
    # 2. AI 서버에 전송
    try:
        response = requests.post('http://localhost:8000/generate', 
                               json={'prompt': enhanced_prompt, 'max_tokens': 500}, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('generated_text', 'No text generated')
            
            if generated_text and generated_text != 'No text generated':
                print("✅ RAG 향상된 AI 응답 생성 성공!")
                return generated_text
            else:
                print("⚠️ AI 모델이 텍스트를 생성하지 못했습니다")
                return f"RAG 검색 결과:\n{enhanced_prompt}"
        else:
            print(f"❌ AI 서버 오류: {response.status_code}")
            return enhanced_prompt
            
    except requests.exceptions.ConnectionError:
        print("❌ AI 서버에 연결할 수 없습니다")
        return enhanced_prompt
    except Exception as e:
        print(f"❌ 오류: {e}")
        return enhanced_prompt

def interactive_mode():
    """대화형 RAG 모드"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🧠🔍 Quick RAG C# 코드 생성기                               ║
║                                                              ║
║  수집된 578개 C# 지식 데이터를 활용한 향상된 코드 생성       ║
║                                                              ║
║  사용법: 원하는 C# 코드를 설명해주세요                       ║
║  종료: 'quit' 또는 'exit' 입력                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    rag = QuickRAG()
    
    if not rag.knowledge_base:
        print("❌ 지식 베이스를 로드할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    while True:
        try:
            user_input = input("\n🎯 생성하고 싶은 C# 코드: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 RAG 시스템을 종료합니다.")
                break
            
            if not user_input:
                continue
            
            print(f"\n🔍 RAG 검색 중: '{user_input}'")
            enhanced_response = enhance_ai_response(user_input)
            
            print(f"\n📝 RAG 향상된 응답:")
            print("─" * 60)
            print(enhanced_response)
            print("─" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_rag_system()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        else:
            query = " ".join(sys.argv[1:])
            result = enhance_ai_response(query)
            print(f"\n🎯 질문: {query}")
            print(f"📝 RAG 향상된 응답:")
            print("─" * 60)
            print(result)
            print("─" * 60)
    else:
        # 기본 테스트 실행
        test_rag_system()
        print("\n" + "="*60)
        print("사용법:")
        print("  python3 quick_rag_enhancer.py test          # RAG 테스트")
        print("  python3 quick_rag_enhancer.py interactive   # 대화형 모드")
        print("  python3 quick_rag_enhancer.py '질문내용'    # 단일 질문")

if __name__ == "__main__":
    main() 