#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠🔍 Enhanced RAG System for AutoCI
수집된 578개 C# 지식 데이터를 실제 구조에 맞춰 활용하는 RAG 시스템
"""

import json
import os
import sys
import re
from typing import List, Dict, Optional
import requests

class EnhancedRAG:
    """실제 데이터 구조에 최적화된 RAG 시스템"""
    
    def __init__(self, data_dir: str = "expert_learning_data"):
        self.data_dir = data_dir
        self.knowledge_base = []
        self.categories = {}
        self.templates = {}
        self.load_knowledge()
    
    def extract_description_from_code(self, code: str) -> str:
        """코드에서 설명 추출"""
        lines = code.strip().split('\n')
        
        # 주석에서 설명 찾기
        comments = []
        for line in lines:
            line = line.strip()
            if line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                comments.append(line.strip('/* '))
        
        if comments:
            return ' '.join(comments[:3])  # 첫 3개 주석 사용
        
        # 클래스/인터페이스 이름에서 추출
        class_matches = re.findall(r'(?:class|interface|struct)\s+([A-Z][a-zA-Z0-9]+)', code)
        if class_matches:
            return f"C# {', '.join(class_matches[:2])} 구현"
        
        # 메서드 이름에서 추출
        method_matches = re.findall(r'(?:public|private|protected).*?\s+([A-Z][a-zA-Z0-9]+)\s*\(', code)
        if method_matches:
            return f"C# {', '.join(method_matches[:2])} 메서드 구현"
        
        return "C# 코드 예제"
    
    def get_template_description(self, template_name: str) -> str:
        """템플릿 이름에서 설명 생성"""
        descriptions = {
            'async_command_pattern': 'C# 비동기 Command 패턴 구현',
            'repository_pattern': 'Repository 패턴을 활용한 데이터 액세스',
            'unity_object_pool': 'Unity Object Pool 패턴으로 성능 최적화',
            'memory_optimization': 'C# 메모리 최적화 기법',
            'advanced_async': 'C# 고급 비동기 프로그래밍',
            'performance_optimization': 'C# 성능 최적화 패턴',
            'unity_optimization': 'Unity 게임 엔진 최적화',
            'architecture_patterns': 'C# 아키텍처 디자인 패턴',
            'unity_advanced': 'Unity 고급 개발 기법',
            'implementation_patterns': 'C# 구현 패턴 및 모범 사례'
        }
        return descriptions.get(template_name, f'{template_name} 패턴')
    
    def load_knowledge(self):
        """지식 베이스 로드 (실제 데이터 구조에 맞춤)"""
        print("🔍 Enhanced RAG 지식 베이스 로딩 중...")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 데이터 디렉토리가 없습니다: {self.data_dir}")
            return
        
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        valid_count = 0
        
        for file_name in json_files:
            try:
                with open(os.path.join(self.data_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 실제 데이터 구조에서 정보 추출
                code = data.get('code', '').strip()
                category = data.get('category', 'general')
                template_name = data.get('template_name', '')
                quality_score = data.get('quality_score', 80)
                
                if len(code) > 100:  # 유효한 코드만
                    # 설명 생성 (여러 방법 시도)
                    description = (
                        data.get('description') or 
                        self.get_template_description(template_name) or
                        self.extract_description_from_code(code)
                    )
                    
                    # 키워드 추출
                    keywords = self.extract_keywords(code, template_name, category)
                    
                    entry = {
                        'id': file_name,
                        'description': description,
                        'code': code,
                        'category': category,
                        'template_name': template_name,
                        'keywords': keywords,
                        'quality_score': quality_score,
                        'search_text': f"{description} {template_name} {category} {' '.join(keywords)}"
                    }
                    
                    self.knowledge_base.append(entry)
                    valid_count += 1
                    
                    # 카테고리별 통계
                    if category not in self.categories:
                        self.categories[category] = 0
                    self.categories[category] += 1
                    
                    # 템플릿별 통계
                    if template_name and template_name not in self.templates:
                        self.templates[template_name] = 0
                    if template_name:
                        self.templates[template_name] += 1
                    
            except Exception as e:
                print(f"❌ 파일 로드 실패 {file_name}: {e}")
        
        print(f"✅ Enhanced RAG 로드 완료: {valid_count}/{len(json_files)}개 유효한 항목")
        print(f"📊 카테고리: {len(self.categories)}개, 템플릿: {len(self.templates)}개")
    
    def extract_keywords(self, code: str, template_name: str, category: str) -> List[str]:
        """코드에서 키워드 추출"""
        keywords = []
        
        # 템플릿명에서 키워드
        if template_name:
            keywords.extend(template_name.split('_'))
        
        # 카테고리에서 키워드
        keywords.extend(category.split('_'))
        
        # 코드에서 C# 키워드 추출
        csharp_patterns = [
            r'\b(async|await|Task|public|private|protected|class|interface|struct)\b',
            r'\b(Unity|GameObject|MonoBehaviour|Component)\b',
            r'\b(List|Dictionary|Array|IEnumerable)\b',
            r'\b(Command|Handler|Repository|Service|Manager)\b'
        ]
        
        for pattern in csharp_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            keywords.extend([m.lower() for m in matches])
        
        # 중복 제거 및 정리
        unique_keywords = list(set([k.strip().lower() for k in keywords if k and len(k) > 2]))
        return unique_keywords[:10]  # 상위 10개만
    
    def search_relevant_code(self, query: str, max_results: int = 3) -> List[Dict]:
        """향상된 검색 알고리즘"""
        if not self.knowledge_base:
            return []
        
        query_lower = query.lower()
        query_words = query_lower.split()
        relevant_codes = []
        
        for entry in self.knowledge_base:
            score = 0
            
            # 1. 템플릿명 매칭 (높은 가중치)
            template_name = entry.get('template_name', '').lower()
            if any(word in template_name for word in query_words):
                score += 10
            
            # 2. 카테고리 매칭
            category = entry.get('category', '').lower()
            if any(word in category for word in query_words):
                score += 8
            
            # 3. 설명 매칭
            description = entry.get('description', '').lower()
            if any(word in description for word in query_words):
                score += 6
            
            # 4. 키워드 매칭
            keywords = [k.lower() for k in entry.get('keywords', [])]
            for word in query_words:
                if any(word in keyword for keyword in keywords):
                    score += 5
            
            # 5. 코드 내용 매칭
            code = entry.get('code', '').lower()
            for word in query_words:
                if word in code:
                    score += 3
            
            # 6. Unity 관련 특별 처리
            if 'unity' in query_lower:
                if 'unity' in template_name or 'unity' in category or 'unity' in code:
                    score += 15
            
            # 7. 비동기 관련 특별 처리
            if any(word in query_lower for word in ['async', 'await', 'task']):
                if any(word in template_name for word in ['async', 'command']):
                    score += 12
            
            # 8. 패턴 관련 특별 처리
            if 'pattern' in query_lower or 'pool' in query_lower:
                if 'pattern' in template_name or 'pool' in template_name:
                    score += 10
            
            # 9. 품질 점수 반영
            quality_bonus = entry.get('quality_score', 80) / 20  # 80점 기준으로 정규화
            score += quality_bonus
            
            if score > 5:  # 최소 임계값
                entry_copy = entry.copy()
                entry_copy['relevance_score'] = round(score, 2)
                relevant_codes.append(entry_copy)
        
        # 점수순으로 정렬하고 상위 결과만 반환
        relevant_codes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_codes[:max_results]
    
    def enhance_prompt(self, user_query: str) -> str:
        """고급 프롬프트 향상"""
        relevant_codes = self.search_relevant_code(user_query, max_results=3)
        
        if not relevant_codes:
            return f"{user_query}\n\n(수집된 C# 지식 베이스에서 관련 예제를 찾지 못했습니다. 일반적인 C# 모범 사례를 사용해주세요.)"
        
        enhanced_prompt = f"""사용자 요청: {user_query}

🔍 관련 C# 코드 예제들 (수집된 {len(self.knowledge_base)}개 지식 베이스에서 검색):

"""
        
        for i, code_entry in enumerate(relevant_codes, 1):
            code_preview = code_entry['code'][:600] + ('...' if len(code_entry['code']) > 600 else '')
            
            enhanced_prompt += f"""
━━━ 예제 {i} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 카테고리: {code_entry['category']}
🏷️ 패턴: {code_entry.get('template_name', 'N/A')}
📊 품질점수: {code_entry.get('quality_score', 'N/A')}/100
🎯 관련도: {code_entry.get('relevance_score', 'N/A')}점
💡 설명: {code_entry['description']}

🔧 코드:
{code_preview}

"""
        
        enhanced_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 요청사항:
위의 관련 예제들을 참고하여 사용자 요청에 맞는 고품질 C# 코드를 생성해주세요.

🎯 중점사항:
• Unity 최적화 및 성능 패턴 적극 활용
• 모던 C# 기법 (async/await, LINQ, Pattern Matching 등) 사용
• 메모리 효율성과 가독성 고려
• 실제 프로덕션에서 사용 가능한 수준의 코드
• 관련 예제의 구조와 패턴을 참고하되, 사용자 요청에 맞게 customization

🚀 활용 가능한 전체 지식: {len(self.knowledge_base)}개 C# 전문 예제"""

        return enhanced_prompt

def test_enhanced_rag():
    """Enhanced RAG 시스템 테스트"""
    rag = EnhancedRAG()
    
    if not rag.knowledge_base:
        print("❌ 지식 베이스가 비어있습니다.")
        return
    
    print(f"""
📊 Enhanced RAG 상태:
• 총 지식 항목: {len(rag.knowledge_base)}개
• 카테고리: {len(rag.categories)}개 - {', '.join(list(rag.categories.keys())[:5])}
• 템플릿 패턴: {len(rag.templates)}개 - {', '.join(list(rag.templates.keys())[:5])}
""")
    
    test_queries = [
        "Unity Object Pool 패턴",
        "C# async await 비동기 프로그래밍",
        "Repository 패턴 구현",
        "Unity 게임 성능 최적화",
        "Command 패턴 구현"
    ]
    
    print("🧪 Enhanced RAG 검색 테스트:")
    for query in test_queries:
        results = rag.search_relevant_code(query, max_results=3)
        print(f"\n🔍 '{query}' -> {len(results)}개 관련 코드")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['template_name']} ({result['category']}) - 점수: {result['relevance_score']}")
            print(f"     {result['description'][:60]}...")

def generate_enhanced_response(user_query: str) -> str:
    """Enhanced RAG로 AI 응답 생성"""
    rag = EnhancedRAG()
    
    if not rag.knowledge_base:
        print("⚠️ Enhanced RAG 지식 베이스가 비어있습니다.")
        return user_query
    
    # 1. Enhanced RAG로 프롬프트 향상
    enhanced_prompt = rag.enhance_prompt(user_query)
    print(f"✨ Enhanced RAG 향상: {len(user_query)}자 -> {len(enhanced_prompt)}자")
    
    # 2. AI 서버에 전송
    try:
        print("🤖 AI 서버에 Enhanced RAG 프롬프트 전송 중...")
        response = requests.post('http://localhost:8000/generate', 
                               json={'prompt': enhanced_prompt, 'max_tokens': 1000}, 
                               timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('generated_text', '').strip()
            
            if generated_text and generated_text != 'No text generated' and len(generated_text) > 50:
                print("🎉 Enhanced RAG AI 응답 생성 성공!")
                return generated_text
            else:
                print("⚠️ AI 모델이 충분한 텍스트를 생성하지 못했습니다")
                print("📋 Enhanced RAG 검색 결과를 직접 제공합니다:")
                return enhanced_prompt
        else:
            print(f"❌ AI 서버 오류: {response.status_code}")
            return enhanced_prompt
            
    except requests.exceptions.ConnectionError:
        print("❌ AI 서버에 연결할 수 없습니다")
        return enhanced_prompt
    except Exception as e:
        print(f"❌ 오류: {e}")
        return enhanced_prompt

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_enhanced_rag()
        else:
            query = " ".join(sys.argv[1:])
            result = generate_enhanced_response(query)
            print(f"\n🎯 질문: {query}")
            print(f"📝 Enhanced RAG 응답:")
            print("═" * 80)
            print(result)
            print("═" * 80)
    else:
        # 기본 테스트 실행
        test_enhanced_rag()
        print("\n" + "="*80)
        print("🚀 사용법:")
        print("  python3 enhanced_rag_system.py test          # Enhanced RAG 테스트")
        print("  python3 enhanced_rag_system.py '질문내용'    # 단일 질문")

if __name__ == "__main__":
    main() 