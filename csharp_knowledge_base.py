#!/usr/bin/env python3
import json
import re
import os
import time
from datetime import datetime

class CSharpCodeAnalyzer:
    def __init__(self):
        self.patterns = [
            {
                'name': 'GetComponent_in_Update',
                'severity': 'high',
                'message': 'Update에서 GetComponent 호출 감지',
                'suggestion': '컴포넌트를 Awake에서 캐싱하세요',
                'pattern': r'void Update.*GetComponent'
            },
            {
                'name': 'string_concatenation',
                'severity': 'medium', 
                'message': '문자열 += 연산자 사용',
                'suggestion': 'StringBuilder 사용 고려',
                'pattern': r'string.*\+=.*\+'
            },
            {
                'name': 'null_check_chain',
                'severity': 'low',
                'message': '중첩 null 체크',
                'suggestion': 'null 조건부 연산자(?.) 사용',
                'pattern': r'if.*!=.*null.*&&.*!=.*null'
            }
        ]
    
    def analyze_code(self, code):
        issues = []
        score = 100
        
        for pattern in self.patterns:
            if re.search(pattern['pattern'], code, re.DOTALL | re.IGNORECASE):
                issues.append({
                    'type': pattern['name'],
                    'severity': pattern['severity'],
                    'message': pattern['message'],
                    'suggestion': pattern['suggestion']
                })
                
                if pattern['severity'] == 'high':
                    score -= 20
                elif pattern['severity'] == 'medium':
                    score -= 10
                else:
                    score -= 5
        
        # 긍정적 패턴 체크
        if 'readonly' in code:
            score += 5
        if '?.' in code:
            score += 5
        if 'StringBuilder' in code:
            score += 10
        
        return {
            'issues': issues,
            'performance_score': max(0, score),
            'suggestions': self.generate_suggestions(code)
        }
    
    def generate_suggestions(self, code):
        suggestions = []
        
        if 'MonoBehaviour' in code:
            suggestions.append('Unity 컴포넌트 캐싱으로 성능 향상')
            suggestions.append('SerializeField로 인스펙터 편의성 개선')
        
        if 'string' in code and '+' in code:
            suggestions.append('문자열 보간($"") 사용으로 가독성 향상')
        
        if 'public ' in code and 'get' in code:
            suggestions.append('자동 속성 및 식 본문 멤버 활용')
        
        return suggestions
    
    def improve_code(self, original_code):
        analysis = self.analyze_code(original_code)
        
        # 개선 헤더 생성
        header = f"""// ==========================================
// 🚀 AI 코드 개선 by C# Expert System  
// 📅 개선 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// 📊 성능 점수: {analysis['performance_score']}/100
// 🔧 발견된 이슈: {len(analysis['issues'])}개
//
// 🎯 적용된 개선사항:"""
        
        improvements = []
        for issue in analysis['issues']:
            improvements.append(f'//   ✅ {issue["message"]} -> {issue["suggestion"]}')
        
        for suggestion in analysis['suggestions']:
            improvements.append(f'//   💡 {suggestion}')
        
        header += '
' + '
'.join(improvements)
        header += '
// ==========================================

'
        
        # using 문 추가
        improved_code = original_code
        if 'using System;' not in improved_code:
            improved_code = 'using System;
' + improved_code
        
        return header + improved_code, analysis

def improve_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        analyzer = CSharpCodeAnalyzer()
        improved_code, analysis = analyzer.improve_code(original_code)
        
        # 백업 생성
        backup_path = f"{file_path}.backup_{int(time.time())}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_code)
        
        # 개선된 코드 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(improved_code)
        
        print(f"✅ 코드 개선 완료: {file_path}")
        print(f"📊 성능 점수: {analysis['performance_score']}/100")
        print(f"🔍 발견된 이슈: {len(analysis['issues'])}개")
        print(f"💡 개선 제안: {len(analysis['suggestions'])}개")
        print(f"🔄 백업: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python3 csharp_knowledge_base.py <파일경로>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    improve_file(file_path)
