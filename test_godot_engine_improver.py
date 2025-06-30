#!/usr/bin/env python3
"""
Godot Engine Improver 테스트 스크립트
"""

import asyncio
import logging
from pathlib import Path
import sys

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from modules.godot_engine_improver import GodotEngineImprover


async def test_engine_improver():
    """엔진 개선 시스템 테스트"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Godot Engine Improver 테스트 ===")
    
    improver = GodotEngineImprover()
    
    # 1. 학습 데이터 로드 테스트
    print("\n1. 학습 데이터 로드 테스트...")
    learning_data = await improver.load_learning_data()
    print(f"  - 로드된 주제: {len(learning_data['topics'])}개")
    print(f"  - 연습 문제: {len(learning_data['exercises'])}개")
    print(f"  - 개선 사항: {len(learning_data['improvements'])}개")
    
    # 일부 주제 출력
    if learning_data['topics']:
        print("\n  주요 학습 주제:")
        for i, (topic, data) in enumerate(list(learning_data['topics'].items())[:5]):
            mastery = data.get('mastery_score', 0) if isinstance(data, dict) else 0
            print(f"    - {topic}: 숙련도 {mastery}%")
    
    # 2. 소스 코드 분석 테스트 (실제 분석은 시간이 걸리므로 간단히)
    print("\n2. 엔진 소스 분석 구조 확인...")
    if improver.godot_source_dir.exists():
        print(f"  - Godot 소스 디렉토리 확인: {improver.godot_source_dir}")
        
        # 주요 디렉토리 확인
        for category, info in improver.improvement_categories.items():
            print(f"\n  {category} ({info['description']}):")
            for path in info['source_paths']:
                full_path = improver.godot_source_dir / path
                exists = "✓" if full_path.exists() else "✗"
                print(f"    {exists} {path}")
    else:
        print(f"  ⚠️  Godot 소스 디렉토리를 찾을 수 없습니다: {improver.godot_source_dir}")
    
    # 3. 개선 카테고리 및 매핑 확인
    print("\n3. 개선 카테고리 확인:")
    for category, info in improver.improvement_categories.items():
        print(f"  - {category}: {info['description']} (우선순위: {info['priority']})")
    
    print("\n4. 학습 주제 매핑 확인:")
    for topic, categories in list(improver.learning_topics_mapping.items())[:5]:
        print(f"  - {topic} → {', '.join(categories)}")
    
    # 5. 샘플 개선 사항 생성 (실제 분석 없이)
    print("\n5. 샘플 개선 사항 생성...")
    sample_improvements = improver._generate_from_learning_data(learning_data)
    print(f"  - 생성된 개선 사항: {len(sample_improvements)}개")
    
    if sample_improvements:
        print("\n  상위 5개 개선 사항:")
        for imp in sample_improvements[:5]:
            print(f"    - [{imp['category']}] {imp['description']} (우선순위: {imp['priority']})")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_engine_improver())