#!/usr/bin/env python3
"""
AutoCI Knowledge Evolution System
Implements evolutionary algorithms to evolve and improve knowledge through natural selection
"""

import os
import sys
import json
import time
import random
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import copy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_evolution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeGene:
    """지식 유전자 - 재사용 가능한 솔루션 패턴"""
    gene_id: str
    dna_sequence: str  # 인코딩된 솔루션 패턴
    trait_type: str  # error_handling, optimization, architecture, etc.
    effectiveness: float = 0.5  # 0.0 ~ 1.0
    mutation_rate: float = 0.1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeOrganism:
    """지식 유기체 - 여러 유전자를 가진 완전한 솔루션"""
    organism_id: str
    generation: int
    genes: List[KnowledgeGene]
    fitness_score: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    birth_time: datetime = field(default_factory=datetime.now)
    survival_time: float = 0.0
    successful_applications: int = 0
    failed_applications: int = 0
    environment_type: str = "general"  # godot, csharp, networking, etc.
    
    def calculate_fitness(self) -> float:
        """적합도 계산"""
        if self.successful_applications + self.failed_applications == 0:
            return 0.5
        
        success_rate = self.successful_applications / (self.successful_applications + self.failed_applications)
        gene_effectiveness = np.mean([gene.effectiveness for gene in self.genes])
        survival_bonus = min(self.survival_time / 86400, 1.0)  # 최대 1일 생존 보너스
        
        return (success_rate * 0.5 + gene_effectiveness * 0.3 + survival_bonus * 0.2)

@dataclass
class Generation:
    """세대 정보"""
    generation_number: int
    organisms: List[KnowledgeOrganism]
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    timestamp: datetime = field(default_factory=datetime.now)
    environment_pressures: Dict[str, float] = field(default_factory=dict)

class KnowledgeEvolutionSystem:
    """지식 진화 시스템"""
    
    def __init__(self):
        self.data_dir = Path("continuous_learning/evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.gene_pool_file = self.data_dir / "gene_pool.json"
        self.population_file = self.data_dir / "population.json"
        self.generation_history_file = self.data_dir / "generation_history.json"
        
        self.gene_pool: Dict[str, KnowledgeGene] = {}
        self.population: List[KnowledgeOrganism] = []
        self.generation_history: List[Generation] = []
        self.current_generation = 0
        
        self.population_size = 100
        self.elite_size = 20  # 상위 20%는 다음 세대로 직접 전달
        self.mutation_base_rate = 0.1
        self.crossover_rate = 0.7
        
        self._load_data()
        logger.info(f"🧬 지식 진화 시스템 초기화 완료 - 현재 세대: {self.current_generation}")
    
    def _load_data(self):
        """저장된 데이터 로드"""
        # 유전자 풀 로드
        if self.gene_pool_file.exists():
            with open(self.gene_pool_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.gene_pool = {
                    k: KnowledgeGene(**v) for k, v in data.items()
                }
        
        # 인구 로드
        if self.population_file.exists():
            with open(self.population_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.population = [
                    KnowledgeOrganism(
                        **{k: v if k != 'genes' else [KnowledgeGene(**g) for g in v] 
                           for k, v in org.items()}
                    ) for org in data
                ]
        
        # 세대 기록 로드
        if self.generation_history_file.exists():
            with open(self.generation_history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current_generation = len(data)
    
    def _save_data(self):
        """데이터 저장"""
        # 유전자 풀 저장
        gene_pool_data = {
            k: asdict(v) for k, v in self.gene_pool.items()
        }
        with open(self.gene_pool_file, 'w', encoding='utf-8') as f:
            json.dump(gene_pool_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 인구 저장
        population_data = [
            {**asdict(org), 'genes': [asdict(g) for g in org.genes]}
            for org in self.population
        ]
        with open(self.population_file, 'w', encoding='utf-8') as f:
            json.dump(population_data, f, ensure_ascii=False, indent=2, default=str)
    
    def encode_solution_to_dna(self, solution: Dict[str, Any]) -> str:
        """솔루션을 DNA 시퀀스로 인코딩"""
        # 솔루션의 주요 특성을 문자열로 인코딩
        solution_str = json.dumps(solution, sort_keys=True)
        dna_hash = hashlib.sha256(solution_str.encode()).hexdigest()[:16]
        
        # 솔루션 타입에 따른 접두사 추가
        if solution.get('type') == 'error_handling':
            prefix = 'EH'
        elif solution.get('type') == 'optimization':
            prefix = 'OP'
        elif solution.get('type') == 'architecture':
            prefix = 'AR'
        else:
            prefix = 'GN'
        
        return f"{prefix}-{dna_hash}"
    
    def decode_dna_to_pattern(self, dna_sequence: str) -> Dict[str, Any]:
        """DNA 시퀀스를 패턴으로 디코딩"""
        prefix, hash_part = dna_sequence.split('-', 1)
        
        pattern_types = {
            'EH': 'error_handling',
            'OP': 'optimization',
            'AR': 'architecture',
            'GN': 'general'
        }
        
        return {
            'type': pattern_types.get(prefix, 'general'),
            'dna': dna_sequence,
            'traits': self._extract_traits_from_dna(dna_sequence)
        }
    
    def _extract_traits_from_dna(self, dna_sequence: str) -> List[str]:
        """DNA에서 특성 추출"""
        traits = []
        
        # DNA 해시 부분 분석
        hash_part = dna_sequence.split('-')[1]
        
        # 16진수 문자의 분포로 특성 결정
        char_counts = defaultdict(int)
        for char in hash_part:
            char_counts[char] += 1
        
        # 특성 매핑
        if char_counts['0'] + char_counts['1'] > 4:
            traits.append('defensive')
        if char_counts['f'] + char_counts['e'] > 4:
            traits.append('aggressive')
        if char_counts['a'] + char_counts['b'] > 4:
            traits.append('adaptive')
        if char_counts['7'] + char_counts['8'] > 4:
            traits.append('efficient')
        
        return traits
    
    def create_gene(self, solution: Dict[str, Any], trait_type: str) -> KnowledgeGene:
        """솔루션에서 유전자 생성"""
        gene_id = f"gene_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        dna_sequence = self.encode_solution_to_dna(solution)
        
        gene = KnowledgeGene(
            gene_id=gene_id,
            dna_sequence=dna_sequence,
            trait_type=trait_type,
            effectiveness=0.5,  # 초기 효과성
            mutation_rate=self.mutation_base_rate,
            metadata=solution
        )
        
        self.gene_pool[gene_id] = gene
        return gene
    
    def mutate_gene(self, gene: KnowledgeGene) -> KnowledgeGene:
        """유전자 변이"""
        if random.random() > gene.mutation_rate:
            return gene  # 변이 없음
        
        mutated_gene = copy.deepcopy(gene)
        mutated_gene.gene_id = f"mutated_{gene.gene_id}_{int(time.time())}"
        
        # DNA 시퀀스 변이
        dna_parts = mutated_gene.dna_sequence.split('-')
        if len(dna_parts) == 2:
            # 해시 부분의 일부 문자 변경
            hash_list = list(dna_parts[1])
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                pos = random.randint(0, len(hash_list) - 1)
                hash_list[pos] = random.choice('0123456789abcdef')
            dna_parts[1] = ''.join(hash_list)
            mutated_gene.dna_sequence = '-'.join(dna_parts)
        
        # 변이율 조정
        mutated_gene.mutation_rate *= random.uniform(0.8, 1.2)
        mutated_gene.mutation_rate = max(0.01, min(0.5, mutated_gene.mutation_rate))
        
        # 메타데이터 변이
        if 'parameters' in mutated_gene.metadata:
            params = mutated_gene.metadata['parameters'].copy()
            for key in params:
                if isinstance(params[key], (int, float)):
                    params[key] *= random.uniform(0.9, 1.1)
            mutated_gene.metadata['parameters'] = params
        
        self.gene_pool[mutated_gene.gene_id] = mutated_gene
        return mutated_gene
    
    def crossbreed_organisms(self, parent1: KnowledgeOrganism, parent2: KnowledgeOrganism) -> KnowledgeOrganism:
        """두 유기체 교배"""
        child_id = f"org_{int(time.time() * 1000)}_{random.randint(10000, 99999)}"
        
        # 유전자 조합
        child_genes = []
        
        # 각 부모로부터 유전자 선택
        for i in range(max(len(parent1.genes), len(parent2.genes))):
            if random.random() < 0.5:
                if i < len(parent1.genes):
                    gene = parent1.genes[i]
                else:
                    gene = random.choice(parent2.genes)
            else:
                if i < len(parent2.genes):
                    gene = parent2.genes[i]
                else:
                    gene = random.choice(parent1.genes)
            
            # 교배 중 변이 가능성
            if random.random() < self.mutation_base_rate:
                gene = self.mutate_gene(gene)
            
            child_genes.append(gene)
        
        # 최적 유전자 수 유지
        optimal_gene_count = 5
        if len(child_genes) > optimal_gene_count:
            # 효과성 기준으로 상위 유전자 선택
            child_genes.sort(key=lambda g: g.effectiveness, reverse=True)
            child_genes = child_genes[:optimal_gene_count]
        
        child = KnowledgeOrganism(
            organism_id=child_id,
            generation=self.current_generation + 1,
            genes=child_genes,
            parent_ids=[parent1.organism_id, parent2.organism_id],
            environment_type=parent1.environment_type if random.random() < 0.5 else parent2.environment_type
        )
        
        return child
    
    def evaluate_fitness(self, organism: KnowledgeOrganism, test_results: Dict[str, Any]) -> float:
        """적합도 평가"""
        # 테스트 결과 기반 평가
        success_rate = test_results.get('success_rate', 0.0)
        error_reduction = test_results.get('error_reduction', 0.0)
        performance_gain = test_results.get('performance_gain', 0.0)
        code_quality = test_results.get('code_quality', 0.5)
        
        # 환경별 가중치
        env_weights = {
            'godot': {'performance': 0.4, 'error': 0.3, 'quality': 0.3},
            'csharp': {'quality': 0.4, 'error': 0.4, 'performance': 0.2},
            'networking': {'performance': 0.5, 'error': 0.4, 'quality': 0.1},
            'general': {'error': 0.4, 'quality': 0.3, 'performance': 0.3}
        }
        
        weights = env_weights.get(organism.environment_type, env_weights['general'])
        
        fitness = (
            success_rate * 0.25 +
            error_reduction * weights['error'] +
            performance_gain * weights['performance'] +
            code_quality * weights['quality']
        )
        
        # 유전자 다양성 보너스
        unique_traits = set()
        for gene in organism.genes:
            unique_traits.add(gene.trait_type)
        diversity_bonus = len(unique_traits) / 10.0  # 최대 0.5 보너스
        
        organism.fitness_score = min(fitness + diversity_bonus, 1.0)
        
        # 유전자 효과성 업데이트
        for gene in organism.genes:
            gene.effectiveness = gene.effectiveness * 0.7 + organism.fitness_score * 0.3
        
        return organism.fitness_score
    
    def natural_selection(self) -> List[KnowledgeOrganism]:
        """자연 선택 - 적합도 기반 생존자 선택"""
        # 적합도 순으로 정렬
        self.population.sort(key=lambda org: org.fitness_score, reverse=True)
        
        # 엘리트 선택
        elite = self.population[:self.elite_size]
        
        # 토너먼트 선택으로 나머지 선택
        selected = elite.copy()
        tournament_size = 5
        
        while len(selected) < self.population_size // 2:
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda org: org.fitness_score)
            selected.append(winner)
        
        return selected
    
    def evolve_generation(self) -> Generation:
        """새로운 세대 진화"""
        logger.info(f"🧬 세대 {self.current_generation} -> {self.current_generation + 1} 진화 시작")
        
        # 현재 세대 평가
        if self.population:
            fitness_scores = [org.fitness_score for org in self.population]
            current_gen = Generation(
                generation_number=self.current_generation,
                organisms=self.population.copy(),
                avg_fitness=np.mean(fitness_scores),
                best_fitness=max(fitness_scores),
                worst_fitness=min(fitness_scores)
            )
            self.generation_history.append(current_gen)
        
        # 자연 선택
        survivors = self.natural_selection()
        
        # 새로운 세대 생성
        new_population = []
        
        # 엘리트 직접 전달
        for elite in survivors[:self.elite_size]:
            new_population.append(elite)
        
        # 교배를 통한 자손 생성
        while len(new_population) < self.population_size:
            if len(survivors) >= 2:
                parent1 = random.choice(survivors)
                parent2 = random.choice([s for s in survivors if s != parent1])
                
                if random.random() < self.crossover_rate:
                    child = self.crossbreed_organisms(parent1, parent2)
                else:
                    # 교배 없이 부모 중 하나 선택
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                    child.organism_id = f"clone_{child.organism_id}_{int(time.time())}"
                    child.generation = self.current_generation + 1
                
                new_population.append(child)
            else:
                # 생존자가 부족한 경우 랜덤 생성
                new_population.append(self.create_random_organism())
        
        self.population = new_population
        self.current_generation += 1
        
        # 데이터 저장
        self._save_data()
        
        logger.info(f"✅ 세대 {self.current_generation} 진화 완료 - 인구: {len(self.population)}")
        return current_gen
    
    def create_random_organism(self) -> KnowledgeOrganism:
        """랜덤 유기체 생성"""
        organism_id = f"random_org_{int(time.time() * 1000)}_{random.randint(10000, 99999)}"
        
        # 랜덤 유전자 선택
        num_genes = random.randint(3, 7)
        genes = []
        
        if self.gene_pool:
            available_genes = list(self.gene_pool.values())
            for _ in range(num_genes):
                gene = random.choice(available_genes)
                if random.random() < 0.3:  # 30% 확률로 변이
                    gene = self.mutate_gene(gene)
                genes.append(gene)
        
        return KnowledgeOrganism(
            organism_id=organism_id,
            generation=self.current_generation,
            genes=genes,
            environment_type=random.choice(['godot', 'csharp', 'networking', 'general'])
        )
    
    def apply_organism_to_problem(self, organism: KnowledgeOrganism, problem: Dict[str, Any]) -> Dict[str, Any]:
        """유기체를 문제에 적용"""
        solutions = []
        
        for gene in organism.genes:
            pattern = self.decode_dna_to_pattern(gene.dna_sequence)
            
            # 유전자 타입에 따른 솔루션 적용
            if gene.trait_type == 'error_handling' and problem.get('has_error'):
                solutions.append({
                    'type': 'error_fix',
                    'pattern': pattern,
                    'gene_id': gene.gene_id,
                    'confidence': gene.effectiveness
                })
            elif gene.trait_type == 'optimization' and problem.get('needs_optimization'):
                solutions.append({
                    'type': 'optimization',
                    'pattern': pattern,
                    'gene_id': gene.gene_id,
                    'confidence': gene.effectiveness
                })
            elif gene.trait_type == 'architecture':
                solutions.append({
                    'type': 'architecture',
                    'pattern': pattern,
                    'gene_id': gene.gene_id,
                    'confidence': gene.effectiveness
                })
        
        return {
            'organism_id': organism.organism_id,
            'solutions': solutions,
            'combined_confidence': np.mean([s['confidence'] for s in solutions]) if solutions else 0.0
        }
    
    def update_organism_success(self, organism_id: str, success: bool):
        """유기체 성공/실패 업데이트"""
        for org in self.population:
            if org.organism_id == organism_id:
                if success:
                    org.successful_applications += 1
                else:
                    org.failed_applications += 1
                org.fitness_score = org.calculate_fitness()
                break
    
    def get_best_organisms(self, environment_type: Optional[str] = None, limit: int = 10) -> List[KnowledgeOrganism]:
        """최고의 유기체들 반환"""
        organisms = self.population
        
        if environment_type:
            organisms = [org for org in organisms if org.environment_type == environment_type]
        
        organisms.sort(key=lambda org: org.fitness_score, reverse=True)
        return organisms[:limit]
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """진화 보고서 생성"""
        if not self.generation_history:
            return {
                'current_generation': self.current_generation,
                'population_size': len(self.population),
                'gene_pool_size': len(self.gene_pool),
                'status': 'initializing'
            }
        
        recent_generations = self.generation_history[-10:]
        fitness_trend = [gen.avg_fitness for gen in recent_generations]
        
        # 환경별 통계
        env_stats = defaultdict(lambda: {'count': 0, 'avg_fitness': 0.0})
        for org in self.population:
            env_stats[org.environment_type]['count'] += 1
            env_stats[org.environment_type]['avg_fitness'] += org.fitness_score
        
        for env in env_stats:
            if env_stats[env]['count'] > 0:
                env_stats[env]['avg_fitness'] /= env_stats[env]['count']
        
        return {
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'gene_pool_size': len(self.gene_pool),
            'average_fitness': np.mean([org.fitness_score for org in self.population]),
            'best_fitness': max([org.fitness_score for org in self.population]) if self.population else 0.0,
            'fitness_improvement': fitness_trend[-1] - fitness_trend[0] if len(fitness_trend) > 1 else 0.0,
            'environment_stats': dict(env_stats),
            'total_organisms_created': sum(len(gen.organisms) for gen in self.generation_history),
            'evolution_rate': self.current_generation / max(1, (datetime.now() - self.generation_history[0].timestamp).days)
        }
    
    async def adaptive_evolution(self, project_metrics: Dict[str, Any]):
        """프로젝트 메트릭에 따른 적응적 진화"""
        # 환경 압력 계산
        pressures = {
            'error_rate': project_metrics.get('error_rate', 0.0),
            'performance_issues': project_metrics.get('performance_issues', 0.0),
            'code_complexity': project_metrics.get('code_complexity', 0.5),
            'user_satisfaction': 1.0 - project_metrics.get('user_satisfaction', 0.5)
        }
        
        # 압력에 따른 변이율 조정
        total_pressure = sum(pressures.values()) / len(pressures)
        self.mutation_base_rate = 0.1 + (total_pressure * 0.2)  # 압력이 높을수록 변이 증가
        
        # 선택압 조정
        if total_pressure > 0.7:
            self.elite_size = max(10, self.population_size // 10)  # 엘리트 축소
        else:
            self.elite_size = min(30, self.population_size // 5)   # 엘리트 확대
        
        # 환경별 적합도 가중치 조정
        for org in self.population:
            if org.environment_type == 'error_handling' and pressures['error_rate'] > 0.5:
                org.fitness_score *= 1.2  # 에러 처리 유기체 선호
            elif org.environment_type == 'optimization' and pressures['performance_issues'] > 0.5:
                org.fitness_score *= 1.2  # 최적화 유기체 선호
        
        logger.info(f"🎯 적응적 진화 적용 - 총 압력: {total_pressure:.2f}, 변이율: {self.mutation_base_rate:.2f}")

# 전역 진화 시스템 인스턴스
evolution_system = None

def initialize_evolution_system():
    """진화 시스템 초기화"""
    global evolution_system
    if evolution_system is None:
        evolution_system = KnowledgeEvolutionSystem()
    return evolution_system

if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    
    async def test_evolution():
        system = initialize_evolution_system()
        
        # 초기 유기체 생성
        if not system.population:
            logger.info("📦 초기 인구 생성 중...")
            for _ in range(system.population_size):
                system.population.append(system.create_random_organism())
        
        # 몇 세대 진화
        for i in range(5):
            logger.info(f"\n🔄 세대 {i+1} 진화 시작")
            
            # 각 유기체 평가 (시뮬레이션)
            for org in system.population:
                test_results = {
                    'success_rate': random.random(),
                    'error_reduction': random.random(),
                    'performance_gain': random.random(),
                    'code_quality': random.random()
                }
                system.evaluate_fitness(org, test_results)
            
            # 진화
            generation = system.evolve_generation()
            
            # 보고서
            report = system.get_evolution_report()
            logger.info(f"📊 진화 보고서: {json.dumps(report, indent=2, ensure_ascii=False)}")
            
            # 적응적 진화
            project_metrics = {
                'error_rate': random.random(),
                'performance_issues': random.random(),
                'code_complexity': random.random(),
                'user_satisfaction': random.random()
            }
            await system.adaptive_evolution(project_metrics)
            
            time.sleep(1)
        
        # 최고 유기체들
        best = system.get_best_organisms(limit=5)
        logger.info(f"\n🏆 최고의 유기체들:")
        for org in best:
            logger.info(f"  - {org.organism_id}: 적합도 {org.fitness_score:.3f}, 유전자 {len(org.genes)}개")
    
    asyncio.run(test_evolution())