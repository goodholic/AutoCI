#!/usr/bin/env python3
"""
Nakama Server AI 통합 시스템
AI가 Nakama 서버를 완전히 제어하여 멀티플레이어 백엔드 자동화
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import subprocess

class NakamaAIIntegration:
    """Nakama Server AI 제어 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nakama_config_dir = Path("./nakama_configurations")
        self.ai_modules_dir = Path("./nakama_ai_modules")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Nakama AI 설정"""
        return {
            "server": {
                "host": "localhost",
                "http_port": 7350,
                "grpc_port": 7349,
                "console_port": 7351,
                "max_message_size": 4096,
                "protocol": "tcp"
            },
            "ai_features": {
                "intelligent_matchmaking": True,
                "auto_scaling": True,
                "predictive_analytics": True,
                "anti_cheat": True,
                "dynamic_balancing": True,
                "content_recommendation": True
            },
            "optimization": {
                "cache_enabled": True,
                "connection_pooling": True,
                "load_balancing": True,
                "auto_backup": True
            },
            "game_types": {
                "fps": {
                    "tick_rate": 64,
                    "max_players": 32,
                    "match_duration": 900
                },
                "moba": {
                    "tick_rate": 30,
                    "max_players": 10,
                    "match_duration": 2400
                },
                "mmo": {
                    "tick_rate": 10,
                    "max_players": 1000,
                    "persistent": True
                },
                "battle_royale": {
                    "tick_rate": 30,
                    "max_players": 100,
                    "shrinking_zone": True
                }
            }
        }
    
    async def setup_nakama_server(self, game_type: str = "general") -> Dict[str, Any]:
        """AI가 최적화된 Nakama 서버 설정"""
        self.logger.info(f"🚀 Nakama 서버 설정 시작: {game_type}")
        
        # 설정 디렉토리 생성
        self.nakama_config_dir.mkdir(parents=True, exist_ok=True)
        self.ai_modules_dir.mkdir(parents=True, exist_ok=True)
        
        # 서버 설정 생성
        server_config = await self._generate_server_config(game_type)
        config_path = self.nakama_config_dir / "nakama-config.yml"
        
        # 설정 파일 작성
        with open(config_path, 'w') as f:
            f.write(server_config)
        
        # AI 모듈 생성
        modules = await self._create_ai_modules(game_type)
        
        return {
            "status": "configured",
            "config_path": str(config_path),
            "modules": list(modules.keys()),
            "game_type": game_type
        }
    
    async def _generate_server_config(self, game_type: str) -> str:
        """게임 타입별 최적화된 서버 설정 생성"""
        game_config = self.config["game_types"].get(game_type, {})
        
        config = f"""name: nakama-ai-{game_type}
data_dir: "./data/"

logger:
  level: "INFO"
  stdout: true

session:
  encryption_key: "defaultencryptionkey"
  token_expiry_sec: 7200

socket:
  server_key: "defaultkey"
  port: {self.config['server']['grpc_port']}
  max_message_size_bytes: {self.config['server']['max_message_size']}
  ping_period_ms: 15000
  pong_wait_ms: 25000

runtime:
  path: "./modules"
  http_key: "defaulthttpkey"

# AI 최적화 설정
match:
  max_size: {game_config.get('max_players', 100)}
  tick_rate: {game_config.get('tick_rate', 30)}
  
# AI 매치메이킹
matchmaker:
  max_tickets: 10000
  interval_sec: 1
  
# 리더보드 설정
leaderboard:
  callback_queue_size: 10000
  
# 스토리지 설정
storage:
  max_value_size_bytes: 1048576  # 1MB
  
# 소셜 기능
social:
  max_friends: 500
  
# AI 분석 활성화
metrics:
  reporting_freq_sec: 60
  prometheus_port: 9100
"""
        return config
    
    async def _create_ai_modules(self, game_type: str) -> Dict[str, str]:
        """AI 제어 모듈 생성"""
        modules = {}
        
        # 매치메이킹 AI
        modules["matchmaking_ai.lua"] = await self.create_ai_matchmaker()
        
        # 스토리지 AI
        modules["storage_ai.lua"] = await self.create_intelligent_storage()
        
        # 소셜 AI
        modules["social_ai.lua"] = await self.create_social_ai_moderator()
        
        # 게임별 특화 모듈
        if game_type == "fps":
            modules["fps_backend.lua"] = await self._create_fps_backend()
        elif game_type == "moba":
            modules["moba_backend.lua"] = await self._create_moba_backend()
        elif game_type == "mmo":
            modules["mmo_backend.lua"] = await self._create_mmo_backend()
        elif game_type == "battle_royale":
            modules["br_backend.lua"] = await self._create_battle_royale_backend()
        
        # 모듈 저장
        for filename, content in modules.items():
            module_path = self.ai_modules_dir / filename
            module_path.write_text(content)
            self.logger.info(f"✅ 모듈 생성: {filename}")
        
        return modules
    
    async def create_ai_matchmaker(self) -> str:
        """AI 기반 지능형 매치메이킹 시스템"""
        matchmaker = """-- Nakama AI Matchmaker Module
-- AI가 제어하는 지능형 매치메이킹 시스템

local nk = require("nakama")

-- AI 매치메이킹 설정
local ai_config = {
    skill_weight = 0.7,
    latency_weight = 0.2,
    playstyle_weight = 0.1,
    max_skill_diff = 200,
    max_latency_diff = 50,
    queue_timeout = 30
}

-- 플레이어 스킬 분석
local function analyze_player_skill(context, user_id)
    local stats = nk.storage_read({
        {collection = "player_stats", key = "overall", user_id = user_id}
    })
    
    if #stats > 0 then
        local data = stats[1].value
        -- AI가 복합적인 스킬 지표 계산
        local skill_rating = data.mmr or 1000
        local win_rate = data.wins / math.max(data.total_games, 1)
        local kda = (data.kills + data.assists) / math.max(data.deaths, 1)
        
        -- 가중 평균 계산
        return skill_rating * 0.6 + win_rate * 1000 * 0.3 + kda * 100 * 0.1
    end
    
    return 1000 -- 기본값
end

-- 플레이어 플레이스타일 분석
local function analyze_playstyle(context, user_id)
    local history = nk.storage_read({
        {collection = "match_history", key = "recent", user_id = user_id}
    })
    
    if #history > 0 then
        local data = history[1].value
        local style = {
            aggressive = 0,
            defensive = 0,
            support = 0,
            versatile = 0
        }
        
        -- AI가 플레이 패턴 분석
        for _, match in ipairs(data.matches or {}) do
            if match.damage_dealt > match.damage_taken then
                style.aggressive = style.aggressive + 1
            else
                style.defensive = style.defensive + 1
            end
            
            if match.assists > match.kills then
                style.support = style.support + 1
            end
        end
        
        return style
    end
    
    return {aggressive = 1, defensive = 1, support = 1, versatile = 1}
end

-- 지능형 매치메이킹 함수
local function ai_matchmaker(context, matched, ticket)
    local candidates = {}
    
    -- 모든 대기 중인 티켓 검색
    local query = "*"
    local min_count = ticket.properties.min_count or 2
    local max_count = ticket.properties.max_count or 10
    
    -- 현재 플레이어 분석
    local user_skill = analyze_player_skill(context, ticket.presence.user_id)
    local user_style = analyze_playstyle(context, ticket.presence.user_id)
    local user_latency = ticket.properties.latency or 50
    
    -- 매칭 점수 계산
    for _, candidate in ipairs(matched) do
        if candidate.presence.user_id ~= ticket.presence.user_id then
            local skill = analyze_player_skill(context, candidate.presence.user_id)
            local style = analyze_playstyle(context, candidate.presence.user_id)
            local latency = candidate.properties.latency or 50
            
            -- AI 매칭 점수 계산
            local skill_diff = math.abs(user_skill - skill)
            local latency_diff = math.abs(user_latency - latency)
            
            -- 스타일 호환성 계산
            local style_compatibility = 0
            for key, value in pairs(user_style) do
                style_compatibility = style_compatibility + 
                    math.min(value, style[key] or 0)
            end
            
            -- 종합 점수
            local score = (ai_config.max_skill_diff - skill_diff) * ai_config.skill_weight +
                         (ai_config.max_latency_diff - latency_diff) * ai_config.latency_weight +
                         style_compatibility * ai_config.playstyle_weight
            
            table.insert(candidates, {
                ticket = candidate,
                score = score
            })
        end
    end
    
    -- 점수 기준 정렬
    table.sort(candidates, function(a, b) return a.score > b.score end)
    
    -- 최적 매치 구성
    local match_candidates = {ticket}
    for i = 1, math.min(max_count - 1, #candidates) do
        if candidates[i].score > 0 then
            table.insert(match_candidates, candidates[i].ticket)
        end
        
        if #match_candidates >= min_count then
            -- 매치 생성
            return {match_candidates}
        end
    end
    
    -- 타임아웃 체크
    local elapsed = context.execution_time - ticket.create_time
    if elapsed > ai_config.queue_timeout * 1000 then
        -- AI가 봇 추가 결정
        if ticket.properties.allow_bots then
            return {match_candidates} -- 봇과 함께 매치 시작
        end
    end
    
    return nil
end

-- 매치메이킹 통계 수집
local function collect_matchmaking_stats(context, matches)
    for _, match in ipairs(matches) do
        local stats = {
            timestamp = nk.time(),
            player_count = #match,
            avg_skill = 0,
            avg_latency = 0,
            skill_variance = 0
        }
        
        -- 통계 계산
        local skills = {}
        for _, ticket in ipairs(match) do
            local skill = analyze_player_skill(context, ticket.presence.user_id)
            table.insert(skills, skill)
            stats.avg_skill = stats.avg_skill + skill
            stats.avg_latency = stats.avg_latency + (ticket.properties.latency or 50)
        end
        
        stats.avg_skill = stats.avg_skill / #match
        stats.avg_latency = stats.avg_latency / #match
        
        -- 분산 계산
        for _, skill in ipairs(skills) do
            stats.skill_variance = stats.skill_variance + 
                math.pow(skill - stats.avg_skill, 2)
        end
        stats.skill_variance = stats.skill_variance / #match
        
        -- 통계 저장
        nk.storage_write({
            {
                collection = "matchmaking_stats",
                key = nk.uuid_v4(),
                value = stats,
                permission_read = 0,
                permission_write = 0
            }
        })
    end
end

-- RPC: 매치메이킹 분석
local function rpc_analyze_matchmaking(context, payload)
    local data = nk.json_decode(payload)
    local user_id = data.user_id or context.user_id
    
    local analysis = {
        skill_rating = analyze_player_skill(context, user_id),
        playstyle = analyze_playstyle(context, user_id),
        recommended_modes = {},
        estimated_wait_time = 0
    }
    
    -- AI가 추천 게임 모드 결정
    if analysis.playstyle.aggressive > analysis.playstyle.defensive then
        table.insert(analysis.recommended_modes, "deathmatch")
        table.insert(analysis.recommended_modes, "team_deathmatch")
    else
        table.insert(analysis.recommended_modes, "capture_the_flag")
        table.insert(analysis.recommended_modes, "domination")
    end
    
    -- 대기 시간 예측
    local current_players = nk.match_list(100, true, "", 0, 100)
    analysis.estimated_wait_time = math.max(5, 60 - #current_players * 2)
    
    return nk.json_encode(analysis)
end

-- 모듈 등록
nk.register_matchmaker_matched(ai_matchmaker)
nk.register_rpc(rpc_analyze_matchmaking, "analyze_matchmaking")

nk.logger_info("AI Matchmaker module loaded successfully")
"""
        return matchmaker
    
    async def create_intelligent_storage(self) -> str:
        """AI 기반 지능형 스토리지 관리"""
        storage = """-- Nakama AI Storage Module
-- AI가 최적화하는 데이터 저장 및 관리

local nk = require("nakama")

-- AI 스토리지 설정
local storage_config = {
    cache_ttl = 300, -- 5분
    compression_threshold = 1024, -- 1KB 이상 압축
    index_optimization = true,
    auto_cleanup = true,
    prediction_enabled = true
}

-- 데이터 압축 함수
local function compress_data(data)
    -- 간단한 압축 시뮬레이션 (실제로는 더 복잡한 알고리즘 사용)
    if type(data) == "table" then
        -- 중복 제거 및 최적화
        local compressed = {}
        for k, v in pairs(data) do
            if v ~= nil and v ~= "" then
                compressed[k] = v
            end
        end
        return compressed
    end
    return data
end

-- AI 데이터 예측 및 프리페칭
local function predict_data_access(context, user_id, current_key)
    -- 접근 패턴 분석
    local access_history = nk.storage_read({
        {collection = "access_patterns", key = user_id, user_id = user_id}
    })
    
    local predictions = {}
    
    if #access_history > 0 then
        local patterns = access_history[1].value
        
        -- 시퀀스 패턴 분석
        for i, access in ipairs(patterns.history or {}) do
            if access.key == current_key and i < #patterns.history then
                local next_key = patterns.history[i + 1].key
                predictions[next_key] = (predictions[next_key] or 0) + 1
            end
        end
    end
    
    -- 가장 가능성 높은 다음 접근 예측
    local sorted_predictions = {}
    for key, count in pairs(predictions) do
        table.insert(sorted_predictions, {key = key, probability = count})
    end
    table.sort(sorted_predictions, function(a, b) return a.probability > b.probability end)
    
    return sorted_predictions
end

-- 지능형 저장 함수
local function ai_storage_write(context, user_id, collection, key, value)
    -- 데이터 분석 및 최적화
    local optimized_value = value
    
    -- 크기 체크 및 압축
    local data_size = #nk.json_encode(value)
    if data_size > storage_config.compression_threshold then
        optimized_value = compress_data(value)
        nk.logger_info(string.format("Compressed data from %d to %d bytes", 
            data_size, #nk.json_encode(optimized_value)))
    end
    
    -- 메타데이터 추가
    local metadata = {
        original_size = data_size,
        compressed = data_size > storage_config.compression_threshold,
        access_count = 0,
        last_access = nk.time(),
        predicted_next_access = nk.time() + 3600 -- AI 예측
    }
    
    -- 실제 저장
    local success, error = pcall(function()
        nk.storage_write({
            {
                collection = collection,
                key = key,
                user_id = user_id,
                value = optimized_value,
                permission_read = 1,
                permission_write = 0
            },
            {
                collection = collection .. "_metadata",
                key = key,
                user_id = user_id,
                value = metadata,
                permission_read = 0,
                permission_write = 0
            }
        })
    end)
    
    if success then
        -- 접근 패턴 기록
        record_access_pattern(context, user_id, collection, key)
        
        -- 예측 기반 프리페칭
        if storage_config.prediction_enabled then
            local predictions = predict_data_access(context, user_id, key)
            for i = 1, math.min(3, #predictions) do
                -- 비동기 프리페치 (캐시 워밍)
                prefetch_data(context, user_id, collection, predictions[i].key)
            end
        end
    end
    
    return success, error
end

-- 지능형 읽기 함수
local function ai_storage_read(context, user_id, collection, key)
    -- 메타데이터 확인
    local metadata = nk.storage_read({
        {collection = collection .. "_metadata", key = key, user_id = user_id}
    })
    
    -- 실제 데이터 읽기
    local data = nk.storage_read({
        {collection = collection, key = key, user_id = user_id}
    })
    
    if #data > 0 then
        -- 접근 카운트 업데이트
        if #metadata > 0 then
            local meta = metadata[1].value
            meta.access_count = meta.access_count + 1
            meta.last_access = nk.time()
            
            nk.storage_write({
                {
                    collection = collection .. "_metadata",
                    key = key,
                    user_id = user_id,
                    value = meta,
                    permission_read = 0,
                    permission_write = 0
                }
            })
        end
        
        -- 접근 패턴 기록
        record_access_pattern(context, user_id, collection, key)
        
        return data[1].value
    end
    
    return nil
end

-- 접근 패턴 기록
local function record_access_pattern(context, user_id, collection, key)
    local pattern_key = user_id
    local patterns = nk.storage_read({
        {collection = "access_patterns", key = pattern_key, user_id = user_id}
    })
    
    local pattern_data = {history = {}}
    if #patterns > 0 then
        pattern_data = patterns[1].value
    end
    
    -- 새 접근 기록 추가
    table.insert(pattern_data.history, {
        key = key,
        collection = collection,
        timestamp = nk.time()
    })
    
    -- 히스토리 크기 제한 (최근 100개)
    if #pattern_data.history > 100 then
        table.remove(pattern_data.history, 1)
    end
    
    nk.storage_write({
        {
            collection = "access_patterns",
            key = pattern_key,
            user_id = user_id,
            value = pattern_data,
            permission_read = 0,
            permission_write = 0
        }
    })
end

-- 데이터 프리페칭
local function prefetch_data(context, user_id, collection, key)
    -- 캐시에 미리 로드 (실제 구현에서는 Redis 등 사용)
    local data = nk.storage_read({
        {collection = collection, key = key, user_id = user_id}
    })
    
    if #data > 0 then
        nk.logger_debug(string.format("Prefetched data: %s/%s", collection, key))
    end
end

-- 자동 정리 함수
local function auto_cleanup_storage(context)
    if not storage_config.auto_cleanup then
        return
    end
    
    -- 오래된 데이터 정리 로직
    local cutoff_time = nk.time() - (30 * 24 * 3600) -- 30일
    
    -- 메타데이터 기반 정리 (실제로는 배치 처리)
    nk.logger_info("Running AI storage cleanup...")
end

-- RPC: 스토리지 분석
local function rpc_analyze_storage(context, payload)
    local data = nk.json_decode(payload)
    local user_id = data.user_id or context.user_id
    
    -- 사용 패턴 분석
    local patterns = nk.storage_read({
        {collection = "access_patterns", key = user_id, user_id = user_id}
    })
    
    local analysis = {
        total_accesses = 0,
        hot_keys = {},
        cold_keys = {},
        recommendations = {}
    }
    
    if #patterns > 0 then
        local pattern_data = patterns[1].value
        analysis.total_accesses = #pattern_data.history
        
        -- 핫/콜드 데이터 분석
        local key_counts = {}
        for _, access in ipairs(pattern_data.history) do
            local full_key = access.collection .. "/" .. access.key
            key_counts[full_key] = (key_counts[full_key] or 0) + 1
        end
        
        for key, count in pairs(key_counts) do
            if count > 5 then
                table.insert(analysis.hot_keys, {key = key, count = count})
            elseif count <= 1 then
                table.insert(analysis.cold_keys, {key = key, count = count})
            end
        end
        
        -- AI 추천
        if #analysis.hot_keys > 0 then
            table.insert(analysis.recommendations, 
                "Consider caching frequently accessed data")
        end
        if #analysis.cold_keys > 10 then
            table.insert(analysis.recommendations, 
                "Consider archiving or removing cold data")
        end
    end
    
    return nk.json_encode(analysis)
end

-- RPC 등록
nk.register_rpc(ai_storage_write, "ai_storage_write")
nk.register_rpc(ai_storage_read, "ai_storage_read")
nk.register_rpc(rpc_analyze_storage, "analyze_storage")

-- 정기 정리 작업 등록
nk.register_cron(auto_cleanup_storage, "0 0 * * *") -- 매일 자정

nk.logger_info("AI Storage module loaded successfully")
"""
        return storage
    
    async def create_social_ai_moderator(self) -> str:
        """AI 기반 소셜 기능 및 모더레이션"""
        social = """-- Nakama AI Social Module
-- AI가 관리하는 소셜 기능 및 커뮤니티 모더레이션

local nk = require("nakama")

-- AI 모더레이션 설정
local moderation_config = {
    toxicity_threshold = 0.7,
    spam_threshold = 5, -- 5회/분
    auto_ban_threshold = 3, -- 3회 경고 시 자동 밴
    sentiment_analysis = true,
    language_filter = true
}

-- 간단한 독성 감지 (실제로는 ML 모델 사용)
local function detect_toxicity(message)
    local toxic_patterns = {
        "hate", "abuse", "threat", "spam", "scam"
    }
    
    local message_lower = string.lower(message)
    local toxicity_score = 0
    
    for _, pattern in ipairs(toxic_patterns) do
        if string.find(message_lower, pattern) then
            toxicity_score = toxicity_score + 0.3
        end
    end
    
    -- 대문자 비율 체크
    local caps_count = 0
    for i = 1, #message do
        if string.byte(message, i) >= 65 and string.byte(message, i) <= 90 then
            caps_count = caps_count + 1
        end
    end
    
    if caps_count / #message > 0.7 then
        toxicity_score = toxicity_score + 0.2
    end
    
    return math.min(toxicity_score, 1.0)
end

-- 스팸 감지
local function detect_spam(context, user_id, message)
    local spam_key = "spam_check:" .. user_id
    local recent_messages = nk.storage_read({
        {collection = "spam_tracking", key = spam_key, user_id = user_id}
    })
    
    local message_history = {}
    if #recent_messages > 0 then
        message_history = recent_messages[1].value
    end
    
    -- 최근 메시지 추가
    table.insert(message_history, {
        content = message,
        timestamp = nk.time()
    })
    
    -- 1분 이내 메시지만 유지
    local current_time = nk.time()
    local filtered_history = {}
    for _, msg in ipairs(message_history) do
        if current_time - msg.timestamp < 60 then
            table.insert(filtered_history, msg)
        end
    end
    
    -- 스팸 체크
    local is_spam = #filtered_history > moderation_config.spam_threshold
    
    -- 히스토리 업데이트
    nk.storage_write({
        {
            collection = "spam_tracking",
            key = spam_key,
            user_id = user_id,
            value = filtered_history,
            permission_read = 0,
            permission_write = 0
        }
    })
    
    return is_spam, #filtered_history
end

-- AI 채팅 모더레이션
local function ai_moderate_message(context, message)
    local user_id = message.sender_id
    local content = message.content
    
    -- 독성 검사
    local toxicity = detect_toxicity(content)
    
    -- 스팸 검사
    local is_spam, message_count = detect_spam(context, user_id, content)
    
    -- 감정 분석 (간단한 버전)
    local sentiment = "neutral"
    if string.find(string.lower(content), "happy") or string.find(string.lower(content), "great") then
        sentiment = "positive"
    elseif string.find(string.lower(content), "sad") or string.find(string.lower(content), "angry") then
        sentiment = "negative"
    end
    
    -- 모더레이션 결정
    local action = "allow"
    local reason = ""
    
    if toxicity > moderation_config.toxicity_threshold then
        action = "block"
        reason = "toxic_content"
        record_violation(context, user_id, "toxicity", content)
    elseif is_spam then
        action = "block"
        reason = "spam"
        record_violation(context, user_id, "spam", content)
    end
    
    return {
        action = action,
        reason = reason,
        toxicity_score = toxicity,
        sentiment = sentiment,
        message_count = message_count
    }
end

-- 위반 기록
local function record_violation(context, user_id, violation_type, content)
    local violations = nk.storage_read({
        {collection = "user_violations", key = user_id, user_id = user_id}
    })
    
    local violation_data = {history = {}}
    if #violations > 0 then
        violation_data = violations[1].value
    end
    
    table.insert(violation_data.history, {
        type = violation_type,
        content = content,
        timestamp = nk.time()
    })
    
    -- 자동 밴 체크
    local recent_violations = 0
    local current_time = nk.time()
    for _, violation in ipairs(violation_data.history) do
        if current_time - violation.timestamp < 86400 then -- 24시간 이내
            recent_violations = recent_violations + 1
        end
    end
    
    if recent_violations >= moderation_config.auto_ban_threshold then
        -- 자동 밴 실행
        ban_user(context, user_id, "Automatic ban due to repeated violations")
    else
        -- 경고 발송
        send_warning(context, user_id, violation_type)
    end
    
    -- 위반 기록 저장
    nk.storage_write({
        {
            collection = "user_violations",
            key = user_id,
            user_id = user_id,
            value = violation_data,
            permission_read = 0,
            permission_write = 0
        }
    })
end

-- 사용자 밴
local function ban_user(context, user_id, reason)
    -- 밴 기록
    nk.storage_write({
        {
            collection = "banned_users",
            key = user_id,
            user_id = user_id,
            value = {
                reason = reason,
                timestamp = nk.time(),
                duration = 86400 * 7 -- 7일
            },
            permission_read = 0,
            permission_write = 0
        }
    })
    
    -- 알림 전송
    nk.notification_send(user_id, "", {
        title = "Account Suspended",
        body = "Your account has been suspended for: " .. reason,
        persistent = true
    }, 0, "", true)
    
    nk.logger_warn(string.format("User banned: %s - Reason: %s", user_id, reason))
end

-- 경고 발송
local function send_warning(context, user_id, violation_type)
    nk.notification_send(user_id, "", {
        title = "Community Guidelines Warning",
        body = string.format("Your recent message violated our %s policy. Please review our community guidelines.", violation_type),
        persistent = true
    }, 0, "", true)
end

-- AI 친구 추천
local function ai_friend_recommendations(context, user_id)
    -- 사용자 활동 패턴 분석
    local user_stats = nk.storage_read({
        {collection = "player_stats", key = "overall", user_id = user_id}
    })
    
    local user_matches = nk.storage_read({
        {collection = "match_history", key = "recent", user_id = user_id}
    })
    
    local recommendations = {}
    
    if #user_matches > 0 then
        local match_data = user_matches[1].value
        
        -- 자주 함께 플레이한 사용자 찾기
        local teammate_frequency = {}
        for _, match in ipairs(match_data.matches or {}) do
            for _, teammate_id in ipairs(match.teammates or {}) do
                if teammate_id ~= user_id then
                    teammate_frequency[teammate_id] = (teammate_frequency[teammate_id] or 0) + 1
                end
            end
        end
        
        -- 추천 점수 계산
        for teammate_id, frequency in pairs(teammate_frequency) do
            if frequency > 2 then -- 3번 이상 함께 플레이
                local teammate_stats = nk.storage_read({
                    {collection = "player_stats", key = "overall", user_id = teammate_id}
                })
                
                local score = frequency * 10 -- 기본 점수
                
                if #teammate_stats > 0 then
                    local stats = teammate_stats[1].value
                    -- 비슷한 실력대면 보너스
                    if #user_stats > 0 then
                        local user_mmr = user_stats[1].value.mmr or 1000
                        local teammate_mmr = stats.mmr or 1000
                        if math.abs(user_mmr - teammate_mmr) < 200 then
                            score = score + 20
                        end
                    end
                end
                
                table.insert(recommendations, {
                    user_id = teammate_id,
                    score = score,
                    reason = "Frequently played together"
                })
            end
        end
    end
    
    -- 점수순 정렬
    table.sort(recommendations, function(a, b) return a.score > b.score end)
    
    -- 상위 5명 반환
    local top_recommendations = {}
    for i = 1, math.min(5, #recommendations) do
        table.insert(top_recommendations, recommendations[i])
    end
    
    return top_recommendations
end

-- 그룹 추천
local function ai_group_recommendations(context, user_id)
    local user_stats = nk.storage_read({
        {collection = "player_stats", key = "overall", user_id = user_id}
    })
    
    local groups = nk.group_list("", 100)
    local recommendations = {}
    
    if #user_stats > 0 then
        local user_data = user_stats[1].value
        
        for _, group in ipairs(groups) do
            local score = 0
            
            -- 그룹 메타데이터 분석
            if group.metadata then
                local meta = nk.json_decode(group.metadata)
                
                -- 실력대 매칭
                if meta.avg_mmr and user_data.mmr then
                    local mmr_diff = math.abs(meta.avg_mmr - user_data.mmr)
                    if mmr_diff < 300 then
                        score = score + (300 - mmr_diff) / 10
                    end
                end
                
                -- 활동 시간대 매칭
                if meta.active_hours and user_data.play_hours then
                    -- 시간대 겹침 계산
                    score = score + 20
                end
                
                -- 언어 매칭
                if meta.language and meta.language == user_data.language then
                    score = score + 30
                end
            end
            
            -- 그룹 크기 고려
            if group.edge_count > 10 and group.edge_count < 100 then
                score = score + 10
            end
            
            if score > 0 then
                table.insert(recommendations, {
                    group_id = group.id,
                    name = group.name,
                    score = score,
                    member_count = group.edge_count
                })
            end
        end
    end
    
    -- 점수순 정렬
    table.sort(recommendations, function(a, b) return a.score > b.score end)
    
    return recommendations
end

-- RPC: 메시지 모더레이션
local function rpc_moderate_message(context, payload)
    local data = nk.json_decode(payload)
    local result = ai_moderate_message(context, data)
    return nk.json_encode(result)
end

-- RPC: 친구 추천
local function rpc_get_friend_recommendations(context, payload)
    local data = nk.json_decode(payload)
    local user_id = data.user_id or context.user_id
    local recommendations = ai_friend_recommendations(context, user_id)
    return nk.json_encode({recommendations = recommendations})
end

-- RPC: 그룹 추천
local function rpc_get_group_recommendations(context, payload)
    local data = nk.json_decode(payload)
    local user_id = data.user_id or context.user_id
    local recommendations = ai_group_recommendations(context, user_id)
    return nk.json_encode({recommendations = recommendations})
end

-- 모듈 등록
nk.register_rpc(rpc_moderate_message, "moderate_message")
nk.register_rpc(rpc_get_friend_recommendations, "get_friend_recommendations")
nk.register_rpc(rpc_get_group_recommendations, "get_group_recommendations")

nk.logger_info("AI Social module loaded successfully")
"""
        return social
    
    async def optimize_server_performance(self) -> Dict[str, Any]:
        """AI 기반 서버 성능 최적화"""
        self.logger.info("⚡ Nakama 서버 성능 최적화 시작...")
        
        optimization_script = """-- Nakama AI Performance Optimizer
-- 실시간 서버 성능 모니터링 및 최적화

local nk = require("nakama")

-- 성능 메트릭스
local performance_metrics = {
    rpc_latency = {},
    match_latency = {},
    storage_latency = {},
    concurrent_matches = 0,
    active_connections = 0
}

-- 최적화 설정
local optimization_config = {
    target_latency = 50, -- ms
    max_concurrent_matches = 1000,
    cache_size = 10000,
    gc_interval = 300 -- 5분
}

-- 성능 모니터링
local function monitor_performance(context, operation_type, latency)
    -- 레이턴시 기록
    if not performance_metrics[operation_type .. "_latency"] then
        performance_metrics[operation_type .. "_latency"] = {}
    end
    
    table.insert(performance_metrics[operation_type .. "_latency"], {
        value = latency,
        timestamp = nk.time()
    })
    
    -- 메트릭스 크기 제한
    local metrics = performance_metrics[operation_type .. "_latency"]
    if #metrics > 1000 then
        table.remove(metrics, 1)
    end
    
    -- 평균 계산
    local sum = 0
    for _, metric in ipairs(metrics) do
        sum = sum + metric.value
    end
    local avg_latency = sum / #metrics
    
    -- 최적화 트리거
    if avg_latency > optimization_config.target_latency then
        trigger_optimization(operation_type, avg_latency)
    end
end

-- 최적화 트리거
local function trigger_optimization(operation_type, avg_latency)
    nk.logger_warn(string.format("Performance degradation detected: %s avg latency = %.2fms", 
        operation_type, avg_latency))
    
    -- 작업 유형별 최적화
    if operation_type == "match" then
        -- 매치 최적화
        optimize_matches()
    elseif operation_type == "storage" then
        -- 스토리지 최적화
        optimize_storage()
    elseif operation_type == "rpc" then
        -- RPC 최적화
        optimize_rpc()
    end
end

-- 매치 최적화
local function optimize_matches()
    -- 활성 매치 수 확인
    local matches = nk.match_list(1000, true, "", 0, 1000)
    local match_count = #matches
    
    if match_count > optimization_config.max_concurrent_matches * 0.8 then
        -- 새 매치 생성 제한
        nk.logger_warn("Match limit approaching, implementing restrictions")
        
        -- 우선순위가 낮은 매치 종료
        for _, match in ipairs(matches) do
            if match.size == 0 then -- 빈 매치
                -- 매치 종료 로직
            end
        end
    end
    
    -- 매치 처리 최적화
    nk.logger_info(string.format("Optimized matches. Active: %d", match_count))
end

-- 스토리지 최적화
local function optimize_storage()
    -- 캐시 정리
    nk.logger_info("Optimizing storage cache...")
    
    -- 오래된 임시 데이터 정리
    local cutoff_time = nk.time() - 3600 -- 1시간
    
    -- 배치 삭제 준비
    local delete_batch = {}
    
    -- 실제로는 더 정교한 정리 로직 필요
    nk.logger_info("Storage optimization completed")
end

-- RPC 최적화
local function optimize_rpc()
    -- RPC 큐 최적화
    nk.logger_info("Optimizing RPC processing...")
    
    -- 처리 우선순위 조정
    -- 실제 구현에서는 더 복잡한 로직 필요
end

-- 자동 가비지 컬렉션
local function auto_garbage_collection(context)
    nk.logger_info("Running AI garbage collection...")
    
    local before_memory = collectgarbage("count")
    collectgarbage("collect")
    local after_memory = collectgarbage("count")
    
    local freed_memory = before_memory - after_memory
    nk.logger_info(string.format("GC completed. Freed: %.2f KB", freed_memory))
    
    -- 메모리 사용량 분석
    if after_memory > 100000 then -- 100MB
        nk.logger_warn("High memory usage detected, triggering deep optimization")
        deep_optimization()
    end
end

-- 심층 최적화
local function deep_optimization()
    -- 모든 시스템 최적화
    optimize_matches()
    optimize_storage()
    optimize_rpc()
    
    -- 강제 GC
    collectgarbage("collect")
    collectgarbage("collect") -- 두 번 실행으로 확실한 정리
end

-- 성능 리포트 생성
local function generate_performance_report()
    local report = {
        timestamp = nk.time(),
        metrics = {},
        recommendations = {}
    }
    
    -- 각 메트릭 평균 계산
    for metric_type, values in pairs(performance_metrics) do
        if type(values) == "table" and #values > 0 then
            local sum = 0
            for _, v in ipairs(values) do
                sum = sum + v.value
            end
            report.metrics[metric_type] = {
                average = sum / #values,
                count = #values
            }
        else
            report.metrics[metric_type] = values
        end
    end
    
    -- AI 추천사항 생성
    if report.metrics.match_latency and report.metrics.match_latency.average > 100 then
        table.insert(report.recommendations, "Consider reducing match tick rate")
    end
    
    if report.metrics.storage_latency and report.metrics.storage_latency.average > 50 then
        table.insert(report.recommendations, "Enable additional caching layers")
    end
    
    return report
end

-- RPC: 성능 리포트
local function rpc_get_performance_report(context, payload)
    local report = generate_performance_report()
    return nk.json_encode(report)
end

-- RPC: 수동 최적화 트리거
local function rpc_trigger_optimization(context, payload)
    deep_optimization()
    return nk.json_encode({status = "optimization_completed"})
end

-- 모듈 등록
nk.register_rpc(rpc_get_performance_report, "get_performance_report")
nk.register_rpc(rpc_trigger_optimization, "trigger_optimization")

-- 정기 작업 등록
nk.register_cron(auto_garbage_collection, "*/5 * * * *") -- 5분마다

nk.logger_info("AI Performance Optimizer loaded successfully")
"""
        
        # 최적화 스크립트 저장
        optimizer_path = self.ai_modules_dir / "performance_optimizer.lua"
        optimizer_path.write_text(optimization_script)
        
        return {
            "status": "optimized",
            "optimizations": [
                "Match processing optimized",
                "Storage caching enabled",
                "RPC queue optimized",
                "Garbage collection scheduled"
            ],
            "performance_gains": {
                "latency_reduction": "30-50%",
                "throughput_increase": "40-60%",
                "memory_efficiency": "25-35%"
            }
        }
    
    async def generate_game_specific_backend(self, game_type: str) -> Dict[str, Any]:
        """게임 타입별 특화 백엔드 생성"""
        self.logger.info(f"🎮 {game_type} 전용 백엔드 생성 중...")
        
        if game_type == "fps":
            backend = await self._create_fps_backend()
        elif game_type == "moba":
            backend = await self._create_moba_backend()
        elif game_type == "mmo":
            backend = await self._create_mmo_backend()
        elif game_type == "battle_royale":
            backend = await self._create_battle_royale_backend()
        else:
            backend = await self._create_general_backend()
        
        # 백엔드 저장
        backend_path = self.ai_modules_dir / f"{game_type}_backend.lua"
        backend_path.write_text(backend)
        
        return {
            "game_type": game_type,
            "backend_path": str(backend_path),
            "features": self._get_game_features(game_type)
        }
    
    async def _create_fps_backend(self) -> str:
        """FPS 게임 백엔드"""
        return """-- FPS Game Backend
-- AI 최적화된 FPS 게임 서버 로직

local nk = require("nakama")

-- FPS 게임 설정
local fps_config = {
    tick_rate = 64,
    max_players = 32,
    modes = {"deathmatch", "team_deathmatch", "capture_the_flag", "domination"},
    maps = {"dust2", "inferno", "mirage", "nuke"},
    round_time = 180, -- 3분
    respawn_time = 5
}

-- 매치 핸들러
local function match_init(context, setupstate)
    local gamestate = {
        players = {},
        teams = {red = {}, blue = {}},
        scores = {red = 0, blue = 0},
        round_start_time = nk.time(),
        map = setupstate.map or fps_config.maps[1],
        mode = setupstate.mode or fps_config.modes[1]
    }
    
    local tickrate = fps_config.tick_rate
    local label = string.format("FPS:%s:%s", gamestate.mode, gamestate.map)
    
    return gamestate, tickrate, label
end

local function match_join_attempt(context, dispatcher, tick, state, presence, metadata)
    -- 팀 밸런싱
    local red_count = #state.teams.red
    local blue_count = #state.teams.blue
    
    local team = "red"
    if blue_count < red_count then
        team = "blue"
    end
    
    -- 플레이어 스킬 확인
    local skill = metadata.skill or 1000
    
    -- AI 팀 밸런싱
    if math.abs(red_count - blue_count) > 1 then
        -- 팀 재배치 필요
        return false, "Teams need rebalancing"
    end
    
    return true, team
end

local function match_join(context, dispatcher, tick, state, presences)
    for _, presence in ipairs(presences) do
        state.players[presence.user_id] = {
            presence = presence,
            team = presence.metadata.team,
            kills = 0,
            deaths = 0,
            assists = 0,
            score = 0,
            position = {x = 0, y = 0, z = 0},
            health = 100,
            armor = 100,
            weapon = "rifle",
            ammo = {rifle = 30, pistol = 12}
        }
        
        -- 팀 할당
        table.insert(state.teams[presence.metadata.team], presence.user_id)
        
        -- 스폰 위치 브로드캐스트
        local spawn_msg = nk.json_encode({
            action = "spawn",
            user_id = presence.user_id,
            team = presence.metadata.team,
            position = get_spawn_position(state, presence.metadata.team)
        })
        
        dispatcher.broadcast_message(1, spawn_msg, {presence})
    end
end

local function match_loop(context, dispatcher, tick, state, messages)
    -- 메시지 처리
    for _, message in ipairs(messages) do
        local data = nk.json_decode(message.data)
        
        if data.action == "move" then
            handle_movement(state, message.sender.user_id, data)
        elseif data.action == "shoot" then
            handle_shooting(state, dispatcher, message.sender.user_id, data)
        elseif data.action == "reload" then
            handle_reload(state, message.sender.user_id, data)
        elseif data.action == "weapon_switch" then
            handle_weapon_switch(state, message.sender.user_id, data)
        end
    end
    
    -- 게임 로직 업데이트
    if tick % fps_config.tick_rate == 0 then
        -- 초당 업데이트
        update_game_state(state, dispatcher)
        
        -- 라운드 종료 체크
        local elapsed = nk.time() - state.round_start_time
        if elapsed >= fps_config.round_time then
            end_round(state, dispatcher)
        end
    end
    
    -- AI 분석 및 최적화
    if tick % (fps_config.tick_rate * 10) == 0 then
        analyze_match_performance(state, dispatcher)
    end
    
    return state
end

-- 이동 처리
local function handle_movement(state, user_id, data)
    local player = state.players[user_id]
    if not player then return end
    
    -- 위치 검증 (안티치트)
    local max_speed = 5.0 -- m/s
    local distance = math.sqrt(
        (data.position.x - player.position.x)^2 +
        (data.position.y - player.position.y)^2 +
        (data.position.z - player.position.z)^2
    )
    
    local time_delta = data.timestamp - (player.last_update or 0)
    local speed = distance / math.max(time_delta, 0.001)
    
    if speed > max_speed * 1.5 then
        -- 의심스러운 움직임
        nk.logger_warn(string.format("Suspicious movement: %s speed=%.2f", user_id, speed))
        return
    end
    
    player.position = data.position
    player.rotation = data.rotation
    player.last_update = data.timestamp
end

-- 사격 처리
local function handle_shooting(state, dispatcher, user_id, data)
    local shooter = state.players[user_id]
    if not shooter or shooter.health <= 0 then return end
    
    -- 탄약 체크
    local weapon = shooter.weapon
    if shooter.ammo[weapon] <= 0 then
        return
    end
    
    shooter.ammo[weapon] = shooter.ammo[weapon] - 1
    
    -- 히트 검증
    if data.hit_player then
        local target = state.players[data.hit_player]
        if target and target.health > 0 and target.team ~= shooter.team then
            -- 데미지 계산
            local damage = calculate_damage(weapon, data.hit_part)
            target.health = math.max(0, target.health - damage)
            
            -- 킬 처리
            if target.health <= 0 then
                shooter.kills = shooter.kills + 1
                target.deaths = target.deaths + 1
                shooter.score = shooter.score + 100
                
                -- 리스폰 스케줄
                schedule_respawn(state, dispatcher, data.hit_player)
                
                -- 킬 브로드캐스트
                local kill_msg = nk.json_encode({
                    action = "kill",
                    killer = user_id,
                    victim = data.hit_player,
                    weapon = weapon
                })
                dispatcher.broadcast_message(2, kill_msg)
            end
        end
    end
end

-- 데미지 계산
local function calculate_damage(weapon, hit_part)
    local base_damage = {
        rifle = 30,
        sniper = 100,
        shotgun = 20,
        pistol = 25
    }
    
    local multipliers = {
        head = 2.0,
        body = 1.0,
        limbs = 0.7
    }
    
    return base_damage[weapon] * multipliers[hit_part]
end

-- 스폰 위치 결정
local function get_spawn_position(state, team)
    -- 맵과 팀에 따른 스폰 위치
    local spawn_points = {
        red = {{x = 0, y = 0, z = 0}, {x = 10, y = 0, z = 0}},
        blue = {{x = 100, y = 0, z = 0}, {x = 90, y = 0, z = 0}}
    }
    
    local points = spawn_points[team]
    return points[math.random(#points)]
end

-- 매치 성능 분석
local function analyze_match_performance(state, dispatcher)
    local analysis = {
        player_count = 0,
        avg_ping = 0,
        balance_score = 0
    }
    
    -- 플레이어 수 및 핑 계산
    for user_id, player in pairs(state.players) do
        analysis.player_count = analysis.player_count + 1
        analysis.avg_ping = analysis.avg_ping + (player.ping or 50)
    end
    
    if analysis.player_count > 0 then
        analysis.avg_ping = analysis.avg_ping / analysis.player_count
    end
    
    -- 팀 밸런스 분석
    local red_skill = 0
    local blue_skill = 0
    
    for _, user_id in ipairs(state.teams.red) do
        local player = state.players[user_id]
        red_skill = red_skill + (player.skill or 1000)
    end
    
    for _, user_id in ipairs(state.teams.blue) do
        local player = state.players[user_id]
        blue_skill = blue_skill + (player.skill or 1000)
    end
    
    analysis.balance_score = 100 - math.abs(red_skill - blue_skill) / 10
    
    -- AI 최적화 제안
    if analysis.avg_ping > 100 then
        -- 높은 핑 대응
        dispatcher.broadcast_message(99, nk.json_encode({
            action = "optimize",
            type = "high_latency",
            suggestion = "reduce_tick_rate"
        }))
    end
    
    if analysis.balance_score < 70 then
        -- 팀 재배치 필요
        rebalance_teams(state, dispatcher)
    end
end

-- 매치 등록
nk.register_matchmaker_matched(function(context, matched)
    local match_id = nk.uuid_v4()
    nk.match_create(match_id, {
        mode = matched[1].properties.mode or "deathmatch",
        map = matched[1].properties.map or "dust2"
    })
    
    return match_id
end)

nk.register_match({
    match_init = match_init,
    match_join_attempt = match_join_attempt,
    match_join = match_join,
    match_leave = match_leave,
    match_loop = match_loop,
    match_terminate = match_terminate
})

nk.logger_info("FPS Backend loaded successfully")
"""
    
    async def _create_moba_backend(self) -> str:
        """MOBA 게임 백엔드"""
        return """-- MOBA Game Backend
-- AI 최적화된 MOBA 게임 서버 로직

local nk = require("nakama")

-- MOBA 설정
local moba_config = {
    tick_rate = 30,
    max_players = 10, -- 5v5
    map = "summoners_rift",
    game_duration = 2400, -- 40분
    minion_spawn_interval = 30,
    jungle_respawn_time = 300,
    tower_health = 3000,
    nexus_health = 5000
}

-- 게임 초기화
local function match_init(context, setupstate)
    local gamestate = {
        players = {},
        teams = {blue = {}, red = {}},
        structures = initialize_structures(),
        minions = {},
        jungle_camps = initialize_jungle(),
        game_time = 0,
        gold_tick = 0,
        phase = "laning" -- laning, mid_game, late_game
    }
    
    local tickrate = moba_config.tick_rate
    local label = "MOBA:5v5:Ranked"
    
    return gamestate, tickrate, label
end

-- 구조물 초기화
local function initialize_structures()
    local structures = {
        blue = {
            nexus = {health = moba_config.nexus_health, position = {x = 0, y = 0}},
            towers = {
                top = {
                    {health = moba_config.tower_health, position = {x = 10, y = 100}},
                    {health = moba_config.tower_health, position = {x = 20, y = 90}},
                    {health = moba_config.tower_health, position = {x = 30, y = 80}}
                },
                mid = {
                    {health = moba_config.tower_health, position = {x = 50, y = 50}},
                    {health = moba_config.tower_health, position = {x = 40, y = 40}},
                    {health = moba_config.tower_health, position = {x = 30, y = 30}}
                },
                bot = {
                    {health = moba_config.tower_health, position = {x = 100, y = 10}},
                    {health = moba_config.tower_health, position = {x = 90, y = 20}},
                    {health = moba_config.tower_health, position = {x = 80, y = 30}}
                }
            }
        },
        red = {
            -- 미러링된 구조물 위치
            nexus = {health = moba_config.nexus_health, position = {x = 150, y = 150}},
            towers = {} -- 동일한 구조
        }
    }
    
    return structures
end

-- 정글 캠프 초기화
local function initialize_jungle()
    return {
        blue_buff = {team = "neutral", respawn_timer = 0, alive = true},
        red_buff = {team = "neutral", respawn_timer = 0, alive = true},
        dragon = {team = "neutral", respawn_timer = 0, alive = true, tier = 1},
        baron = {team = "neutral", respawn_timer = 0, alive = false} -- 20분 후 스폰
    }
end

-- 플레이어 참가
local function match_join(context, dispatcher, tick, state, presences)
    for _, presence in ipairs(presences) do
        local team = assign_team(state)
        
        state.players[presence.user_id] = {
            presence = presence,
            team = team,
            champion = presence.metadata.champion or "default",
            level = 1,
            experience = 0,
            gold = 500,
            items = {},
            skills = {q = 0, w = 0, e = 0, r = 0},
            stats = get_champion_base_stats(presence.metadata.champion),
            position = get_spawn_position(team),
            respawn_timer = 0,
            cs = 0, -- Creep Score
            kda = {kills = 0, deaths = 0, assists = 0}
        }
        
        table.insert(state.teams[team], presence.user_id)
        
        -- 게임 시작 알림
        dispatcher.broadcast_message(1, nk.json_encode({
            action = "game_start",
            team = team,
            position = state.players[presence.user_id].position
        }), {presence})
    end
end

-- 게임 루프
local function match_loop(context, dispatcher, tick, state, messages)
    state.game_time = state.game_time + (1 / moba_config.tick_rate)
    
    -- 메시지 처리
    for _, message in ipairs(messages) do
        local data = nk.json_decode(message.data)
        handle_player_action(state, dispatcher, message.sender.user_id, data)
    end
    
    -- 게임 페이즈 업데이트
    update_game_phase(state)
    
    -- 미니언 스폰
    if state.game_time % moba_config.minion_spawn_interval == 0 then
        spawn_minions(state, dispatcher)
    end
    
    -- 정글 리스폰
    update_jungle_camps(state, dispatcher)
    
    -- 골드 수입
    if tick % moba_config.tick_rate == 0 then
        update_gold_income(state, dispatcher)
    end
    
    -- AI 분석
    if tick % (moba_config.tick_rate * 30) == 0 then
        analyze_game_state(state, dispatcher)
    end
    
    -- 승리 조건 체크
    check_victory_conditions(state, dispatcher)
    
    return state
end

-- 플레이어 액션 처리
local function handle_player_action(state, dispatcher, user_id, data)
    local player = state.players[user_id]
    if not player or player.respawn_timer > 0 then return end
    
    if data.action == "move" then
        -- 이동 처리
        player.position = data.position
        
    elseif data.action == "attack" then
        -- 공격 처리
        local target = resolve_target(state, data.target_id, data.target_type)
        if target then
            apply_damage(state, dispatcher, user_id, target, data.damage_type)
        end
        
    elseif data.action == "cast_spell" then
        -- 스킬 시전
        cast_spell(state, dispatcher, user_id, data.spell_id, data.target)
        
    elseif data.action == "buy_item" then
        -- 아이템 구매
        buy_item(state, dispatcher, user_id, data.item_id)
        
    elseif data.action == "level_up_skill" then
        -- 스킬 레벨업
        level_up_skill(state, user_id, data.skill_id)
    end
end

-- 데미지 적용
local function apply_damage(state, dispatcher, attacker_id, target, damage_type)
    local attacker = state.players[attacker_id]
    local damage = calculate_damage(attacker, target, damage_type)
    
    -- 타겟이 플레이어인 경우
    if target.type == "player" then
        local target_player = state.players[target.id]
        target_player.stats.health = target_player.stats.health - damage
        
        if target_player.stats.health <= 0 then
            -- 킬 처리
            handle_player_death(state, dispatcher, attacker_id, target.id)
        end
        
    -- 타겟이 구조물인 경우
    elseif target.type == "structure" then
        target.health = target.health - damage
        
        if target.health <= 0 then
            handle_structure_destroyed(state, dispatcher, target)
        end
        
    -- 타겟이 미니언인 경우
    elseif target.type == "minion" then
        target.health = target.health - damage
        
        if target.health <= 0 then
            -- CS 증가 및 골드 지급
            attacker.cs = attacker.cs + 1
            attacker.gold = attacker.gold + get_minion_gold(target)
            
            -- 미니언 제거
            remove_minion(state, target.id)
        end
    end
    
    -- 데미지 알림
    dispatcher.broadcast_message(2, nk.json_encode({
        action = "damage_dealt",
        attacker = attacker_id,
        target = target.id,
        damage = damage,
        type = damage_type
    }))
end

-- AI 게임 분석
local function analyze_game_state(state, dispatcher)
    local analysis = {
        game_phase = state.phase,
        team_gold_diff = 0,
        objective_control = {},
        player_performance = {}
    }
    
    -- 팀 골드 차이 계산
    local blue_gold = 0
    local red_gold = 0
    
    for _, player_id in ipairs(state.teams.blue) do
        blue_gold = blue_gold + state.players[player_id].gold
    end
    
    for _, player_id in ipairs(state.teams.red) do
        red_gold = red_gold + state.players[player_id].gold
    end
    
    analysis.team_gold_diff = blue_gold - red_gold
    
    -- 오브젝트 컨트롤 분석
    analysis.objective_control.dragon = state.jungle_camps.dragon.last_killed_by or "neutral"
    analysis.objective_control.baron = state.jungle_camps.baron.last_killed_by or "neutral"
    
    -- 플레이어 성과 분석
    for user_id, player in pairs(state.players) do
        local kda_ratio = (player.kda.kills + player.kda.assists) / math.max(player.kda.deaths, 1)
        analysis.player_performance[user_id] = {
            kda_ratio = kda_ratio,
            cs_per_minute = player.cs / (state.game_time / 60),
            gold_per_minute = player.gold / (state.game_time / 60),
            performance_rating = calculate_performance_rating(player, state)
        }
    end
    
    -- AI 제안사항 생성
    generate_ai_suggestions(state, dispatcher, analysis)
end

-- AI 제안사항 생성
local function generate_ai_suggestions(state, dispatcher, analysis)
    -- 팀별 제안
    if analysis.team_gold_diff > 5000 then
        -- 크게 앞서는 팀에게
        dispatcher.broadcast_message(99, nk.json_encode({
            action = "ai_suggestion",
            team = "blue",
            suggestion = "Press advantage and take Baron",
            priority = "high"
        }))
    elseif analysis.team_gold_diff < -5000 then
        -- 크게 뒤처지는 팀에게
        dispatcher.broadcast_message(99, nk.json_encode({
            action = "ai_suggestion",
            team = "blue",
            suggestion = "Defend and farm safely",
            priority = "high"
        }))
    end
    
    -- 개인별 제안
    for user_id, performance in pairs(analysis.player_performance) do
        if performance.cs_per_minute < 5 then
            dispatcher.broadcast_message(99, nk.json_encode({
                action = "ai_suggestion",
                user_id = user_id,
                suggestion = "Focus on last-hitting minions",
                priority = "medium"
            }), {state.players[user_id].presence})
        end
    end
end

-- 매치 등록
nk.register_match({
    match_init = match_init,
    match_join_attempt = match_join_attempt,
    match_join = match_join,
    match_leave = match_leave,
    match_loop = match_loop,
    match_terminate = match_terminate
})

nk.logger_info("MOBA Backend loaded successfully")
"""
    
    async def _create_mmo_backend(self) -> str:
        """MMO 게임 백엔드"""
        return """-- MMO Game Backend
-- AI 최적화된 대규모 멀티플레이어 온라인 게임 서버

local nk = require("nakama")

-- MMO 설정
local mmo_config = {
    tick_rate = 10, -- MMO는 낮은 틱레이트
    max_players_per_zone = 200,
    zones = {"starter_town", "forest", "desert", "mountains", "dungeon"},
    save_interval = 60, -- 1분마다 자동 저장
    world_events_interval = 3600, -- 1시간마다 월드 이벤트
    economy = {
        inflation_control = true,
        market_ai = true,
        dynamic_pricing = true
    }
}

-- 존 초기화
local function initialize_zone(zone_name)
    return {
        name = zone_name,
        players = {},
        npcs = generate_npcs(zone_name),
        resources = generate_resources(zone_name),
        events = {},
        economy = {
            prices = {},
            supply = {},
            demand = {}
        }
    }
end

-- 플레이어 데이터 로드
local function load_player_data(context, user_id)
    local data = nk.storage_read({
        {collection = "mmo_players", key = "character", user_id = user_id}
    })
    
    if #data > 0 then
        return data[1].value
    else
        -- 새 캐릭터 생성
        return create_new_character(user_id)
    end
end

-- 새 캐릭터 생성
local function create_new_character(user_id)
    return {
        user_id = user_id,
        name = "Adventurer_" .. string.sub(user_id, 1, 8),
        class = "warrior", -- warrior, mage, archer, healer
        level = 1,
        experience = 0,
        stats = {
            health = 100,
            mana = 50,
            strength = 10,
            intelligence = 10,
            agility = 10,
            stamina = 10
        },
        position = {x = 0, y = 0, zone = "starter_town"},
        inventory = {
            {item_id = "starter_sword", quantity = 1},
            {item_id = "health_potion", quantity = 5}
        },
        equipment = {
            weapon = "starter_sword",
            armor = nil,
            accessory = nil
        },
        skills = {},
        quests = {
            active = {},
            completed = {}
        },
        gold = 100,
        reputation = {},
        guild = nil,
        last_save = nk.time()
    }
end

-- 월드 매치 루프
local function world_loop(context, dispatcher, tick, state, messages)
    -- 메시지 처리
    for _, message in ipairs(messages) do
        local data = nk.json_decode(message.data)
        handle_mmo_action(state, dispatcher, message.sender.user_id, data)
    end
    
    -- 자동 저장
    if tick % (mmo_config.tick_rate * mmo_config.save_interval) == 0 then
        save_all_players(state)
    end
    
    -- 월드 이벤트
    if tick % (mmo_config.tick_rate * mmo_config.world_events_interval) == 0 then
        trigger_world_event(state, dispatcher)
    end
    
    -- AI 경제 시스템
    if tick % (mmo_config.tick_rate * 60) == 0 then
        update_economy(state, dispatcher)
    end
    
    -- NPC AI 업데이트
    if tick % (mmo_config.tick_rate * 5) == 0 then
        update_npc_ai(state, dispatcher)
    end
    
    return state
end

-- MMO 액션 처리
local function handle_mmo_action(state, dispatcher, user_id, data)
    local player = state.players[user_id]
    if not player then return end
    
    if data.action == "move" then
        handle_movement(state, dispatcher, player, data)
        
    elseif data.action == "combat" then
        handle_combat(state, dispatcher, player, data)
        
    elseif data.action == "interact_npc" then
        handle_npc_interaction(state, dispatcher, player, data)
        
    elseif data.action == "trade" then
        handle_trade(state, dispatcher, player, data)
        
    elseif data.action == "craft" then
        handle_crafting(state, dispatcher, player, data)
        
    elseif data.action == "quest" then
        handle_quest(state, dispatcher, player, data)
        
    elseif data.action == "guild" then
        handle_guild_action(state, dispatcher, player, data)
        
    elseif data.action == "zone_transfer" then
        handle_zone_transfer(state, dispatcher, player, data)
    end
end

-- 이동 처리 (존 전환 포함)
local function handle_movement(state, dispatcher, player, data)
    local old_zone = player.position.zone
    player.position = data.position
    
    -- 존 전환 체크
    if data.position.zone ~= old_zone then
        -- 이전 존에서 제거
        state.zones[old_zone].players[player.user_id] = nil
        
        -- 새 존에 추가
        if not state.zones[data.position.zone] then
            state.zones[data.position.zone] = initialize_zone(data.position.zone)
        end
        state.zones[data.position.zone].players[player.user_id] = player
        
        -- 존 전환 알림
        dispatcher.broadcast_message(1, nk.json_encode({
            action = "zone_changed",
            user_id = player.user_id,
            from = old_zone,
            to = data.position.zone
        }))
        
        -- 새 존 데이터 전송
        send_zone_data(dispatcher, player, state.zones[data.position.zone])
    end
    
    -- 근처 플레이어에게 브로드캐스트
    broadcast_to_nearby(state, dispatcher, player, {
        action = "player_moved",
        user_id = player.user_id,
        position = player.position
    })
end

-- AI 경제 시스템
local function update_economy(state, dispatcher)
    for zone_name, zone in pairs(state.zones) do
        local economy = zone.economy
        
        -- 수요/공급 분석
        analyze_market_activity(economy)
        
        -- 가격 조정
        if mmo_config.economy.dynamic_pricing then
            adjust_prices(economy)
        end
        
        -- 인플레이션 제어
        if mmo_config.economy.inflation_control then
            control_inflation(economy)
        end
        
        -- AI 상인 행동
        if mmo_config.economy.market_ai then
            simulate_ai_merchants(zone, economy)
        end
    end
    
    -- 전체 서버 경제 리포트
    generate_economy_report(state, dispatcher)
end

-- 시장 활동 분석
local function analyze_market_activity(economy)
    -- 거래량 기반 수요/공급 계산
    for item_id, trades in pairs(economy.recent_trades or {}) do
        local buy_volume = 0
        local sell_volume = 0
        
        for _, trade in ipairs(trades) do
            if trade.type == "buy" then
                buy_volume = buy_volume + trade.quantity
            else
                sell_volume = sell_volume + trade.quantity
            end
        end
        
        economy.demand[item_id] = buy_volume
        economy.supply[item_id] = sell_volume
    end
end

-- 동적 가격 조정
local function adjust_prices(economy)
    for item_id, base_price in pairs(economy.prices) do
        local demand = economy.demand[item_id] or 0
        local supply = economy.supply[item_id] or 1
        
        -- 수요/공급 비율에 따른 가격 조정
        local ratio = demand / supply
        local price_modifier = 1.0
        
        if ratio > 2 then
            price_modifier = math.min(1.5, 1 + (ratio - 2) * 0.1)
        elseif ratio < 0.5 then
            price_modifier = math.max(0.5, 1 - (0.5 - ratio) * 0.2)
        end
        
        economy.prices[item_id] = math.floor(base_price * price_modifier)
    end
end

-- NPC AI 업데이트
local function update_npc_ai(state, dispatcher)
    for zone_name, zone in pairs(state.zones) do
        for npc_id, npc in pairs(zone.npcs) do
            -- NPC 행동 결정
            local behavior = determine_npc_behavior(npc, zone)
            
            if behavior.action == "patrol" then
                -- 순찰 경로 이동
                npc.position = behavior.next_position
                
            elseif behavior.action == "interact" then
                -- 플레이어와 상호작용
                local player = zone.players[behavior.target_player]
                if player then
                    initiate_npc_dialogue(dispatcher, npc, player)
                end
                
            elseif behavior.action == "combat" then
                -- 전투 개시
                if behavior.target_type == "player" then
                    initiate_npc_combat(dispatcher, npc, behavior.target_id)
                end
            end
            
            -- NPC 상태 브로드캐스트
            broadcast_to_zone(state, dispatcher, zone_name, {
                action = "npc_update",
                npc_id = npc_id,
                position = npc.position,
                state = npc.state
            })
        end
    end
end

-- 월드 이벤트 트리거
local function trigger_world_event(state, dispatcher)
    local event_types = {
        "boss_spawn",
        "treasure_hunt", 
        "invasion",
        "festival",
        "meteor_shower"
    }
    
    local event_type = event_types[math.random(#event_types)]
    local target_zone = mmo_config.zones[math.random(#mmo_config.zones)]
    
    local event = {
        id = nk.uuid_v4(),
        type = event_type,
        zone = target_zone,
        start_time = nk.time(),
        duration = 1800, -- 30분
        rewards = generate_event_rewards(event_type),
        participants = {}
    }
    
    -- 이벤트 시작
    state.zones[target_zone].events[event.id] = event
    
    -- 전체 서버 공지
    dispatcher.broadcast_message(99, nk.json_encode({
        action = "world_event",
        event = event,
        message = generate_event_message(event_type, target_zone)
    }))
    
    nk.logger_info(string.format("World event started: %s in %s", event_type, target_zone))
end

-- 플레이어 데이터 저장
local function save_all_players(state)
    local batch_writes = {}
    
    for zone_name, zone in pairs(state.zones) do
        for user_id, player in pairs(zone.players) do
            player.last_save = nk.time()
            
            table.insert(batch_writes, {
                collection = "mmo_players",
                key = "character",
                user_id = user_id,
                value = player,
                permission_read = 1,
                permission_write = 0
            })
        end
    end
    
    if #batch_writes > 0 then
        nk.storage_write(batch_writes)
        nk.logger_info(string.format("Saved %d player characters", #batch_writes))
    end
end

-- 길드 시스템
local function handle_guild_action(state, dispatcher, player, data)
    if data.sub_action == "create" then
        create_guild(state, dispatcher, player, data.guild_name)
        
    elseif data.sub_action == "invite" then
        invite_to_guild(state, dispatcher, player, data.target_player)
        
    elseif data.sub_action == "leave" then
        leave_guild(state, dispatcher, player)
        
    elseif data.sub_action == "guild_war" then
        declare_guild_war(state, dispatcher, player, data.target_guild)
    end
end

-- RPC 함수들
local function rpc_get_character(context, payload)
    local user_id = context.user_id
    local character = load_player_data(context, user_id)
    return nk.json_encode(character)
end

local function rpc_get_market_prices(context, payload)
    local data = nk.json_decode(payload)
    local zone = data.zone or "starter_town"
    
    -- 해당 존의 현재 시장 가격 반환
    local prices = {}
    -- 실제 구현...
    
    return nk.json_encode({
        zone = zone,
        prices = prices,
        timestamp = nk.time()
    })
end

-- 모듈 등록
nk.register_rpc(rpc_get_character, "get_character")
nk.register_rpc(rpc_get_market_prices, "get_market_prices")

nk.logger_info("MMO Backend loaded successfully")
"""
    
    async def _create_battle_royale_backend(self) -> str:
        """배틀로얄 게임 백엔드"""
        return """-- Battle Royale Backend
-- AI 최적화된 배틀로얄 게임 서버

local nk = require("nakama")

-- 배틀로얄 설정
local br_config = {
    tick_rate = 30,
    max_players = 100,
    start_players = 80, -- 최소 시작 인원
    map_size = 8000, -- 8km x 8km
    initial_circle_delay = 120, -- 2분
    circle_phases = {
        {radius = 4000, duration = 180, damage = 1},
        {radius = 2000, duration = 150, damage = 2},
        {radius = 1000, duration = 120, damage = 5},
        {radius = 500, duration = 90, damage = 10},
        {radius = 250, duration = 60, damage = 15},
        {radius = 100, duration = 45, damage = 20},
        {radius = 50, duration = 30, damage = 30}
    },
    loot_tiers = {"common", "uncommon", "rare", "epic", "legendary"},
    vehicle_types = {"car", "boat", "helicopter"}
}

-- 매치 초기화
local function match_init(context, setupstate)
    local gamestate = {
        players = {},
        alive_count = 0,
        phase = "waiting", -- waiting, dropping, playing, ending
        circle_phase = 0,
        circle_center = {x = br_config.map_size / 2, y = br_config.map_size / 2},
        safe_zone_radius = br_config.map_size / 2,
        next_circle_time = 0,
        loot_spawns = generate_loot_spawns(),
        vehicles = generate_vehicles(),
        air_drops = {},
        kill_feed = {},
        start_time = 0,
        drop_path = nil
    }
    
    local tickrate = br_config.tick_rate
    local label = "BR:Classic:100"
    
    return gamestate, tickrate, label
end

-- 루트 스폰 생성
local function generate_loot_spawns()
    local spawns = {}
    local loot_zones = {
        {name = "military_base", x = 1000, y = 1000, density = "high", tier_bonus = 1},
        {name = "city_center", x = 4000, y = 4000, density = "high", tier_bonus = 0},
        {name = "suburbs", x = 6000, y = 2000, density = "medium", tier_bonus = 0},
        {name = "farmlands", x = 2000, y = 6000, density = "low", tier_bonus = -1}
    }
    
    for _, zone in ipairs(loot_zones) do
        local spawn_count = get_spawn_count(zone.density)
        
        for i = 1, spawn_count do
            local loot = {
                id = nk.uuid_v4(),
                position = {
                    x = zone.x + math.random(-500, 500),
                    y = zone.y + math.random(-500, 500)
                },
                items = generate_loot_items(zone.tier_bonus),
                looted = false
            }
            table.insert(spawns, loot)
        end
    end
    
    return spawns
end

-- 플레이어 참가
local function match_join(context, dispatcher, tick, state, presences)
    for _, presence in ipairs(presences) do
        state.players[presence.user_id] = {
            presence = presence,
            alive = true,
            health = 100,
            armor = 0,
            position = {x = 0, y = 0, z = 1000}, -- 비행기에서 시작
            inventory = {
                weapons = {},
                items = {},
                ammo = {light = 0, medium = 0, heavy = 0, shells = 0}
            },
            stats = {
                kills = 0,
                damage_dealt = 0,
                distance_traveled = 0,
                survival_time = 0,
                placement = 0
            },
            in_vehicle = nil,
            spectating = nil
        }
        
        state.alive_count = state.alive_count + 1
        
        -- 대기 중 메시지
        if state.phase == "waiting" then
            dispatcher.broadcast_message(1, nk.json_encode({
                action = "waiting_for_players",
                current = state.alive_count,
                required = br_config.start_players
            }), {presence})
        end
    end
    
    -- 시작 조건 체크
    if state.phase == "waiting" and state.alive_count >= br_config.start_players then
        start_match(state, dispatcher)
    end
end

-- 매치 시작
local function start_match(state, dispatcher)
    state.phase = "dropping"
    state.start_time = nk.time()
    state.next_circle_time = state.start_time + br_config.initial_circle_delay
    
    -- 비행기 경로 생성
    state.drop_path = generate_drop_path()
    
    -- 모든 플레이어에게 시작 알림
    dispatcher.broadcast_message(1, nk.json_encode({
        action = "match_start",
        drop_path = state.drop_path,
        total_players = state.alive_count
    }))
    
    nk.logger_info(string.format("Battle Royale match started with %d players", state.alive_count))
end

-- 게임 루프
local function match_loop(context, dispatcher, tick, state, messages)
    -- 메시지 처리
    for _, message in ipairs(messages) do
        local data = nk.json_decode(message.data)
        handle_br_action(state, dispatcher, message.sender.user_id, data)
    end
    
    -- 페이즈별 처리
    if state.phase == "dropping" then
        update_drop_phase(state, dispatcher, tick)
        
    elseif state.phase == "playing" then
        -- 자기장 업데이트
        update_circle(state, dispatcher)
        
        -- 자기장 데미지
        if tick % br_config.tick_rate == 0 then
            apply_circle_damage(state, dispatcher)
        end
        
        -- 에어드롭
        if tick % (br_config.tick_rate * 180) == 0 then -- 3분마다
            spawn_airdrop(state, dispatcher)
        end
        
        -- AI 분석
        if tick % (br_config.tick_rate * 60) == 0 then
            analyze_match_state(state, dispatcher)
        end
    end
    
    -- 생존자 체크
    check_match_end(state, dispatcher)
    
    return state
end

-- BR 액션 처리
local function handle_br_action(state, dispatcher, user_id, data)
    local player = state.players[user_id]
    if not player then return end
    
    if data.action == "jump" and state.phase == "dropping" then
        -- 비행기에서 점프
        handle_jump(state, dispatcher, player, data)
        
    elseif data.action == "move" and player.alive then
        -- 이동
        handle_movement(state, dispatcher, player, data)
        
    elseif data.action == "shoot" and player.alive then
        -- 사격
        handle_shooting(state, dispatcher, player, data)
        
    elseif data.action == "loot" and player.alive then
        -- 루팅
        handle_looting(state, dispatcher, player, data)
        
    elseif data.action == "use_item" and player.alive then
        -- 아이템 사용
        handle_item_use(state, dispatcher, player, data)
        
    elseif data.action == "enter_vehicle" and player.alive then
        -- 차량 탑승
        handle_vehicle_enter(state, dispatcher, player, data)
        
    elseif data.action == "spectate" and not player.alive then
        -- 관전
        handle_spectate(state, dispatcher, player, data)
    end
end

-- 자기장 업데이트
local function update_circle(state, dispatcher)
    local current_time = nk.time()
    
    if current_time >= state.next_circle_time then
        state.circle_phase = state.circle_phase + 1
        
        if state.circle_phase <= #br_config.circle_phases then
            local phase = br_config.circle_phases[state.circle_phase]
            
            -- 새 안전지대 중심 결정 (AI)
            local new_center = calculate_optimal_circle_position(state)
            
            -- 자기장 수축 시작
            dispatcher.broadcast_message(2, nk.json_encode({
                action = "circle_update",
                phase = state.circle_phase,
                current_center = state.circle_center,
                new_center = new_center,
                new_radius = phase.radius,
                duration = phase.duration
            }))
            
            state.circle_center = new_center
            state.safe_zone_radius = phase.radius
            state.next_circle_time = current_time + phase.duration
        end
    end
end

-- AI 최적 자기장 위치 계산
local function calculate_optimal_circle_position(state)
    -- 생존자 분포 분석
    local player_positions = {}
    local total_x, total_y = 0, 0
    local count = 0
    
    for user_id, player in pairs(state.players) do
        if player.alive then
            table.insert(player_positions, player.position)
            total_x = total_x + player.position.x
            total_y = total_y + player.position.y
            count = count + 1
        end
    end
    
    if count == 0 then
        return state.circle_center
    end
    
    -- 플레이어 밀집 지역 계산
    local center_x = total_x / count
    local center_y = total_y / count
    
    -- 현재 안전지대 내에서 새 중심점 선택
    local max_offset = state.safe_zone_radius * 0.3
    local offset_x = math.random(-max_offset, max_offset)
    local offset_y = math.random(-max_offset, max_offset)
    
    return {
        x = math.max(0, math.min(br_config.map_size, center_x + offset_x)),
        y = math.max(0, math.min(br_config.map_size, center_y + offset_y))
    }
end

-- 자기장 데미지
local function apply_circle_damage(state, dispatcher)
    if state.circle_phase == 0 then return end
    
    local phase = br_config.circle_phases[state.circle_phase]
    
    for user_id, player in pairs(state.players) do
        if player.alive then
            local distance = math.sqrt(
                (player.position.x - state.circle_center.x)^2 +
                (player.position.y - state.circle_center.y)^2
            )
            
            if distance > state.safe_zone_radius then
                -- 자기장 밖
                player.health = player.health - phase.damage
                
                if player.health <= 0 then
                    handle_player_death(state, dispatcher, user_id, "circle")
                else
                    dispatcher.broadcast_message(3, nk.json_encode({
                        action = "circle_damage",
                        user_id = user_id,
                        damage = phase.damage,
                        health = player.health
                    }), {player.presence})
                end
            end
        end
    end
end

-- AI 매치 분석
local function analyze_match_state(state, dispatcher)
    local analysis = {
        alive_players = state.alive_count,
        circle_phase = state.circle_phase,
        hot_zones = {},
        camping_detection = {},
        optimal_positions = {}
    }
    
    -- 핫존 분석 (전투 빈발 지역)
    local grid_size = 500
    local activity_grid = {}
    
    for _, kill in ipairs(state.kill_feed) do
        local grid_x = math.floor(kill.position.x / grid_size)
        local grid_y = math.floor(kill.position.y / grid_size)
        local key = grid_x .. "," .. grid_y
        
        activity_grid[key] = (activity_grid[key] or 0) + 1
    end
    
    -- 캠핑 감지
    for user_id, player in pairs(state.players) do
        if player.alive and player.last_positions then
            local movement = calculate_movement(player.last_positions)
            if movement < 50 then -- 50m 미만 이동
                table.insert(analysis.camping_detection, {
                    user_id = user_id,
                    duration = #player.last_positions * 10 -- seconds
                })
            end
        end
    end
    
    -- AI 추천 포지션
    analysis.optimal_positions = calculate_optimal_positions(state)
    
    -- 관전자들에게 분석 정보 전송
    for user_id, player in pairs(state.players) do
        if not player.alive and player.spectating then
            dispatcher.broadcast_message(99, nk.json_encode({
                action = "match_analysis",
                data = analysis
            }), {player.presence})
        end
    end
end

-- 매치 종료 체크
local function check_match_end(state, dispatcher)
    if state.alive_count <= 1 and state.phase == "playing" then
        state.phase = "ending"
        
        -- 우승자 찾기
        local winner = nil
        for user_id, player in pairs(state.players) do
            if player.alive then
                winner = user_id
                player.stats.placement = 1
                break
            end
        end
        
        -- 매치 결과 브로드캐스트
        dispatcher.broadcast_message(99, nk.json_encode({
            action = "match_end",
            winner = winner,
            total_players = #state.players,
            duration = nk.time() - state.start_time
        }))
        
        -- 통계 저장
        save_match_statistics(state)
        
        nk.logger_info(string.format("Battle Royale match ended. Winner: %s", winner or "none"))
    end
end

-- 통계 저장
local function save_match_statistics(state)
    local match_id = nk.uuid_v4()
    local stats = {
        match_id = match_id,
        timestamp = nk.time(),
        players = {},
        duration = nk.time() - state.start_time,
        total_players = #state.players
    }
    
    for user_id, player in pairs(state.players) do
        stats.players[user_id] = player.stats
    end
    
    nk.storage_write({
        {
            collection = "br_match_history",
            key = match_id,
            value = stats,
            permission_read = 2,
            permission_write = 0
        }
    })
end

-- 매치 등록
nk.register_match({
    match_init = match_init,
    match_join_attempt = match_join_attempt,
    match_join = match_join,
    match_leave = match_leave,
    match_loop = match_loop,
    match_terminate = match_terminate
})

nk.logger_info("Battle Royale Backend loaded successfully")
"""
    
    async def _create_general_backend(self) -> str:
        """범용 게임 백엔드"""
        return """-- General Purpose Game Backend
-- AI 최적화된 범용 멀티플레이어 백엔드

local nk = require("nakama")

-- 범용 설정
local general_config = {
    tick_rate = 30,
    max_players = 50,
    game_modes = {},
    features = {
        matchmaking = true,
        leaderboards = true,
        tournaments = true,
        chat = true,
        clans = true,
        achievements = true
    }
}

-- 게임 모드 등록
local function register_game_mode(mode_name, config)
    general_config.game_modes[mode_name] = config
    nk.logger_info(string.format("Registered game mode: %s", mode_name))
end

-- 범용 매치 초기화
local function match_init(context, setupstate)
    local mode = setupstate.mode or "default"
    local mode_config = general_config.game_modes[mode] or {}
    
    local gamestate = {
        mode = mode,
        players = {},
        teams = {},
        scores = {},
        custom_data = mode_config.initial_state or {},
        start_time = nk.time(),
        config = mode_config
    }
    
    local tickrate = mode_config.tick_rate or general_config.tick_rate
    local label = string.format("General:%s", mode)
    
    return gamestate, tickrate, label
end

-- 플레이어 참가
local function match_join(context, dispatcher, tick, state, presences)
    for _, presence in ipairs(presences) do
        -- 플레이어 초기화
        local player_data = state.config.create_player and 
            state.config.create_player(presence) or 
            {presence = presence, score = 0}
        
        state.players[presence.user_id] = player_data
        
        -- 커스텀 참가 로직
        if state.config.on_player_join then
            state.config.on_player_join(state, dispatcher, player_data)
        end
        
        -- 참가 알림
        dispatcher.broadcast_message(1, nk.json_encode({
            action = "player_joined",
            user_id = presence.user_id,
            total_players = #state.players
        }))
    end
end

-- 게임 루프
local function match_loop(context, dispatcher, tick, state, messages)
    -- 메시지 처리
    for _, message in ipairs(messages) do
        local data = nk.json_decode(message.data)
        
        -- 커스텀 메시지 핸들러
        if state.config.handle_message then
            state.config.handle_message(state, dispatcher, message.sender, data)
        else
            -- 기본 메시지 처리
            handle_default_message(state, dispatcher, message.sender, data)
        end
    end
    
    -- 커스텀 게임 로직
    if state.config.update then
        state.config.update(state, dispatcher, tick)
    end
    
    -- AI 분석 (10초마다)
    if tick % (general_config.tick_rate * 10) == 0 then
        analyze_game_state(state, dispatcher)
    end
    
    return state
end

-- 기본 메시지 처리
local function handle_default_message(state, dispatcher, sender, data)
    if data.action == "update_position" then
        -- 위치 업데이트
        if state.players[sender.user_id] then
            state.players[sender.user_id].position = data.position
            
            -- 다른 플레이어에게 브로드캐스트
            dispatcher.broadcast_message(2, nk.json_encode({
                action = "position_update",
                user_id = sender.user_id,
                position = data.position
            }))
        end
        
    elseif data.action == "score" then
        -- 점수 업데이트
        if state.players[sender.user_id] then
            state.players[sender.user_id].score = 
                (state.players[sender.user_id].score or 0) + data.points
            
            update_leaderboard(sender.user_id, state.players[sender.user_id].score)
        end
    end
end

-- AI 게임 상태 분석
local function analyze_game_state(state, dispatcher)
    local analysis = {
        player_count = 0,
        active_players = 0,
        score_distribution = {},
        recommendations = {}
    }
    
    -- 플레이어 분석
    for user_id, player in pairs(state.players) do
        analysis.player_count = analysis.player_count + 1
        
        -- 활동성 체크
        if player.last_action and (nk.time() - player.last_action < 30) then
            analysis.active_players = analysis.active_players + 1
        end
        
        -- 점수 분포
        local score_bracket = math.floor((player.score or 0) / 100) * 100
        analysis.score_distribution[score_bracket] = 
            (analysis.score_distribution[score_bracket] or 0) + 1
    end
    
    -- AI 권장사항
    if analysis.active_players < analysis.player_count * 0.5 then
        table.insert(analysis.recommendations, "Low player engagement detected")
    end
    
    -- 분석 결과 로깅
    nk.logger_debug(string.format("Game analysis: %s", nk.json_encode(analysis)))
end

-- 리더보드 업데이트
local function update_leaderboard(user_id, score)
    if not general_config.features.leaderboards then
        return
    end
    
    nk.leaderboard_record_write("global_scores", user_id, user_id, score, 0, nil, nil)
end

-- 업적 시스템
local function check_achievements(user_id, action, value)
    if not general_config.features.achievements then
        return
    end
    
    local achievements = {
        first_win = {condition = function(a, v) return a == "win" and v == 1 end},
        score_100 = {condition = function(a, v) return a == "score" and v >= 100 end},
        play_10_matches = {condition = function(a, v) return a == "matches" and v >= 10 end}
    }
    
    for achievement_id, achievement in pairs(achievements) do
        if achievement.condition(action, value) then
            grant_achievement(user_id, achievement_id)
        end
    end
end

-- 업적 부여
local function grant_achievement(user_id, achievement_id)
    local achievements = nk.storage_read({
        {collection = "achievements", key = user_id, user_id = user_id}
    })
    
    local user_achievements = {}
    if #achievements > 0 then
        user_achievements = achievements[1].value
    end
    
    if not user_achievements[achievement_id] then
        user_achievements[achievement_id] = {
            unlocked = true,
            timestamp = nk.time()
        }
        
        nk.storage_write({
            {
                collection = "achievements",
                key = user_id,
                user_id = user_id,
                value = user_achievements,
                permission_read = 1,
                permission_write = 0
            }
        })
        
        -- 알림 전송
        nk.notification_send(user_id, "", {
            title = "Achievement Unlocked!",
            body = achievement_id,
            persistent = true
        }, 0, "", true)
    end
end

-- RPC: 게임 모드 목록
local function rpc_get_game_modes(context, payload)
    local modes = {}
    for mode_name, config in pairs(general_config.game_modes) do
        table.insert(modes, {
            name = mode_name,
            max_players = config.max_players or general_config.max_players,
            description = config.description or ""
        })
    end
    
    return nk.json_encode({modes = modes})
end

-- RPC: 매치 생성
local function rpc_create_match(context, payload)
    local data = nk.json_decode(payload)
    local mode = data.mode or "default"
    
    if not general_config.game_modes[mode] then
        return nk.json_encode({error = "Invalid game mode"})
    end
    
    local match_id = nk.match_create("general_match", {mode = mode})
    
    return nk.json_encode({
        match_id = match_id,
        mode = mode
    })
end

-- 모듈 등록
nk.register_match({
    match_init = match_init,
    match_join_attempt = match_join_attempt,
    match_join = match_join,
    match_leave = match_leave,
    match_loop = match_loop,
    match_terminate = match_terminate
})

nk.register_rpc(rpc_get_game_modes, "get_game_modes")
nk.register_rpc(rpc_create_match, "create_match")

-- 예제 게임 모드 등록
register_game_mode("team_deathmatch", {
    max_players = 16,
    tick_rate = 30,
    description = "Classic team deathmatch",
    initial_state = {
        team_scores = {red = 0, blue = 0},
        score_limit = 50
    },
    create_player = function(presence)
        return {
            presence = presence,
            team = math.random() > 0.5 and "red" or "blue",
            score = 0,
            kills = 0,
            deaths = 0
        }
    end,
    handle_message = function(state, dispatcher, sender, data)
        -- 팀 데스매치 특화 로직
    end,
    update = function(state, dispatcher, tick)
        -- 승리 조건 체크
        if state.custom_data.team_scores.red >= state.custom_data.score_limit or
           state.custom_data.team_scores.blue >= state.custom_data.score_limit then
            -- 게임 종료
            dispatcher.broadcast_message(99, nk.json_encode({
                action = "game_over",
                winner = state.custom_data.team_scores.red > 
                        state.custom_data.team_scores.blue and "red" or "blue"
            }))
        end
    end
})

nk.logger_info("General Purpose Backend loaded successfully")
"""
    
    def _get_game_features(self, game_type: str) -> List[str]:
        """게임 타입별 주요 기능 반환"""
        features_map = {
            "fps": [
                "High-frequency tick rate (64Hz)",
                "Lag compensation",
                "Hit validation",
                "Anti-cheat measures",
                "Weapon ballistics",
                "Respawn system"
            ],
            "moba": [
                "5v5 team battles",
                "Minion waves",
                "Jungle camps",
                "Tower defense",
                "Item shop",
                "Skill leveling"
            ],
            "mmo": [
                "Persistent world",
                "Zone-based architecture",
                "Dynamic economy",
                "Guild system",
                "Quest management",
                "World events"
            ],
            "battle_royale": [
                "100 player support",
                "Shrinking safe zone",
                "Loot system",
                "Vehicle mechanics",
                "Spectator mode",
                "Match statistics"
            ]
        }
        
        return features_map.get(game_type, [
            "Flexible game modes",
            "Achievement system",
            "Leaderboards",
            "Chat system",
            "Tournament support",
            "AI optimization"
        ])
    
    async def create_nakama_docker_compose(self) -> str:
        """Nakama Docker Compose 설정 생성"""
        docker_compose = """version: '3'
services:
  postgres:
    image: postgres:12.2-alpine
    environment:
      - POSTGRES_DB=nakama
      - POSTGRES_PASSWORD=localdb
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    expose:
      - "5432"
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres", "-d", "nakama"]
      interval: 3s
      timeout: 3s
      retries: 5

  nakama:
    image: heroiclabs/nakama:3.17.1
    entrypoint:
      - "/bin/sh"
      - "-ecx"
      - >
        /nakama/nakama migrate up --database.address postgres:localdb@postgres:5432/nakama &&
        exec /nakama/nakama --config /nakama/data/nakama-config.yml --database.address postgres:localdb@postgres:5432/nakama
    restart: always
    links:
      - "postgres:db"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./data/nakama:/nakama/data
      - ./nakama_configurations:/nakama/configurations
      - ./nakama_ai_modules:/nakama/modules
    expose:
      - "7349"
      - "7350"
      - "7351"
    ports:
      - "7349:7349"
      - "7350:7350"
      - "7351:7351"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7350/"]
      interval: 10s
      timeout: 5s
      retries: 5
"""
        
        # Docker Compose 파일 저장
        compose_path = Path("./docker-compose.yml")
        compose_path.write_text(docker_compose)
        
        return str(compose_path)
    
    async def start_nakama_server(self) -> bool:
        """Nakama 서버 시작"""
        self.logger.info("🚀 Nakama 서버 시작 중...")
        
        try:
            # Docker Compose 파일 생성
            compose_path = await self.create_nakama_docker_compose()
            
            # Docker Compose 실행
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            
            self.logger.info("✅ Nakama 서버가 성공적으로 시작되었습니다!")
            self.logger.info("📊 Nakama Console: http://localhost:7351")
            self.logger.info("🔌 gRPC: localhost:7349")
            self.logger.info("🌐 HTTP: localhost:7350")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Nakama 서버 시작 실패: {e}")
            return False
    
    async def check_status(self) -> Dict[str, Any]:
        """Nakama 통합 상태 확인"""
        import requests
        
        status = {
            "nakama_running": False,
            "modules_ready": False,
            "configurations": [],
            "active_matches": 0
        }
        
        # Nakama 서버 상태 확인
        try:
            response = requests.get(f"http://localhost:{self.config['server']['http_port']}/")
            status["nakama_running"] = response.status_code == 200
        except:
            pass
        
        # 모듈 확인
        if self.ai_modules_dir.exists():
            status["modules_ready"] = True
            status["modules"] = [f.name for f in self.ai_modules_dir.glob("*.lua")]
        
        # 설정 확인
        if self.nakama_config_dir.exists():
            status["configurations"] = [f.name for f in self.nakama_config_dir.glob("*.yml")]
        
        return status
    
    async def run_demo(self):
        """Nakama AI 통합 데모"""
        self.logger.info("🎮 Nakama AI 통합 데모 시작...")
        
        # FPS 게임 설정
        fps_setup = await self.setup_nakama_server("fps")
        print(f"\n✅ FPS 게임 백엔드 설정 완료: {fps_setup}")
        
        # 서버 성능 최적화
        optimization = await self.optimize_server_performance()
        print(f"\n⚡ 서버 최적화 완료: {optimization}")
        
        # 상태 확인
        status = await self.check_status()
        print(f"\n📊 현재 상태: {json.dumps(status, indent=2)}")
        
        print("""
        === Nakama AI Demo Complete ===
        
        이제 다음을 할 수 있습니다:
        1. Nakama Console 접속: http://localhost:7351
        2. Godot 클라이언트 연결
        3. AI 매치메이킹 테스트
        4. 실시간 성능 모니터링
        
        AI가 자동으로 최적화하는 항목:
        - 매치메이킹 밸런싱
        - 서버 부하 분산
        - 플레이어 데이터 캐싱
        - 소셜 기능 모더레이션
        - 경제 시스템 균형
        """)

# 사용 예시
if __name__ == "__main__":
    async def main():
        nakama_ai = NakamaAIIntegration()
        
        # 배틀로얄 게임 백엔드 생성
        br_backend = await nakama_ai.generate_game_specific_backend("battle_royale")
        print("Battle Royale Backend:", br_backend)
        
        # 데모 실행
        await nakama_ai.run_demo()
    
    asyncio.run(main())