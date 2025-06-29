#!/usr/bin/env python3
"""
AI 리소스 및 에셋 자동 관리 시스템
AI가 게임 리소스를 자동으로 생성, 최적화, 관리
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import wave
import struct
import math

@dataclass
class ResourceMetadata:
    """리소스 메타데이터"""
    name: str
    type: str  # texture, audio, model, animation, shader, etc.
    path: Path
    size_bytes: int
    format: str
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None  # 오디오/애니메이션용
    compression: Optional[str] = None
    tags: List[str] = None
    dependencies: List[str] = None
    usage_count: int = 0
    last_modified: float = 0.0
    checksum: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class GenerationRequest:
    """리소스 생성 요청"""
    type: str
    name: str
    parameters: Dict[str, Any]
    output_path: Path
    quality: str = "medium"  # low, medium, high, ultra
    optimization: bool = True

class AIResourceManager:
    """AI 리소스 자동 관리 시스템"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.assets_path = project_path / "assets"
        self.cache_path = project_path / ".godot" / "ai_cache"
        self.logger = logging.getLogger("AIResourceManager")
        
        # 디렉토리 초기화
        self._initialize_directories()
        
        # 리소스 데이터베이스
        self.resource_db: Dict[str, ResourceMetadata] = {}
        self._load_resource_database()
        
        # 생성기 맵핑
        self.generators = {
            "texture": self._generate_texture,
            "sprite": self._generate_sprite,
            "audio": self._generate_audio,
            "music": self._generate_music,
            "material": self._generate_material,
            "animation": self._generate_animation,
            "particle": self._generate_particle_effect,
            "shader": self._generate_shader,
            "font": self._generate_font,
            "tileset": self._generate_tileset
        }
        
        # 최적화기 맵핑
        self.optimizers = {
            "texture": self._optimize_texture,
            "audio": self._optimize_audio,
            "model": self._optimize_model,
            "animation": self._optimize_animation
        }
        
    def _initialize_directories(self):
        """디렉토리 구조 초기화"""
        directories = [
            self.assets_path,
            self.assets_path / "textures",
            self.assets_path / "sprites", 
            self.assets_path / "audio",
            self.assets_path / "music",
            self.assets_path / "materials",
            self.assets_path / "animations",
            self.assets_path / "effects",
            self.assets_path / "shaders",
            self.assets_path / "fonts",
            self.assets_path / "tilesets",
            self.cache_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _load_resource_database(self):
        """리소스 데이터베이스 로드"""
        db_path = self.cache_path / "resource_db.json"
        
        if db_path.exists():
            try:
                with open(db_path, 'r') as f:
                    data = json.load(f)
                    
                for name, metadata in data.items():
                    metadata['path'] = Path(metadata['path'])
                    self.resource_db[name] = ResourceMetadata(**metadata)
                    
            except Exception as e:
                self.logger.warning(f"리소스 DB 로드 실패: {e}")
                
    def _save_resource_database(self):
        """리소스 데이터베이스 저장"""
        db_path = self.cache_path / "resource_db.json"
        
        try:
            data = {}
            for name, metadata in self.resource_db.items():
                metadata_dict = asdict(metadata)
                metadata_dict['path'] = str(metadata.path)
                data[name] = metadata_dict
                
            with open(db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"리소스 DB 저장 실패: {e}")
            
    async def generate_resource(self, request: GenerationRequest) -> Optional[ResourceMetadata]:
        """리소스 자동 생성"""
        self.logger.info(f"리소스 생성 시작: {request.name} ({request.type})")
        
        try:
            generator = self.generators.get(request.type)
            if not generator:
                self.logger.error(f"지원하지 않는 리소스 타입: {request.type}")
                return None
                
            # 생성 실행
            success = await generator(request)
            
            if success and request.output_path.exists():
                # 메타데이터 생성
                metadata = await self._create_metadata(request.name, request.type, request.output_path)
                
                # 최적화 적용
                if request.optimization:
                    await self._optimize_resource(metadata)
                    
                # 데이터베이스에 등록
                self.resource_db[request.name] = metadata
                self._save_resource_database()
                
                self.logger.info(f"리소스 생성 완료: {request.name}")
                return metadata
            else:
                self.logger.error(f"리소스 생성 실패: {request.name}")
                return None
                
        except Exception as e:
            self.logger.error(f"리소스 생성 중 오류: {e}")
            return None
            
    async def _generate_texture(self, request: GenerationRequest) -> bool:
        """텍스처 생성"""
        params = request.parameters
        width = params.get("width", 256)
        height = params.get("height", 256)
        texture_type = params.get("texture_type", "solid")
        color = params.get("color", (255, 255, 255, 255))
        
        try:
            if texture_type == "solid":
                image = Image.new("RGBA", (width, height), color)
                
            elif texture_type == "gradient":
                image = self._create_gradient_texture(width, height, params)
                
            elif texture_type == "noise":
                image = self._create_noise_texture(width, height, params)
                
            elif texture_type == "pattern":
                image = self._create_pattern_texture(width, height, params)
                
            elif texture_type == "procedural":
                image = self._create_procedural_texture(width, height, params)
                
            else:
                image = Image.new("RGBA", (width, height), color)
                
            # 저장
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(request.output_path, "PNG")
            
            return True
            
        except Exception as e:
            self.logger.error(f"텍스처 생성 실패: {e}")
            return False
            
    def _create_gradient_texture(self, width: int, height: int, params: Dict[str, Any]) -> Image.Image:
        """그라디언트 텍스처 생성"""
        start_color = params.get("start_color", (255, 0, 0, 255))
        end_color = params.get("end_color", (0, 0, 255, 255))
        direction = params.get("direction", "horizontal")  # horizontal, vertical, diagonal
        
        image = Image.new("RGBA", (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                if direction == "horizontal":
                    ratio = x / (width - 1)
                elif direction == "vertical":
                    ratio = y / (height - 1)
                else:  # diagonal
                    ratio = (x + y) / (width + height - 2)
                    
                r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
                g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
                b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
                a = int(start_color[3] * (1 - ratio) + end_color[3] * ratio)
                
                pixels.append((r, g, b, a))
                
        image.putdata(pixels)
        return image
        
    def _create_noise_texture(self, width: int, height: int, params: Dict[str, Any]) -> Image.Image:
        """노이즈 텍스처 생성"""
        noise_type = params.get("noise_type", "perlin")
        scale = params.get("scale", 50.0)
        octaves = params.get("octaves", 4)
        
        # 간단한 노이즈 구현 (실제로는 더 복잡한 알고리즘 사용)
        import random
        random.seed(params.get("seed", 42))
        
        image = Image.new("RGBA", (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # 간단한 노이즈 생성
                noise_value = 0
                frequency = 1.0
                amplitude = 1.0
                
                for _ in range(octaves):
                    noise_value += (random.random() - 0.5) * amplitude
                    frequency *= 2
                    amplitude *= 0.5
                    
                # 노이즈 값을 0-255 범위로 정규화
                noise_value = max(0, min(255, int((noise_value + 1) * 127.5)))
                
                pixels.append((noise_value, noise_value, noise_value, 255))
                
        image.putdata(pixels)
        return image
        
    def _create_pattern_texture(self, width: int, height: int, params: Dict[str, Any]) -> Image.Image:
        """패턴 텍스처 생성"""
        pattern_type = params.get("pattern_type", "checkerboard")
        tile_size = params.get("tile_size", 32)
        color1 = params.get("color1", (255, 255, 255, 255))
        color2 = params.get("color2", (0, 0, 0, 255))
        
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        
        if pattern_type == "checkerboard":
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    tile_x = x // tile_size
                    tile_y = y // tile_size
                    
                    if (tile_x + tile_y) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                        
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill=color)
                    
        elif pattern_type == "stripes":
            vertical = params.get("vertical", True)
            
            if vertical:
                for x in range(0, width, tile_size):
                    color = color1 if (x // tile_size) % 2 == 0 else color2
                    draw.rectangle([x, 0, x + tile_size, height], fill=color)
            else:
                for y in range(0, height, tile_size):
                    color = color1 if (y // tile_size) % 2 == 0 else color2
                    draw.rectangle([0, y, width, y + tile_size], fill=color)
                    
        elif pattern_type == "dots":
            dot_radius = params.get("dot_radius", tile_size // 4)
            
            for y in range(tile_size // 2, height, tile_size):
                for x in range(tile_size // 2, width, tile_size):
                    draw.ellipse([x - dot_radius, y - dot_radius, 
                                x + dot_radius, y + dot_radius], fill=color2)
                    
        return image
        
    def _create_procedural_texture(self, width: int, height: int, params: Dict[str, Any]) -> Image.Image:
        """절차적 텍스처 생성"""
        texture_style = params.get("style", "organic")
        complexity = params.get("complexity", 5)
        base_color = params.get("base_color", (100, 150, 200, 255))
        
        image = Image.new("RGBA", (width, height), base_color)
        draw = ImageDraw.Draw(image)
        
        if texture_style == "organic":
            # 유기적 패턴 생성
            import random
            random.seed(params.get("seed", 42))
            
            for _ in range(complexity * 10):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(5, 50)
                
                color = (
                    random.randint(base_color[0] - 50, base_color[0] + 50),
                    random.randint(base_color[1] - 50, base_color[1] + 50),
                    random.randint(base_color[2] - 50, base_color[2] + 50),
                    random.randint(100, 255)
                )
                
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                           fill=color)
                           
        elif texture_style == "geometric":
            # 기하학적 패턴 생성
            import random
            random.seed(params.get("seed", 42))
            
            for _ in range(complexity * 5):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(50, 200)
                )
                
                draw.polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], 
                           fill=color)
                           
        return image
        
    async def _generate_sprite(self, request: GenerationRequest) -> bool:
        """스프라이트 생성"""
        params = request.parameters
        sprite_type = params.get("sprite_type", "character")
        size = params.get("size", (64, 64))
        style = params.get("style", "pixel")
        
        try:
            if style == "pixel":
                image = self._create_pixel_sprite(size, sprite_type, params)
            else:
                image = self._create_vector_sprite(size, sprite_type, params)
                
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(request.output_path, "PNG")
            
            return True
            
        except Exception as e:
            self.logger.error(f"스프라이트 생성 실패: {e}")
            return False
            
    def _create_pixel_sprite(self, size: Tuple[int, int], sprite_type: str, 
                           params: Dict[str, Any]) -> Image.Image:
        """픽셀 아트 스프라이트 생성"""
        width, height = size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        if sprite_type == "character":
            # 간단한 캐릭터 스프라이트
            primary_color = params.get("primary_color", (100, 150, 255, 255))
            secondary_color = params.get("secondary_color", (50, 100, 200, 255))
            
            # 머리
            draw.ellipse([width//4, height//8, 3*width//4, 3*height//8], fill=primary_color)
            
            # 몸
            draw.rectangle([3*width//8, 3*height//8, 5*width//8, 7*height//8], fill=secondary_color)
            
            # 팔
            draw.rectangle([width//8, 4*height//8, 3*width//8, 6*height//8], fill=primary_color)
            draw.rectangle([5*width//8, 4*height//8, 7*width//8, 6*height//8], fill=primary_color)
            
            # 다리
            draw.rectangle([3*width//8, 7*height//8, 4*width//8, height], fill=primary_color)
            draw.rectangle([4*width//8, 7*height//8, 5*width//8, height], fill=primary_color)
            
        elif sprite_type == "item":
            # 아이템 스프라이트
            item_color = params.get("item_color", (255, 215, 0, 255))
            
            # 보석 모양
            points = []
            for i in range(8):
                angle = i * math.pi / 4
                if i % 2 == 0:
                    radius = width // 3
                else:
                    radius = width // 4
                    
                x = width // 2 + radius * math.cos(angle)
                y = height // 2 + radius * math.sin(angle)
                points.append((x, y))
                
            draw.polygon(points, fill=item_color)
            
        elif sprite_type == "enemy":
            # 적 스프라이트
            enemy_color = params.get("enemy_color", (255, 50, 50, 255))
            
            # 간단한 몬스터 모양
            draw.ellipse([width//6, height//4, 5*width//6, 3*height//4], fill=enemy_color)
            
            # 눈
            eye_color = (255, 255, 255, 255)
            draw.ellipse([width//3, height//3, width//2, height//2], fill=eye_color)
            draw.ellipse([width//2, height//3, 2*width//3, height//2], fill=eye_color)
            
            # 동공
            pupil_color = (0, 0, 0, 255)
            draw.ellipse([3*width//8, 3*height//8, 4*width//8, 4*height//8], fill=pupil_color)
            draw.ellipse([5*width//8, 3*height//8, 6*width//8, 4*height//8], fill=pupil_color)
            
        return image
        
    def _create_vector_sprite(self, size: Tuple[int, int], sprite_type: str,
                            params: Dict[str, Any]) -> Image.Image:
        """벡터 스타일 스프라이트 생성"""
        width, height = size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 벡터 스타일은 더 부드러운 곡선과 그라디언트 사용
        # 여기서는 간단한 구현만 제공
        
        if sprite_type == "character":
            primary_color = params.get("primary_color", (100, 150, 255, 255))
            
            # 부드러운 캐릭터 모양
            draw.ellipse([width//6, height//10, 5*width//6, 4*height//10], fill=primary_color)
            draw.ellipse([width//4, 3*height//10, 3*width//4, 8*height//10], fill=primary_color)
            
        return image
        
    async def _generate_audio(self, request: GenerationRequest) -> bool:
        """오디오 생성"""
        params = request.parameters
        audio_type = params.get("audio_type", "tone")
        duration = params.get("duration", 1.0)
        sample_rate = params.get("sample_rate", 44100)
        
        try:
            if audio_type == "tone":
                audio_data = self._create_tone(duration, sample_rate, params)
            elif audio_type == "noise":
                audio_data = self._create_noise_audio(duration, sample_rate, params)
            elif audio_type == "melody":
                audio_data = self._create_simple_melody(duration, sample_rate, params)
            else:
                audio_data = self._create_tone(duration, sample_rate, params)
                
            # WAV 파일로 저장
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_wav_file(audio_data, request.output_path, sample_rate)
            
            return True
            
        except Exception as e:
            self.logger.error(f"오디오 생성 실패: {e}")
            return False
            
    def _create_tone(self, duration: float, sample_rate: int, 
                    params: Dict[str, Any]) -> List[float]:
        """톤 생성"""
        frequency = params.get("frequency", 440.0)
        amplitude = params.get("amplitude", 0.5)
        wave_type = params.get("wave_type", "sine")
        
        samples = int(duration * sample_rate)
        audio_data = []
        
        for i in range(samples):
            t = i / sample_rate
            
            if wave_type == "sine":
                sample = amplitude * math.sin(2 * math.pi * frequency * t)
            elif wave_type == "square":
                sample = amplitude * (1 if math.sin(2 * math.pi * frequency * t) > 0 else -1)
            elif wave_type == "triangle":
                sample = amplitude * (2 * abs(2 * (frequency * t - math.floor(frequency * t + 0.5))) - 1)
            elif wave_type == "sawtooth":
                sample = amplitude * (2 * (frequency * t - math.floor(frequency * t + 0.5)))
            else:
                sample = amplitude * math.sin(2 * math.pi * frequency * t)
                
            audio_data.append(sample)
            
        return audio_data
        
    def _create_noise_audio(self, duration: float, sample_rate: int,
                           params: Dict[str, Any]) -> List[float]:
        """노이즈 오디오 생성"""
        import random
        
        noise_type = params.get("noise_type", "white")
        amplitude = params.get("amplitude", 0.1)
        
        samples = int(duration * sample_rate)
        audio_data = []
        
        for i in range(samples):
            if noise_type == "white":
                sample = amplitude * (random.random() * 2 - 1)
            elif noise_type == "pink":
                # 간단한 핑크 노이즈 근사
                sample = amplitude * (random.random() * 2 - 1) * (1 / math.sqrt(i + 1))
            else:
                sample = amplitude * (random.random() * 2 - 1)
                
            audio_data.append(sample)
            
        return audio_data
        
    def _create_simple_melody(self, duration: float, sample_rate: int,
                            params: Dict[str, Any]) -> List[float]:
        """간단한 멜로디 생성"""
        notes = params.get("notes", [440, 494, 523, 587, 659, 698, 784])  # C major scale
        note_duration = duration / len(notes)
        
        audio_data = []
        
        for note_freq in notes:
            note_samples = self._create_tone(note_duration, sample_rate, {
                "frequency": note_freq,
                "amplitude": 0.3,
                "wave_type": "sine"
            })
            audio_data.extend(note_samples)
            
        return audio_data
        
    def _save_wav_file(self, audio_data: List[float], file_path: Path, sample_rate: int):
        """WAV 파일 저장"""
        with wave.open(str(file_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # 오디오 데이터를 16-bit 정수로 변환
            for sample in audio_data:
                packed_sample = struct.pack('<h', int(sample * 32767))
                wav_file.writeframes(packed_sample)
                
    async def _generate_material(self, request: GenerationRequest) -> bool:
        """머티리얼 생성"""
        params = request.parameters
        material_type = params.get("material_type", "standard")
        
        try:
            if material_type == "standard":
                content = self._create_standard_material(params)
            elif material_type == "unshaded":
                content = self._create_unshaded_material(params)
            elif material_type == "canvas":
                content = self._create_canvas_material(params)
            else:
                content = self._create_standard_material(params)
                
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(request.output_path, 'w') as f:
                f.write(content)
                
            return True
            
        except Exception as e:
            self.logger.error(f"머티리얼 생성 실패: {e}")
            return False
            
    def _create_standard_material(self, params: Dict[str, Any]) -> str:
        """표준 머티리얼 생성"""
        albedo_color = params.get("albedo_color", [1.0, 1.0, 1.0, 1.0])
        metallic = params.get("metallic", 0.0)
        roughness = params.get("roughness", 0.5)
        emission = params.get("emission", [0.0, 0.0, 0.0])
        
        return f'''[gd_resource type="StandardMaterial3D" format=3]

[resource]
albedo_color = Color({albedo_color[0]}, {albedo_color[1]}, {albedo_color[2]}, {albedo_color[3]})
metallic = {metallic}
roughness = {roughness}
emission = Color({emission[0]}, {emission[1]}, {emission[2]}, 1.0)
'''

    async def _generate_animation(self, request: GenerationRequest) -> bool:
        """애니메이션 생성"""
        params = request.parameters
        animation_type = params.get("animation_type", "movement")
        duration = params.get("duration", 1.0)
        
        try:
            if animation_type == "movement":
                content = self._create_movement_animation(duration, params)
            elif animation_type == "rotation":
                content = self._create_rotation_animation(duration, params)
            elif animation_type == "scale":
                content = self._create_scale_animation(duration, params)
            else:
                content = self._create_movement_animation(duration, params)
                
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(request.output_path, 'w') as f:
                f.write(content)
                
            return True
            
        except Exception as e:
            self.logger.error(f"애니메이션 생성 실패: {e}")
            return False
            
    def _create_movement_animation(self, duration: float, params: Dict[str, Any]) -> str:
        """이동 애니메이션 생성"""
        start_pos = params.get("start_position", [0, 0])
        end_pos = params.get("end_position", [100, 0])
        loop_mode = params.get("loop", "linear")
        
        return f'''[gd_resource type="Animation" format=3]

[resource]
resource_name = "movement"
length = {duration}
loop_mode = 1
step = 0.1

[sub_resource type="AnimationTrack" id="1"]
type = "position"
path = NodePath(".")
interp = 1
loop_wrap = true
keys = {{
"times": PackedFloat32Array(0, {duration}),
"transitions": PackedFloat32Array(1, 1),
"update": 0,
"values": [Vector2({start_pos[0]}, {start_pos[1]}), Vector2({end_pos[0]}, {end_pos[1]})]
}}

[node name="AnimationPlayer" type="AnimationPlayer"]
libraries = {{
"": SubResource("AnimationLibrary_1")
}}
'''

    async def _optimize_resource(self, metadata: ResourceMetadata):
        """리소스 최적화"""
        optimizer = self.optimizers.get(metadata.type)
        if optimizer:
            await optimizer(metadata)
            
    async def _optimize_texture(self, metadata: ResourceMetadata):
        """텍스처 최적화"""
        try:
            image = Image.open(metadata.path)
            
            # 크기 최적화
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            # 압축 적용
            if metadata.format.upper() == "PNG":
                image.save(metadata.path, "PNG", optimize=True, compress_level=9)
            elif metadata.format.upper() in ["JPG", "JPEG"]:
                image.save(metadata.path, "JPEG", optimize=True, quality=85)
                
            # 메타데이터 업데이트
            metadata.size_bytes = metadata.path.stat().st_size
            metadata.dimensions = image.size
            metadata.compression = "optimized"
            
        except Exception as e:
            self.logger.warning(f"텍스처 최적화 실패: {e}")
            
    async def _optimize_audio(self, metadata: ResourceMetadata):
        """오디오 최적화"""
        # 간단한 구현 - 실제로는 더 복잡한 오디오 처리 필요
        try:
            # OGG 변환 등의 압축 작업
            metadata.compression = "optimized"
            
        except Exception as e:
            self.logger.warning(f"오디오 최적화 실패: {e}")
            
    async def _create_metadata(self, name: str, resource_type: str, 
                             file_path: Path) -> ResourceMetadata:
        """메타데이터 생성"""
        stat = file_path.stat()
        
        # 체크섬 계산
        checksum = self._calculate_checksum(file_path)
        
        # 타입별 추가 정보
        dimensions = None
        duration = None
        
        if resource_type == "texture" or resource_type == "sprite":
            try:
                with Image.open(file_path) as img:
                    dimensions = img.size
            except:
                pass
                
        elif resource_type == "audio" or resource_type == "music":
            try:
                with wave.open(str(file_path), 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / rate
            except:
                pass
                
        return ResourceMetadata(
            name=name,
            type=resource_type,
            path=file_path,
            size_bytes=stat.st_size,
            format=file_path.suffix[1:].upper(),
            dimensions=dimensions,
            duration=duration,
            last_modified=stat.st_mtime,
            checksum=checksum
        )
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                
        return hasher.hexdigest()
        
    async def batch_generate_resources(self, requests: List[GenerationRequest]) -> List[ResourceMetadata]:
        """리소스 일괄 생성"""
        results = []
        
        for request in requests:
            result = await self.generate_resource(request)
            if result:
                results.append(result)
                
        return results
        
    async def cleanup_unused_resources(self) -> int:
        """사용하지 않는 리소스 정리"""
        cleaned_count = 0
        
        for name, metadata in list(self.resource_db.items()):
            if metadata.usage_count == 0:
                try:
                    if metadata.path.exists():
                        metadata.path.unlink()
                    del self.resource_db[name]
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"리소스 삭제 실패 {name}: {e}")
                    
        self._save_resource_database()
        return cleaned_count
        
    def get_resource_stats(self) -> Dict[str, Any]:
        """리소스 통계"""
        total_size = sum(r.size_bytes for r in self.resource_db.values())
        type_counts = {}
        
        for resource in self.resource_db.values():
            type_counts[resource.type] = type_counts.get(resource.type, 0) + 1
            
        return {
            "total_resources": len(self.resource_db),
            "total_size_mb": total_size / (1024 * 1024),
            "type_distribution": type_counts,
            "cache_size_mb": sum(f.stat().st_size for f in self.cache_path.rglob("*") if f.is_file()) / (1024 * 1024)
        }

# 나머지 생성기들은 비슷한 패턴으로 구현...
# _generate_music, _generate_particle_effect, _generate_shader, 
# _generate_font, _generate_tileset 등

def main():
    """메인 실행 함수"""
    print("🎨 AI 리소스 자동 관리 시스템")
    print("=" * 60)
    
    async def test_resource_generation():
        project_path = Path("/tmp/test_project")
        manager = AIResourceManager(project_path)
        
        # 텍스처 생성 테스트
        texture_request = GenerationRequest(
            type="texture",
            name="test_gradient",
            parameters={
                "width": 256,
                "height": 256,
                "texture_type": "gradient",
                "start_color": (255, 0, 0, 255),
                "end_color": (0, 0, 255, 255),
                "direction": "diagonal"
            },
            output_path=project_path / "assets" / "textures" / "test_gradient.png"
        )
        
        result = await manager.generate_resource(texture_request)
        if result:
            print(f"텍스처 생성 성공: {result.name}")
            
        # 오디오 생성 테스트
        audio_request = GenerationRequest(
            type="audio",
            name="test_tone",
            parameters={
                "audio_type": "tone",
                "frequency": 440,
                "duration": 2.0,
                "wave_type": "sine"
            },
            output_path=project_path / "assets" / "audio" / "test_tone.wav"
        )
        
        result = await manager.generate_resource(audio_request)
        if result:
            print(f"오디오 생성 성공: {result.name}")
            
        # 스프라이트 생성 테스트
        sprite_request = GenerationRequest(
            type="sprite",
            name="test_character",
            parameters={
                "sprite_type": "character",
                "size": (64, 64),
                "style": "pixel",
                "primary_color": (100, 150, 255, 255),
                "secondary_color": (50, 100, 200, 255)
            },
            output_path=project_path / "assets" / "sprites" / "test_character.png"
        )
        
        result = await manager.generate_resource(sprite_request)
        if result:
            print(f"스프라이트 생성 성공: {result.name}")
            
        # 리소스 통계
        stats = manager.get_resource_stats()
        print(f"\n리소스 통계:")
        print(f"  총 리소스: {stats['total_resources']}개")
        print(f"  총 크기: {stats['total_size_mb']:.2f}MB")
        print(f"  타입별 분포: {stats['type_distribution']}")
        
    asyncio.run(test_resource_generation())

if __name__ == "__main__":
    main()