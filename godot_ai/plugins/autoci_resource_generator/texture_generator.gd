@tool
extends Node

# AutoCI Texture Generator
# AI 기반 텍스처 자동 생성

enum TextureType {
    SOLID,
    GRADIENT,
    NOISE,
    PATTERN,
    SPRITE
}

func generate_texture(type: TextureType, width: int, height: int, params: Dictionary = {}) -> ImageTexture:
    """텍스처 생성"""
    var image = Image.create(width, height, false, Image.FORMAT_RGBA8)
    
    match type:
        TextureType.SOLID:
            _fill_solid_color(image, params.get("color", Color.WHITE))
        TextureType.GRADIENT:
            _fill_gradient(image, params)
        TextureType.NOISE:
            _fill_noise(image, params)
        TextureType.PATTERN:
            _fill_pattern(image, params)
        TextureType.SPRITE:
            _draw_sprite(image, params)
    
    var texture = ImageTexture.new()
    texture.set_image(image)
    return texture

func _fill_solid_color(image: Image, color: Color) -> void:
    """단색 채우기"""
    image.fill(color)

func _fill_gradient(image: Image, params: Dictionary) -> void:
    """그라디언트 채우기"""
    var start_color = params.get("start_color", Color.BLACK)
    var end_color = params.get("end_color", Color.WHITE)
    var direction = params.get("direction", "horizontal")
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var ratio: float
            
            if direction == "horizontal":
                ratio = float(x) / (width - 1)
            else:  # vertical
                ratio = float(y) / (height - 1)
            
            var color = start_color.lerp(end_color, ratio)
            image.set_pixel(x, y, color)

func _fill_noise(image: Image, params: Dictionary) -> void:
    """노이즈 채우기"""
    var noise_intensity = params.get("intensity", 0.5)
    var base_color = params.get("base_color", Color.GRAY)
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var noise_value = randf_range(-noise_intensity, noise_intensity)
            var color = Color(
                clamp(base_color.r + noise_value, 0.0, 1.0),
                clamp(base_color.g + noise_value, 0.0, 1.0),
                clamp(base_color.b + noise_value, 0.0, 1.0),
                base_color.a
            )
            image.set_pixel(x, y, color)

func _fill_pattern(image: Image, params: Dictionary) -> void:
    """패턴 채우기"""
    var pattern_type = params.get("pattern", "checkerboard")
    var color1 = params.get("color1", Color.BLACK)
    var color2 = params.get("color2", Color.WHITE)
    var scale = params.get("scale", 8)
    
    var width = image.get_width()
    var height = image.get_height()
    
    for y in range(height):
        for x in range(width):
            var color: Color
            
            match pattern_type:
                "checkerboard":
                    if (x / scale + y / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                "stripes_horizontal":
                    if (y / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                "stripes_vertical":
                    if (x / scale) % 2 == 0:
                        color = color1
                    else:
                        color = color2
                _:
                    color = color1
            
            image.set_pixel(x, y, color)

func _draw_sprite(image: Image, params: Dictionary) -> void:
    """스프라이트 그리기"""
    var sprite_type = params.get("sprite_type", "player")
    
    match sprite_type:
        "player":
            _draw_player_sprite(image)
        "enemy":
            _draw_enemy_sprite(image)
        "collectible":
            _draw_collectible_sprite(image)
        "platform":
            _draw_platform_sprite(image)

func _draw_player_sprite(image: Image) -> void:
    """플레이어 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    var center_x = width / 2
    var center_y = height / 2
    
    # 머리 (원)
    _draw_circle(image, center_x, center_y - height / 4, width / 6, Color.BEIGE)
    
    # 몸 (사각형)
    _draw_rectangle(image, center_x - width / 6, center_y - height / 8, 
                   width / 3, height / 2, Color.BLUE)
    
    # 팔과 다리 (선)
    _draw_line(image, center_x - width / 4, center_y, center_x + width / 4, center_y, Color.BEIGE)
    _draw_line(image, center_x, center_y + height / 4, center_x - width / 8, center_y + height / 2, Color.BLUE)
    _draw_line(image, center_x, center_y + height / 4, center_x + width / 8, center_y + height / 2, Color.BLUE)

func _draw_enemy_sprite(image: Image) -> void:
    """적 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 간단한 적 스프라이트 (삼각형)
    _draw_triangle(image, width / 2, height / 4, width / 4, height * 3 / 4, 
                  width * 3 / 4, height * 3 / 4, Color.RED)

func _draw_collectible_sprite(image: Image) -> void:
    """수집 아이템 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 별 모양
    _draw_star(image, width / 2, height / 2, width / 3, Color.YELLOW)

func _draw_platform_sprite(image: Image) -> void:
    """플랫폼 스프라이트 그리기"""
    var width = image.get_width()
    var height = image.get_height()
    
    # 플랫폼 (그라디언트 사각형)
    _draw_rectangle(image, 0, 0, width, height, Color.BROWN)

func _draw_circle(image: Image, cx: int, cy: int, radius: int, color: Color) -> void:
    """원 그리기"""
    for y in range(max(0, cy - radius), min(image.get_height(), cy + radius + 1)):
        for x in range(max(0, cx - radius), min(image.get_width(), cx + radius + 1)):
            var dx = x - cx
            var dy = y - cy
            if dx * dx + dy * dy <= radius * radius:
                image.set_pixel(x, y, color)

func _draw_rectangle(image: Image, x: int, y: int, w: int, h: int, color: Color) -> void:
    """사각형 그리기"""
    for py in range(max(0, y), min(image.get_height(), y + h)):
        for px in range(max(0, x), min(image.get_width(), x + w)):
            image.set_pixel(px, py, color)

func _draw_line(image: Image, x1: int, y1: int, x2: int, y2: int, color: Color) -> void:
    """선 그리기 (Bresenham 알고리즘)"""
    var dx = abs(x2 - x1)
    var dy = abs(y2 - y1)
    var sx = 1 if x1 < x2 else -1
    var sy = 1 if y1 < y2 else -1
    var err = dx - dy
    
    var x = x1
    var y = y1
    
    while true:
        if x >= 0 and x < image.get_width() and y >= 0 and y < image.get_height():
            image.set_pixel(x, y, color)
        
        if x == x2 and y == y2:
            break
            
        var e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

func _draw_triangle(image: Image, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, color: Color) -> void:
    """삼각형 그리기"""
    # 간단한 구현: 세 변을 선으로 그리기
    _draw_line(image, x1, y1, x2, y2, color)
    _draw_line(image, x2, y2, x3, y3, color)
    _draw_line(image, x3, y3, x1, y1, color)

func _draw_star(image: Image, cx: int, cy: int, radius: int, color: Color) -> void:
    """별 그리기"""
    var points = []
    for i in range(10):
        var angle = i * PI / 5
        var r = radius if i % 2 == 0 else radius / 2
        var x = cx + int(cos(angle) * r)
        var y = cy + int(sin(angle) * r)
        points.append(Vector2(x, y))
    
    # 별의 선들 그리기
    for i in range(points.size()):
        var next_i = (i + 1) % points.size()
        _draw_line(image, points[i].x, points[i].y, points[next_i].x, points[next_i].y, color)
