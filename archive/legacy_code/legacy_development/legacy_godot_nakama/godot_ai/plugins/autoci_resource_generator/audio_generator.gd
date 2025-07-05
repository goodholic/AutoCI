@tool
extends Node

# AutoCI Audio Generator
# AI 기반 오디오 자동 생성

enum AudioType {
    TONE,
    NOISE,
    MELODY,
    EFFECT
}

func generate_audio(type: AudioType, duration: float, params: Dictionary = {}) -> AudioStream:
    """오디오 생성"""
    var sample_rate = 44100
    var samples = int(duration * sample_rate)
    var data = PackedFloat32Array()
    data.resize(samples)
    
    match type:
        AudioType.TONE:
            _generate_tone(data, sample_rate, params)
        AudioType.NOISE:
            _generate_noise(data, params)
        AudioType.MELODY:
            _generate_melody(data, sample_rate, params)
        AudioType.EFFECT:
            _generate_effect(data, sample_rate, params)
    
    var audio_stream = AudioStreamWAV.new()
    audio_stream.data = data.to_byte_array()
    audio_stream.format = AudioStreamWAV.FORMAT_32_FLOAT
    audio_stream.mix_rate = sample_rate
    audio_stream.stereo = false
    
    return audio_stream

func _generate_tone(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """톤 생성"""
    var frequency = params.get("frequency", 440.0)  # A4
    var amplitude = params.get("amplitude", 0.5)
    
    for i in range(data.size()):
        var t = float(i) / sample_rate
        data[i] = amplitude * sin(2 * PI * frequency * t)

func _generate_noise(data: PackedFloat32Array, params: Dictionary) -> void:
    """노이즈 생성"""
    var amplitude = params.get("amplitude", 0.3)
    
    for i in range(data.size()):
        data[i] = amplitude * randf_range(-1.0, 1.0)

func _generate_melody(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """멜로디 생성"""
    var notes = params.get("notes", [261.63, 293.66, 329.63, 349.23])  # C, D, E, F
    var note_duration = params.get("note_duration", 0.5)
    var amplitude = params.get("amplitude", 0.4)
    
    var samples_per_note = int(note_duration * sample_rate)
    
    for i in range(data.size()):
        var note_index = (i / samples_per_note) % notes.size()
        var frequency = notes[note_index]
        var t = float(i % samples_per_note) / sample_rate
        data[i] = amplitude * sin(2 * PI * frequency * t)

func _generate_effect(data: PackedFloat32Array, sample_rate: int, params: Dictionary) -> void:
    """효과음 생성"""
    var effect_type = params.get("effect_type", "jump")
    var amplitude = params.get("amplitude", 0.6)
    
    match effect_type:
        "jump":
            _generate_jump_sound(data, sample_rate, amplitude)
        "collect":
            _generate_collect_sound(data, sample_rate, amplitude)
        "hit":
            _generate_hit_sound(data, sample_rate, amplitude)
        "explosion":
            _generate_explosion_sound(data, sample_rate, amplitude)

func _generate_jump_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """점프 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var frequency = 300 + 200 * exp(-t * 5)  # 주파수가 감소
        data[i] = amplitude * sin(2 * PI * frequency * t) * exp(-t * 3)

func _generate_collect_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """수집 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var frequency = 400 + 300 * t  # 주파수가 증가
        data[i] = amplitude * sin(2 * PI * frequency * t) * exp(-t * 2)

func _generate_hit_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """타격 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var noise = randf_range(-0.3, 0.3)
        var tone = 150 * sin(2 * PI * 100 * t)
        data[i] = amplitude * (noise + tone) * exp(-t * 8)

func _generate_explosion_sound(data: PackedFloat32Array, sample_rate: int, amplitude: float) -> void:
    """폭발 사운드"""
    for i in range(data.size()):
        var t = float(i) / sample_rate
        var noise = randf_range(-1.0, 1.0)
        var bass = 0.3 * sin(2 * PI * 60 * t)
        data[i] = amplitude * (noise * 0.7 + bass) * exp(-t * 2)
