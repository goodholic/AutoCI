# AutoCI Modules Package
"""
AutoCI 모듈들을 포함하는 패키지
"""

__version__ = "5.0"
__author__ = "AutoCI Team"

# 핵심 모듈들을 import 가능하도록 설정
from . import korean_conversation
from . import game_factory_24h  
from . import ai_model_controller
from . import self_evolution_system

__all__ = [
    'korean_conversation',
    'game_factory_24h',
    'ai_model_controller', 
    'self_evolution_system'
] 