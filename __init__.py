from .EgyTTS import EgyTTSGenerator

NODE_CLASS_MAPPINGS = {
    "EgyTTSGenerator": EgyTTSGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgyTTSGenerator": "🇪🇬 Egyptian TTS (EgyTTS)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
