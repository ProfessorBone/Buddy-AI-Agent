# agents/fallback_manager.py
"""
Handles fallback logic between local and cloud LLMs
"""
class FallbackManager:
    def __init__(self, primary_model, fallback_model=None):
        self.primary = primary_model
        self.fallback = fallback_model

    def get_response(self, prompt: str, context: str = "") -> str:
        try:
            return self.primary.generate_response(prompt, context)
        except Exception as e:
            print(f"⚠️ Primary model failed: {e}")
            if self.fallback:
                return self.fallback.generate_response(prompt, context)
            return "Buddy encountered an error and no fallback model is available."

