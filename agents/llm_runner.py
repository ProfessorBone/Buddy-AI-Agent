# agents/llm_runner.py
from llama_cpp import Llama
import os

class LLMRunner:
    def __init__(self, model_path=None, n_threads=6, context_size=4096):
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH")
        self.n_threads = n_threads
        self.context_size = context_size
        self.llm = self._load_model()

    def _load_model(self):
        return Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_threads=self.n_threads,
            n_gpu_layers=0,
            verbose=False
        )

    def generate_response(self, prompt: str, system_prompt: str = "You are Buddy, a helpful assistant.") -> str:
        formatted_prompt = f"""<|system|>\n{system_prompt}\n<|end|>\n<|user|>\n{prompt}\n<|end|>\n<|assistant|>"""
        output = self.llm(formatted_prompt, stop=["<|end|>"])
        return output["choices"][0]["text"].strip()
