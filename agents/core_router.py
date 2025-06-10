class CoreRouter:
    def __init__(self, llm_runner, voice_interface, vision_handler=None):
        self.llm = llm_runner
        self.voice = voice_interface
        self.vision = vision_handler

    def route_text(self, prompt: str, context: str = "") -> str:
        return self.llm.generate_response(prompt, context)

    def route_voice(self):
        self.voice.listen_for_wake_word()

    def route_image(self, image_path: str) -> str:
        if self.vision:
            return self.vision.analyze(image_path)
        return "Vision module not enabled."
