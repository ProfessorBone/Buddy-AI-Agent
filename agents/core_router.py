class CoreRouter:
    def __init__(self, llm_runner, voice_interface, vision_handler=None):
        self.llm = llm_runner
        self.voice = voice_interface
        self.vision = vision_handler

    def route_text(self, prompt: str = "", context: str = "") -> str:
        return self.llm.generate_response(prompt, context)

    def route_voice(self):
        self.voice.listen_for_wake_word()

    def route_image(self, image_path: str) -> str:
        if self.vision:
            return self.vision.analyze(image_path)
        return "Vision module not enabled."

    def route_command(self, command_text):
        """
        Basic router for Buddy voice/text commands.
        Currently logs input and returns a placeholder response.
        """
        command_text = command_text.lower()
        print(f"🛠️ Routing command: {command_text}")

        if "trip" in command_text or "odometer" in command_text:
            return "✅ Trip command received!"
        elif "fuel" in command_text:
            return "⛽ Fuel log received!"
        elif "stop" in command_text:
            return "🛑 Stopping loop!"
        else:
            return "🤖 Sorry, I didn’t understand that command."


