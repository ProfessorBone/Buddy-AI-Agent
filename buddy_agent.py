# buddy_agent.py
"""
Buddy AI Agent - Voice & Multimodal Assistant for Mobile Deployment
"""
import argparse
import os
from dotenv import load_dotenv
from agents.llm_runner import LLMRunner
from agents.voice_interface import VoiceInterface
from agents.core_router import CoreRouter
from agents.vision_handler import VisionHandler
from agents.fallback_manager import FallbackManager

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Buddy AI Agent")
    parser.add_argument("--mode", choices=["voice", "text", "image"], default="text", help="Interaction mode")
    parser.add_argument("--query", type=str, help="Text prompt")
    parser.add_argument("--image", type=str, help="Image file path")
    args = parser.parse_args()

    # Initialize components
    llm = LLMRunner()
    voice = VoiceInterface()
    vision = VisionHandler(model_path="~/models/moondream2-q5_1.gguf")
    router = CoreRouter(llm_runner=llm, voice_interface=voice, vision_handler=vision)

    if args.mode == "voice":
        router.route_voice()
    elif args.mode == "text" and args.query:
        response = router.route_text(args.query)
        print("🧠 Buddy:", response)
    elif args.mode == "image" and args.image:
        response = router.route_image(args.image)
        print("🖼️ Buddy (vision):", response)
    else:
        print("❌ Invalid usage. Provide --query for text or --image for image mode.")

if __name__ == "__main__":
    main()