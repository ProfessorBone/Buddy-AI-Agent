# buddy_agent.py
"""
Buddy AI Agent - Voice & Multimodal Assistant for Mobile Deployment
"""
import argparse
import os
from dotenv import load_dotenv
from agents.llm_runner import LLMRunner
from agents.voice_interface import VoiceInterface
from agents.core_router imporfrom agents.voice_interface import transcribe_audio_whisper from agents.core_router import route_command

#Buddy voice-powered entry point

def main():
     print("\U0001F3A4 Say your command...")
     text = transcribe_audio_whisper()
     if text:
         response = route_command(text)
         print(f"\U0001F9E0 Buddy: {response}")
     else: print("❌ No voice input detected.")

if__ name__ == "__main__":
     main()

t CoreRouter
from agents.vision_handler import VisionHandler
from agents.fallback_manager import FallbackManager
from agents.voice_interface import transcribe_audio_whisper
from agents.voice_interface import transcribe_audio_whisper from agents.core_router import route_command

Buddy voice-powered entry point

def main(): print("\U0001F3A4 Say your command...") text = transcribe_audio_whisper() if text: response = route_command(text) print(f"\U0001F9E0 Buddy: {response}") else: print("❌ No voice input d
