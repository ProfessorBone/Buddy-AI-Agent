# TechMentor AI Agent

🚛 **Your On-Device, Hybrid AI Copilot for Trucking Productivity**

---

## 🔍 Overview
TechMentor is a modular, dual-agent system optimized for truckers. It combines a lightweight on-device LLM (Phi-4) with intelligent agent routing and cloud fallback (OpenAI/Anthropic) to manage:

- Trip logging and data capture  
- Real-time route generation via Google Maps  
- Local model-based weekly summaries and trip stats  
- Context-aware agent routing (TripLogger, NavigationAgent, TripAnalysis)  
- Secure .env configuration for local/cloud models  

---

## ✅ Current MVP Capabilities

| Feature             | Status   | Example Query                              |
|---------------------|----------|--------------------------------------------|
| Trip Logger Agent   | ✅ Active | `Start trip 49220. Odometer 720110...`     |
| Navigation Agent    | ✅ Active | `Route to Store 2244 in Tulsa`             |
| Trip Analysis Agent | 🛠 Partial | `Weekly summary` / `Trip summary 49100`    |
| Agent Router        | ✅ Active | Auto-directs prompt to correct agent       |
| Phi-4 Local Model   | ✅ Active | On-device, fast inference                   |

---

## 🔒 Environment Setup

Make sure your `.env` includes:
```env
LOCAL_MODEL_PATH=/data/data/com.termux/files/home/models/Phi-4-mini-128k-instruct-q4_k_m.gguf
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
LOG_LEVEL=INFO
MAX_TOKENS_CLOUD=1000
MAX_TOKENS_LOCAL=500
```

---

## ✅ Core Capabilities (as of June 2025)

### 🟢 Trip Logger Agent
- Logs trip number, odometer readings, trailer numbers
- Captures trip type (Drop, Live Load, etc.)
- Tracks origin, stops, and break suitability
- Stores all data in a structured local SQLite database

### 🟢 Navigation Agent
- Accepts natural language routing prompts
- Generates truck-legal Google Maps URLs
- Future integration with rest stops, hazards, and offline mapping

### 🟢 Agent Routing Engine
- Keyword-based domain detection
- Routes queries to the correct specialist agent
- Falls back to Phi-4 or cloud GPT-4/Claude as needed

### 🛠 Trip Analysis Agent (In Progress)
- Weekly summaries, trip totals, mileage breakdowns
- Database stats and structured insights
- Future anomaly detection and predictive trends

---

## 🟡 Upcoming Features

### Compliance Agent
- HOS tracking (10/14/70-hour rules)
- DOT reminders and inspection readiness

### Voice Input via Whisper
- Voice-to-text logging for drivers on the move
- Planned future TTS for read-back replies

### Streamlit / GUI Dashboard
- Touch-based visual interface for trip review
- Mobile-friendly frontend (planned React Native alt)

---

## 💾 Installation (Coming Soon)
For now, follow the tutorial in the `/docs/` folder or reference:
- `techmentor_agent.py` for the main CLI launcher
- `.env.example` for configuring your model/API paths
- `trucking_database.py` for the full database schema

---

## 🧱 Architecture Summary

┌────────────┐
│  User CLI  │ ← Voice & GUI in future
└────┬───────┘
↓
┌────────────────────┐
│  TechMentor Agent  │ ← Dispatcher
└────┬───────┬───────┘
↓       ↓
TripLogger  Navigation
↓       ↓
SQLite   Google Maps

---

## 👤 Maintainer
Developed by **Faheem** for personal productivity, AI education, and trucker-tech innovation.

- `#mvp` milestones complete
- `#in-progress` = current work
- `#planned` = features in roadmap

GitHub: [https://github.com/ProfessorBone/TechMentor-AI-Agent](https://github.com/ProfessorBone/TechMentor-AI-Agent)

---

📁 _This README auto-syncs with Obsidian notes (TechMentor_Capability_Log.md)._
