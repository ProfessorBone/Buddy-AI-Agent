#!/usr/bin/env python3
"""
TechMentor AI Agent - Comprehensive Learning Platform
Combines: Trucking Logistics + Data Science + Excel/Python + AI/ML + Teaching
Optimized for Samsung Galaxy S24 Ultra
"""

import os
import sys
import json
import time
import argparse
import textwrap
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

import requests
import openai
import anthropic
from llama_cpp import Llama
from dotenv import load_dotenv
from trucking_database import TruckingDatabase, TruckingCommandParser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechMentorAgent:
    def __init__(self):
        self.setup_models()
        self.setup_routing_rules()
        self.setup_specialists()
        self.setup_trucking_database()
   
    def setup_trucking_database(self):
        """Initialize trucking database for lightning-fast trip logging"""
        self.trucking_db = TruckingDatabase()
        self.trucking_parser = TruckingCommandParser(self.trucking_db)
        logger.info("Trucking database initialized - ready for trip logging!")
        
    def setup_models(self):
        """Initialize all AI model clients"""
        # Cloud models
        self.openai_client = openai.OpenAI(timeout=20)
        self.anthropic_client = anthropic.Anthropic(timeout=25)
        
        # Local model (lazy loading for better startup performance)
        self._local_llm = None
        self.local_model_path = os.getenv('LOCAL_MODEL_PATH')
        
        logger.info("TechMentor Agent initialized successfully")
    
    @property
    def local_llm(self):
        """Lazy load local model to improve startup time"""
        if self._local_llm is None:
            logger.info("Loading local AI model...")
            self._local_llm = Llama(
                model_path=self.local_model_path,
                n_ctx=2048,
                n_threads=8,  # Optimized for S24 Ultra's Snapdragon 8 Gen 3
                n_gpu_layers=0,  # CPU inference
                verbose=False
            )
            logger.info("Local AI model loaded successfully")
        return self._local_llm
    
    def setup_routing_rules(self):
        """Define intelligent routing rules for different domains"""
        self.routing_rules = {
            'local_keywords': [
                # Quick calculations and basic tasks
                'calculate', 'convert', 'math', 'percentage',
                'gps', 'navigation', 'directions', 'route', 'map',
                'bluetooth', 'phone', 'call', 'text', 'sms',
                'time', 'timer', 'alarm', 'weather',
                'fuel', 'mpg', 'miles', 'distance',
                'quick', 'fast', 'urgent',# Trucking database commands (instant local responses)
                'start trip', 'end trip', 'trip ended', 'add stop', 
                'hook trailer', 'drop trailer', 'trip summary', 
                'weekly summary', 'database stats', 'odometer',
                'trailer', 'store', 'layover', 'summary'
             ],
            'excel_python_keywords': [
                # Excel and Python integration
                '=py()', 'excel formula', 'spreadsheet', 'xlwings',
                'pandas excel', 'dataframe', 'pivot table',
                'excel automation', 'office script', 'vba replacement'
            ],
            'data_science_keywords': [
                # Data science and analytics
                'pandas', 'numpy', 'matplotlib', 'seaborn',
                'data analysis', 'visualization', 'statistics',
                'machine learning', 'ml model', 'algorithm',
                'regression', 'classification', 'clustering'
            ],
            'ai_ml_keywords': [
                # Advanced AI/ML topics
                'transformer', 'neural network', 'deep learning',
                'pytorch', 'tensorflow', 'hugging face',
                'gpt', 'bert', 'llm', 'fine-tuning',
                'gan', 'vae', 'diffusion', 'attention mechanism'
            ],
            'trucking_keywords': [
                # Trucking industry specific
                'dot', 'fmcsa', 'hos', 'hours of service',
                'cdl', 'truck', 'trailer', 'freight',
                'load board', 'broker', 'shipper',
                'compliance', 'inspection', 'logbook'
            ],
            'teaching_keywords': [
                # Educational and tutorial requests
                'explain', 'tutorial', 'learn', 'teach me',
                'step by step', 'beginner', 'example',
                'how to', 'guide', 'walkthrough'
            ],
            'anthropic_keywords': [
                # Complex analysis tasks
                'analyze', 'review code', 'detailed explanation',
                'comprehensive', 'document analysis', 'contract',
                'legal', 'regulation', 'compliance review'
            ]
        }
    
    def setup_specialists(self):
        """Initialize specialist modules for different domains"""
        self.specialists = {
            'excel_python': ExcelPythonSpecialist(),
            'data_science': DataScienceSpecialist(),
            'ai_ml': AIMLSpecialist(),
            'trucking': TruckingSpecialist(),
            'teaching': TeachingSpecialist()
        }
    
    def select_provider(self, prompt: str, default: str = "openai") -> str:
        """Intelligent provider selection based on prompt analysis"""
        prompt_lower = prompt.lower()
        
        # Check for local processing keywords
        if any(keyword in prompt_lower for keyword in self.routing_rules['local_keywords']):
            return "local"
        
        # Check for Anthropic-specific tasks (complex analysis)
        if any(keyword in prompt_lower for keyword in self.routing_rules['anthropic_keywords']):
            return "anthropic"
        
        # Route based on prompt length and complexity
        if len(prompt) > 2000:
            return "anthropic"  # Better for long-form analysis
        elif len(prompt) < 100 and any(word in prompt_lower for word in ['what', 'how', 'when', 'where']):
            return "local"  # Quick questions
        
        return default
    
    def identify_domain(self, prompt: str) -> List[str]:
        """Identify which domains are relevant to the prompt"""
        prompt_lower = prompt.lower()
        relevant_domains = []
        
        for domain, keywords in self.routing_rules.items():
            if domain.endswith('_keywords'):
                domain_name = domain.replace('_keywords', '')
                if any(keyword in prompt_lower for keyword in keywords):
                    relevant_domains.append(domain_name)
        
        return relevant_domains if relevant_domains else ['general']
    
    def enhance_prompt_with_context(self, prompt: str, domains: List[str]) -> str:
        """Add domain-specific context to prompts"""
        context_additions = []
        
        if 'trucking' in domains:
            context_additions.append("Consider trucking industry context, DOT regulations, and practical driver needs.")
        
        if 'excel_python' in domains:
            context_additions.append("Focus on Excel-Python integration using =PY() functions and pandas.")
        
        if 'data_science' in domains:
            context_additions.append("Provide practical data science guidance with code examples.")
        
        if 'teaching' in domains:
            context_additions.append("Explain concepts step-by-step for someone learning the topic.")
        
        if context_additions:
            enhanced_prompt = f"{prompt}\n\nContext: {' '.join(context_additions)}"
            return enhanced_prompt
        
        return prompt
    
    def call_openai(self, prompt: str, domains: List[str]) -> str:
        """Call OpenAI GPT-4o with domain-specific system prompt"""
        system_prompts = {
            'trucking': "You are TechMentor, specializing in trucking logistics, DOT compliance, and driver technology needs.",
            'excel_python': "You are TechMentor, an expert in Excel-Python integration, =PY() functions, and office automation.",
            'data_science': "You are TechMentor, a data science educator focusing on practical pandas, numpy, and ML applications.",
            'ai_ml': "You are TechMentor, an AI/ML specialist helping with transformers, PyTorch, and advanced AI concepts.",
            'teaching': "You are TechMentor, an adaptive tutor who explains complex topics clearly with examples.",
            'general': "You are TechMentor, a comprehensive technology mentor combining trucking, data science, and AI expertise."
        }
        
        # Select most relevant system prompt
        primary_domain = domains[0] if domains else 'general'
        system_prompt = system_prompts.get(primary_domain, system_prompts['general'])
        
        enhanced_prompt = self.enhance_prompt_with_context(prompt, domains)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.7,
            max_tokens=int(os.getenv('MAX_TOKENS_CLOUD', 1000))
        )
        return response.choices[0].message.content.strip()
    
    def call_anthropic(self, prompt: str, domains: List[str]) -> str:
        """Call Anthropic Claude with domain-specific focus"""
        system_prompts = {
            'trucking': "You are TechMentor, specializing in detailed trucking analysis, compliance review, and industry insights.",
            'excel_python': "You are TechMentor, providing comprehensive Excel-Python integration guidance and code review.",
            'data_science': "You are TechMentor, offering detailed data science analysis and comprehensive statistical guidance.",
            'ai_ml': "You are TechMentor, providing in-depth AI/ML explanations, code review, and research insights.",
            'general': "You are TechMentor, providing detailed technical analysis across trucking, data science, and AI domains."
        }
        
        primary_domain = domains[0] if domains else 'general'
        system_prompt = system_prompts.get(primary_domain, system_prompts['general'])
        
        enhanced_prompt = self.enhance_prompt_with_context(prompt, domains)
        
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=int(os.getenv('MAX_TOKENS_CLOUD', 1000)),
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        return response.content[0].text.strip()
    
    
    def call_local(self, prompt: str, domains: List[str]) -> str:
        """Call local Phi-3 model with domain context OR handle trucking database commands"""
        
        # Check if this is a trucking database command first
        if any(keyword in prompt.lower() for keyword in ['start trip', 'end trip', 'trip ended', 'add stop', 'hook trailer', 'trip summary', 'weekly summary', 'database stats']):
            try:
                trucking_result = self.trucking_parser.parse_command(prompt)
                if trucking_result.get('success'):
                    return trucking_result['message']
                elif trucking_result.get('success') is False:
                    return trucking_result['message']
            except Exception as e:
                logger.warning(f"Trucking database error: {e}")
        
        # Regular AI response for non-trucking queries
        domain_context = "You are TechMentor, a helpful assistant"
        if 'trucking' in domains:
            domain_context += " specializing in trucking and logistics"
        if 'data_science' in domains:
            domain_context += " with data science expertise"
        
        formatted_prompt = f"""<|system|>
{domain_context}. Provide concise, practical answers.
<|end|>
<|user|>
{prompt}
<|end|>
<|assistant|>"""
        
        response = self.local_llm(
            formatted_prompt,
            max_tokens=int(os.getenv('MAX_TOKENS_LOCAL', 500)),
            temperature=0.7,
            stop=["<|end|>", "<|user|>"]
        )
        return response["choices"][0]["text"].strip()


    def get_response(self, prompt: str, provider_override: Optional[str] = None) -> Dict:
        """Main response generation with intelligent routing and domain awareness"""
        start_time = time.time()
        
        # Identify relevant domains
        domains = self.identify_domain(prompt)
        
        # Determine provider
        if provider_override and provider_override != "auto":
            provider = provider_override
        else:
            provider = self.select_provider(prompt)
        
        # Provider mapping
        provider_functions = {
            "openai": self.call_openai,
            "anthropic": self.call_anthropic,
            "local": self.call_local
        }
        
        # Attempt primary provider
        try:
            logger.info(f"Using {provider} provider for {domains} domain(s)")
            response_text = provider_functions[provider](prompt, domains)
            
            response_time = time.time() - start_time
            
            return {
                "response": response_text,
                "provider_used": provider,
                "domains_detected": domains,
                "response_time": response_time,
                "fallback_used": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"{provider} provider failed: {str(e)}")
            
            # Fail-over to local if cloud provider failed
            if provider != "local":
                try:
                    logger.info("Falling back to local AI model")
                    response_text = self.call_local(prompt, domains)
                    response_time = time.time() - start_time
                    
                    return {
                        "response": response_text,
                        "provider_used": "local",
                        "domains_detected": domains,
                        "response_time": response_time,
                        "fallback_used": True,
                        "original_provider": provider,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as local_error:
                    logger.error(f"Local fallback also failed: {str(local_error)}")
                    return {
                        "response": "I'm having trouble with all AI services right now. Please check the setup and try again.",
                        "provider_used": "error",
                        "domains_detected": domains,
                        "fallback_used": True,
                        "error": f"All providers failed. Last error: {str(local_error)}",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "response": "Local AI model is not available. Please check the model file path.",
                    "provider_used": "error",
                    "domains_detected": domains,
                    "fallback_used": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }



            
    
# Specialist Classes for Domain-Specific Functionality
class ExcelPythonSpecialist:
    """Handles Excel-Python integration queries"""
    
    def convert_formula(self, excel_formula: str) -> str:
        """Convert Excel formulas to Python =PY() equivalents"""
        conversions = {
            "SUM": "sum",
            "AVERAGE": "numpy.mean",
            "COUNT": "len",
            "MAX": "max",
            "MIN": "min"
        }
        # Basic conversion logic - can be expanded
        return f"Converted formula guidance for: {excel_formula}"
    
    def generate_tutorial(self, topic: str) -> str:
        """Generate Excel-Python tutorials"""
        return f"Excel-Python tutorial for: {topic}"

class DataScienceSpecialist:
    """Handles data science and analytics queries"""
    
    def optimize_pandas(self, code: str) -> str:
        """Provide pandas optimization suggestions"""
        return f"Pandas optimization suggestions for your code"
    
    def explain_concept(self, concept: str) -> str:
        """Explain data science concepts"""
        return f"Data science explanation for: {concept}"

class AIMLSpecialist:
    """Handles AI/ML and deep learning queries"""
    
    def explain_architecture(self, model_type: str) -> str:
        """Explain AI/ML architectures"""
        return f"AI/ML architecture explanation for: {model_type}"
    
    def debug_model(self, issue: str) -> str:
        """Help debug ML models"""
        return f"ML debugging guidance for: {issue}"

class TruckingSpecialist:
    """Handles trucking industry specific queries"""
    
    def calculate_fuel_efficiency(self, miles: float, gallons: float) -> str:
        """Calculate fuel efficiency metrics"""
        if gallons > 0:
            mpg = miles / gallons
            return f"Fuel efficiency: {mpg:.2f} MPG"
        return "Invalid fuel data"
    
    def check_hos_compliance(self, hours: float) -> str:
        """Check hours of service compliance"""
        return f"HOS compliance check for {hours} hours"

class TeachingSpecialist:
    """Handles educational and tutorial requests"""
    
    def create_lesson_plan(self, topic: str, level: str) -> str:
        """Create structured lesson plans"""
        return f"Lesson plan for {topic} at {level} level"
    
    def generate_examples(self, concept: str) -> str:
        """Generate practical examples"""
        return f"Practical examples for: {concept}"

def main():
    """Main CLI interface for TechMentor Agent"""
    parser = argparse.ArgumentParser(description="TechMentor AI Agent - Comprehensive Learning Platform")
    parser.add_argument("--provider", default="auto", 
                       choices=["auto", "openai", "anthropic", "local"],
                       help="AI provider to use")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--domain", help="Force specific domain context")
    
    args = parser.parse_args()
    
    # Initialize agent
    try:
        agent = TechMentorAgent()
        print("🎓 TechMentor AI Agent Ready!")
        print("Multi-domain support: Trucking | Data Science | Excel-Python | AI/ML | Teaching")
        print("Type 'exit' to quit, 'help' for commands, or ask any question.\n")
    except Exception as e:
        print(f"❌ Failed to initialize TechMentor: {e}")
        sys.exit(1)
    
    # Single query mode
    if args.query:
        result = agent.get_response(args.query, args.provider)
        print(f"\n{result['response']}")
        if result.get('domains_detected'):
            print(f"\n🎯 Domains: {', '.join(result['domains_detected'])}")
        if result.get('fallback_used'):
            print(f"⚠️  Used fallback: {result['provider_used']}")
        return
    
    # Interactive mode
    while True:
        try:
            user_input = input("\n🎓 You: ").strip()
            
            if not user_input:
                continue
            elif user_input.lower() in ['exit', 'quit']:
                print("Keep learning and stay curious! 🚀")
                break
            elif user_input.lower() == 'help':
                print("""
🎓 TechMentor Commands:
• Ask questions about: trucking, data science, Excel-Python, AI/ML, programming
• 'provider <openai|anthropic|local>' - Change AI provider
• 'domains' - See detected domains for last query
• 'status' - Show system status
• 'examples' - Show example queries
• 'exit' - Quit TechMentor

Example queries:
• "Calculate fuel cost for 500 miles at 6 MPG"
• "Convert Excel SUM formula to Python =PY()"
• "Explain pandas groupby with trucking data example"
• "How do transformer attention mechanisms work?"
• "Create a beginner tutorial for Python basics"
                """)
                continue
            elif user_input.lower().startswith('provider '):
                new_provider = user_input.split()[1]
                if new_provider in ['openai', 'anthropic', 'local', 'auto']:
                    args.provider = new_provider
                    print(f"✅ Provider set to: {new_provider}")
                else:
                    print("❌ Invalid provider. Use: openai, anthropic, local, or auto")
                continue
            elif user_input.lower() == 'status':
                print(f"""
🎓 TechMentor Status:
• Provider: {args.provider}
• Local model: {'✅ Available' if os.path.exists(agent.local_model_path) else '❌ Not found'}
• OpenAI: {'✅ Configured' if os.getenv('OPENAI_API_KEY') else '❌ No API key'}
• Anthropic: {'✅ Configured' if os.getenv('ANTHROPIC_API_KEY') else '❌ No API key'}
• Domains: Trucking, Data Science, Excel-Python, AI/ML, Teaching
                """)
                continue
            elif user_input.lower() == 'examples':
                print("""
🎓 Example Queries by Domain:

🚛 Trucking:
• "What are the current HOS regulations?"
• "Calculate fuel cost for 1200 miles at $3.50/gallon"
• "Best route planning for oversized loads"

📊 Data Science:
• "How to optimize pandas DataFrame performance?"
• "Explain machine learning classification algorithms"
• "Create a data visualization with matplotlib"

📈 Excel-Python:
• "Convert VLOOKUP to Python =PY() function"
• "Automate Excel reporting with pandas"
• "Create dynamic charts using Python in Excel"

🤖 AI/ML:
• "Explain transformer attention mechanisms"
• "Debug PyTorch model training issues"
• "Fine-tune a language model for trucking data"

👨‍🏫 Teaching:
• "Create a beginner Python tutorial"
• "Explain statistical concepts with examples"
• "Step-by-step guide to data analysis"
                """)
                continue
            
            # Process query
            print("🤔 Analyzing and thinking...")
            result = agent.get_response(user_input, args.provider if args.provider != "auto" else None)
            
            # Display response
            print(f"\n🎓 TechMentor ({result['provider_used']}):")
            print(textwrap.fill(result['response'], width=80))
            
            # Show additional info
            if result.get('domains_detected'):
                print(f"\n🎯 Domains detected: {', '.join(result['domains_detected'])}")
            
            if result.get('fallback_used'):
                print(f"⚠️  Fallback used: {result.get('original_provider', 'unknown')} → {result['provider_used']}")
            
            print(f"⏱️  Response time: {result['response_time']:.2f}s")
                    
        except KeyboardInterrupt:
            print("\n\nKeep learning and stay curious! 🚀")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
