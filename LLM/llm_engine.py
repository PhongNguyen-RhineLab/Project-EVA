"""
LLM Engine for EVA Project

Supports:
1. API backends (free tier):
   - Google Gemini (free API)
   - Groq (free tier - fast inference)
   - OpenRouter (free models available)

2. Local backends (offline):
   - Ollama (easiest setup)
   - Transformers (HuggingFace models)
"""

import os
import json
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Generator
from dataclasses import dataclass
from pathlib import Path
import time

# Add project root to path for console import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from console import console, Colors
except ImportError:
    # Fallback minimal console
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'

    class Console:
        def info(self, msg, indent=0): print(f"{'  '*indent}[*] {msg}")
        def success(self, msg, indent=0): print(f"{'  '*indent}[+] {msg}")
        def warning(self, msg, indent=0): print(f"{'  '*indent}[!] {msg}")
        def error(self, msg, indent=0): print(f"{'  '*indent}[-] {msg}")
        def header(self, title, width=50): print(f"\n{'='*width}\n{title}\n{'='*width}")
        def item(self, label, value, indent=1): print(f"{'  '*indent}{label}: {value}")
        def divider(self, width=50): print("-" * width)
    console = Console()


# Load environment variables from .env file if exists
def load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"\'')

load_env()


@dataclass
class LLMResponse:
    """LLM response container"""
    text: str
    model: str
    tokens_used: Optional[int] = None
    latency: float = 0.0
    finish_reason: Optional[str] = None


# --------------------------
# Base LLM Class
# --------------------------
class BaseLLM(ABC):
    """Abstract base class for LLM backends"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


# --------------------------
# Google Gemini (Free API)
# --------------------------
class GeminiLLM(BaseLLM):
    """
    Google Gemini API

    Free tier: 60 requests/minute
    Get API key: https://makersuite.google.com/app/apikey
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-flash"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._client = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            console.success(f"Gemini initialized ({self.model})")
        except ImportError:
            console.warning("Install google-generativeai: pip install google-generativeai")
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Gemini not available. Set GEMINI_API_KEY.")

        start_time = time.time()

        response = self._client.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )

        latency = time.time() - start_time

        return LLMResponse(
            text=response.text,
            model=self.model,
            latency=latency,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
        )


# --------------------------
# Groq (Free API - Fast)
# --------------------------
class GroqLLM(BaseLLM):
    """
    Groq API - Very fast inference

    Free tier: 14,400 requests/day for smaller models
    Get API key: https://console.groq.com/keys

    Available models:
    - llama-3.3-70b-versatile (best quality)
    - llama-3.1-8b-instant (fastest)
    - gemma2-9b-it (good for Vietnamese)
    - mixtral-8x7b-32768 (good balance)
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.1-8b-instant"
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self._client = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            console.success(f"Groq initialized ({self.model})")
        except ImportError:
            console.warning("Install groq: pip install groq")
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Groq not available. Set GROQ_API_KEY.")

        start_time = time.time()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        latency = time.time() - start_time

        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else None,
            latency=latency,
            finish_reason=response.choices[0].finish_reason
        )


# --------------------------
# OpenRouter (Multiple Free Models)
# --------------------------
class OpenRouterLLM(BaseLLM):
    """
    OpenRouter API - Access multiple models

    Free models available (with rate limits)
    Get API key: https://openrouter.ai/keys

    Some free models:
    - meta-llama/llama-3.2-3b-instruct:free
    - google/gemma-2-9b-it:free
    - mistralai/mistral-7b-instruct:free
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "meta-llama/llama-3.2-3b-instruct:free"
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        if self.api_key:
            console.success(f"OpenRouter initialized ({self.model})")

    def is_available(self) -> bool:
        return self.api_key is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenRouter not available. Set OPENROUTER_API_KEY.")

        import requests

        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

        latency = time.time() - start_time

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error: {response.text}")

        data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            tokens_used=data.get("usage", {}).get("total_tokens"),
            latency=latency,
            finish_reason=data["choices"][0].get("finish_reason")
        )


# --------------------------
# Ollama (Local)
# --------------------------
class OllamaLLM(BaseLLM):
    """
    Ollama - Local LLM runner

    Setup:
    1. Install: https://ollama.ai/download
    2. Pull model: ollama pull llama3.2:3b
    3. Start server: ollama serve

    Good models for Vietnamese:
    - llama3.2:3b (fast, decent)
    - gemma2:9b (better quality)
    - qwen2.5:7b (good multilingual)
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            self._available = response.status_code == 200
            if self._available:
                console.success(f"Ollama connected ({self.model})")
        except:
            self._available = False

        return self._available

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Ollama not available. Is it running?")

        import requests

        start_time = time.time()

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
        )

        latency = time.time() - start_time

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        data = response.json()

        return LLMResponse(
            text=data["response"],
            model=self.model,
            tokens_used=data.get("eval_count"),
            latency=latency,
            finish_reason="stop" if data.get("done") else None
        )


# --------------------------
# Transformers (HuggingFace Local)
# --------------------------
class TransformersLLM(BaseLLM):
    """
    HuggingFace Transformers - Local inference

    Requires GPU for reasonable speed.
    Models are downloaded automatically.

    Good small models:
    - microsoft/phi-2 (2.7B, fast)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, very fast)
    - Qwen/Qwen2.5-1.5B-Instruct (1.5B, multilingual)
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Load model (call explicitly to avoid slow startup)"""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            console.info(f"Loading {self.model_name}...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            )

            self._loaded = True
            console.success(f"Transformers model loaded")

        except ImportError:
            console.warning("Install transformers: pip install transformers torch")
        except Exception as e:
            console.error(f"Failed to load model: {e}")

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        import torch

        start_time = time.time()

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # Decode
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        latency = time.time() - start_time

        return LLMResponse(
            text=generated_text,
            model=self.model_name,
            tokens_used=len(outputs[0]),
            latency=latency,
            finish_reason="stop"
        )


# --------------------------
# Unified LLM Engine
# --------------------------
class LLMEngine:
    """
    Unified LLM Engine with automatic fallback

    Priority order:
    1. Specified backend
    2. Any available API backend
    3. Local backend (Ollama)

    Usage:
        # Auto-select best available
        llm = LLMEngine()

        # Force specific backend
        llm = LLMEngine(backend="groq")

        # Generate
        response = llm.generate(prompt)
        print(response.text)
    """

    BACKENDS = {
        "gemini": GeminiLLM,
        "groq": GroqLLM,
        "openrouter": OpenRouterLLM,
        "ollama": OllamaLLM,
        "transformers": TransformersLLM
    }

    def __init__(
        self,
        backend: str = None,
        model: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize LLM Engine

        Args:
            backend: Specific backend to use (gemini, groq, openrouter, ollama, transformers)
            model: Model name/ID
            api_key: API key (if applicable)
            **kwargs: Additional backend-specific arguments
        """
        self.backend_name = backend
        self._backend = None

        console.header("Initializing LLM Engine")

        if backend:
            self._init_specific_backend(backend, model, api_key, **kwargs)
        else:
            self._init_auto_backend()

        if self._backend and self._backend.is_available():
            console.success(f"LLM Engine ready ({self.backend_name})")
        else:
            console.warning("No LLM backend available!")
            self._print_setup_help()

        console.divider()
        print()

    def _init_specific_backend(self, backend: str, model: str, api_key: str, **kwargs):
        """Initialize a specific backend"""
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

        backend_class = self.BACKENDS[backend]

        init_kwargs = {**kwargs}
        if model:
            init_kwargs["model"] = model
        if api_key:
            init_kwargs["api_key"] = api_key

        self._backend = backend_class(**init_kwargs)
        self.backend_name = backend

    def _init_auto_backend(self):
        """Auto-detect and initialize best available backend"""
        console.info("Auto-detecting available backends...")

        # Try API backends first
        api_backends = ["groq", "gemini", "openrouter"]

        for backend in api_backends:
            try:
                self._backend = self.BACKENDS[backend]()
                if self._backend.is_available():
                    self.backend_name = backend
                    console.item(backend, f"{Colors.GREEN}available{Colors.RESET}")
                    return
            except Exception:
                console.item(backend, "not available")

        # Try Ollama
        try:
            self._backend = OllamaLLM()
            if self._backend.is_available():
                self.backend_name = "ollama"
                console.item("ollama", f"{Colors.GREEN}available{Colors.RESET}")
                return
        except:
            console.item("ollama", "not available")

        self._backend = None
        self.backend_name = None

    def _print_setup_help(self):
        """Print help for setting up backends"""
        print("\nSetup instructions:")
        print("\n[Option 1] Groq (Recommended - Fast & Free)")
        print("   1. Get API key: https://console.groq.com/keys")
        print("   2. Set: export GROQ_API_KEY=your_key")
        print("   3. Or add to .env file: GROQ_API_KEY=your_key")

        print("\n[Option 2] Google Gemini (Free)")
        print("   1. Get API key: https://makersuite.google.com/app/apikey")
        print("   2. Set: export GEMINI_API_KEY=your_key")

        print("\n[Option 3] Ollama (Local/Offline)")
        print("   1. Install: https://ollama.ai/download")
        print("   2. Run: ollama pull llama3.2:3b")
        print("   3. Start: ollama serve")

    def is_available(self) -> bool:
        """Check if any backend is available"""
        return self._backend is not None and self._backend.is_available()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMResponse with generated text
        """
        if not self.is_available():
            raise RuntimeError("No LLM backend available. See setup instructions.")

        return self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def list_backends(self) -> Dict[str, bool]:
        """List all backends and their availability"""
        status = {}

        for name, backend_class in self.BACKENDS.items():
            try:
                backend = backend_class()
                status[name] = backend.is_available()
            except:
                status[name] = False

        return status


# --------------------------
# Test Function
# --------------------------
def test_llm():
    """Test LLM engine"""
    console.header("LLM Engine Test")

    # Check available backends
    console.info("Checking backends...")
    engine = LLMEngine()

    status = engine.list_backends()
    for backend, available in status.items():
        status_str = f"{Colors.GREEN}yes{Colors.RESET}" if available else f"{Colors.RED}no{Colors.RESET}"
        console.item(backend, status_str)

    if not engine.is_available():
        console.error("No backend available. Follow setup instructions above.")
        return

    # Test generation
    console.info(f"Testing generation with {engine.backend_name}...")

    test_prompt = """You are a helpful assistant. Respond briefly.

User: Hello! How are you today?
Assistant:"""

    try:
        response = engine.generate(test_prompt, max_tokens=100)

        console.success("Response received!")
        console.item("Model", response.model)
        console.item("Latency", f"{response.latency:.2f}s")
        console.item("Tokens", response.tokens_used or "N/A")
        print(f"\nResponse:\n{response.text}")

    except Exception as e:
        console.error(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_llm()
    else:
        print("LLM Engine for EVA Project")
        print("=" * 50)
        print("\nUsage:")
        print("  python llm_engine.py --test    # Test LLM")
        print("\nBackends available:")
        print("  - gemini (Google Gemini - free)")
        print("  - groq (Groq - free, fast)")
        print("  - openrouter (Multiple models)")
        print("  - ollama (Local)")
        print("  - transformers (HuggingFace local)")
        print("\nSetup:")
        print("  1. Get free API key from Groq or Gemini")
        print("  2. Create .env file with: GROQ_API_KEY=your_key")
        print("  3. Run: python llm_engine.py --test")