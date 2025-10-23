import os
import time
import logging
from functools import wraps
from dotenv import load_dotenv
from together import Together
# Import provider SDKs
import openai
import requests
import anthropic
from google import genai

# Load environment variables from .env file with override
load_dotenv(override=True)

# Together API does not have an official SDK, use requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = None
        try:
            from threading import Lock
            self.lock = Lock()
        except ImportError:
            self.lock = None

    def acquire(self):
        if self.lock:
            self.lock.acquire()
        try:
            now = time.time()
            time_passed = now - self.last_update
            if time_passed >= 60:
                self.tokens = self.requests_per_minute
                self.last_update = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            wait_time = 60 - time_passed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_update = time.time()
            self.tokens = self.requests_per_minute - 1
            return True
        finally:
            if self.lock:
                self.lock.release()

def rate_limit(requests_per_minute: int):
    limiter = RateLimiter(requests_per_minute)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class AIClient:
    def generate_completion(self, prompt, **kwargs):
        raise NotImplementedError

class OpenAIClient(AIClient):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        self.client = openai.OpenAI(api_key=self.api_key)

    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", "gpt-3.5-turbo")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        if "o4" in model:
            # o4 does not support temperature
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                # max_completion_tokens=max_tokens,
                n=n,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
        return [choice.message.content for choice in response.choices]

class TogetherClient(AIClient):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not found")
        # Using Together's official client library instead of direct API calls
        self.client = Together(api_key=self.api_key)

    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", "togethercomputer/llama-2-70b-chat")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "n": n
        }
        if "o4" in model:
            # payload["max_completion_tokens"] = max_tokens
            # o4 does not support temperature
            pass
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        response = self.client.chat.completions.create(**payload)
        return [choice.message.content for choice in response.choices]

class ClaudeClient(AIClient):
    def __init__(self, api_key=None):
        if anthropic is None:
            raise ImportError("Please install the 'anthropic' package for Claude support.")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @retry_on_error()
    def generate_completion(self, prompt, **kwargs):
        model = kwargs.get("model", "claude-3-opus-20240229")
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        n = kwargs.get("n", 1)
        messages = kwargs.get("messages")
        if messages is not None:
            # Convert OpenAI-style messages to Anthropic format
            system_prompt = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                elif m["role"] == "user":
                    user_content += m["content"] + "\n"
            prompt = user_content.strip()
        else:
            system_prompt = ""
        completions = []
        for _ in range(n):
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            completions.append(response.content[0].text if hasattr(response.content[0], 'text') else response.content[0]["text"])
        return completions

class GeminiClient(AIClient):
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google Gemini API key not found")
        self.client = genai.Client(api_key=self.api_key)

    @retry_on_error()
    @rate_limit(requests_per_minute=15)  # Gemini's rate limit is 15 RPM
    def generate_completion(self, prompt, **kwargs):
        completions = []
        for _ in range(kwargs.get("n", 1)):
            if messages := kwargs.get("messages"):
                # Convert messages to content format
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                response = self.client.models.generate_content(
                    model=kwargs.get("model", "gemini-pro"),
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.0),
                    )
                )
            else:
                response = self.client.models.generate_content(
                    model=kwargs.get("model", "gemini-pro"),
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=kwargs.get("max_tokens", 2048),
                        temperature=kwargs.get("temperature", 0.0),
                    )
                )
            completions.append(response.text)
                
        return completions

def get_client(client_type, api_key=None):
    client_type = client_type.lower()
    if client_type == "openai":
        return OpenAIClient(api_key)
    elif client_type == "together":
        return TogetherClient(api_key)
    elif client_type in ["claude", "anthropic"]:
        return ClaudeClient(api_key)
    elif client_type == "gemini":
        return GeminiClient(api_key)
    else:
        raise ValueError(f"Unsupported client type: {client_type}") 