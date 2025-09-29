import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    # OpenAI config
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Anthropic config
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # Common config
    max_retries: int = 3
    timeout: float = 30.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion"""
        pass
    
    
    @abstractmethod
    async def classify(self, prompt: str, options: List[str], **kwargs) -> str:
        """Return one of the provided options"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        
    async def _get_client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion using OpenAI API"""
        client = await self._get_client()
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', 0.1),
                    max_tokens=kwargs.get('max_tokens', 500)
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"OpenAI API failed after {self.config.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    
    async def classify(self, prompt: str, options: List[str], **kwargs) -> str:
        """Return one of the provided options"""
        options_str = ", ".join(options)
        classification_prompt = f"{prompt}\n\nRespond with exactly one of these options: {options_str}"
        
        response = await self.complete(classification_prompt, temperature=0.0, max_tokens=20)
        response_clean = response.strip().lower()
        
        # Find best match from options
        for option in options:
            if option.lower() in response_clean:
                return option
        
        # Fallback to first option if no match
        return options[0]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        
    async def _get_client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.anthropic_api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion using Anthropic API"""
        client = await self._get_client()
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=kwargs.get('max_tokens', 500),
                    temperature=kwargs.get('temperature', 0.1),
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Anthropic API failed after {self.config.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    
    async def classify(self, prompt: str, options: List[str], **kwargs) -> str:
        """Return one of the provided options"""
        options_str = ", ".join(options)
        classification_prompt = f"{prompt}\n\nRespond with exactly one of these options: {options_str}"
        
        response = await self.complete(classification_prompt, temperature=0.0, max_tokens=20)
        response_clean = response.strip().lower()
        
        # Find best match from options
        for option in options:
            if option.lower() in response_clean:
                return option
        
        # Fallback to first option if no match
        return options[0]




def load_config_from_env() -> LLMConfig:
    """Load LLM configuration from environment variables"""
    return LLMConfig(
        # OpenAI config
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        openai_base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        
        # Anthropic config
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        anthropic_model=os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
        
        # Common config
        max_retries=int(os.getenv('LLM_MAX_RETRIES', '3')),
        timeout=float(os.getenv('LLM_TIMEOUT', '30.0'))
    )


def create_llm_provider(config: Optional[LLMConfig] = None) -> Optional[BaseLLMProvider]:
    """Factory function to create appropriate LLM provider"""
    if config is None:
        config = load_config_from_env()
    
    # Try Anthropic first if API key is available
    if config.anthropic_api_key:
        try:
            return AnthropicProvider(config)
        except ImportError:
            pass  # Try next option
    
    # Try OpenAI if API key is available
    if config.openai_api_key:
        try:
            return OpenAIProvider(config)
        except ImportError:
            pass  # No fallback - return None
    
    # No LLM provider available
    return None