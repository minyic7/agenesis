import pytest
from dotenv import load_dotenv

from agenesis.providers import create_llm_provider

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_llm_provider_simple_question():
    """Test LLM provider with a simple question using .env configuration"""
    provider = create_llm_provider()
    
    if provider is None:
        pytest.skip("No API keys available in .env - set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    print(f"\n✅ Using provider: {type(provider).__name__}")
    
    # Test simple completion
    result = await provider.complete("What is 2+2? Respond with just the number.")
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"✅ Question: 'What is 2+2?'")
    print(f"✅ Answer: '{result.strip()}'")
    
    # Test that it works with longer prompts too
    result2 = await provider.complete("Hello! Can you say hello back?")
    assert isinstance(result2, str)
    assert len(result2) > 0
    print(f"✅ Greeting test: '{result2.strip()}'")