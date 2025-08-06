import os
import requests
from typing import List, Dict, Any, Optional, Union, Mapping, Sequence, TypeVar, cast
import logging
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun, Callbacks, CallbackManager
from langchain.schema import LLMResult
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.output import Generation, LLMResult
from pydantic import Field, root_validator
from langchain.load.serializable import Serializable

logger = logging.getLogger(__name__)

T = TypeVar("T")

class OpenRouterLLM(LLM, Runnable):
    """LLM wrapper for OpenRouter API."""
    
    api_key: str
    model: str = "anthropic/claude-3-opus:beta"
    temperature: float = 0.1
    max_tokens: int = 2048
    base_url: str = "https://openrouter.ai/api/v1"
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs
    ) -> str:
        """Call the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            data["stop"] = stop
            
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error calling OpenRouter API: {e}")
            if e.response.status_code == 401:
                raise Exception("Invalid OpenRouter API key. Please check your API key.")
            elif e.response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"OpenRouter API error: {e}")
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise Exception(f"Failed to call OpenRouter API: {str(e)}")
    
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string."""
        # Simple approximation - 1 token ~= 4 chars
        return len(text) // 4
        
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> str:
        """Invoke the OpenRouter API with a prompt."""
        config = config or {}
        callbacks = config.get("callbacks")
        return self._call(input, callbacks=callbacks)

class OpenRouterChatModel(Serializable, Runnable):
    """Chat model wrapper for OpenRouter API that implements Runnable interface."""
    
    api_key: str = Field(..., description="OpenRouter API key")
    model: str = Field(default="anthropic/claude-3-opus:beta", description="Model name")
    temperature: float = Field(default=0.2, description="Temperature for sampling - lowered for more factual responses")
    max_tokens: int = Field(default=2048, description="Maximum number of tokens to generate")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="Base URL for API")
    name: str = Field(default="openrouter_chat_model", description="Model name for identification")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        """Invoke the OpenRouter API with a prompt."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": input}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return OpenRouterResponse(content=content)
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error calling OpenRouter API: {e}")
            if e.response.status_code == 401:
                raise Exception("Invalid OpenRouter API key. Please check your API key.")
            elif e.response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"OpenRouter API error: {e}")
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise Exception(f"Failed to call OpenRouter API: {str(e)}")
    
    def batch(self, inputs: List[str], config: Optional[RunnableConfig] = None) -> List[Any]:
        """Process multiple inputs with the model."""
        return [self.invoke(input, config) for input in inputs]
    
    def stream(self, input: str, config: Optional[RunnableConfig] = None) -> Any:
        """Stream output from the model."""
        # OpenRouter doesn't support streaming in this implementation
        return self.invoke(input, config)

class OpenRouterResponse:
    """Response object to mimic LangChain response format."""
    
    def __init__(self, content: str):
        self.content = content