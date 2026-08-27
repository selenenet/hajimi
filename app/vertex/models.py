from pydantic import BaseModel, ConfigDict  # Field removed
from typing import List, Dict, Any, Optional, Union, Literal
from app.utils.logging import vertex_log


# Define data models
class ImageUrl(BaseModel):
    url: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class OpenAIMessage(BaseModel):
    role: str
    content: Optional[
        Union[str, List[Union[ContentPartText, ContentPartImage, Dict[str, Any]]]]
    ] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Tool messages and newer OpenAI clients may attach provider-specific fields.
    model_config = ConfigDict(extra="allow")


class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    logprobs: Optional[int] = None
    response_logprobs: Optional[bool] = None
    n: Optional[int] = None  # Maps to candidate_count in Vertex AI
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Any] = None

    # Allow extra fields to pass through without causing validation errors
    model_config = ConfigDict(extra="allow")


class TokenUsage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


class GeminiMessage(BaseModel):
    role: str  # 'user' or 'model'
    content: str
    name: Optional[str] = None


class GeminiChatRequest(BaseModel):
    model: str
    messages: List[GeminiMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_output_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

    def log_request(self):
        vertex_log("info", f"Chat request for model: {self.model}")
        vertex_log(
            "debug",
            f"Request parameters: temp={self.temperature}, top_p={self.top_p}, max_tokens={self.max_output_tokens}",
        )


class GeminiCompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_output_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

    def log_request(self):
        vertex_log("info", f"Completion request for model: {self.model}")
        vertex_log(
            "debug",
            f"Request parameters: temp={self.temperature}, top_p={self.top_p}, max_tokens={self.max_output_tokens}",
        )
        prompt_preview = (
            self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
        )
        vertex_log("debug", f"Prompt preview: {prompt_preview}")
