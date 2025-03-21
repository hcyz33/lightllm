import time

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Union, Literal
import uuid


class ChatCompletionRequest(BaseModel):
    # The openai api native parameters
    model: str
    messages: List[Dict[str, str]]
    function_call: Optional[str] = "none"
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Additional parameters supported by LightLLM
    do_sample: Optional[bool] = False
    top_k: Optional[int] = -1
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    role_settings: Optional[Dict[str, str]] = None
    character_settings: Optional[List[Dict[str, str]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str
    reasoning_content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

    @field_validator("id", mode="before")
    def ensure_id_is_str(cls, v):
        return str(v)


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    session_params: Optional[Dict] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None

