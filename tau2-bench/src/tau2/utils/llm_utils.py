import hashlib
import json
import re
from typing import Any, Optional

import litellm


def _make_deterministic_tool_id(name: str, arguments: str, index: int) -> str:
    """
    Generate a deterministic tool call ID based on the tool name and arguments.
    This ensures reproducibility when running the same conversation multiple times.
    """
    content = f"{name}:{arguments}:{index}"
    hash_hex = hashlib.md5(content.encode()).hexdigest()[:16]
    return f"tool-{hash_hex}"
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

# Initialize cache based on config
if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


_tokenizer_cache: dict = {}


def _extract_model_name(model: str) -> str:
    """Extract the HuggingFace model name from various formats."""
    if model.startswith("openai/"):
        return model[7:]
    if model.startswith("vllm://"):
        return model[7:]
    if model.startswith("hf://"):
        return model[5:]
    if model.startswith("huggingface://"):
        return model[14:]
    return model


def _get_tokenizer(model: str):
    """Load and cache the tokenizer for the model. Raises error if not found."""
    model_name = _extract_model_name(model)

    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _tokenizer_cache[model_name] = tokenizer
    logger.debug(f"Loaded tokenizer for {model_name}")
    return tokenizer


def _count_tokens(model: str, messages: list[dict], tools: Optional[list] = None) -> int:
    """Count tokens using the model's tokenizer."""
    tokenizer = _get_tokenizer(model)

    text_parts = []
    for m in messages:
        content = m.get("content") or ""
        role = m.get("role", "")
        text_parts.append(f"{role}: {content}")
        if m.get("tool_calls"):
            text_parts.append(json.dumps(m["tool_calls"]))
    if tools:
        text_parts.append(json.dumps(tools))

    full_text = "\n".join(text_parts)
    tokens = tokenizer.encode(full_text)
    # Add overhead for chat formatting (~10%)
    return int(len(tokens) * 1.1)


def truncate_messages_to_fit_context(
    model: str,
    messages: list[dict],
    max_context_length: int,
    tools: Optional[list] = None,
) -> list[dict]:
    """
    Truncate messages to fit within the max context length.
    Keeps system messages and removes older conversation messages from the middle.

    Args:
        model: The model name for token counting.
        messages: The list of litellm-formatted messages.
        max_context_length: The maximum number of tokens allowed.
        tools: Optional tools list (affects token count).

    Returns:
        Truncated list of messages that fits within max_context_length.
    """
    # Apply a safety margin (5%) to account for tokenizer differences
    effective_max = int(max_context_length * 0.95)

    # Count current tokens
    current_tokens = _count_tokens(model, messages, tools)
    logger.debug(f"Current token count: {current_tokens}, max allowed: {effective_max}")

    if current_tokens <= effective_max:
        return messages

    logger.warning(
        f"Context length ({current_tokens}) exceeds max ({effective_max}). Truncating messages."
    )

    # Separate system messages from conversation messages
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation_messages = [m for m in messages if m.get("role") != "system"]

    if not conversation_messages:
        logger.warning("No conversation messages to truncate, returning as-is.")
        return messages

    # Binary search for the number of recent messages to keep
    left, right = 1, len(conversation_messages)
    result_messages = system_messages  # Start with just system messages

    while left <= right:
        mid = (left + right) // 2
        # Keep the last 'mid' messages
        candidate_messages = system_messages + conversation_messages[-mid:]

        candidate_tokens = _count_tokens(model, candidate_messages, tools)

        if candidate_tokens <= effective_max:
            result_messages = candidate_messages
            left = mid + 1  # Try to keep more messages
        else:
            right = mid - 1  # Need to keep fewer messages

    removed_count = len(conversation_messages) - (len(result_messages) - len(system_messages))
    if removed_count > 0:
        final_tokens = _count_tokens(model, result_messages, tools)
        logger.info(f"Truncated {removed_count} messages to fit context window. Final token count: {final_tokens}")

    return result_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.
            - max_context_length: Optional maximum context length in tokens.
              If the messages exceed this limit, older conversation messages
              will be truncated (system messages are preserved).

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    # Extract max_context_length if provided (don't pass to litellm)
    max_context_length = kwargs.pop("max_context_length", None)
    # Extract tokenizer_model if provided (for LoRA adapters whose name isn't a valid HF model)
    tokenizer_model = kwargs.pop("tokenizer_model", None)

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Truncate messages if max_context_length is specified
    if max_context_length is not None:
        litellm_messages = truncate_messages_to_fit_context(
            model=tokenizer_model or model,
            messages=litellm_messages,
            max_context_length=max_context_length,
            tools=tools_schema,
        )

    # Debug: log if seed is present
    if 'seed' in kwargs:
        logger.debug(f"LLM call with seed={kwargs['seed']}")
    else:
        logger.debug("LLM call without seed")

    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=_make_deterministic_tool_id(tool_call.function.name, tool_call.function.arguments, i),
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for i, tool_call in enumerate(tool_calls)
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
