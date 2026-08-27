import asyncio
import json
from types import SimpleNamespace

from google.genai import types

from app.vertex.api_helpers import (
    _base_fake_stream_engine,
    create_generation_config,
    is_response_valid,
)
from app.vertex.message_processing import (
    convert_to_openai_format,
    create_gemini_prompt,
)
from app.vertex.models import OpenAIRequest
from app.models.schemas import ChatCompletionRequest


wrapper_request = ChatCompletionRequest.model_validate(
    {
        "model": "[PAY]gemini-3.7-flash",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
    }
)
assert wrapper_request.tool_choice == "required"
assert wrapper_request.tools[0]["function"]["name"] == "get_weather"


request = OpenAIRequest.model_validate(
    {
        "model": "[PAY]gemini-3.7-flash",
        "messages": [{"role": "user", "content": "Weather in Beijing?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "$defs": {
                            "City": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "type": "object",
                        "properties": {"city": {"$ref": "#/$defs/City"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
)
config = create_generation_config(request)
declaration = config["tools"][0].function_declarations[0]
assert declaration.name == "get_weather"
assert declaration.parameters.properties["city"].type.value == "STRING"
assert config["tool_config"].function_calling_config.mode.value == "AUTO"
assert config["automatic_function_calling"].disable is True


history_request = OpenAIRequest.model_validate(
    {
        "model": "[PAY]gemini-3.7-flash",
        "messages": [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Beijing"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": '{"temperature":26}',
            },
        ],
    }
)
contents = create_gemini_prompt(history_request.messages)
assert len(contents) == 3
assert contents[1].role == "model"
assert contents[1].parts[0].function_call.name == "get_weather"
assert contents[1].parts[0].function_call.args == {"city": "Beijing"}
assert contents[2].parts[0].function_response.name == "get_weather"
assert contents[2].parts[0].function_response.response == {"temperature": 26}


candidate = SimpleNamespace(
    content=types.Content(
        role="model",
        parts=[
            types.Part(
                thought_signature=b"test-signature",
                function_call=types.FunctionCall(
                    id="call_weather_1",
                    name="get_weather",
                    args={"city": "Beijing"},
                )
            )
        ],
    ),
    finish_reason="STOP",
)
response = SimpleNamespace(candidates=[candidate])
assert is_response_valid(response)
converted = convert_to_openai_format(response, request.model)
choice = converted["choices"][0]
assert choice["finish_reason"] == "tool_calls"
assert choice["message"]["content"] is None
assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
    "city": "Beijing"
}
assert choice["message"]["tool_calls"][0]["thought_signature"]

signed_history = OpenAIRequest.model_validate(
    {
        "model": request.model,
        "messages": [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": choice["message"]["tool_calls"],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": '{"temperature":26}',
            },
        ],
    }
)
signed_contents = create_gemini_prompt(signed_history.messages)
assert signed_contents[1].parts[0].thought_signature == b"test-signature"


async def collect_fake_stream():
    chunks = []
    async for chunk in _base_fake_stream_engine(
        api_call_task_creator=lambda: asyncio.create_task(
            asyncio.sleep(0, result=response)
        ),
        extract_text_from_response_func=lambda _: "",
        response_id="chatcmpl-test",
        sse_model_name=request.model,
        is_auto_attempt=False,
        is_valid_response_func=is_response_valid,
        keep_alive_interval_seconds=0,
        reasoning_text_to_yield="",
        actual_content_text_to_yield="",
        tool_calls_to_yield=choice["message"]["tool_calls"],
    ):
        chunks.append(chunk)
    return chunks


stream_chunks = asyncio.run(collect_fake_stream())
assert any('"tool_calls"' in chunk for chunk in stream_chunks)
assert any('"finish_reason": "tool_calls"' in chunk for chunk in stream_chunks)
assert stream_chunks[-1] == "data: [DONE]\n\n"

print("tool-call smoke tests passed")
