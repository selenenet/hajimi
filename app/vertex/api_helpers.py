import json
import time
import math
import asyncio
import re
from typing import List, Dict, Any, Callable, Union, Optional
from fastapi.responses import JSONResponse, StreamingResponse

from google.genai import types
from app.vertex.message_processing import (
    parse_gemini_response_for_reasoning_and_content,
    extract_gemini_tool_calls,
)

# Local module imports
from app.vertex.models import OpenAIRequest, OpenAIMessage  # Changed from relative
from app.vertex.message_processing import (
    deobfuscate_text,
    convert_to_openai_format,
    convert_chunk_to_openai,
    create_final_chunk,
)  # Changed from relative
import app.vertex.config as app_config  # Changed from relative
from app.config import settings  # 导入settings模块


def create_openai_error_response(
    status_code: int, message: str, error_type: str
) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
            "param": None,
        }
    }


_FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


def _resolve_local_schema_ref(ref: str, root: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    current: Any = root
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or segment not in current:
            return {}
        current = current[segment]
    return current if isinstance(current, dict) else {}


def _sanitize_json_schema(
    schema: Any,
    root: Optional[Dict[str, Any]] = None,
    seen_refs: Optional[set] = None,
) -> Dict[str, Any]:
    """Convert common OpenAI JSON Schema into google-genai 1.11 Schema."""
    if not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}
    if root is None:
        root = schema
    if seen_refs is None:
        seen_refs = set()

    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref in seen_refs:
            return {"type": "OBJECT", "properties": {}}
        resolved = _resolve_local_schema_ref(ref, root)
        if resolved:
            return _sanitize_json_schema(resolved, root, seen_refs | {ref})

    working = dict(schema)
    all_of = working.pop("allOf", None)
    if isinstance(all_of, list):
        merged_properties = dict(working.get("properties") or {})
        merged_required = list(working.get("required") or [])
        for item in all_of:
            normalized_item = _sanitize_json_schema(item, root, seen_refs)
            merged_properties.update(normalized_item.get("properties") or {})
            for required_name in normalized_item.get("required") or []:
                if required_name not in merged_required:
                    merged_required.append(required_name)
            for key, value in normalized_item.items():
                working.setdefault(key, value)
        if merged_properties:
            working["properties"] = merged_properties
        if merged_required:
            working["required"] = merged_required

    result: Dict[str, Any] = {}
    schema_type = working.get("type")
    nullable = bool(working.get("nullable", False))
    if isinstance(schema_type, list):
        nullable = nullable or "null" in schema_type
        schema_type = next((item for item in schema_type if item != "null"), None)
    if schema_type:
        result["type"] = str(schema_type).upper()
    elif "properties" in working:
        result["type"] = "OBJECT"
    elif "items" in working:
        result["type"] = "ARRAY"

    for key in (
        "description",
        "title",
        "format",
        "pattern",
        "default",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ):
        if key in working and working[key] is not None:
            result[key] = working[key]

    if nullable:
        result["nullable"] = True
    if "const" in working and "enum" not in working:
        working["enum"] = [working["const"]]
    if isinstance(working.get("enum"), list):
        result["enum"] = [str(value) for value in working["enum"]]
    if isinstance(working.get("properties"), dict):
        result["properties"] = {
            str(name): _sanitize_json_schema(value, root, seen_refs)
            for name, value in working["properties"].items()
            if isinstance(value, dict)
        }
    if isinstance(working.get("required"), list):
        result["required"] = [str(name) for name in working["required"]]
    if isinstance(working.get("items"), dict):
        result["items"] = _sanitize_json_schema(working["items"], root, seen_refs)

    any_of = working.get("anyOf") or working.get("oneOf")
    if isinstance(any_of, list):
        normalized_any_of = []
        for item in any_of:
            if isinstance(item, dict) and item.get("type") == "null":
                result["nullable"] = True
                continue
            normalized_any_of.append(_sanitize_json_schema(item, root, seen_refs))
        if normalized_any_of:
            result["anyOf"] = normalized_any_of

    if "type" not in result and "anyOf" not in result:
        result["type"] = "OBJECT"
        result.setdefault("properties", {})
    return result


def _openai_tools_to_gemini(request: OpenAIRequest):
    raw_tools = list(request.tools or [])
    if not raw_tools and request.functions:
        raw_tools = [
            {"type": "function", "function": function}
            for function in request.functions
        ]
    if not raw_tools:
        return None, None

    declarations = []
    declared_names = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict) or raw_tool.get("type", "function") != "function":
            continue
        function = raw_tool.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        if not _FUNCTION_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"Invalid function name '{name}'. Gemini function names must start with a letter or underscore, "
                "contain only letters, digits, underscores, dots or dashes, and be at most 64 characters."
            )
        parameters = _sanitize_json_schema(
            function.get("parameters") or {"type": "object", "properties": {}}
        )
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=str(function.get("description") or ""),
                parameters=types.Schema.model_validate(parameters),
            )
        )
        declared_names.append(name)

    if not declarations:
        return None, None

    tool_choice = request.tool_choice
    if tool_choice is None and request.function_call is not None:
        tool_choice = request.function_call
    mode = "AUTO"
    allowed_function_names = None
    if isinstance(tool_choice, str):
        mode = {"none": "NONE", "required": "ANY", "auto": "AUTO"}.get(
            tool_choice.lower(), "AUTO"
        )
    elif isinstance(tool_choice, dict):
        function_choice = tool_choice.get("function") or tool_choice
        chosen_name = (
            function_choice.get("name") if isinstance(function_choice, dict) else None
        )
        if chosen_name:
            if chosen_name not in declared_names:
                raise ValueError(f"tool_choice references undeclared function '{chosen_name}'.")
            mode = "ANY"
            allowed_function_names = [chosen_name]

    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=mode, allowed_function_names=allowed_function_names
        )
    )
    return types.Tool(function_declarations=declarations), tool_config


def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config = {}
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None:
        config["top_p"] = request.top_p
    if request.top_k is not None:
        config["top_k"] = request.top_k
    if request.stop is not None:
        config["stop_sequences"] = request.stop
    if request.seed is not None:
        config["seed"] = request.seed
    if request.presence_penalty is not None:
        config["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        config["frequency_penalty"] = request.frequency_penalty
    if request.n is not None:
        config["candidate_count"] = request.n
    gemini_tool, tool_config = _openai_tools_to_gemini(request)
    if gemini_tool is not None:
        config["tools"] = [gemini_tool]
        config["tool_config"] = tool_config
        config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
    config["safety_settings"] = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
        ),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
    ]
    return config


def is_response_valid(response):
    if response is None:
        print("DEBUG: Response is None, therefore invalid.")
        return False

    # Check for direct text attribute
    if (
        hasattr(response, "text")
        and isinstance(response.text, str)
        and response.text.strip()
    ):
        # print("DEBUG: Response valid due to response.text")
        return True

    # Check candidates for text content
    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:  # Iterate through all candidates
            if (
                hasattr(candidate, "text")
                and isinstance(candidate.text, str)
                and candidate.text.strip()
            ):
                # print(f"DEBUG: Response valid due to candidate.text in candidate")
                return True
            if (
                hasattr(candidate, "content")
                and hasattr(candidate.content, "parts")
                and candidate.content.parts
            ):
                for part in candidate.content.parts:
                    if getattr(part, "function_call", None) and getattr(
                        part.function_call, "name", None
                    ):
                        return True
                    if (
                        hasattr(part, "text")
                        and isinstance(part.text, str)
                        and part.text.strip()
                    ):
                        # print(f"DEBUG: Response valid due to part.text in candidate's content part")
                        return True

    # Removed prompt_feedback as a sole criterion for validity.
    # It should only be valid if actual text content is found.
    # Block reasons will be checked explicitly by callers if they need to treat it as an error for retries.
    print(
        "DEBUG: Response is invalid, no usable text content found by is_response_valid."
    )
    return False


async def _base_fake_stream_engine(
    api_call_task_creator: Callable[[], asyncio.Task],
    extract_text_from_response_func: Callable[[Any], str],
    response_id: str,
    sse_model_name: str,
    is_auto_attempt: bool,
    is_valid_response_func: Callable[[Any], bool],
    keep_alive_interval_seconds: float,
    process_text_func: Optional[Callable[[str, str], str]] = None,
    check_block_reason_func: Optional[Callable[[Any], None]] = None,
    reasoning_text_to_yield: Optional[str] = None,
    actual_content_text_to_yield: Optional[str] = None,
    tool_calls_to_yield: Optional[List[Dict[str, Any]]] = None,
):
    api_call_task = api_call_task_creator()

    if keep_alive_interval_seconds > 0:
        while not api_call_task.done():
            keep_alive_data = {
                "id": "chatcmpl-keepalive",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": sse_model_name,
                "choices": [
                    {
                        "delta": {"reasoning_content": ""},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(keep_alive_interval_seconds)

    try:
        full_api_response = await api_call_task

        if check_block_reason_func:
            check_block_reason_func(full_api_response)

        if not is_valid_response_func(full_api_response):
            raise ValueError(
                f"Invalid/empty API response in fake stream for model {sse_model_name}: {str(full_api_response)[:200]}"
            )

        final_reasoning_text = reasoning_text_to_yield
        final_actual_content_text = actual_content_text_to_yield

        if final_reasoning_text is None and final_actual_content_text is None:
            extracted_full_text = extract_text_from_response_func(full_api_response)
            if process_text_func:
                final_actual_content_text = process_text_func(
                    extracted_full_text, sse_model_name
                )
            else:
                final_actual_content_text = extracted_full_text
        else:
            if process_text_func:
                if final_reasoning_text is not None:
                    final_reasoning_text = process_text_func(
                        final_reasoning_text, sse_model_name
                    )
                if final_actual_content_text is not None:
                    final_actual_content_text = process_text_func(
                        final_actual_content_text, sse_model_name
                    )

        if final_reasoning_text:
            reasoning_delta_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": sse_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": final_reasoning_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(reasoning_delta_data)}\n\n"
            if final_actual_content_text:
                await asyncio.sleep(0.05)

        if tool_calls_to_yield:
            tool_call_delta = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": sse_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls_to_yield,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(tool_call_delta, ensure_ascii=False)}\n\n"

        content_to_chunk = final_actual_content_text or ""
        chunk_size = (
            max(20, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 0
        )

        if not content_to_chunk and not tool_calls_to_yield:
            empty_delta_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": sse_model_name,
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(empty_delta_data)}\n\n"
        elif content_to_chunk and not tool_calls_to_yield:
            for i in range(0, len(content_to_chunk), chunk_size):
                chunk_text = content_to_chunk[i : i + chunk_size]
                content_delta_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": sse_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(content_delta_data)}\n\n"
                if len(content_to_chunk) > chunk_size:
                    await asyncio.sleep(0.05)

        yield create_final_chunk(
            sse_model_name,
            response_id,
            finish_reason="tool_calls" if tool_calls_to_yield else "stop",
        )
        yield "data: [DONE]\n\n"

    except Exception as e:
        err_msg_detail = f"Error in _base_fake_stream_engine (model: '{sse_model_name}'): {type(e).__name__} - {str(e)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e)
        if len(sse_err_msg_display) > 512:
            sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_for_sse = create_openai_error_response(
            500, sse_err_msg_display, "server_error"
        )
        json_payload_for_fake_stream_error = json.dumps(err_resp_for_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_for_fake_stream_error}\n\n"
            yield "data: [DONE]\n\n"
        raise


async def gemini_fake_stream_generator(
    gemini_client_instance: Any,
    model_for_api_call: str,
    prompt_for_api_call: Union[types.Content, List[types.Content]],
    gen_config_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool,
):
    model_name_for_log = getattr(
        gemini_client_instance, "model_name", "unknown_gemini_model_object"
    )
    print(
        f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}', client obj: '{model_name_for_log}') with reasoning separation."
    )
    response_id = f"chatcmpl-{int(time.time())}"

    # 1. Create and await the API call task
    api_call_task = asyncio.create_task(
        gemini_client_instance.aio.models.generate_content(
            model=model_for_api_call,
            contents=prompt_for_api_call,
            config=gen_config_for_api_call,
        )
    )

    # Keep-alive loop while the main API call is in progress
    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {
                "id": "chatcmpl-keepalive",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request_obj.model,
                "choices": [
                    {
                        "delta": {"reasoning_content": ""},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)

    try:
        raw_response = await api_call_task  # Get the full Gemini response

        # 2. Parse the response for reasoning and content using the centralized parser
        separated_reasoning_text = ""
        separated_actual_content_text = ""
        tool_calls = []
        if hasattr(raw_response, "candidates") and raw_response.candidates:
            # Typically, fake streaming would focus on the first candidate
            separated_reasoning_text, separated_actual_content_text = (
                parse_gemini_response_for_reasoning_and_content(
                    raw_response.candidates[0]
                )
            )
            tool_calls = extract_gemini_tool_calls(raw_response.candidates[0])
        elif (
            hasattr(raw_response, "text") and raw_response.text is not None
        ):  # Fallback for simpler response structures
            separated_actual_content_text = raw_response.text

        # 3. Define a text processing function (e.g., for deobfuscation)
        def _process_gemini_text_if_needed(text: str, model_name: str) -> str:
            if model_name.endswith("-encrypt-full"):
                return deobfuscate_text(text)
            return text

        final_reasoning_text = _process_gemini_text_if_needed(
            separated_reasoning_text, request_obj.model
        )
        final_actual_content_text = _process_gemini_text_if_needed(
            separated_actual_content_text, request_obj.model
        )

        # Define block checking for the raw response
        def _check_gemini_block_wrapper(response_to_check: Any):
            if (
                hasattr(response_to_check, "prompt_feedback")
                and hasattr(response_to_check.prompt_feedback, "block_reason")
                and response_to_check.prompt_feedback.block_reason
            ):
                block_message = f"Response blocked by Gemini safety filter: {response_to_check.prompt_feedback.block_reason}"
                if (
                    hasattr(response_to_check.prompt_feedback, "block_reason_message")
                    and response_to_check.prompt_feedback.block_reason_message
                ):
                    block_message += f" (Message: {response_to_check.prompt_feedback.block_reason_message})"
                raise ValueError(block_message)

        # Call _base_fake_stream_engine with pre-split and processed texts
        async for chunk in _base_fake_stream_engine(
            api_call_task_creator=lambda: asyncio.create_task(
                asyncio.sleep(0, result=raw_response)
            ),  # Dummy task
            extract_text_from_response_func=lambda r: "",  # Not directly used as text is pre-split
            is_valid_response_func=is_response_valid,  # Validates raw_response
            check_block_reason_func=_check_gemini_block_wrapper,  # Checks raw_response
            process_text_func=None,  # Text processing already done above
            response_id=response_id,
            sse_model_name=request_obj.model,
            keep_alive_interval_seconds=0,  # Keep-alive for this inner call is 0
            is_auto_attempt=is_auto_attempt,
            reasoning_text_to_yield=final_reasoning_text,
            actual_content_text_to_yield=final_actual_content_text,
            tool_calls_to_yield=tool_calls,
        ):
            yield chunk

    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        print(f"ERROR: {err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512:
            sse_err_msg_display = sse_err_msg_display[:512] + "..."
        err_resp_sse = create_openai_error_response(
            500, sse_err_msg_display, "server_error"
        )
        json_payload_error = json.dumps(err_resp_sse)
        if not is_auto_attempt:
            yield f"data: {json_payload_error}\n\n"
            yield "data: [DONE]\n\n"


async def execute_gemini_call(
    current_client: Any,
    model_to_call: str,
    prompt_func: Callable[
        [List[OpenAIMessage]], Union[types.Content, List[types.Content]]
    ],
    gen_config_for_call: Dict[str, Any],
    request_obj: OpenAIRequest,
    is_auto_attempt: bool = False,
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    client_model_name_for_log = getattr(
        current_client, "model_name", "unknown_direct_client_object"
    )
    print(
        f"INFO: execute_gemini_call for requested API model '{model_to_call}', using client object with internal name '{client_model_name_for_log}'. Original request model: '{request_obj.model}'"
    )

    # 每次调用时直接从settings获取最新的FAKE_STREAMING值
    fake_streaming_enabled = False
    if hasattr(settings, "FAKE_STREAMING"):
        fake_streaming_enabled = settings.FAKE_STREAMING
    else:
        fake_streaming_enabled = app_config.FAKE_STREAMING_ENABLED

    print(
        f"DEBUG: FAKE_STREAMING setting is {fake_streaming_enabled} for model {request_obj.model}"
    )

    if request_obj.stream:
        if fake_streaming_enabled:
            return StreamingResponse(
                gemini_fake_stream_generator(
                    current_client,
                    model_to_call,
                    actual_prompt_for_call,
                    gen_config_for_call,
                    request_obj,
                    is_auto_attempt,
                ),
                media_type="text/event-stream",
            )

        response_id_for_stream = f"chatcmpl-{int(time.time())}"
        cand_count_stream = request_obj.n or 1

        async def _gemini_real_stream_generator_inner():
            try:
                saw_tool_call = False
                async for (
                    chunk_item_call
                ) in await current_client.aio.models.generate_content_stream(
                    model=model_to_call,
                    contents=actual_prompt_for_call,
                    config=gen_config_for_call,
                ):
                    if (
                        getattr(chunk_item_call, "candidates", None)
                        and extract_gemini_tool_calls(chunk_item_call.candidates[0])
                    ):
                        saw_tool_call = True
                    yield convert_chunk_to_openai(
                        chunk_item_call, request_obj.model, response_id_for_stream, 0
                    )
                yield create_final_chunk(
                    request_obj.model,
                    response_id_for_stream,
                    cand_count_stream,
                    finish_reason="tool_calls" if saw_tool_call else "stop",
                )
                yield "data: [DONE]\n\n"
            except Exception as e_stream_call:
                err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                print(f"ERROR: {err_msg_detail_stream}")
                s_err = str(e_stream_call)
                s_err = s_err[:1024] + "..." if len(s_err) > 1024 else s_err
                err_resp = create_openai_error_response(500, s_err, "server_error")
                j_err = json.dumps(err_resp)
                if not is_auto_attempt:
                    yield f"data: {j_err}\n\n"
                    yield "data: [DONE]\n\n"
                raise e_stream_call

        return StreamingResponse(
            _gemini_real_stream_generator_inner(), media_type="text/event-stream"
        )
    else:
        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call,
            contents=actual_prompt_for_call,
            config=gen_config_for_call,
        )
        if (
            hasattr(response_obj_call, "prompt_feedback")
            and hasattr(response_obj_call.prompt_feedback, "block_reason")
            and response_obj_call.prompt_feedback.block_reason
        ):
            block_msg = (
                f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            )
            if (
                hasattr(response_obj_call.prompt_feedback, "block_reason_message")
                and response_obj_call.prompt_feedback.block_reason_message
            ):
                block_msg += (
                    f" ({response_obj_call.prompt_feedback.block_reason_message})"
                )
            raise ValueError(block_msg)

        if not is_response_valid(response_obj_call):
            raise ValueError(
                f"Invalid non-streaming Gemini response for model string '{model_to_call}'. Resp: {str(response_obj_call)[:200]}"
            )
        return JSONResponse(
            content=convert_to_openai_format(response_obj_call, request_obj.model)
        )
