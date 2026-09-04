"""Vertex-backed image generation and editing handlers."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Sequence

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.utils.logging import vertex_log
from app.vertex.image_processing import (
    InputImage,
    ImageProxyError,
    MAX_INPUT_IMAGE_BYTES,
    MAX_INPUT_IMAGES,
    MAX_TOTAL_INPUT_BYTES,
    build_image_generation_config,
    augment_openai_image_prompt,
    create_image_contents,
    map_upstream_exception,
    native_error_payload,
    openai_error_payload,
    openai_image_response,
    prepare_native_request,
    serialize_native_response,
    validate_input_image,
    validate_input_images,
)
from app.vertex.vertex_ai_init import get_vertex_ai_client


async def _get_client(request: Request) -> Any:
    credential_manager = getattr(request.app.state, "credential_manager", None)
    if credential_manager is None:
        raise ImageProxyError(
            503,
            "Vertex credential manager is not initialized.",
            "server_error",
        )
    client = await get_vertex_ai_client(credential_manager)
    if client is None:
        raise ImageProxyError(
            401,
            "No usable Vertex service-account credential is available.",
            "authentication_error",
        )
    return client


async def read_uploaded_images(uploads: Sequence[UploadFile]) -> List[InputImage]:
    if not uploads:
        raise ImageProxyError(400, "At least one image is required for editing.")
    if len(uploads) > MAX_INPUT_IMAGES:
        raise ImageProxyError(
            413,
            f"At most {MAX_INPUT_IMAGES} input images are allowed.",
            "request_too_large",
        )

    images: List[InputImage] = []
    total_bytes = 0
    for upload in uploads:
        try:
            data = await upload.read(MAX_INPUT_IMAGE_BYTES + 1)
        finally:
            await upload.close()
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_INPUT_BYTES:
            raise ImageProxyError(
                413,
                f"Combined input images must be at most {MAX_TOTAL_INPUT_BYTES} bytes.",
                "request_too_large",
            )
        images.append(validate_input_image(data, upload.content_type))

    validate_input_images(images)
    return images


async def generate_openai_image(
    request: Request,
    *,
    model_name: str,
    prompt: str,
    size: str,
    aspect_ratio: str | None,
    image_size: str | None,
    quality: str | None = "auto",
    output_format: str | None = "png",
    output_compression: int | None = None,
    background: str | None = "auto",
    moderation: str | None = "auto",
    style: str | None = None,
    input_fidelity: str | None = None,
    n: int = 1,
    response_format: str = "b64_json",
    stream: bool | None = False,
    partial_images: int | None = 0,
    mask_provided: bool = False,
    input_images: Sequence[InputImage] = (),
) -> Dict[str, Any]:
    options, config = build_image_generation_config(
        model_name,
        size=size,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        quality=quality,
        output_format=output_format,
        output_compression=output_compression,
        background=background,
        moderation=moderation,
        style=style,
        input_fidelity=input_fidelity,
        n=n,
        response_format=response_format,
        stream=stream,
        partial_images=partial_images,
        mask_provided=mask_provided,
    )
    augmented_prompt = augment_openai_image_prompt(
        prompt,
        options,
        is_edit=bool(input_images),
    )
    contents = create_image_contents(augmented_prompt, input_images)
    client = await _get_client(request)
    input_bytes = sum(len(image.data) for image in input_images)
    started = time.monotonic()
    vertex_log(
        "info",
        f"Vertex image request started: model={options.model}, "
        f"input_images={len(input_images)}, input_bytes={input_bytes}",
    )
    try:
        response = await client.aio.models.generate_content(
            model=options.model,
            contents=contents,
            config=config,
        )
        result = openai_image_response(
            response,
            int(time.time()),
            options=options,
        )
    except Exception as exc:
        error = map_upstream_exception(exc)
        vertex_log(
            "error",
            f"Vertex image request failed: model={options.model}, "
            f"status={error.status_code}, error_type={error.error_type}",
        )
        raise error from exc

    vertex_log(
        "info",
        f"Vertex image request completed: model={options.model}, "
        f"output_images={len(result['data'])}, "
        f"latency_ms={int((time.monotonic() - started) * 1000)}",
    )
    return result


async def generate_native_image(
    request: Request,
    *,
    model_name: str,
    payload: Dict[str, Any],
    stream: bool,
):
    model, contents, config, input_count, input_bytes = prepare_native_request(
        model_name,
        payload,
    )
    client = await _get_client(request)
    vertex_log(
        "info",
        f"Vertex native image request started: model={model}, "
        f"stream={stream}, input_images={input_count}, input_bytes={input_bytes}",
    )

    if stream:

        async def event_stream():
            started = time.monotonic()
            try:
                response_stream = await client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
                async for chunk in response_stream:
                    payload_chunk = serialize_native_response(chunk)
                    yield f"data: {json.dumps(payload_chunk, ensure_ascii=False)}\n\n"
                vertex_log(
                    "info",
                    f"Vertex native image stream completed: model={model}, "
                    f"latency_ms={int((time.monotonic() - started) * 1000)}",
                )
            except Exception as exc:
                error = map_upstream_exception(exc)
                vertex_log(
                    "error",
                    f"Vertex native image stream failed: model={model}, "
                    f"status={error.status_code}, error_type={error.error_type}",
                )
                yield (
                    "data: "
                    + json.dumps(native_error_payload(error), ensure_ascii=False)
                    + "\n\n"
                )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    started = time.monotonic()
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        result = serialize_native_response(response)
    except Exception as exc:
        error = map_upstream_exception(exc)
        vertex_log(
            "error",
            f"Vertex native image request failed: model={model}, "
            f"status={error.status_code}, error_type={error.error_type}",
        )
        raise error from exc

    vertex_log(
        "info",
        f"Vertex native image request completed: model={model}, "
        f"latency_ms={int((time.monotonic() - started) * 1000)}",
    )
    return JSONResponse(content=result)


def openai_error_response(error: ImageProxyError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=openai_error_payload(error),
    )


def native_error_response(error: ImageProxyError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=native_error_payload(error),
    )
