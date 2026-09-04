"""Shared validation and conversion helpers for Vertex image models."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from google.genai import types

from app.vertex.model_loader import OFFICIAL_VERTEX_IMAGE_MODELS


PAY_PREFIX = "[PAY]"
EXPRESS_PREFIX = "[EXPRESS]"
DEFAULT_IMAGE_MODEL = f"{PAY_PREFIX}gemini-3.1-flash-image"

MAX_INPUT_IMAGES = 14
MAX_INPUT_IMAGE_BYTES = 7 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 20 * 1024 * 1024

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "9:21",
}

OPENAI_SIZE_MAP: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "auto": (None, None),
    "1024x1024": ("1:1", "1K"),
    "1536x1024": ("3:2", "1K"),
    "1024x1536": ("2:3", "1K"),
}

MODEL_IMAGE_SIZES = {
    "gemini-3.1-flash-image": {"512", "1K", "2K", "4K"},
    "gemini-3.1-flash-lite-image": {"1K"},
    "gemini-3-pro-image": {"1K", "2K", "4K"},
    "gemini-2.5-flash-image": {"1K"},
}


class ImageProxyError(Exception):
    """An error safe to expose through an image API response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type


@dataclass(frozen=True)
class InputImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GeneratedImage:
    b64_json: str
    mime_type: str


def normalize_image_model(model_name: str) -> str:
    """Return a bare supported Vertex image model ID."""
    normalized = (model_name or "").strip()
    if normalized.startswith(EXPRESS_PREFIX):
        raise ImageProxyError(
            400,
            "Image generation is not exposed through Vertex Express credentials.",
        )
    if normalized.startswith(PAY_PREFIX):
        normalized = normalized[len(PAY_PREFIX) :].strip()
    if normalized not in OFFICIAL_VERTEX_IMAGE_MODELS:
        allowed = ", ".join(OFFICIAL_VERTEX_IMAGE_MODELS)
        raise ImageProxyError(
            400,
            f"Unsupported image model '{model_name}'. Supported models: {allowed}.",
        )
    return normalized


def is_image_model(model_name: str) -> bool:
    try:
        normalize_image_model(model_name)
        return True
    except ImageProxyError:
        return False


def resolve_image_options(
    model_name: str,
    size: str = "1024x1024",
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Validate model/options and resolve OpenAI sizes to Gemini options."""
    model = normalize_image_model(model_name)
    if size not in OPENAI_SIZE_MAP:
        allowed_sizes = ", ".join(OPENAI_SIZE_MAP)
        raise ImageProxyError(
            400,
            f"Unsupported size '{size}'. Supported values: {allowed_sizes}.",
        )

    mapped_aspect_ratio, mapped_image_size = OPENAI_SIZE_MAP[size]
    resolved_aspect_ratio = aspect_ratio or mapped_aspect_ratio
    resolved_image_size = image_size or mapped_image_size

    if (
        resolved_aspect_ratio is not None
        and resolved_aspect_ratio not in SUPPORTED_ASPECT_RATIOS
    ):
        raise ImageProxyError(
            400,
            f"Unsupported aspect_ratio '{resolved_aspect_ratio}'.",
        )

    if resolved_image_size is not None:
        resolved_image_size = resolved_image_size.upper()
        if resolved_image_size == "0.5K":
            resolved_image_size = "512"
        supported_sizes = MODEL_IMAGE_SIZES[model]
        if resolved_image_size not in supported_sizes:
            allowed = ", ".join(sorted(supported_sizes))
            raise ImageProxyError(
                400,
                f"Model '{model}' does not support image_size "
                f"'{resolved_image_size}'. Supported values: {allowed}.",
            )

    return model, size, resolved_aspect_ratio, resolved_image_size


def build_image_generation_config(
    model_name: str,
    size: str = "1024x1024",
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None,
) -> Tuple[str, types.GenerateContentConfig]:
    model, _, resolved_aspect_ratio, resolved_image_size = resolve_image_options(
        model_name,
        size=size,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )
    image_config: Dict[str, Any] = {"image_output_options": {"mime_type": "image/png"}}
    if resolved_aspect_ratio is not None:
        image_config["aspect_ratio"] = resolved_aspect_ratio
    if resolved_image_size is not None:
        image_config["image_size"] = resolved_image_size

    return model, types.GenerateContentConfig(
        candidate_count=1,
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig.model_validate(image_config),
    )


def detect_image_mime(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
            return "image/heic"
        if brand in {b"mif1", b"msf1"}:
            return "image/heif"
    return None


def validate_input_image(
    data: bytes,
    declared_mime_type: Optional[str] = None,
) -> InputImage:
    if not data:
        raise ImageProxyError(400, "Uploaded image is empty.")
    if len(data) > MAX_INPUT_IMAGE_BYTES:
        raise ImageProxyError(
            413,
            f"Each uploaded image must be at most {MAX_INPUT_IMAGE_BYTES} bytes.",
            "request_too_large",
        )

    detected_mime_type = detect_image_mime(data)
    if detected_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ImageProxyError(
            400,
            "Unsupported or invalid image. Use PNG, JPEG, WebP, HEIC, or HEIF.",
        )
    if (
        declared_mime_type
        and declared_mime_type != "application/octet-stream"
        and declared_mime_type not in SUPPORTED_IMAGE_MIME_TYPES
    ):
        raise ImageProxyError(
            400,
            f"Unsupported declared image MIME type '{declared_mime_type}'.",
        )
    if (
        declared_mime_type in SUPPORTED_IMAGE_MIME_TYPES
        and declared_mime_type != detected_mime_type
        and {declared_mime_type, detected_mime_type} != {"image/heic", "image/heif"}
    ):
        raise ImageProxyError(
            400,
            "The uploaded image content does not match its declared MIME type.",
        )
    return InputImage(data=data, mime_type=detected_mime_type)


def validate_input_images(images: Sequence[InputImage]) -> None:
    if not images:
        raise ImageProxyError(400, "At least one image is required for editing.")
    if len(images) > MAX_INPUT_IMAGES:
        raise ImageProxyError(
            413,
            f"At most {MAX_INPUT_IMAGES} input images are allowed.",
            "request_too_large",
        )
    total_bytes = sum(len(image.data) for image in images)
    if total_bytes > MAX_TOTAL_INPUT_BYTES:
        raise ImageProxyError(
            413,
            f"Combined input images must be at most {MAX_TOTAL_INPUT_BYTES} bytes.",
            "request_too_large",
        )


def create_image_contents(
    prompt: str,
    input_images: Sequence[InputImage] = (),
) -> List[types.Content]:
    stripped_prompt = (prompt or "").strip()
    if not stripped_prompt:
        raise ImageProxyError(400, "prompt must not be empty.")
    if input_images:
        validate_input_images(input_images)
    parts = [types.Part.from_text(text=stripped_prompt)]
    parts.extend(
        types.Part.from_bytes(data=image.data, mime_type=image.mime_type)
        for image in input_images
    )
    return [types.Content(role="user", parts=parts)]


def _iter_response_parts(response: Any) -> Iterable[Any]:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            yield part


def _base64_from_inline_data(data: Any) -> str:
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageProxyError(
                502,
                "Vertex returned malformed image data.",
                "upstream_error",
            ) from exc
        return base64.b64encode(decoded).decode("ascii")
    raise ImageProxyError(
        502,
        "Vertex returned image data in an unsupported format.",
        "upstream_error",
    )


def extract_generated_images(
    response: Any,
) -> Tuple[List[GeneratedImage], Optional[str]]:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason:
        block_value = getattr(block_reason, "value", block_reason)
        raise ImageProxyError(
            400,
            f"Image request was blocked by the Vertex safety filter: {block_value}.",
            "content_policy_violation",
        )
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_value = str(getattr(finish_reason, "value", finish_reason) or "").upper()
        if finish_value in {
            "SAFETY",
            "IMAGE_SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
        }:
            raise ImageProxyError(
                400,
                f"Image request was blocked by the Vertex safety filter: {finish_value}.",
                "content_policy_violation",
            )

    images: List[GeneratedImage] = []
    text_parts: List[str] = []
    for part in _iter_response_parts(response):
        text = getattr(part, "text", None)
        if (
            isinstance(text, str)
            and text.strip()
            and not getattr(part, "thought", False)
        ):
            text_parts.append(text.strip())
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue
        raw_data = getattr(inline_data, "data", None)
        if raw_data is None:
            continue
        images.append(
            GeneratedImage(
                b64_json=_base64_from_inline_data(raw_data),
                mime_type=getattr(inline_data, "mime_type", None) or "image/png",
            )
        )
    if not images:
        raise ImageProxyError(
            502,
            "Vertex returned no generated image.",
            "upstream_error",
        )
    revised_prompt = "\n".join(text_parts) or None
    return images, revised_prompt


def openai_image_response(response: Any, created: int) -> Dict[str, Any]:
    images, revised_prompt = extract_generated_images(response)
    return {
        "created": created,
        "data": [
            {
                "b64_json": image.b64_json,
                "revised_prompt": revised_prompt,
            }
            for image in images
        ],
    }


def serialize_native_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", by_alias=True, exclude_none=True)
    raise ImageProxyError(
        502,
        "Vertex returned an unsupported response object.",
        "upstream_error",
    )


def prepare_native_request(
    model_name: str,
    payload: Dict[str, Any],
) -> Tuple[str, List[types.Content], types.GenerateContentConfig, int, int]:
    model = normalize_image_model(model_name)
    raw_contents = payload.get("contents")
    if not isinstance(raw_contents, list) or not raw_contents:
        raise ImageProxyError(400, "contents must be a non-empty array.")
    try:
        contents = [types.Content.model_validate(item) for item in raw_contents]
    except Exception as exc:
        raise ImageProxyError(400, f"Invalid Gemini contents: {exc}") from exc

    input_images: List[InputImage] = []
    for content in contents:
        for part in content.parts or []:
            inline_data = part.inline_data
            if inline_data is None or inline_data.data is None:
                continue
            raw_data = inline_data.data
            if isinstance(raw_data, str):
                try:
                    raw_data = base64.b64decode(raw_data, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ImageProxyError(
                        400, "Invalid inlineData base64 image."
                    ) from exc
            input_images.append(validate_input_image(raw_data, inline_data.mime_type))
    if input_images:
        validate_input_images(input_images)

    raw_config = dict(payload.get("generationConfig") or {})
    raw_config["responseModalities"] = ["TEXT", "IMAGE"]
    if payload.get("safetySettings") is not None:
        raw_config["safetySettings"] = payload["safetySettings"]
    system_instruction = payload.get("systemInstruction") or payload.get(
        "system_instruction"
    )
    if system_instruction is not None:
        raw_config["systemInstruction"] = system_instruction
    if payload.get("tools") is not None:
        raw_config["tools"] = payload["tools"]
    try:
        config = types.GenerateContentConfig.model_validate(raw_config)
    except Exception as exc:
        raise ImageProxyError(
            400,
            f"Invalid Gemini generationConfig: {exc}",
        ) from exc

    return (
        model,
        contents,
        config,
        len(input_images),
        sum(len(image.data) for image in input_images),
    )


_BASE64_LIKE_PATTERN = re.compile(r"[A-Za-z0-9+/=_-]{256,}")


def safe_upstream_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    message = _BASE64_LIKE_PATTERN.sub("[binary data omitted]", message)
    return (message or exc.__class__.__name__)[:1024]


def map_upstream_exception(exc: Exception) -> ImageProxyError:
    if isinstance(exc, ImageProxyError):
        return exc
    raw_status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    try:
        status_code = int(raw_status)
    except (TypeError, ValueError):
        status_code = 500
    message = safe_upstream_message(exc)
    if status_code in {400, 401, 403, 404, 409, 429}:
        error_type = "rate_limit_error" if status_code == 429 else "upstream_error"
        return ImageProxyError(status_code, message, error_type)
    if status_code >= 500:
        return ImageProxyError(502, message, "upstream_error")
    return ImageProxyError(500, message, "server_error")


def openai_error_payload(error: ImageProxyError) -> Dict[str, Any]:
    return {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "param": None,
            "code": error.status_code,
        }
    }


def native_error_payload(error: ImageProxyError) -> Dict[str, Any]:
    status_name = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "ABORTED",
        413: "INVALID_ARGUMENT",
        429: "RESOURCE_EXHAUSTED",
        500: "INTERNAL",
        502: "UNAVAILABLE",
        503: "UNAVAILABLE",
    }.get(error.status_code, "UNKNOWN")
    return {
        "error": {
            "code": error.status_code,
            "message": error.message,
            "status": status_name,
        }
    }
