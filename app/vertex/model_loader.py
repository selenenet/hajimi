import httpx
import asyncio
import json
from typing import List, Dict, Optional
from app.utils.logging import vertex_log

# 导入settings和app_config
from app.config import settings
import app.vertex.config as app_config

_model_cache: Optional[Dict[str, List[str]]] = None
_cache_lock = asyncio.Lock()

# Google Cloud model lifecycle documentation snapshot (2026-09-02 UTC):
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
#
# Image models are listed alongside text models, but are handled only by the
# dedicated image and Gemini-native endpoints. Live, video, and embedding models
# use different contracts and remain intentionally omitted.
OFFICIAL_VERTEX_TEXT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

OFFICIAL_VERTEX_IMAGE_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]


def get_builtin_models_config() -> Dict[str, List[str]]:
    """Return an isolated copy of the Google-official built-in model list."""
    return {
        "vertex_models": list(
            dict.fromkeys(OFFICIAL_VERTEX_TEXT_MODELS + OFFICIAL_VERTEX_IMAGE_MODELS)
        ),
        "vertex_express_models": [
            f"[EXPRESS] {model_name}" for model_name in OFFICIAL_VERTEX_TEXT_MODELS
        ],
    }


async def fetch_and_parse_models_config() -> Optional[Dict[str, List[str]]]:
    """
    Fetches the model configuration JSON from the URL specified in app_config.
    Parses it and returns a dictionary with 'vertex_models' and 'vertex_express_models'.
    Returns None if fetching or parsing fails.
    """
    # 优先从settings中获取MODELS_CONFIG_URL
    models_config_url = None
    if hasattr(settings, "MODELS_CONFIG_URL") and settings.MODELS_CONFIG_URL:
        models_config_url = settings.MODELS_CONFIG_URL
        vertex_log("info", "使用settings中的MODELS_CONFIG_URL")
    else:
        models_config_url = app_config.MODELS_CONFIG_URL
        vertex_log("info", "使用app_config中的MODELS_CONFIG_URL")

    if not models_config_url:
        vertex_log(
            "info",
            "Using the built-in Google-official Vertex model list "
            "(documentation snapshot: 2026-09-02 UTC).",
        )
        return get_builtin_models_config()

    vertex_log("info", f"Fetching model configuration from: {models_config_url}")

    # 添加重试机制
    max_retries = 3
    retry_delay = 1  # 初始延迟1秒

    for retry in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # 增加超时时间
                vertex_log("info", f"尝试获取模型配置，第{retry + 1}次尝试")
                response = await client.get(models_config_url)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # 记录原始响应内容，便于调试
                response_text = response.text
                vertex_log(
                    "debug", f"接收到原始响应: {response_text[:200]}..."
                )  # 只记录前200个字符

                data = response.json()

                # 更详细的验证和日志
                if not isinstance(data, dict):
                    vertex_log("error", f"模型配置不是有效的JSON对象: {type(data)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue

                if "vertex_models" not in data:
                    vertex_log("error", "模型配置缺少'vertex_models'字段")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                if "vertex_express_models" not in data:
                    vertex_log("error", "模型配置缺少'vertex_express_models'字段")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                if not isinstance(data["vertex_models"], list):
                    vertex_log(
                        "error",
                        f"'vertex_models'不是列表: {type(data['vertex_models'])}",
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                if not isinstance(data["vertex_express_models"], list):
                    vertex_log(
                        "error",
                        f"'vertex_express_models'不是列表: {type(data['vertex_express_models'])}",
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                vertex_log(
                    "info",
                    f"成功获取和解析模型配置。找到 {len(data['vertex_models'])} 个标准模型和 {len(data['vertex_express_models'])} 个Express模型。",
                )

                # Add [EXPRESS] prefix to express models
                prefixed_express_models = [
                    f"[EXPRESS] {model_name}"
                    for model_name in data["vertex_express_models"]
                ]

                return {
                    "vertex_models": data["vertex_models"],
                    "vertex_express_models": prefixed_express_models,
                }

        except httpx.RequestError as e:
            vertex_log("error", f"HTTP请求失败({retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                vertex_log("info", f"将在{retry_delay}秒后重试...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                vertex_log(
                    "error", f"HTTP请求在{max_retries}次尝试后仍然失败，放弃尝试"
                )
                return None

        except json.JSONDecodeError as e:
            vertex_log("error", f"JSON解析失败({retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                vertex_log("info", f"将在{retry_delay}秒后重试...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                vertex_log(
                    "error", f"JSON解析在{max_retries}次尝试后仍然失败，放弃尝试"
                )
                return None

        except Exception as e:
            vertex_log(
                "error",
                f"获取/解析模型配置时发生意外错误({retry + 1}/{max_retries}): {str(e)}",
            )
            if retry < max_retries - 1:
                vertex_log("info", f"将在{retry_delay}秒后重试...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                vertex_log(
                    "error", f"获取/解析在{max_retries}次尝试后仍然失败，放弃尝试"
                )
                return None

    # 如果所有重试都失败，返回内置配置
    vertex_log("warning", "获取模型配置失败，使用内置Google官方模型列表")
    return get_builtin_models_config()


async def get_models_config() -> Dict[str, List[str]]:
    """
    Returns the cached model configuration.
    If not cached, fetches and caches it.
    Returns a default empty structure if fetching fails.
    """
    global _model_cache
    async with _cache_lock:
        if _model_cache is None:
            vertex_log("info", "Model cache is empty. Fetching configuration...")
            _model_cache = await fetch_and_parse_models_config()
            if (
                _model_cache is None
            ):  # If fetching failed, use a default empty structure
                vertex_log(
                    "warning",
                    "Using built-in Google-official model configuration after "
                    "remote fetch/parse failure.",
                )
                _model_cache = get_builtin_models_config()
    return _model_cache


async def get_vertex_models() -> List[str]:
    config = await get_models_config()
    configured_models = config.get("vertex_models", [])
    # Image models require SA-backed Vertex access. Keep them present even when a
    # remote text-model configuration is in use, but never add Express aliases.
    return list(dict.fromkeys(configured_models + OFFICIAL_VERTEX_IMAGE_MODELS))


async def get_vertex_express_models() -> List[str]:
    config = await get_models_config()
    return config.get("vertex_express_models", [])


async def refresh_models_config_cache() -> bool:
    """
    Forces a refresh of the model configuration cache.
    Returns True if successful, False otherwise.
    """
    global _model_cache
    vertex_log("info", "Attempting to refresh model configuration cache...")
    async with _cache_lock:
        new_config = await fetch_and_parse_models_config()
        if new_config is not None:
            _model_cache = new_config
            vertex_log("info", "Model configuration cache refreshed successfully.")
            return True
        else:
            vertex_log("error", "Failed to refresh model configuration cache.")
            # Optionally, decide if we want to clear the old cache or keep it
            # _model_cache = {"vertex_models": [], "vertex_express_models": []} # To clear
            return False
