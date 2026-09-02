import re
from typing import Dict


LEGACY_PRO_THINKING_MODEL = "gemini-2.5-pro-preview-06-05"
_GEMINI_FLASH_MODEL_PATTERN = re.compile(
    r"^gemini-\d+(?:\.\d+)*-flash(?:$|-)", re.IGNORECASE
)


def is_gemini_flash_model(model_name: str) -> bool:
    """Return whether a base model ID belongs to a Gemini Flash family."""
    return bool(_GEMINI_FLASH_MODEL_PATTERN.match(model_name))


def supports_nothinking_variant(model_name: str) -> bool:
    """Return whether Hajimi can expose its legacy ``-nothinking`` alias."""
    return is_gemini_flash_model(model_name) or model_name == LEGACY_PRO_THINKING_MODEL


def supports_max_thinking_variant(model_name: str) -> bool:
    """Limit numeric max-budget aliases to Gemini 2.5-era models."""
    return model_name.startswith("gemini-2.5-flash") or (
        model_name == LEGACY_PRO_THINKING_MODEL
    )


def thinking_config_for_variant(model_name: str, variant: str) -> Dict[str, int]:
    """Build the google-genai thinking config for a validated model alias."""
    if variant == "nothinking" and supports_nothinking_variant(model_name):
        # The legacy 2.5 Pro preview cannot disable thinking completely.
        budget = 128 if model_name == LEGACY_PRO_THINKING_MODEL else 0
        return {"thinking_budget": budget}

    if variant == "max" and supports_max_thinking_variant(model_name):
        budget = 32768 if model_name == LEGACY_PRO_THINKING_MODEL else 24576
        return {"thinking_budget": budget}

    raise ValueError(f"Unsupported {variant!r} thinking variant for {model_name!r}")
