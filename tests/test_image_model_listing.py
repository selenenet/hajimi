import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.vertex.routes import models_api


class ImageModelListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_models_are_pay_only_without_chat_aliases(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    credential_manager=SimpleNamespace(get_total_credentials=lambda: 1)
                )
            )
        )
        base_models = ["gemini-3.8-flash", "gemini-3.1-flash-image"]
        with (
            patch.object(
                models_api,
                "refresh_models_config_cache",
                AsyncMock(return_value=True),
            ),
            patch.object(
                models_api,
                "get_vertex_models",
                AsyncMock(return_value=base_models),
            ),
            patch.object(
                models_api,
                "get_vertex_express_models",
                AsyncMock(return_value=[]),
            ),
            patch.object(models_api.app_config, "VERTEX_EXPRESS_API_KEY_VAL", []),
            patch.object(models_api.settings, "VERTEX_EXPRESS_API_KEY", ""),
        ):
            response = await models_api.list_models(request, "test-key")

        model_ids = {item["id"] for item in response["data"]}
        self.assertIn("[PAY]gemini-3.1-flash-image", model_ids)
        self.assertIn("[PAY]gemini-3.8-flash-nothinking", model_ids)
        self.assertFalse(
            any(
                model_id.startswith("[PAY]gemini-3.1-flash-image-")
                for model_id in model_ids
            )
        )


if __name__ == "__main__":
    unittest.main()
