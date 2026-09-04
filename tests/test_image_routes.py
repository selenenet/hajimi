import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import types

from app.api import routes as api_routes
from app.api.routes import router
from app.config import settings
from app.vertex.routes import images_api


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"route-test"


class FakeModels:
    def __init__(self):
        self.generate_calls = 0
        self.last_kwargs = None

    async def generate_content(self, **kwargs):
        self.generate_calls += 1
        self.last_kwargs = kwargs
        image_config = kwargs["config"].image_config
        output_options = image_config.image_output_options if image_config else None
        mime_type = output_options.mime_type if output_options else "image/png"
        return types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Revised"},
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": PNG_BYTES,
                                    }
                                },
                            ],
                        }
                    }
                ]
            }
        )


class ImageRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_password = settings.PASSWORD
        cls.previous_vertex = settings.ENABLE_VERTEX
        cls.previous_whitelist = settings.WHITELIST_USER_AGENT
        settings.PASSWORD = "route-test-key"
        settings.ENABLE_VERTEX = True
        settings.WHITELIST_USER_AGENT = []
        app = FastAPI()
        app.state.credential_manager = object()
        app.include_router(router)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        settings.PASSWORD = cls.previous_password
        settings.ENABLE_VERTEX = cls.previous_vertex
        settings.WHITELIST_USER_AGENT = cls.previous_whitelist

    def setUp(self):
        self.models = FakeModels()
        self.client_object = SimpleNamespace(aio=SimpleNamespace(models=self.models))
        self.client_patch = patch.object(
            images_api,
            "get_vertex_ai_client",
            AsyncMock(return_value=self.client_object),
        )
        self.client_patch.start()

    def tearDown(self):
        self.client_patch.stop()

    @property
    def headers(self):
        return {"Authorization": "Bearer route-test-key"}

    def test_openai_generation_route(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers,
            json={
                "model": "[PAY]gemini-3.1-flash-image",
                "prompt": "A blue square",
                "response_format": "b64_json",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.models.generate_calls, 1)
        self.assertEqual(
            base64.b64decode(response.json()["data"][0]["b64_json"]),
            PNG_BYTES,
        )

    def test_openai_edit_route(self):
        response = self.client.post(
            "/v1/images/edits",
            headers=self.headers,
            data={
                "model": "[PAY]gemini-3.1-flash-image",
                "prompt": "Add a border",
                "response_format": "b64_json",
            },
            files=[("image", ("input.png", PNG_BYTES, "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.models.generate_calls, 1)

    def test_openai_edit_accepts_bracketed_image_field(self):
        response = self.client.post(
            "/v1/images/edits",
            headers=self.headers,
            data={"prompt": "Add a border"},
            files=[("image[]", ("input.png", PNG_BYTES, "image/png"))],
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.models.generate_calls, 1)

    def test_openai_generation_maps_quality_format_and_prompt_controls(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers,
            json={
                "prompt": "A blue square",
                "quality": "low",
                "output_format": "webp",
                "output_compression": 75,
                "background": "transparent",
                "style": "vivid",
                "moderation": "low",
                "user": "must-not-be-forwarded",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        config = self.models.last_kwargs["config"]
        self.assertEqual(config.image_config.image_size, "512")
        self.assertEqual(
            config.image_config.image_output_options.mime_type,
            "image/webp",
        )
        self.assertEqual(
            config.image_config.image_output_options.compression_quality,
            75,
        )
        prompt = self.models.last_kwargs["contents"][0].parts[0].text
        self.assertIn("transparent background", prompt)
        self.assertIn("vivid", prompt)
        self.assertNotIn("must-not-be-forwarded", repr(self.models.last_kwargs))

    def test_openai_edit_rejects_mask(self):
        response = self.client.post(
            "/v1/images/edits",
            headers=self.headers,
            data={"prompt": "Add a border"},
            files=[
                ("image", ("input.png", PNG_BYTES, "image/png")),
                ("mask", ("mask.png", PNG_BYTES, "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["param"], "mask")
        self.assertEqual(self.models.generate_calls, 0)

    def test_openai_generation_rejects_streaming(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers,
            json={"prompt": "A blue square", "stream": True},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["param"], "stream")
        self.assertEqual(self.models.generate_calls, 0)

    def test_rejects_multiple_candidates_before_upstream_call(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=self.headers,
            json={"prompt": "A blue square", "n": 2},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.models.generate_calls, 0)

    def test_native_image_route(self):
        response = self.client.post(
            "/gemini/v1/models/gemini-3.1-flash-image:generateContent",
            headers={"x-goog-api-key": "route-test-key"},
            json={
                "contents": [{"role": "user", "parts": [{"text": "A blue square"}]}],
                "generationConfig": {},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        inline_data = response.json()["candidates"][0]["content"]["parts"][1][
            "inlineData"
        ]
        self.assertEqual(base64.b64decode(inline_data["data"]), PNG_BYTES)

    def test_model_listing_does_not_require_ai_studio_key(self):
        with (
            patch.object(api_routes, "current_api_key", None),
            patch.object(
                api_routes.models_api,
                "list_models",
                AsyncMock(return_value={"object": "list", "data": []}),
            ) as list_models,
        ):
            response = self.client.get("/v1/models", headers=self.headers)

        self.assertEqual(response.status_code, 200, response.text)
        list_models.assert_awaited_once()
        self.assertEqual(list_models.await_args.args[1], "")


if __name__ == "__main__":
    unittest.main()
