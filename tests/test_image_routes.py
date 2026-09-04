import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import types

from app.api.routes import router
from app.config import settings
from app.vertex.routes import images_api


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"route-test"


class FakeModels:
    def __init__(self):
        self.generate_calls = 0

    async def generate_content(self, **kwargs):
        self.generate_calls += 1
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
                                        "mimeType": "image/png",
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


if __name__ == "__main__":
    unittest.main()
