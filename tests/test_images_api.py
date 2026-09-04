import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.genai import types

from app.vertex.image_processing import ImageProxyError
from app.vertex.routes import images_api


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"generated"


def image_response():
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Revised"),
                        types.Part(
                            inline_data=types.Blob(
                                data=PNG_BYTES,
                                mime_type="image/png",
                            )
                        ),
                    ],
                )
            )
        ]
    )


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.generate_calls = 0

    async def generate_content(self, **kwargs):
        self.generate_calls += 1
        if self.error:
            raise self.error
        return self.response

    async def generate_content_stream(self, **kwargs):
        async def chunks():
            yield self.response

        return chunks()


class FakeClient:
    def __init__(self, models):
        self.aio = SimpleNamespace(models=models)


class ImagesApiTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(credential_manager=object()))
        )

    async def test_openai_generation_uses_one_upstream_call(self):
        models = FakeModels(response=image_response())
        client = FakeClient(models)
        with patch.object(
            images_api,
            "get_vertex_ai_client",
            AsyncMock(return_value=client),
        ):
            result = await images_api.generate_openai_image(
                self.request(),
                model_name="[PAY]gemini-3.1-flash-image",
                prompt="Generate a blue square",
                size="1024x1024",
                aspect_ratio=None,
                image_size=None,
            )
        self.assertEqual(models.generate_calls, 1)
        self.assertEqual(result["data"][0]["revised_prompt"], "Revised")

    async def test_failure_is_not_retried(self):
        error = RuntimeError("upstream failed")
        models = FakeModels(error=error)
        client = FakeClient(models)
        with patch.object(
            images_api,
            "get_vertex_ai_client",
            AsyncMock(return_value=client),
        ):
            with self.assertRaises(ImageProxyError):
                await images_api.generate_openai_image(
                    self.request(),
                    model_name="[PAY]gemini-3.1-flash-image",
                    prompt="Generate a blue square",
                    size="1024x1024",
                    aspect_ratio=None,
                    image_size=None,
                )
        self.assertEqual(models.generate_calls, 1)

    async def test_native_stream_emits_camel_case_inline_data(self):
        models = FakeModels(
            response=types.GenerateContentResponse.model_validate(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": PNG_BYTES,
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        )
        client = FakeClient(models)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Image"}]}],
            "generationConfig": {},
        }
        with patch.object(
            images_api,
            "get_vertex_ai_client",
            AsyncMock(return_value=client),
        ):
            response = await images_api.generate_native_image(
                self.request(),
                model_name="gemini-3.1-flash-image",
                payload=payload,
                stream=True,
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        body = "".join(chunks)
        self.assertIn("inlineData", body)
        self.assertIn("mimeType", body)
        self.assertNotIn("[DONE]", body)
        data_line = next(
            line for line in body.splitlines() if line.startswith("data: ")
        )
        json.loads(data_line[6:])


if __name__ == "__main__":
    unittest.main()
