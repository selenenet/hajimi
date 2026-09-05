"""Exercise authenticated chat routes with only Vertex service-account credentials."""

import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from google.genai import types

from app.api import routes
from app.config import settings
from app.vertex import config as vertex_config


class VertexChatRouteTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for obj, name, value in [
            (settings, "PASSWORD", "vertex-route-test-token"),
            (settings, "ENABLE_VERTEX", True),
            (settings, "GEMINI_API_KEYS", ""),
            (settings, "VERTEX_EXPRESS_API_KEY", ""),
            (settings, "WHITELIST_USER_AGENT", []),
            (settings, "FAKE_STREAMING", False),
            (vertex_config, "VERTEX_EXPRESS_API_KEY_VAL", []),
            (vertex_config, "FAKE_STREAMING_INTERVAL_SECONDS", 0),
            (routes, "current_api_key", None),
        ]:
            self.patches.enter_context(patch.object(obj, name, value))

        self.credentials = object()
        self.manager = Mock()
        self.manager.get_random_credentials.return_value = (
            self.credentials,
            "test-vertex-project",
        )
        self.response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    index=0,
                    finish_reason="STOP",
                    content=types.Content(role="model", parts=[types.Part(text="OK")]),
                )
            ],
        )
        self.models = SimpleNamespace(
            generate_content=AsyncMock(side_effect=lambda **kw: self.response),
            generate_content_stream=AsyncMock(side_effect=self.make_stream),
        )
        self.client_factory = self.patches.enter_context(
            patch.object(
                routes.chat_api.genai,
                "Client",
                return_value=SimpleNamespace(aio=SimpleNamespace(models=self.models)),
            )
        )
        app = FastAPI()
        app.state.credential_manager = self.manager
        app.include_router(routes.router)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer vertex-route-test-token"}

    async def make_stream(self, **kwargs):
        async def chunks():
            yield self.response

        return chunks()

    def post(self, path="/v1/chat/completions", **params):
        return self.client.post(
            path,
            headers=self.headers,
            json={
                "model": "[PAY]gemini-3.8-flash",
                "messages": [{"role": "user", "content": "Say OK"}],
                **params,
            },
        )

    def test_nonstream_chat_and_aliases_without_ai_studio_key(self):
        for path in [
            "/v1/chat/completions",
            "/chat/completions",
            "/vertex/chat/completions",
        ]:
            with self.subTest(path=path):
                self.models.generate_content.reset_mock()
                response = self.post(path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json()["choices"][0]["message"]["content"], "OK"
                )
                self.models.generate_content.assert_awaited_once()
                self.assertEqual(
                    self.models.generate_content.call_args.kwargs["model"],
                    "gemini-3.8-flash",
                )
                self.assertIs(
                    self.client_factory.call_args.kwargs["credentials"],
                    self.credentials,
                )
                self.assertNotIn("api_key", self.client_factory.call_args.kwargs)

    def test_real_and_fake_stream_without_ai_studio_key(self):
        for fake_streaming in [False, True]:
            with (
                self.subTest(fake_streaming=fake_streaming),
                patch.object(settings, "FAKE_STREAMING", fake_streaming),
            ):
                self.models.generate_content.reset_mock()
                self.models.generate_content_stream.reset_mock()
                response = self.post(stream=True)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("text/event-stream", response.headers["content-type"])
                self.assertIn("data: [DONE]", response.text)
                chunks = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                self.assertTrue(chunks)
                self.assertTrue(all("error" not in chunk for chunk in chunks))
                text = "".join(
                    c["choices"][0]["delta"].get("content", "") for c in chunks
                )
                self.assertEqual(text, "OK")
                self.assertEqual(
                    self.models.generate_content.await_count
                    + self.models.generate_content_stream.await_count,
                    1,
                )

    def test_tool_call_and_tool_response_without_ai_studio_key(self):
        self.response.candidates[0].content.parts = [
            types.Part(
                function_call=types.FunctionCall(
                    name="get_weather", args={"city": "London"}
                ),
                thought_signature=b"test-signature",
            )
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        response = self.post(tools=tools, tool_choice="required")
        self.assertEqual(response.status_code, 200, response.text)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.models.generate_content.assert_awaited_once()
        config = self.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(
            config["tools"][0].function_declarations[0].name, "get_weather"
        )

        self.response.candidates[0].content.parts = [types.Part(text="OK")]
        response = self.post(
            tools=tools,
            messages=[
                {"role": "user", "content": "Weather in London?"},
                choice["message"],
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": '{"temperature":20}',
                },
            ],
        )
        self.assertEqual(response.status_code, 200, response.text)
        contents = self.models.generate_content.call_args.kwargs["contents"]
        replies = [
            p.function_response
            for c in contents
            for p in c.parts
            if p.function_response
        ]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].name, "get_weather")
        self.assertEqual(self.models.generate_content.await_count, 2)

    def test_missing_and_invalid_client_tokens_are_rejected(self):
        for path in [
            "/v1/chat/completions",
            "/chat/completions",
            "/vertex/chat/completions",
        ]:
            for headers in [{}, {"Authorization": "Bearer wrong-token"}]:
                with self.subTest(path=path, headers_present=bool(headers)):
                    response = self.client.post(
                        path,
                        headers=headers,
                        json={
                            "model": "[PAY]gemini-3.8-flash",
                            "messages": [{"role": "user", "content": "Say OK"}],
                        },
                    )
                    self.assertEqual(response.status_code, 401)
        self.client_factory.assert_not_called()

    def test_missing_vertex_credentials_returns_structured_error(self):
        self.manager.get_random_credentials.return_value = (None, None)
        response = self.post()
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(response.json()["error"]["type"], "authentication_error")
        self.client_factory.assert_not_called()

    def test_ai_studio_dispatch_is_unchanged(self):
        with (
            patch.object(settings, "ENABLE_VERTEX", False),
            patch.object(
                routes,
                "aistudio_chat_completions",
                AsyncMock(return_value=JSONResponse({"backend": "aistudio"})),
            ) as aistudio,
        ):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        aistudio.assert_awaited_once()
        self.client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
