import unittest

from app.vertex.message_processing import create_gemini_prompt
from app.vertex.models import OpenAIRequest


def tool_call(call_id, name, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class GeminiToolResponseGroupingTests(unittest.TestCase):
    def test_parallel_tool_responses_share_one_user_content(self):
        request = OpenAIRequest.model_validate(
            {
                "model": "gemini-test",
                "messages": [
                    {"role": "user", "content": "Inspect the domain"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            tool_call("call_wayback", "webfetch"),
                            tool_call("call_urlscan", "webfetch"),
                            tool_call("call_search", "bash"),
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_wayback",
                        "content": '{"archived": false}',
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_urlscan",
                        "content": '{"results": []}',
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_search",
                        "content": "401 Unauthorized",
                    },
                ],
            }
        )

        contents = create_gemini_prompt(request.messages)

        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(len(contents[1].parts), 3)
        self.assertEqual(contents[2].role, "user")
        self.assertEqual(len(contents[2].parts), 3)
        self.assertEqual(
            [part.function_response.id for part in contents[2].parts],
            ["call_wayback", "call_urlscan", "call_search"],
        )
        self.assertEqual(
            [part.function_response.name for part in contents[2].parts],
            ["webfetch", "webfetch", "bash"],
        )

    def test_separate_tool_call_turns_form_separate_response_groups(self):
        request = OpenAIRequest.model_validate(
            {
                "model": "gemini-test",
                "messages": [
                    {"role": "user", "content": "Inspect the domain"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call("call_skill", "skill")],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_skill",
                        "content": "skill loaded",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            tool_call("call_one", "webfetch"),
                            tool_call("call_two", "webfetch"),
                            tool_call("call_three", "bash"),
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_one",
                        "content": "one",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_two",
                        "content": "two",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_three",
                        "content": "three",
                    },
                ],
            }
        )

        contents = create_gemini_prompt(request.messages)

        self.assertEqual([content.role for content in contents], [
            "user",
            "model",
            "user",
            "model",
            "user",
        ])
        self.assertEqual(len(contents[2].parts), 1)
        self.assertEqual(len(contents[4].parts), 3)


if __name__ == "__main__":
    unittest.main()
