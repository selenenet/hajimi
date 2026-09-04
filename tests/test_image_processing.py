import base64
import unittest
from types import SimpleNamespace

from google.genai import types

from app.vertex.image_processing import (
    ImageProxyError,
    InputImage,
    build_image_generation_config,
    create_image_contents,
    extract_generated_images,
    normalize_image_model,
    prepare_native_request,
    resolve_image_options,
    validate_input_image,
    validate_input_images,
)


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test-image"


class ImageModelValidationTests(unittest.TestCase):
    def test_normalizes_pay_model(self):
        self.assertEqual(
            normalize_image_model("[PAY]gemini-3.1-flash-image"),
            "gemini-3.1-flash-image",
        )

    def test_rejects_express_and_text_models(self):
        with self.assertRaises(ImageProxyError):
            normalize_image_model("[EXPRESS] gemini-3.1-flash-image")
        with self.assertRaises(ImageProxyError):
            normalize_image_model("[PAY]gemini-3.8-flash")

    def test_openai_size_mapping_and_explicit_override(self):
        model, _, aspect_ratio, image_size = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            size="1536x1024",
        )
        self.assertEqual(model, "gemini-3.1-flash-image")
        self.assertEqual(aspect_ratio, "3:2")
        self.assertEqual(image_size, "1K")

        _, _, aspect_ratio, image_size = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            size="1024x1024",
            aspect_ratio="16:9",
            image_size="2K",
        )
        self.assertEqual(aspect_ratio, "16:9")
        self.assertEqual(image_size, "2K")

    def test_rejects_resolution_not_supported_by_model(self):
        with self.assertRaises(ImageProxyError):
            resolve_image_options(
                "[PAY]gemini-3.1-flash-lite-image",
                image_size="2K",
            )

    def test_generation_config_requests_text_and_image(self):
        model, config = build_image_generation_config(
            "[PAY]gemini-3.1-flash-image",
            image_size="512",
        )
        self.assertEqual(model, "gemini-3.1-flash-image")
        self.assertEqual(config.response_modalities, ["TEXT", "IMAGE"])
        self.assertEqual(config.image_config.image_size, "512")


class ImageInputTests(unittest.TestCase):
    def test_validates_magic_bytes_instead_of_trusting_mime(self):
        image = validate_input_image(PNG_BYTES, "image/png")
        self.assertEqual(image.mime_type, "image/png")
        with self.assertRaises(ImageProxyError):
            validate_input_image(PNG_BYTES, "image/jpeg")
        with self.assertRaises(ImageProxyError):
            validate_input_image(b"not-an-image", "image/png")

    def test_builds_prompt_and_image_parts(self):
        contents = create_image_contents(
            "Add a blue border",
            [InputImage(PNG_BYTES, "image/png")],
        )
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Add a blue border")
        self.assertEqual(contents[0].parts[1].inline_data.data, PNG_BYTES)

    def test_combined_input_limit(self):
        images = [InputImage(b"x" * (4 * 1024 * 1024), "image/png") for _ in range(6)]
        with self.assertRaises(ImageProxyError) as raised:
            validate_input_images(images)
        self.assertEqual(raised.exception.status_code, 413)


class ImageResponseTests(unittest.TestCase):
    def test_accepts_image_only_response(self):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    data=PNG_BYTES,
                                    mime_type="image/png",
                                )
                            )
                        ],
                    )
                )
            ]
        )
        images, revised_prompt = extract_generated_images(response)
        self.assertIsNone(revised_prompt)
        self.assertEqual(base64.b64decode(images[0].b64_json), PNG_BYTES)

    def test_preserves_text_from_mixed_response(self):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(text="Refined description"),
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
        images, revised_prompt = extract_generated_images(response)
        self.assertEqual(len(images), 1)
        self.assertEqual(revised_prompt, "Refined description")

    def test_rejects_response_without_image(self):
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="No image")],
                    )
                )
            ]
        )
        with self.assertRaises(ImageProxyError) as raised:
            extract_generated_images(response)
        self.assertEqual(raised.exception.status_code, 502)

    def test_surfaces_safety_block(self):
        response = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="PROHIBITED_CONTENT"),
            candidates=[],
        )
        with self.assertRaises(ImageProxyError) as raised:
            extract_generated_images(response)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            raised.exception.error_type,
            "content_policy_violation",
        )


class NativeImageRequestTests(unittest.TestCase):
    def test_native_request_forces_text_and_image_modalities(self):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Edit this image"},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(PNG_BYTES).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"},
            },
        }
        model, contents, config, image_count, input_bytes = prepare_native_request(
            "gemini-3.1-flash-image",
            payload,
        )
        self.assertEqual(model, "gemini-3.1-flash-image")
        self.assertEqual(config.response_modalities, ["TEXT", "IMAGE"])
        self.assertEqual(config.image_config.aspect_ratio, "1:1")
        self.assertEqual(contents[0].parts[1].inline_data.data, PNG_BYTES)
        self.assertEqual(image_count, 1)
        self.assertEqual(input_bytes, len(PNG_BYTES))


if __name__ == "__main__":
    unittest.main()
