import base64
import unittest
from types import SimpleNamespace

from google.genai import types

from app.vertex.image_processing import (
    ImageProxyError,
    InputImage,
    augment_openai_image_prompt,
    build_image_generation_config,
    create_image_contents,
    extract_generated_images,
    normalize_image_model,
    openai_image_response,
    prepare_native_request,
    resolve_image_options,
    serialize_native_response,
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
        options = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            size="1536x1024",
        )
        self.assertEqual(options.model, "gemini-3.1-flash-image")
        self.assertEqual(options.aspect_ratio, "3:2")
        self.assertEqual(options.image_size, "1K")

        options = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            size="1024x1024",
            aspect_ratio="16:9",
            image_size="2K",
        )
        self.assertEqual(options.aspect_ratio, "16:9")
        self.assertEqual(options.image_size, "2K")

    def test_quality_mapping_and_explicit_image_size_precedence(self):
        options = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            quality="high",
        )
        self.assertEqual(options.image_size, "2K")

        options = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            quality="low",
            image_size="4K",
        )
        self.assertEqual(options.image_size, "4K")

    def test_quality_rejects_unsupported_model_resolution(self):
        with self.assertRaises(ImageProxyError) as raised:
            resolve_image_options(
                "[PAY]gemini-3.1-flash-lite-image",
                quality="high",
            )
        self.assertEqual(raised.exception.param, "quality")

    def test_rejects_resolution_not_supported_by_model(self):
        with self.assertRaises(ImageProxyError):
            resolve_image_options(
                "[PAY]gemini-3.1-flash-lite-image",
                image_size="2K",
            )

    def test_generation_config_requests_text_and_image(self):
        options, config = build_image_generation_config(
            "[PAY]gemini-3.1-flash-image",
            image_size="512",
        )
        self.assertEqual(options.model, "gemini-3.1-flash-image")
        self.assertEqual(config.response_modalities, ["TEXT", "IMAGE"])
        self.assertEqual(config.image_config.image_size, "512")

    def test_output_format_and_compression_mapping(self):
        options, config = build_image_generation_config(
            "[PAY]gemini-3.1-flash-image",
            output_format="jpeg",
            output_compression=82,
        )
        self.assertEqual(options.output_mime_type, "image/jpeg")
        self.assertEqual(
            config.image_config.image_output_options.mime_type,
            "image/jpeg",
        )
        self.assertEqual(
            config.image_config.image_output_options.compression_quality,
            82,
        )

    def test_rejects_invalid_openai_controls_with_parameter_name(self):
        cases = [
            ({"n": 2}, "n"),
            ({"response_format": "url"}, "response_format"),
            ({"stream": True}, "stream"),
            ({"partial_images": 1}, "partial_images"),
            ({"mask_provided": True}, "mask"),
            ({"output_format": "png", "output_compression": 80}, "output_compression"),
            ({"background": "transparent", "output_format": "jpeg"}, "background"),
        ]
        for kwargs, expected_param in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ImageProxyError) as raised:
                    resolve_image_options(
                        "[PAY]gemini-3.1-flash-image",
                        **kwargs,
                    )
                self.assertEqual(raised.exception.param, expected_param)

    def test_prompt_augmentation_is_deterministic(self):
        options = resolve_image_options(
            "[PAY]gemini-3.1-flash-image",
            background="transparent",
            style="vivid",
            input_fidelity="high",
        )
        prompt = augment_openai_image_prompt(
            "Add a blue border",
            options,
            is_edit=True,
        )
        self.assertIn("transparent background", prompt)
        self.assertIn("vivid", prompt)
        self.assertIn("preserve identities", prompt)


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

    def test_rejects_mismatched_output_mime_type(self):
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
        with self.assertRaises(ImageProxyError) as raised:
            openai_image_response(
                response,
                created=0,
                expected_mime_type="image/webp",
            )
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.param, "output_format")

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
    def test_native_response_uses_standard_base64(self):
        image_bytes = b"\xfb\xff\x00"
        response = types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_bytes,
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )

        serialized = serialize_native_response(response)
        encoded = serialized["candidates"][0]["content"]["parts"][0]["inlineData"][
            "data"
        ]
        self.assertEqual(encoded, base64.b64encode(image_bytes).decode("ascii"))
        self.assertEqual(base64.b64decode(encoded, validate=True), image_bytes)

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
