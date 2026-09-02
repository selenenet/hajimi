import unittest

from app.vertex.model_variants import (
    is_gemini_flash_model,
    supports_max_thinking_variant,
    supports_nothinking_variant,
    thinking_config_for_variant,
)


class ModelVariantTests(unittest.TestCase):
    def test_recognizes_current_flash_families(self):
        for model_name in (
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview-05-20",
        ):
            with self.subTest(model_name=model_name):
                self.assertTrue(is_gemini_flash_model(model_name))
                self.assertTrue(supports_nothinking_variant(model_name))

    def test_rejects_non_flash_models(self):
        for model_name in ("gemini-2.5-pro", "gemini-pro-flash", "other-3.8-flash"):
            with self.subTest(model_name=model_name):
                self.assertFalse(is_gemini_flash_model(model_name))
                self.assertFalse(supports_nothinking_variant(model_name))

    def test_nothinking_uses_zero_budget_for_flash(self):
        self.assertEqual(
            thinking_config_for_variant("gemini-3.8-flash", "nothinking"),
            {"thinking_budget": 0},
        )
        self.assertEqual(
            thinking_config_for_variant("gemini-3.7-flash", "nothinking"),
            {"thinking_budget": 0},
        )

    def test_max_alias_remains_limited_to_numeric_budget_models(self):
        self.assertTrue(supports_max_thinking_variant("gemini-2.5-flash"))
        self.assertFalse(supports_max_thinking_variant("gemini-3.8-flash"))
        with self.assertRaises(ValueError):
            thinking_config_for_variant("gemini-3.8-flash", "max")

    def test_legacy_pro_preview_behavior_is_preserved(self):
        model_name = "gemini-2.5-pro-preview-06-05"
        self.assertEqual(
            thinking_config_for_variant(model_name, "nothinking"),
            {"thinking_budget": 128},
        )
        self.assertEqual(
            thinking_config_for_variant(model_name, "max"),
            {"thinking_budget": 32768},
        )


if __name__ == "__main__":
    unittest.main()
