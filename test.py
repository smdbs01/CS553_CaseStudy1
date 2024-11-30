import unittest
import os

from app import sd_2_1_base

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
TOKEN_MSG = "Set HF_TOKEN to your Hugging Face token for using InferenceClient"
SKIP_MSG = "Skipping test because HF_TOKEN is not set"


class TestGeneralSetup(unittest.TestCase):
    def test_setup(self):
        self.assertTrue(HF_TOKEN != "YOUR_HF_TOKEN", TOKEN_MSG)


# class TestLocalModel(unittest.TestCase):
#     @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
#     # @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
#     def test_url_1(self):
#         image, _ = sd_2_1_base(
#             "A photo of a cat", True, "A photo of a dog", 42, False, 0.1, 10, 128, 128
#         )
#         self.assertIsNotNone(image)


class TestAPIModel(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_url_1(self):
        image, _ = sd_2_1_base(
            "A photo of a cat", False, "A photo of a dog", 42, False, 0.1, 10, 128, 128
        )
        self.assertIsNotNone(image)


if __name__ == "__main__":
    unittest.main()
