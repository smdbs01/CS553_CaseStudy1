import unittest
from huggingface_hub import InferenceClient
from os import path
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
TOKEN_MSG = "Set HF_TOKEN to your Hugging Face token for using InferenceClient"
SKIP_MSG = "Skipping test because HF_TOKEN is not set"

client = InferenceClient("microsoft/resnet-50", token=HF_TOKEN)

class TestGeneralSetup(unittest.TestCase):
    def test_setup(self):
        self.assertTrue(HF_TOKEN != "YOUR_HF_TOKEN", TOKEN_MSG)

class TestLocalImageBytes(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_bytes_1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        with open(p, "rb") as f:
            image_bytes = f.read()
        prediction = client.image_classification(image_bytes)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_bytes_2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        prediction = client.image_classification(p)
        self.assertTrue(len(prediction))
    
class TestLocalImagePath(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_path_1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        prediction = client.image_classification(p)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_path_2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        prediction = client.image_classification(p)
        self.assertTrue(len(prediction))
        
class TestRemoteImageUrl(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_url_1(self):
        url = "https://i.ytimg.com/vi/AQSTQ4VUPH4/maxresdefault.jpg"
        prediction = client.image_classification(url)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_url_2(self):
        url = "https://img.thrfun.com/img/200/610/breed_information_persian_x1.jpg"
        prediction = client.image_classification(url)
        self.assertTrue(len(prediction))
        
if __name__ == "__main__":
    unittest.main()
