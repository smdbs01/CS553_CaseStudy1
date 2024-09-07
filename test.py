import unittest
from huggingface_hub import InferenceClient
from os import path
import os

from app import resnet50
from PIL import Image
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
TOKEN_MSG = "Set HF_TOKEN to your Hugging Face token for using InferenceClient"
SKIP_MSG = "Skipping test because HF_TOKEN is not set"


class TestGeneralSetup(unittest.TestCase):
    def test_setup(self):
        self.assertTrue(HF_TOKEN != "YOUR_HF_TOKEN", TOKEN_MSG)

class TestLocalModelImage(unittest.TestCase):
    def test_local_model_image1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        image = Image.open(p)
        prediction = resnet50(image, is_local=True)
        self.assertTrue(len(prediction))
        
    def test_local_model_image2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        image = Image.open(p)
        prediction = resnet50(image, is_local=True)
        self.assertTrue(len(prediction))

class TestLocalImageArray(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_array_1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        image = Image.open(p)
        image = np.array(image)
        prediction = resnet50(image, is_local=False)
        self.assertTrue(len(prediction))
        
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_array_2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        image = Image.open(p)
        image = np.array(image)
        prediction = resnet50(image, is_local=False)
        self.assertTrue(len(prediction))
    
class TestLocalImagePIL(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_pil_1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        image = Image.open(p)
        prediction = resnet50(image, is_local=False)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_pil_2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        image = Image.open(p)
        prediction = resnet50(image, is_local=False)
        self.assertTrue(len(prediction))

class TestLocalImageStr(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_str_1(self):
        p = path.join(path.dirname(__file__), "sample", "pomdog.jpg")
        prediction = resnet50(p, is_local=False)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_str_2(self):
        p = path.join(path.dirname(__file__), "sample", "persiancat.jpg")
        prediction = resnet50(p, is_local=False)
        self.assertTrue(len(prediction))

class TestRemoteImageUrl(unittest.TestCase):
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_url_1(self):
        url = "https://i.ytimg.com/vi/AQSTQ4VUPH4/maxresdefault.jpg"
        prediction = resnet50(url, is_local=False)
        self.assertTrue(len(prediction))
        
    @unittest.skipIf(HF_TOKEN == "YOUR_HF_TOKEN", SKIP_MSG)
    def test_url_2(self):
        url = "https://img.thrfun.com/img/200/610/breed_information_persian_x1.jpg"
        prediction = resnet50(url, is_local=False)
        self.assertTrue(len(prediction))
        
if __name__ == "__main__":
    unittest.main()
