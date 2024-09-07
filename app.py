from PIL import Image
import gradio as gr
from huggingface_hub import InferenceClient
import numpy as np
from io import BytesIO
import os
from typing import Union
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import torch.nn.functional as F

# Make sure to set the environment variable HF_TOKEN to your Hugging Face token for using InferenceClient
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
client = InferenceClient("microsoft/resnet-50", token=HF_TOKEN)

processor = None
model = None

def load_model():
    global processor, model
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    print("Model loaded.")

def resnet50(image: Union[np.ndarray, Image.Image, str, None], is_local: bool = False) -> dict[str, float]:
    if image is None:
        return {}
    
    if is_local:
        if processor is None or model is None:
            load_model()
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)[0]

        # Get the top 5 predicted classes with probabilities
        top_probs, top_labels = torch.topk(probabilities, 5)

        # Map the label IDs to human-readable class names
        top_predictions = {model.config.id2label[top_labels[i].item()]: top_probs[i].item() for i in range(5)}

        return top_predictions
    else:
        image_in = image
        # Convert the image to bytes if it is a PIL image or numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_in = image_bytes.getvalue()
        
        # Get the model's prediction
        prediction = client.image_classification(image_in)
        # Return the top 5 results
        return {pred.label: pred.score for pred in prediction}


# Define the Gradio interface
ui = gr.Interface(
    fn=resnet50,
    inputs=[gr.Image(), gr.Checkbox(label="Use local model")],
    outputs=gr.Label(num_top_classes=5),  # Display top 5 results
    title="ResNet-50 ImageNet",
    description="Identify the main object in an image with probabilities. This model is a ResNet-50 neural network pre-trained on ImageNet."
)

if __name__ == "__main__":
    ui.launch(share=True)