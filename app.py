from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
from io import BytesIO
import os
from typing import Union
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import torch.nn.functional as F
from huggingface_hub import InferenceClient

# Make sure to set the environment variable HF_TOKEN to your Hugging Face token for using InferenceClient
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
model_name = "stabilityai/stable-diffusion-2-1-base"
client = InferenceClient(model_name, token=HF_TOKEN)

processor = None
model = None


def load_model():
    global processor, model
    print("Loading model...")
    # processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    print("Model loaded.")


def sd_2_1_base(
    prompt: str, is_local: bool
) -> Union[np.ndarray, Image.Image, str, Path, None]:
    if is_local:
        if processor is None or model is None:
            load_model()
        pass
    else:
        output = client.text_to_image(prompt)
        return output


with gr.Blocks() as ui:
    with gr.Column():
        gr.HTML(
            "<h1 style='text-align: center;font-size: 48px;margin-bottom: 24px;'>😎 Cool Image Generator 😎</h1>"
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    "Enter a prompt to generate an image", label="Prompt"
                )
                submit = gr.Button("Generate Image", variant="primary")
            with gr.Column():
                is_local = gr.Checkbox(
                    label="Check this box to use a local model 🖥️",
                )
                output = gr.Image(label="Generated Image")
        gr.HTML(
            f"<p style='text-align: center;'>This app uses the <a href='https://huggingface.co/{model_name}'>{model_name}</a> model.</p>"
        )

    submit.click(sd_2_1_base, inputs=[prompt, is_local], outputs=output)

if __name__ == "__main__":
    ui.launch(share=True)
