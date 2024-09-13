from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import os
from typing import Union
import torch
import torch.nn.functional as F
from huggingface_hub import InferenceClient
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Make sure to set the environment variable HF_TOKEN to your Hugging Face token for using InferenceClient
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
model_name = "stabilityai/stable-diffusion-2-1-base"
client = InferenceClient(model_name, token=HF_TOKEN)

scheduler = None
pipe = None
config = {
    "negative_prompt": "",
    "width": 256,
    "height": 256,
    "num_inference_steps": 20,
}


def load_model():
    global scheduler, pipe
    print("Loading model...")
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(model_name, scheduler=scheduler)
    pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    print("Model loaded.")


def sd_2_1_base(
    prompt: str, is_local: bool
) -> Union[np.ndarray, Image.Image, str, Path, None]:
    if is_local:
        if scheduler is None or pipe is None:
            load_model()

        image = pipe(
            prompt,
        ).images[0]
        return image

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
                    placeholder="Enter a prompt to generate an image",
                    label="Prompt",
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
