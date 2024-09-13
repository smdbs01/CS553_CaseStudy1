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

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1280


# Make sure to set the environment variable HF_TOKEN to your Hugging Face token for using InferenceClient
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
model_name = "stabilityai/stable-diffusion-2-1-base"
client = InferenceClient(model_name, token=HF_TOKEN)

scheduler = None
pipe = None


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
    prompt: str,
    is_local: bool,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    num_inference_steps: int,
    width: int,
    height: int,
) -> Union[np.ndarray, Image.Image, str, Path, None]:
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    if is_local:
        if scheduler is None or pipe is None:
            load_model()

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        return image, seed

    else:
        output = client.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )
        return output, seed


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
                # This part is adopted from Stabilityai's stable-diffusion-3-medium space
                # https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium
                with gr.Accordion("Advanced Settings", open=False):
                    negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                    )

                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )

                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                    with gr.Row():

                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=50,
                            value=512,
                        )

                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=50,
                            value=512,
                        )

                    with gr.Row():
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=5.0,
                        )

                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28,
                        )
            with gr.Column():
                is_local = gr.Checkbox(
                    label="Check this box to use a local model 🖥️",
                )
                img_out = gr.Image(label="Generated Image")
        gr.HTML(
            f"<p style='text-align: center;'>This app uses the <a href='https://huggingface.co/{model_name}'>{model_name}</a> model.</p>"
        )

    submit.click(
        sd_2_1_base,
        inputs=[
            prompt,
            is_local,
            negative_prompt,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
            width,
            height,
        ],
        outputs=[img_out, seed],
    )

if __name__ == "__main__":
    ui.launch(share=True)
