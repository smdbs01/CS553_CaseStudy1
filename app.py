import gradio as gr

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

print("Loading model...")
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
print("Model loaded.")

def resnet50(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

ui = gr.Interface(
    fn=resnet50,
    inputs=gr.Image(height=224, width=224),
    outputs=gr.Text(),
    title="ResNet-50 ImageNet",
    description="Identify the main object in an image. This model is a ResNet-50 neural network pre-trained on ImageNet."
)

if __name__ == "__main__":
    ui.launch()
