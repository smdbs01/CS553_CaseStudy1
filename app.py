import gradio as gr
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import torch.nn.functional as F

print("Loading model...")
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
print("Model loaded.")

def resnet50(image):
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

# Define the Gradio interface
ui = gr.Interface(
    fn=resnet50,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),  # Display top 5 results
    title="ResNet-50 ImageNet",
    description="Identify the main object in an image with probabilities. This model is a ResNet-50 neural network pre-trained on ImageNet."
)

if __name__ == "__main__":
    ui.launch()
