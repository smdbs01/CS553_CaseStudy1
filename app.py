import gradio as gr

# Sample Gradio usage
def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"#sssss
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)
if __name__ == "__main__":
    demo.launch()