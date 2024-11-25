import gradio as gr
import numpy as np

import spaces
import torch
import spaces
import random 

from diffusers import FluxFillPipeline

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, revision="refs/pr/4").to("cuda")

def calculate_optimal_dimensions(image: Image.Image):
    # Extract the original dimensions
    original_width, original_height = image.size
    
    # Set constants
    MIN_ASPECT_RATIO = 9 / 16
    MAX_ASPECT_RATIO = 16 / 9
    FIXED_DIMENSION = 1024

    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height

    # Determine which dimension to fix
    if original_aspect_ratio > 1:  # Wider than tall
        width = FIXED_DIMENSION
        height = round(FIXED_DIMENSION / original_aspect_ratio)
    else:  # Taller than wide
        height = FIXED_DIMENSION
        width = round(FIXED_DIMENSION * original_aspect_ratio)

    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Enforce aspect ratio limits
    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = (height * MAX_ASPECT_RATIO // 8) * 8
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = (width / MIN_ASPECT_RATIO // 8) * 8

    # Ensure width and height remain above the minimum dimensions
    width = max(width, 576) if width == FIXED_DIMENSION else width
    height = max(height, 576) if height == FIXED_DIMENSION else height

    return width, height

@spaces.GPU
def infer(edit_images, prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    image = edit_images["background"]
    width, height = calculate_optimal_dimensions(image)
    mask = edit_images["layers"][0]
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=width,
        width=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    return image, seed
    
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Fill [dev]
12B param rectified flow transformer structural conditioning tuned, guidance-distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/)  
[[non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)] [[blog](https://blackforestlabs.ai/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-dev)]
        """)
        with gr.Row():
            with gr.Column():
                edit_image = gr.ImageEditor(
                    label='Upload and draw mask for inpainting',
                    type='pil',
                    sources=["upload", "webcam"],
                    image_mode='RGB',
                    layers=False,
                    brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                    height=600
                )
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run")
                
            result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
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
                    step=32,
                    value=1024,
                    visible=False
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                    visible=False
                )
            
            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [edit_image, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch()