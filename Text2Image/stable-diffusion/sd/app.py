"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2026/1/10-20:57
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

# app.py
import gradio as gr
import torch
import tempfile
from PIL import Image
import numpy as np
import os
import model_loader
import pipeline
from transformers import CLIPTokenizer

# ----------------------------
# 配置路径（请根据您的项目结构调整）
# ----------------------------
WEIGHTS_DIR = "../weights"
IMAGE_OUTPUT_DIR = "./outputs"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 设备设置
# ----------------------------
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cpu"
elif torch.has_mps or torch.backends.mps.is_available():
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# ----------------------------
# 加载模型（只加载一次）
# ----------------------------
tokenizer = CLIPTokenizer(os.path.join(WEIGHTS_DIR, "vocab.json"),
                          merges_file=os.path.join(WEIGHTS_DIR, "merges.txt"))
model_file = os.path.join(WEIGHTS_DIR, "v1-5-pruned-emaonly.ckpt")
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
print("Models loaded successfully!")

# ----------------------------
# 核心生成函数
# ----------------------------
def generate_image(
        prompt,
        negative_prompt,
        init_image,
        strength,
        cfg_scale,
        steps,
        seed,
        progress=gr.Progress()
):
    progress(0, desc="Starting generation...")

    print('prompt: {} type: {}'.format(prompt, type(prompt)))
    print('negative prompt: {} type: {}'.format(negative_prompt, type(negative_prompt)))
    print('strength: {} type: {}'.format(strength, type(strength)))
    print('cfg_scale: {} type: {}'.format(cfg_scale, type(cfg_scale)))
    print('sample steps: {} type: {}'.format(steps, type(steps)))
    print('random seed: {} type: {}'.format(seed, type(seed)))

    # 处理输入图像
    input_image = None
    if init_image is not None:
        # Gradio 传入的是 numpy array (H, W, C) uint8
        input_image = Image.fromarray(init_image.astype('uint8'))
        print(f"Using input image with strength={strength}")
    else:
        print("ext-to-image mode")

    # 设置随机种子
    if seed == -1:
        seed = int(torch.randint(0, 1000000, (1,)).item())
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)

    # 调用原始 pipeline
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # 转为 PIL Image
    pil_image = Image.fromarray(output_image)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=IMAGE_OUTPUT_DIR) as tmp:
        pil_image.save(tmp.name)
        temp_file = tmp.name

    print(f"Returning: image={type(pil_image)}, seed={type(seed)}, path={type(temp_file)}")
    print(f"Image mode: {pil_image.mode}, size: {pil_image.size}")

    return pil_image, seed, temp_file


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="Diffusion Model Demo") as demo:
    gr.Markdown("# Diffusion Model: Text-to-Image & Image-to-Image")
    gr.Markdown("基于 Stable Diffusion v1.5 的本地推理界面")

    with gr.Row():
        with gr.Column(scale=2):
            # 输入区域
            prompt = gr.Textbox(
                label="Prompt",
                value="A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.",
                placeholder="Enter your prompt here..."
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="",
                placeholder="What to avoid (e.g., blurry, deformed)"
            )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    cfg_scale = gr.Slider(1, 14, value=8, step=0.5, label="CFG Scale")
                    steps = gr.Slider(10, 100, value=50, step=5, label="Sampling Steps")
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    strength = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Denoising Strength (for img2img)")
            # 上传图像
            init_image = gr.Image(label="Upload Image for Image-to-Image (Optional)", type="numpy")
            run_btn = gr.Button("Generate", variant="primary")
        # 将生成的图像进行显示
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Image", interactive=False)
            used_seed = gr.Number(label="Used Seed", interactive=False)
            download_btn = gr.File(label="Download Image", visible=True, interactive=False)

    # 绑定事件
    run_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, init_image, strength, cfg_scale, steps, seed],
        outputs=[output_image, used_seed, download_btn],
        show_progress="full"
    )

# ----------------------------
# 启动
# ----------------------------
if __name__ == "__main__":
    demo.launch()