import torch
from utils.io_tools import login_hf
from diffusers import StableDiffusionXLPipeline


def load_model(model_name="stabilityai/stable-diffusion-xl-base-1.0", cache_dir=None, device_map='balanced', enable_offload=False):
    login_hf()
    model = StableDiffusionXLPipeline.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map=device_map)
    if enable_offload:
        model.enable_model_cpu_offload()
    return model

@torch.no_grad()
def generate(model, prompt, height=1024, width=1024, guidance_scale=5.0, 
             num_inference_steps=50, progress_bar=False, seed=23, timesteps=None,
             **kwargs): 
    model.set_progress_bar_config(disable=(not progress_bar))
    image = model(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
        timesteps=timesteps,
        **kwargs,
    )
    return image.images[0]