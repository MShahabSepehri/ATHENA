import torch
from utils.io_tools import login_hf
from diffusers import FluxPipeline


def load_model(model_name="black-forest-labs/FLUX.1-dev", cache_dir=None, device_map='balanced', enable_offload=True):
    login_hf()
    model = FluxPipeline.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map=device_map)
    if enable_offload:
        model.enable_model_cpu_offload()
    return model

@torch.no_grad()
def generate(model, prompt, height=1024, width=1024, guidance_scale=3.5, 
             num_inference_steps=50, max_sequence_length=512,
             progress_bar=False, seed=23, sigmas=None): 
    model.set_progress_bar_config(disable=(not progress_bar))
    image = model(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
        sigmas=sigmas,
    )
    return image.images[0]
