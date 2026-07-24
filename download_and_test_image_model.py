import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig

# 1. Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("⚡ Loading 4-Bit Quantized FLUX.1 Pipeline safely into CPU RAM...")

# 2. Configure 8-bit T5 Text Encoder using bfloat16
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

text_encoder_8bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# 3. Load 4-Bit NF4 Quantized Transformer on CPU first
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    "Keffisor21/flux1-schnell-bnb-nf4",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)

# 4. Assemble Pipeline in bfloat16
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    text_encoder_2=text_encoder_8bit,
    transformer=transformer_4bit,
    torch_dtype=torch.bfloat16
)

# 5. Enable Sequential CPU Offloading
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print("✅ Model loaded cleanly!")

# 6. Generate Image
prompt = "A beautiful girl with a laptop looks like an HR manager in an office, digital art, trending on artstation, cinematic lighting, 8k, ultra-detailed, realistic"

image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256
).images[0]

# Save output
image.save("test_local_flux_4bit.png")
print("🎉 Test image generated successfully and saved as 'test_local_flux_4bit.png'!")