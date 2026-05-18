# 🎬 Slasher-Vision-35mm (SDXL LoRA)

## 🔗 Model Card

[View the Model Card on Hugging Face](https://huggingface.co/Ilyankhan69/slasher-vision-35mm)

A custom Low-Rank Adaptation (LoRA) for Stable Diffusion XL (SDXL), designed to strip away the smooth, artificial "AI look" and infuse your generations with the raw, visceral, and gritty aesthetic of 1970s grindhouse slasher films.

Trained on high-quality movie stills, this adapter specializes in generating authentic 35mm film grain, harsh optical flash photography lighting, unpolished skin textures, and dark cinematic atmospheres.

---

## 🎞️ Aesthetic Profile

**Core Style:**  
1970s/1980s Grindhouse, B-Movie Horror, Slasher Film Stills.

**Key Textures:**  
- Heavy 35mm grain  
- Harsh camera flash  
- Deep casting shadows  
- Fake movie blood  
- Muted retro color grading

**Best Aspect Ratio:**  
Widescreen Cinematic (1216 x 832)

---

## 🚀 Usage & Inference

You can easily run this LoRA using the [diffusers](https://github.com/huggingface/diffusers) library in Python.

**Recommended:** For the best color accuracy, use the `madebyollin/sdxl-vae-fp16-fix` VAE.

### 1. Install Dependencies

```bash
pip install -U diffusers transformers accelerate
```

### 2. Generate an Image (Python Script)

```python
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

HF_MODEL_ID = "Ilyankhan69/slasher-vision-35mm"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Load FP16 VAE for color stability
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Initialize SDXL Pipeline
pipe = DiffusionPipeline.from_pretrained(
    BASE_MODEL,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Load the Slasher-Vision LoRA
pipe.load_lora_weights(HF_MODEL_ID, weight_name="pytorch_lora_weights.safetensors")

# Define the Prompt
prompt = (
    "A full-length wide-angle shot establishing a raw, grainy 35mm flash photograph of "
    "a female teenager with pale skin and unkempt messy hair. "
    "Harsh direct optical camera flash, heavy dramatic casting shadows, dark blood splatters visible on the skin. "
    "True 16:9 cinematic horizontal widescreen movie still frame layout, 35mm anamorphic lens, "
    "raw unpolished hyper-detailed skin textures, grim visceral high-tension atmosphere."
)

negative_prompt = (
    "close up, portrait, headshot, zoomed-in frame, smooth plastic skin, 3d render, "
    "cartoon, anime, illustration, artwork, low resolution, blurry, generic digital art"
)

# Generate
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=7.5,
        width=1216,   # Widescreen framing
        height=832,
        cross_attention_kwargs={"scale": 0.90} # LoRA style weight
    ).images[0]

image.save("grindhouse_still.png")
```

---

## 📝 Prompting Guide

To get the most out of **slasher-vision-35mm**, use these keywords in your prompts:

### **Recommended Positive Keywords**
- raw
- grainy 35mm flash photograph
- vintage 1970s grindhouse slasher movie still
- eerie dramatic shadows
- unpolished cinematography
- hyper-detailed skin textures

### **Recommended Negative Keywords**
- smooth plastic skin
- 3d render
- clean
- generic digital art
- cartoon
- anime
- perfect lighting

### **Framing Anchors**
- Because base SDXL has a strong bias toward close-up portraits, use these anchors to force full-body or wide shots:

**Positive:**  
`full-length wide-angle shot`, `establishing shot`, `full-body visible from head to toe`

**Dimensions:**  
Always use vertical (`832x1216`) or horizontal (`1216x832`) formats for cinematic framing.

---

## ⚙️ Training Details

- **Base Model:** stabilityai/stable-diffusion-xl-base-1.0
- **Dataset:** [takara-ai/MovieStills_Captioned_SmolVLM](https://www.kaggle.com/datasets/takara-ai/moviestills-captioned-smolvlm)
- **Resolution:** 1024x1024
- **Steps:** 1,000
- **Learning Rate:** 1e-4
- **Optimizer:** 8-bit Adam ([bitsandbytes](https://github.com/TimDettmers/bitsandbytes))
- **Hardware:** 2x NVIDIA T4 GPUs (Kaggle), gradient checkpointing & expandable_segments memory management

---

## ⚠️ Limitations & Safety

This LoRA inherits the anatomical knowledge and safety filters of SDXL 1.0.  
It primarily acts as a style filter for lighting and texture.

- **Explicit Requests:**  
  Explicit generation requests (e.g., NSFW/nudity) are governed by the base model's safety guardrails, often resulting in distorted geometry or cropping to face-only framing if triggered.

---
