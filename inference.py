import argparse
import os
from PIL import Image
import torch
from diffusers import (
    QwenImageInpaintPipeline,
    QwenImageTransformer2DModel,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)

def find_mask(mask_dir: str, stem: str):
    if not mask_dir:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[Device] {device}, dtype={torch_dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load basic pipeline (most stable: batch load including text encoder/vae)
    pipe = QwenImageInpaintPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )

    # 2) (Optional) Replace only transformer with 4-bit quantization
    if args.quantize_transformer:
        print("[Quantize] Loading transformer in 4-bit NF4…")
        quant_cfg = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.model_name,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            quantization_config=quant_cfg,
        )
        pipe.transformer = transformer  # Inject replacement into pipeline

    # 3) Load LoRA weights
    if args.lora_weights:
        print(f"[LoRA] Loading: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights)

    # 4) Optimization options
    if device == "cuda":
        pipe.to(device)
        if args.attention_slicing:
            pipe.enable_attention_slicing()
        if args.vae_tiling:
            pipe.vae.enable_tiling()
        if args.cpu_offload:
            # CPU offloading saves VRAM but reduces speed. Use only when needed.
            pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    # 5) Iterate through input images
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    names = sorted([n for n in os.listdir(args.input_dir) if os.path.splitext(n)[1].lower() in exts])

    if not names:
        print(f"[Warn] No images in: {args.input_dir}")
        return

    print(f"[Run] {len(names)} images")
    for idx, name in enumerate(names, start=0):
        stem, _ = os.path.splitext(name)
        img_path = os.path.join(args.input_dir, name)
        mask_path = find_mask(args.mask_dir, stem)

        if not mask_path:
            print(f"[Skip] mask not found for {name} (expected in {args.mask_dir})")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # Accept various formats: use alpha if available, otherwise convert to L
        if mask.mode in ("RGBA", "LA"):
            mask = mask.split()[-1]
        elif mask.mode != "L":
            mask = mask.convert("L")

        # Fix seed (use idx to make different for each image)
        generator = torch.Generator(device=device)
        if args.seed >= 0:
            generator = generator.manual_seed(args.seed + idx)

        print(f"[Gen] {name} with mask {os.path.basename(mask_path)}")
        out = pipe(
            image=image,
            mask_image=mask,                 # White(255)=edit, Black(0)=preserve
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
        )
        out_img = out.images[0]
        save_path = os.path.join(args.output_dir, f"{stem}_inpaint.png")
        out_img.save(save_path)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    neg_default = (
        "cartoon, anime, CGI, 3D render, illustration, painting, unreal engine, "
        "lowres, blurry, out of focus, motion blur, jpeg artifacts, "
        "watermark, text overlay, duplicate objects, warped geometry, distorted perspective, "
        "melted metal, pattern tiling, repetitive textures, graffiti, "
        "background changes, lighting changes, color cast"
    )

    p = argparse.ArgumentParser()
    # Model/weights/paths
    p.add_argument("--model_name", type=str, default="Qwen/Qwen-Image-Edit")
    p.add_argument("--lora_weights", type=str, default="", help="LoRA checkpoint directory")
    p.add_argument("--input_dir", type=str, required=True, help="Folder of input images")
    p.add_argument("--mask_dir", type=str, required=True, help="Folder of masks (same stem as images)")
    p.add_argument("--output_dir", type=str, required=True, help="Folder to save results")

    # Generation parameters
    p.add_argument("--prompt", type=str, default="man in the city")
    p.add_argument("--negative_prompt", type=str, default=neg_default)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--true_cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # Optimization options
    p.add_argument("--quantize_transformer", action="store_true", help="Use 4-bit NF4 for transformer")
    p.add_argument("--attention_slicing", action="store_true")
    p.add_argument("--vae_tiling", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")

    args = p.parse_args()
    main(args)
