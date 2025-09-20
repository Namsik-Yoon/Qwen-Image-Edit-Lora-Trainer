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

    # 1) 기본 파이프라인 로드 (가장 안정적: 텍스트 인코더/vae 포함 일괄 로드)
    pipe = QwenImageInpaintPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    )

    # 2) (옵션) 트랜스포머만 4bit 양자화로 교체
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
        pipe.transformer = transformer  # 파이프라인에 교체 주입

    # 3) LoRA 가중치 로드
    if args.lora_weights:
        print(f"[LoRA] Loading: {args.lora_weights}")
        pipe.load_lora_weights(args.lora_weights)

    # 4) 최적화 옵션
    if device == "cuda":
        pipe.to(device)
        if args.attention_slicing:
            pipe.enable_attention_slicing()
        if args.vae_tiling:
            pipe.vae.enable_tiling()
        if args.cpu_offload:
            # CPU 오프로딩은 VRAM 절약 대신 속도 저하. 필요할 때만.
            pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    # 5) 입력 이미지 순회
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
        # 다양한 포맷 수용: 알파 있으면 알파를 사용, 아니면 L로 변환
        if mask.mode in ("RGBA", "LA"):
            mask = mask.split()[-1]
        elif mask.mode != "L":
            mask = mask.convert("L")

        # 시드 고정(이미지별로 다르게 하려면 idx 사용)
        generator = torch.Generator(device=device)
        if args.seed >= 0:
            generator = generator.manual_seed(args.seed + idx)

        print(f"[Gen] {name} with mask {os.path.basename(mask_path)}")
        out = pipe(
            image=image,
            mask_image=mask,                 # 흰색(255)=수정, 검정(0)=보존
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
    # 모델/가중치/경로
    p.add_argument("--model_name", type=str, default="Qwen/Qwen-Image-Edit")
    p.add_argument("--lora_weights", type=str, default="", help="LoRA checkpoint directory")
    p.add_argument("--input_dir", type=str, required=True, help="Folder of input images")
    p.add_argument("--mask_dir", type=str, required=True, help="Folder of masks (same stem as images)")
    p.add_argument("--output_dir", type=str, required=True, help="Folder to save results")

    # 생성 파라미터
    p.add_argument("--prompt", type=str, default="man in the city")
    p.add_argument("--negative_prompt", type=str, default=neg_default)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--true_cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)

    # 최적화 옵션
    p.add_argument("--quantize_transformer", action="store_true", help="Use 4-bit NF4 for transformer")
    p.add_argument("--attention_slicing", action="store_true")
    p.add_argument("--vae_tiling", action="store_true")
    p.add_argument("--cpu_offload", action="store_true")

    args = p.parse_args()
    main(args)
