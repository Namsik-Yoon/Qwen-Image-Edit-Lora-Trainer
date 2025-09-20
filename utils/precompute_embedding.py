from omegaconf import OmegaConf
import argparse
from diffusers import (
    QwenImageEditPipeline,       # 텍스트 임베딩용 encode_prompt 사용
    AutoencoderKLQwenImage,      # VAE 로드
)
import torch
import os
import math
from PIL import Image
from tqdm import tqdm
import numpy as np

# -----------------------
# Device / dtype
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.bfloat16  # 캐시 호환성 위해 고정(원하면 bf16/float16로 변경 가능)

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config yaml")
    return parser.parse_args().config

# -----------------------
# Utils
# -----------------------
def round_to_multiple_of_32(x: int) -> int:
    return int(round(x / 32) * 32)

def list_images(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in exts])

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    # [H,W,C] uint8 -> [C,H,W] float [-1,1]
    np_img = np.array(image.convert("RGB"))
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    torch_img = (torch_img * 2.0) - 1.0
    return torch_img

def resize_keep_ar_to_multiple_of_32(image: Image.Image, target_short: int | None) -> Image.Image:
    w, h = image.size
    if target_short and target_short > 0:
        if w <= h:
            new_w = target_short
            new_h = int(round(h * (new_w / w)))
        else:
            new_h = target_short
            new_w = int(round(w * (new_h / h)))
    else:
        new_w, new_h = w, h
    new_w = round_to_multiple_of_32(new_w)
    new_h = round_to_multiple_of_32(new_h)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image

def load_mask_image(mask_path: str) -> Image.Image:
    """
    마스크 규칙:
    - 흰색(255) = 편집/인페인트 영역
    - 검은색(0) = 보존 영역
    """
    m = Image.open(mask_path)
    # 다양한 포맷 지원: RGBA, L, RGB 등
    if m.mode in ("RGBA", "LA"):
        # alpha 채널 있으면 alpha를 마스크로 사용
        m = m.split()[-1]
    elif m.mode != "L":
        # RGB 등은 그레이스케일로 변환
        m = m.convert("L")
    return m

def mask_to_tensor(mask_img: Image.Image) -> torch.Tensor:
    """
    PIL(L, 0~255) -> [1,1,H,W] float(0~1)
    """
    arr = np.array(mask_img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr)[None, None, ...]  # [1,1,H,W]
    return ten

# -----------------------
# Precompute text embeddings (QwenImageEdit encode_prompt)
# -----------------------
def precompute_text_embeddings(cfg):
    """
    저장 구조:
    - 디스크: {output_dir}/cache/text_embs/{stem}.pt
      => dict(prompt_embeds, prompt_embeds_mask)
    - 메모리 반환: {stem: {...}, ...}
    """
    pipe = QwenImageEditPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        transformer=None,  # 텍스트 인코더만 쓰도록 큰 weight 로드는 최소화
        vae=None,
        torch_dtype=compute_dtype,
    ).to(device)

    txt_cache_dir = None
    cached = None
    if cfg.save_cache_on_disk:
        txt_cache_dir = os.path.join(cfg.output_dir, "cache", "text_embs")
        os.makedirs(txt_cache_dir, exist_ok=True)
    else:
        cached = {}

    img_dir = cfg.data_config.img_dir
    prompt_dir = cfg.data_config.prompt_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"prompt_dir not found: {prompt_dir}")

    image_names = list_images(img_dir)
    max_seq_len = int(getattr(cfg.data_config, "max_sequence_length", 256))

    for img_name in tqdm(image_names, desc="Precompute Text Embeds"):
        stem = os.path.splitext(img_name)[0]
        txt_path = os.path.join(prompt_dir, f"{stem}.txt")
        if not os.path.isfile(txt_path):
            continue

        img_path = os.path.join(img_dir, img_name)
        try:
            # 일부 구현은 encode_prompt에서 이미지 컨텍스트를 사용하므로 함께 전달
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        prompt_text = load_text(txt_path)
        with torch.no_grad():
            # 반환: (prompt_embeds, prompt_embeds_mask)
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                image=[img],                      # 컨텍스트 참조 가능(모델 구현에 따라)
                prompt=[prompt_text],           # 배치 가능
                device=pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=max_seq_len,
            )
            item = {
                "prompt_embeds": prompt_embeds[0].to("cpu").contiguous(),
                "prompt_embeds_mask": prompt_embeds_mask[0].to("cpu").contiguous(),
            }

        if cfg.save_cache_on_disk:
            torch.save(item, os.path.join(txt_cache_dir, f"{stem}.pt"))
        else:
            cached[stem] = item

    return cached

# -----------------------
# Precompute image embeddings (VAE latents) + optional inpaint masks
# -----------------------
def precompute_image_and_mask_embeddings(cfg):
    """
    저장 구조:
    - 이미지 라텐트: {output_dir}/cache/image_latents/{stem}.pt
      => dict(latents, size(W,H), scaling_factor)
    - (선택) 마스크: {output_dir}/cache/masks/{stem}.pt
      => dict(mask_image: [1,1,H,W], mask_latent: [1,1,H/8,W/8], size(W,H))
    """
    vae = AutoencoderKLQwenImage.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=compute_dtype,
    ).to(device)
    vae.eval()
    vae.enable_tiling()
    
    img_cache_dir = None
    mask_cache_dir = None
    cached_img = None

    if cfg.save_cache_on_disk:
        img_cache_dir = os.path.join(cfg.output_dir, "cache", "image_latents")
        os.makedirs(img_cache_dir, exist_ok=True)
        # 마스크 캐시 디렉토리 (선택)
        if hasattr(cfg.data_config, "mask_dir") and os.path.isdir(cfg.data_config.mask_dir):
            mask_cache_dir = os.path.join(cfg.output_dir, "cache", "masks")
            os.makedirs(mask_cache_dir, exist_ok=True)
    else:
        cached_img = {}

    img_dir = cfg.data_config.img_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    mask_dir = getattr(cfg.data_config, "mask_dir", None)
    img_size = getattr(cfg.data_config, "img_size", 1024)   # 짧은 변 기준 리사이즈
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)

    image_names = list_images(img_dir)

    for img_name in tqdm(image_names, desc="Precompute Image (VAE) Latents"):
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # 1) 리사이즈(32 배수)
        image = resize_keep_ar_to_multiple_of_32(image, img_size)
        W, H = image.size

        # 2) 텐서 변환 [-1,1], [1,T,C,H,W]
        image_tensor = pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=vae.dtype)
        image_tensor = image_tensor.unsqueeze(2)

        # 3) VAE encode -> latents
        with torch.no_grad():
            posterior = vae.encode(image_tensor.to(vae.dtype)).latent_dist
            latents = posterior.sample()

            # Qwen-Image 표준 정규화
            latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std  = (1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(1, vae.config.z_dim, 1, 1, 1)
            latents = (latents - latents_mean) * latents_std
            latents = latents.to("cpu").contiguous()

        img_item = {
            "latents": latents,                  # [1,T,C,H/8,W/8]
            "size": (W, H),                      # 리사이즈된 최종 해상도
            "scaling_factor": scaling_factor,
        }

        if cfg.save_cache_on_disk:
            torch.save(img_item, os.path.join(img_cache_dir, f"{stem}.pt"))
        else:
            cached_img[stem] = img_item

        # ----- (옵션) 인페인트 마스크 캐시 -----
        if mask_dir and os.path.isdir(mask_dir):
            mask_path = os.path.join(mask_dir, f"{stem}.png")
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(mask_dir, f"{stem}.jpg")
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(mask_dir, f"{stem}.jpeg")

            if os.path.isfile(mask_path):
                try:
                    raw_mask = load_mask_image(mask_path)
                    # 이미지와 동일 해상도로 리사이즈 (픽셀-공간 마스크)
                    mask_img_resized = raw_mask.resize((W, H), Image.NEAREST)
                    mask_image_tensor = mask_to_tensor(mask_img_resized)  # [1,1,H,W], float 0~1

                    # 라텐트 해상도(1/8)로 다운샘플링한 마스크
                    lat_h, lat_w = latents.shape[-2:]
                    mask_latent_img = mask_img_resized.resize((lat_w, lat_h), Image.NEAREST)
                    mask_latent_tensor = mask_to_tensor(mask_latent_img)  # [1,1,latH,latW]

                    mask_item = {
                        "mask_image": mask_image_tensor.to("cpu").contiguous(),
                        "mask_latent": mask_latent_tensor.to("cpu").contiguous(),
                        "size": (W, H),
                    }

                    if cfg.save_cache_on_disk and mask_cache_dir:
                        torch.save(mask_item, os.path.join(mask_cache_dir, f"{stem}.pt"))
                    # 메모리 캐시는 필요 시 확장
                except Exception:
                    pass

    return cached_img

# -----------------------
# Main
# -----------------------
def main():
    config_path = parse_args()
    cfg = OmegaConf.load(config_path)

    # 필수 키 확인
    for k in ["pretrained_model_name_or_path", "output_dir", "data_config"]:
        if k not in cfg:
            raise ValueError(f"Config missing required key: {k}")
    for k in ["img_dir", "prompt_dir"]:
        if k not in cfg.data_config:
            raise ValueError(f"Config.data_config missing required key: {k}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "cache"), exist_ok=True)

    if "save_cache_on_disk" not in cfg:
        cfg.save_cache_on_disk = True

    # 캐시 생성
    _ = precompute_text_embeddings(cfg)
    _ = precompute_image_and_mask_embeddings(cfg)

    print("✅ Precompute finished.")

if __name__ == "__main__":
    main()
