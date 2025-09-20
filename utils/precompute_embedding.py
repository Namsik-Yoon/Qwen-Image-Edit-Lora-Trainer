from omegaconf import OmegaConf
import argparse
from diffusers import (
    QwenImageInpaintPipeline,    # 인페인트 파이프라인: encode_prompt 사용
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
    - 흰색(255) = 인페인트(채워 넣을) 영역
    - 검은색(0) = 보존 영역
    """
    m = Image.open(mask_path)
    if m.mode in ("RGBA", "LA"):
        m = m.split()[-1]
    elif m.mode != "L":
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
# Precompute text embeddings (QwenImageInpaint encode_prompt)
# -----------------------
def precompute_text_embeddings(cfg):
    """
    저장 구조:
    - 디스크: {output_dir}/cache/text_embs/{stem}.pt
      => dict(prompt_embeds, prompt_embeds_mask)
    - 메모리 반환: {stem: {...}, ...}
    """
    pipe = QwenImageInpaintPipeline.from_pretrained(
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
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        prompt_text = load_text(txt_path)
        with torch.no_grad():
            try:
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                    image=[img],                      # 구현에 따라 이미지 컨텍스트 허용
                    prompt=[prompt_text],
                    device=pipe.device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_seq_len,
                )
            except TypeError:
                # encode_prompt가 image 인자를 받지 않는 구현일 경우 fallback
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                    prompt=[prompt_text],
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
# Precompute image embeddings (VAE latents) + inpaint masks & masked_image_latents
# -----------------------
def precompute_image_and_mask_embeddings(cfg):
    """
    저장 구조:
    - 이미지 라텐트: {output_dir}/cache/image_latents/{stem}.pt
      => dict(latents, size(W,H), scaling_factor)
    - 인페인트 마스크/마스킹 라텐트: {output_dir}/cache/masks/{stem}.pt
      => dict(
            mask_image: [1,1,H,W],
            mask_latent: [1,1,h,w],
            masked_image_latents: [1,T,C,h,w],
            size: (W,H)
         )

    참고: masked_image_latents는 픽셀 공간에서
          image_tensor * (1 - mask) + (-1 * mask) 로 “구멍”을 만들고
          VAE encode 한 결과를 저장한다. (-1은 [-1,1] 정규화에서 ‘검정’)
    """
    vae = AutoencoderKLQwenImage.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=compute_dtype,
    ).to(device)
    vae.eval()
    vae.enable_tiling()

    if cfg.save_cache_on_disk:
        img_cache_dir = os.path.join(cfg.output_dir, "cache", "image_latents")
        mask_cache_dir = os.path.join(cfg.output_dir, "cache", "masks")
        os.makedirs(img_cache_dir, exist_ok=True)
        os.makedirs(mask_cache_dir, exist_ok=True)
    else:
        img_cache_dir = mask_cache_dir = None

    cached_img = {} if not cfg.save_cache_on_disk else None

    img_dir = cfg.data_config.img_dir
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    mask_dir = getattr(cfg.data_config, "mask_dir", None)
    img_size = getattr(cfg.data_config, "img_size", 1024)   # 짧은 변 기준 리사이즈

    # VAE 스케일/정규화 파라미터
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    z_dim = getattr(vae.config, "z_dim", 16)
    latents_mean = torch.tensor(
        getattr(vae.config, "latents_mean", [0.0] * z_dim),
        device=device, dtype=vae.dtype
    ).view(1, z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(
        getattr(vae.config, "latents_std", [1.0] * z_dim),
        device=device, dtype=vae.dtype
    ).view(1, z_dim, 1, 1, 1)

    image_names = list_images(img_dir)

    for img_name in tqdm(image_names, desc="Precompute Image/Mask (VAE) Latents"):
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
        image_tensor = pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=vae.dtype)  # [1,C,H,W]
        image_tensor = image_tensor.unsqueeze(2)  # [1,1,C,H,W] — QwenImage VAE 입력 형태에 맞춤

        with torch.no_grad():
            # 3) VAE encode -> latents (원본)
            posterior = vae.encode(image_tensor).latent_dist
            latents = posterior.sample()
            latents = (latents - latents_mean) * latents_std
            latents = latents.to("cpu").contiguous()

        img_item = {
            "latents": latents,                  # [1,T,C,h,w]
            "size": (W, H),                      # 리사이즈된 최종 해상도
            "scaling_factor": scaling_factor,
        }

        if cfg.save_cache_on_disk:
            torch.save(img_item, os.path.join(img_cache_dir, f"{stem}.pt"))
        else:
            cached_img[stem] = img_item

        # ----- 인페인트 마스크/마스킹 라텐트 -----
        if mask_dir and os.path.isdir(mask_dir):
            # 파일명 우선순위 탐색
            mask_path = None
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                p = os.path.join(mask_dir, f"{stem}{ext}")
                if os.path.isfile(p):
                    mask_path = p
                    break

            if mask_path:
                try:
                    raw_mask = load_mask_image(mask_path)

                    # (A) 픽셀 공간 마스크(이미지 해상도)
                    mask_img_resized = raw_mask.resize((W, H), Image.NEAREST)
                    mask_image_tensor = mask_to_tensor(mask_img_resized).to(device=device, dtype=vae.dtype)  # [1,1,H,W]

                    # (B) 라텐트 해상도 마스크
                    #   라텐트 해상도는 VAE 다운샘플(보통 1/8)
                    #   먼저 원본 라텐트 크기를 알아야 하므로 위에서 만든 latents 기반으로 크기 추출
                    lat_h, lat_w = latents.shape[-2], latents.shape[-1]
                    mask_latent_img = mask_img_resized.resize((lat_w, lat_h), Image.NEAREST)
                    mask_latent_tensor = mask_to_tensor(mask_latent_img).to(device=device, dtype=vae.dtype)  # [1,1,latH,latW]

                    # (C) masked image 만들기 (픽셀 공간)
                    #   구멍(hole)은 -1(정규화 공간에서 검정)로 채움.
                    #   image_tensor: [1,1,C,H,W], mask_image_tensor: [1,1,H,W]
                    hole = (-1.0) * mask_image_tensor  # [1,1,H,W]
                    comp = image_tensor * (1.0 - mask_image_tensor) + hole  # 브로드캐스트로 채워짐

                    with torch.no_grad():
                        masked_post = vae.encode(comp).latent_dist
                        masked_image_latents = masked_post.sample()
                        masked_image_latents = (masked_image_latents - latents_mean) * latents_std
                        masked_image_latents = masked_image_latents.to("cpu").contiguous()

                    mask_item = {
                        "mask_image": mask_image_tensor.to("cpu").contiguous(),         # [1,1,H,W]
                        "mask_latent": mask_latent_tensor.to("cpu").contiguous(),       # [1,1,h,w]
                        "masked_image_latents": masked_image_latents,                   # [1,T,C,h,w]
                        "size": (W, H),
                    }

                    if cfg.save_cache_on_disk:
                        torch.save(mask_item, os.path.join(mask_cache_dir, f"{stem}.pt"))
                    # 필요시 메모리 캐시도 추가 가능

                except Exception as e:
                    # 마스크 처리 실패 시 건너뜀
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
