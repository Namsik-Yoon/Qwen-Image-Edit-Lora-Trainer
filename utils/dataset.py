import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _load_pt(path: str):
    return torch.load(path, map_location="cpu")

class PrecomputedInpaintDataset(Dataset):
    """
    Returns:
      pixel_latents (C,T,h,w), prompt_embeds (L,D), prompt_embeds_mask (L,),
      control_latents (C,T,h,w), mask_latent (1,h,w)
    """
    def __init__(
        self,
        img_dir: str,
        # Optional args: infer from output_dir if not provided
        text_cache_dir: str | None = None,
        image_latents_dir: str | None = None,
        masks_cache_dir: str | None = None,
        # Base path needed for inference
        output_dir: str | None = None,
        cache_dir: str | None = None,  # Priority when directly specified
        caption_dropout_rate: float = 0.0,
        use_infinite: bool = True,
        **unused,   # Ignore train_batch_size, num_workers, img_size etc.
    ):
        # ---- Auto-infer cache directories ----
        if cache_dir is None:
            if output_dir is None and (text_cache_dir is None or image_latents_dir is None or masks_cache_dir is None):
                raise ValueError(
                    "Either provide (text_cache_dir, image_latents_dir, masks_cache_dir) "
                    "or provide output_dir (so I can infer cache paths)."
                )
            base_cache = os.path.join(output_dir, "cache") if output_dir else None
        else:
            base_cache = cache_dir

        if text_cache_dir is None:
            text_cache_dir = os.path.join(base_cache, "text_embs")
        if image_latents_dir is None:
            image_latents_dir = os.path.join(base_cache, "image_latents")
        if masks_cache_dir is None:
            masks_cache_dir = os.path.join(base_cache, "masks")

        self.img_dir = img_dir
        self.text_cache_dir = text_cache_dir
        self.image_latents_dir = image_latents_dir
        self.masks_cache_dir = masks_cache_dir
        self.caption_dropout_rate = float(caption_dropout_rate)
        self.use_infinite = use_infinite

        # ---- Collect valid stems ----
        stems = [os.path.splitext(n)[0] for n in os.listdir(self.image_latents_dir) if n.lower().endswith(".pt")]
        stems.sort()
        valid = []
        for s in stems:
            if (os.path.isfile(os.path.join(self.image_latents_dir, f"{s}.pt"))
                and os.path.isfile(os.path.join(self.text_cache_dir, f"{s}.pt"))
                and os.path.isfile(os.path.join(self.masks_cache_dir, f"{s}.pt"))):
                valid.append(s)

        if not valid:
            raise RuntimeError(
                "No valid samples found.\n"
                f"  image_latents_dir: {self.image_latents_dir}\n"
                f"  text_cache_dir   : {self.text_cache_dir}\n"
                f"  masks_cache_dir  : {self.masks_cache_dir}\n"
                "Check that files are saved as <stem>.pt in each directory."
            )

        self.stems = valid
        print(f"[Dataset] valid samples: {len(self.stems)}")

        # optional empty prompt cache
        self.global_empty_txt = None
        for cand in ["_empty.pt", "empty.pt"]:
            p = os.path.join(self.text_cache_dir, cand)
            if os.path.isfile(p):
                self.global_empty_txt = p
                print(f"[Dataset] found global empty prompt cache: {cand}")
                break

    def __len__(self):
        return 10**9 if self.use_infinite else len(self.stems)

    def _choose_stem(self, idx):
        return random.choice(self.stems) if self.use_infinite else self.stems[idx]

    def _load_text_embeds(self, stem: str, do_drop: bool):
        base = os.path.join(self.text_cache_dir, f"{stem}.pt")
        empty_local = os.path.join(self.text_cache_dir, f"{stem}__empty.pt")
        if do_drop and os.path.isfile(empty_local):
            obj = _load_pt(empty_local)
        elif do_drop and self.global_empty_txt is not None:
            obj = _load_pt(self.global_empty_txt)
        else:
            obj = _load_pt(base)
        pe = obj["prompt_embeds"]
        pm = obj["prompt_embeds_mask"]
        if pe.dim() == 3 and pe.size(0) == 1: pe = pe[0]
        if pm.dim() == 2 and pm.size(0) == 1: pm = pm[0]
        return pe.contiguous(), pm.to(dtype=torch.int64).contiguous()

    def _load_image_latents(self, stem: str):
        lat = _load_pt(os.path.join(self.image_latents_dir, f"{stem}.pt"))["latents"]  # [1,T,C,h,w]
        lat = lat[0].permute(1, 0, 2, 3).contiguous()  # (C,T,h,w)
        return lat

    def _load_mask_and_control(self, stem: str):
        obj = _load_pt(os.path.join(self.masks_cache_dir, f"{stem}.pt"))
        mlat = obj.get("mask_latent", obj.get("mask_image"))
        if mlat is None:
            raise RuntimeError(f"mask_latent not found in {stem}.pt")
        if mlat.dim() == 4 and mlat.size(0) == 1 and mlat.size(1) == 1:
            mlat = mlat[0]  # (1,h,w)
        ctrl = obj["masked_image_latents"]  # [1,T,C,h,w]
        ctrl = ctrl[0].permute(1, 0, 2, 3).contiguous()  # (C,T,h,w)
        return mlat.clamp(0, 1).contiguous(), ctrl

    def __getitem__(self, idx):
        try:
            stem = self._choose_stem(idx)
            do_drop = (random.random() < self.caption_dropout_rate)
            prompt_embeds, prompt_mask = self._load_text_embeds(stem, do_drop)
            pixel_latents = self._load_image_latents(stem)
            mask_latent, control_latents = self._load_mask_and_control(stem)
            return pixel_latents, prompt_embeds, prompt_mask, control_latents, mask_latent
        except Exception as e:
            print(f"[Dataset] error @ idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.stems) - 1))

def loader(**cfg):
    """
    cfg example:
      {
        "img_dir": ".../images",
        # (Optional) Infer from output_dir if below are not provided:
        "text_cache_dir": ".../cache/text_embs",
        "image_latents_dir": ".../cache/image_latents",
        "masks_cache_dir": ".../cache/masks",
        # dataloader options:
        "train_batch_size": 1,
        "num_workers": 0,
        # dataset options:
        "caption_dropout_rate": 0.1,
        "use_infinite": True,
        # (Optional) output_dir: for inference
        "output_dir": ".../outputs/exp1"
      }
    """
    batch_size = int(cfg.pop("train_batch_size", 1))
    num_workers = int(cfg.pop("num_workers", 0))

    dataset = PrecomputedInpaintDataset(**cfg)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(num_workers, 2),  # Prevent CPU OOM: recommend 0~2
        shuffle=True,
        pin_memory=False,                 # CPU tight → False
        prefetch_factor=1,
        persistent_workers=False,
        drop_last=True,
        multiprocessing_context="spawn",
    )
