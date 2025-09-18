import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2  # [ADD]

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {"16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1)}
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h
    current_ratio = width / height
    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)
    cropped_img = image.crop(crop_box)
    return cropped_img

# [ADD] Green(0,255,0) mask extraction: np.uint8 HxWx3 → uint8 HxW (0/255)
def green_mask_from_rgb(rgb_np: np.ndarray, g_thr=200, r_thr=50, b_thr=50) -> np.ndarray:
    r, g, b = rgb_np[..., 0], rgb_np[..., 1], rgb_np[..., 2]
    m = ((g > g_thr) & (r < r_thr) & (b < b_thr)).astype(np.uint8) * 255
    return m


class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir,
        img_size=512,
        caption_type='txt',
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=None,
        cached_image_embeddings=None,
        control_dir=None,
        cached_image_embeddings_control=None
    ):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        print('cached_text_embeddings', type(cached_text_embeddings))

    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            img_path = self.images[idx]
            base = os.path.basename(img_path)

            # ------------ (1) target img / latent ------------
            if self.cached_image_embeddings is None:
                img = Image.open(img_path).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
            else:
                img = self.cached_image_embeddings[base]

            # ------------ (2) control RGB → control_img + control_mask_img ------------
            # control image is "original + green(0,255,0) overlay" image (use as is)
            ctrl_rgb = Image.open(os.path.join(self.control_dir, base)).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    ctrl_rgb = crop_to_aspect_ratio(ctrl_rgb, ratio)

            ctrl_rgb = image_resize(ctrl_rgb, self.img_size)
            w, h = ctrl_rgb.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            ctrl_rgb = ctrl_rgb.resize((new_w, new_h))

            # control_img for model input ([-1,1], C,H,W)
            control_img = torch.from_numpy((np.array(ctrl_rgb) / 127.5) - 1).permute(2, 0, 1)

            # Binary mask for loss weighting [1,H,W] float32 in [0,1]
            ctrl_np = np.array(ctrl_rgb)  # uint8
            m_bin = green_mask_from_rgb(ctrl_np)                # 0/255
            m_f = (m_bin.astype(np.float32) / 255.0)[None, ...] # [1,H,W]
            control_mask_img = torch.from_numpy(m_f)            # float32

            # ------------ (3) prompt / prompt_embeds ------------
            txt_path = img_path.split('.')[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None:
                prompt = open(txt_path, encoding='utf-8').read()
                if throw_one(self.caption_dropout_rate):
                    # Caption drop: empty space
                    return img, " ", control_img, control_mask_img
                else:
                    return img, prompt, control_img, control_mask_img
            else:
                txt = os.path.basename(txt_path)
                if throw_one(self.caption_dropout_rate):
                    return (
                        img,
                        self.cached_text_embeddings[txt + 'empty_embedding']['prompt_embeds'],
                        self.cached_text_embeddings[txt + 'empty_embedding']['prompt_embeds_mask'],
                        control_img,
                        control_mask_img,
                    )
                else:
                    return (
                        img,
                        self.cached_text_embeddings[txt]['prompt_embeds'],
                        self.cached_text_embeddings[txt]['prompt_embeds_mask'],
                        control_img,
                        control_mask_img,
                    )

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
