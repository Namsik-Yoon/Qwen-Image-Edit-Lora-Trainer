import argparse
import cv2  # [ADD] for green-mask extraction
import copy
from copy import deepcopy
import logging
import os
import shutil

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from utils.dataset import loader, image_resize
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from PIL import Image
from PIL import ImageDraw, ImageFont
import textwrap
import numpy as np
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb
logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPipeline
import gc
import math
from PIL import Image
import numpy as np
import psutil
import GPUtil
from torchvision.transforms import ToTensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import json
from datetime import datetime
import re  # [ADD] (skip if already exists)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.data = torch.randn(num_samples, input_dim)    # random features
        self.labels = torch.randint(0, 2, (num_samples,))  # random labels: 0 or 1

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

def get_gpu_memory_usage():
    """Return GPU memory usage"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_memory_percent': gpu.memoryUtil * 100
            }
    except:
        pass
    return {'gpu_memory_used': 0, 'gpu_memory_total': 0, 'gpu_memory_percent': 0}

def get_system_metrics():
    """Return system metrics"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }

def calculate_lora_statistics(model):
    """Calculate LoRA parameter statistics"""
    lora_params = []
    lora_grad_norms = []
    
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params.append(param.data.flatten())
            if param.grad is not None:
                lora_grad_norms.append(param.grad.data.norm().item())
    
    if lora_params:
        all_params = torch.cat(lora_params)
        return {
            'lora_param_count': sum(p.numel() for p in lora_params),
            'lora_param_mean': all_params.mean().item(),
            'lora_param_std': all_params.std().item(),
            'lora_param_max': all_params.max().item(),
            'lora_param_min': all_params.min().item(),
            'lora_grad_norm_mean': np.mean(lora_grad_norms) if lora_grad_norms else 0,
            'lora_grad_norm_max': np.max(lora_grad_norms) if lora_grad_norms else 0
        }
    return {}


def log_images_to_tensorboard(accelerator, step, input_img, generated_img, target_img, prompt, cfg_scale):
    """Log images to TensorBoard"""
    try:
        from torchvision.utils import make_grid
        import torchvision.transforms as transforms
        import torchvision.transforms.functional as TF
        from PIL import Image, ImageDraw, ImageFont
        import textwrap
        
        # Convert images to tensors
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        images_to_log = []
        image_labels = []
        
        # Input image (control folder)
        if input_img is not None:
            input_tensor = transform(input_img).unsqueeze(0)
            images_to_log.append(input_tensor)
            image_labels.append("Input (Control)")
        
        # Generated image
        if generated_img is not None:
            generated_tensor = transform(generated_img).unsqueeze(0)
            images_to_log.append(generated_tensor)
            image_labels.append("Generated")
            
        # Target image (original from images_edit folder)
        if target_img is not None:
            target_tensor = transform(target_img).unsqueeze(0)
            images_to_log.append(target_tensor)
            image_labels.append("Target (Original)")
        
        if images_to_log:
            # Generate prompt text image
            prompt_img = create_prompt_image(prompt, width=512 * len(images_to_log), height=100)
            prompt_tensor = transform(prompt_img).unsqueeze(0)
            
            # Combine all images vertically (images + prompt)
            all_images = torch.cat(images_to_log, dim=0)
            combined_images = torch.cat([all_images, prompt_tensor], dim=0)
            
            # Combine into grid (images horizontally, prompt below)
            grid = make_grid(combined_images, nrow=len(images_to_log), padding=10)
            
            # Log to TensorBoard
            accelerator.log({
                "images/input_generated_target_comparison": grid,
                "inference/prompt": prompt,
                "inference/cfg_scale": cfg_scale,
                "inference/image_labels": " | ".join(image_labels)
            }, step=step)
            
    except Exception as e:
        logger.warning(f"TensorBoard image logging failed: {e}")

def create_prompt_image(prompt, width=1536, height=100):
    """Generate prompt text as white background image"""
    try:
        # Generate white background image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Font settings (use default font)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Handle text line wrapping
        max_chars_per_line = width // 8  # Approximate character count
        wrapped_text = textwrap.fill(prompt, width=max_chars_per_line)
        
        # Draw text in the center of the image
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), wrapped_text, fill='black', font=font)
        
        return img
        
    except Exception as e:
        logger.warning(f"Prompt image generation failed: {e}")
        # Return default white background image on failure
        return Image.new('RGB', (width, height), color='white')

def log_comprehensive_metrics(accelerator, step, model, generated_img=None, target_img=None, device=None):
    """Log comprehensive metrics to TensorBoard"""
    try:
        metrics = {}
        
        # LoRA statistics
        lora_stats = calculate_lora_statistics(model)
        for key, value in lora_stats.items():
            metrics[f"lora/{key}"] = value
        
        # GPU memory usage
        gpu_metrics = get_gpu_memory_usage()
        for key, value in gpu_metrics.items():
            metrics[f"system/gpu_{key}"] = value
        
        # System metrics
        system_metrics = get_system_metrics()
        for key, value in system_metrics.items():
            metrics[f"system/{key}"] = value
        
        # Log to TensorBoard
        if metrics:
            accelerator.log(metrics, step=step)
            
    except Exception as e:
        logger.warning(f"Metrics logging failed: {e}")

def run_inference_after_checkpoint(args, accelerator, global_step, weight_dtype, flux_transformer, checkpoint_path):
    """Run inference after checkpoint save (OOM prevention)"""
    try:
        # Check if inference should be run
        test_configs = getattr(args, 'inference_config', {})
        enable_inference = test_configs.get('enable_inference', True)
        
        if not enable_inference:
            logger.info("Inference execution disabled, skipping")
            return
        
        logger.info("Running inference from checkpoint...")
        
        # Remove existing training model from memory (OOM prevention)
        logger.info("Removing existing training model from memory...")
        del flux_transformer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Additional memory cleanup
        import time
        time.sleep(2)  # Wait for memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create inference pipeline (refer to inference_edit.py)
        inference_pipeline = QwenImageEditPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        inference_pipeline.to("cuda")
        
        # Load LoRA weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading LoRA weights from checkpoint: {checkpoint_path}")
            inference_pipeline.load_lora_weights(checkpoint_path)
        
        # Offload model to CPU to save VRAM, parts will be moved to GPU as needed
        inference_pipeline.enable_model_cpu_offload()
        
        # Set up test images and prompts
        test_configs = getattr(args, 'inference_config', {})
        test_images = test_configs.get('test_images', [])
        num_inference_steps = test_configs.get('num_inference_steps', 50)
        cfg_scale = test_configs.get('cfg_scale', 7.0)
        seed = test_configs.get('seed', 42)
        
        # If no test images, randomly select 3 from control folder
        if not test_images and os.path.exists(args.data_config.control_dir):
            available_images = [f for f in os.listdir(args.data_config.control_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if available_images:
                # Randomly select 3 (maximum 3)
                import random
                selected_images = random.sample(available_images, min(3, len(available_images)))
                test_images = [os.path.join(args.data_config.control_dir, img) for img in selected_images]
        
        # Read prompts from txt files for each test image
        test_prompts = []
        for test_img_path in test_images:
            if os.path.exists(test_img_path):
                # Remove extension from image filename and add .txt
                img_name = os.path.basename(test_img_path)
                img_name_without_ext = os.path.splitext(img_name)[0]
                txt_path = os.path.join(args.data_config.img_dir, img_name_without_ext + '.txt')
                
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            prompt = f.read().strip()
                        test_prompts.append(prompt)
                    except Exception as e:
                        logger.warning(f"Failed to read prompt file {txt_path}: {e}")
                        test_prompts.append("man in the city")  # Default prompt
                else:
                    logger.warning(f"Prompt file does not exist: {txt_path}")
                    test_prompts.append("man in the city")  # Default prompt
        
        # Run inference (refer to inference_edit.py)
        for i, (test_img_path, test_prompt) in enumerate(zip(test_images, test_prompts)):
            if os.path.exists(test_img_path):
                try:
                    # Load input image
                    input_img = Image.open(test_img_path).convert("RGB")
                    
                    # Set up generator
                    generator = torch.Generator(device="cuda").manual_seed(seed + i)
                    
                    # Run inference
                    with torch.no_grad():
                        generated_img = inference_pipeline(
                            image=input_img,
                            prompt=test_prompt,
                            num_inference_steps=num_inference_steps,
                            true_cfg_scale=cfg_scale,
                            generator=generator,
                        ).images[0]
                    
                    prompt = test_prompt
                except Exception as e:
                    logger.error(f"Inference execution failed: {e}")
                    generated_img, input_img, prompt = None, None, None
                
                if generated_img is not None:
                    # Find target image (from images_edit folder)
                    target_img = None
                    img_name = os.path.basename(test_img_path)
                    target_path = os.path.join(args.data_config.img_dir, img_name)
                    if os.path.exists(target_path):
                        target_img = Image.open(target_path).convert("RGB")
                    
                    # Log to TensorBoard (including input, generated, and target images)
                    log_images_to_tensorboard(
                        accelerator, global_step, input_img, generated_img, target_img, prompt, cfg_scale
                    )
                    
                    logger.info(f"Inference results logged to TensorBoard (image {i+1})")
        
        # Memory cleanup
        del inference_pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error occurred during inference execution: {e}")

def reload_model_after_inference(args, accelerator, weight_dtype, lora_config, optimizer, lr_scheduler, lora_layers_model, checkpoint_path):
    """Reload model after inference to resume training"""
    try:
        logger.info("Reloading model after inference completion...")
        
        # Additional memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        import time
        time.sleep(2)
        
        # Reload model
        flux_transformer = QwenImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
        )
        
        # Apply quantization (if configured)
        if args.quantize:
            torch_dtype = weight_dtype
            device = accelerator.device
            all_blocks = list(flux_transformer.transformer_blocks)
            K = getattr(args, "quantize_last_k_blocks", 0) # New config value (e.g., 8)
            selected_blocks = all_blocks[-K:] if K > 0 else []

            for block in tqdm(selected_blocks, desc=f"Quantizing last {K} blocks"):
                block.to(device, dtype=torch_dtype)
                quantize(block, weights=qfloat8)
                freeze(block)
                block.to(device)

            flux_transformer.to(device, dtype=torch_dtype)
            # quantize(flux_transformer, weights=qfloat8)
            # freeze(flux_transformer)
        
        # Add LoRA adapter
        flux_transformer.to(accelerator.device)
        flux_transformer.add_adapter(lora_config)
        
        # Load saved LoRA weights
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading LoRA weights from checkpoint: {checkpoint_path}")
            try:
                # Load LoRA weights
                from diffusers import QwenImagePipeline
                QwenImagePipeline.load_lora_weights(flux_transformer, checkpoint_path)
                logger.info("LoRA weights loading completed")
            except Exception as e:
                logger.warning(f"LoRA weights loading failed: {e}")
        
        # Training setup
        flux_transformer.requires_grad_(False)
        flux_transformer.train()
        
        # Set only LoRA parameters to be trainable
        for n, param in flux_transformer.named_parameters():
            if 'lora' not in n:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Enable gradient checkpointing
        flux_transformer.enable_gradient_checkpointing()
        
        # Connect optimizer and scheduler to new model
        lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
        lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
        
        # Prepare again with Accelerator
        lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
            lora_layers_model, optimizer, None, lr_scheduler
        )
        
        logger.info("Model reload completed, resuming training")
        return flux_transformer, lora_layers_model, optimizer, lr_scheduler
        
    except Exception as e:
        logger.error(f"Error occurred during model reload: {e}")
        return None, None, None, None

# [ADD] --- Green mask → soft mask (latent resolution) generation utility ---
def extract_green_mask_from_rgb(img_pil, down_h, down_w, blur_ks=11, blur_sigma=3.0):
    """
    img_pil: PIL.Image in RGB (control image)
    (down_h, down_w): latent resolution
    return: torch.FloatTensor (1,1,down_h,down_w) in [0,1]
    """
    rgb = np.array(img_pil.convert("RGB"))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (g > 200) & (r < 50) & (b < 50)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (down_w, down_h), interpolation=cv2.INTER_AREA)
    if blur_ks > 0:
        if blur_ks % 2 == 0:
            blur_ks += 1
        mask = cv2.GaussianBlur(mask, (blur_ks, blur_ks), blur_sigma)
    m = torch.from_numpy(mask).float() / 255.0
    m = m.clamp(0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
    return m

def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=[args.report_to],
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_warning()
    #     diffusers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()
    #     diffusers.utils.logging.set_verbosity_error()
    
    if args.precompute_text_embeddings:
        precompute_text_embeddings(args)
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=args.
    )
    text_encoding_pipeline.to(accelerator.device)
    cached_text_embeddings = None
    txt_cache_dir = None
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        if accelerator.is_main_process:
            cache_dir = os.path.join(args.output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
    if args.precompute_text_embeddings:
        with torch.no_grad():
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
            else:
                cached_text_embeddings = {}
            for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or '.jpg' in i]):
                img_path = os.path.join(args.data_config.control_dir, img_name)
                txt_path = os.path.join(args.data_config.img_dir, ".".join(img_name.split('.')[:-1]) + '.txt')

                img = Image.open(img_path).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                prompt_image = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
                
                prompt = open(txt_path, encoding='utf-8').read()
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )
                if args.save_cache_on_disk:
                    torch.save({'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}, os.path.join(txt_cache_dir, img_name.split('.')[0] + '.pt'))
                else:
                    cached_text_embeddings[img_name.split('.')[0] + '.txt'] = {'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}
            # compute empty embedding
                prompt_embeds_empty, prompt_embeds_mask_empty = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[' '],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )
                cached_text_embeddings[img_name.split('.')[0] + '.txt' + 'empty_embedding'] = {'prompt_embeds': prompt_embeds_empty[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu')}
                    


    
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    cached_image_embeddings = None
    img_cache_dir = None
    cached_image_embeddings_control = None
    if args.precompute_image_embeddings:
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs")
            os.makedirs(img_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings = {}
        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir) if ".png" in i or ".jpg" in i]):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings[img_name] = pixel_latents
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs_control")
            os.makedirs(img_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings_control = {}
        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or ".jpg" in i]):
                img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
                img = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype).to(accelerator.device)
        
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings_control[img_name] = pixel_latents
        vae.to('cuda')
        torch.cuda.empty_cache()
        text_encoding_pipeline.to("cuda")
        torch.cuda.empty_cache()
    del text_encoding_pipeline
    gc.collect()
    #del vae
    gc.collect()
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",    )
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        K = getattr(args, "quantize_last_k_blocks", 0) # 새 config 값 (예: 8)
        selected_blocks = all_blocks[-K:] if K > 0 else []

        for block in tqdm(selected_blocks, desc=f"Quantizing last {K} blocks"):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to(device)
        flux_transformer.to(device, dtype=torch_dtype)
        # quantize(flux_transformer, weights=qfloat8)
        # freeze(flux_transformer)
        #quantize(flux_transformer, weights=qint8, activations=qint8)
        #freeze(flux_transformer)
        
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    flux_transformer.to(accelerator.device)
    #flux_transformer.add_adapter(lora_config)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    flux_transformer.requires_grad_(False)


    flux_transformer.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
            pass
        else:
            param.requires_grad = True
            print(n)
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),)
    else:
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    train_dataloader = loader(
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings, 
        cached_image_embeddings_control=cached_image_embeddings_control,
        # [CHANGE] If possible, modify loader to support flags like 'return_control_mask=True'
        # And make it receive control_mask (or control_rgb_path) additionally in the batch.
        **args.data_config
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    lora_layers_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    lora_layers_model, optimizer, train_dataloader, lr_scheduler
)

    initial_global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    vae_scale_factor = 8
    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask, control_img, control_mask_img = batch
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device)
                    prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                    control_img = control_img.to(dtype=weight_dtype).to(accelerator.device)
                    
                else:
                    img, prompts, control_img, control_mask_img = batch
                with torch.no_grad():
                    if not args.precompute_image_embeddings:
                        # ----- img → latents -----
                        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)   # (B,C,H,W)
                        pixel_values = pixel_values.unsqueeze(2)                            # (B,C,1,H,W)
                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()       # (B,1,z,H',W')

                        # ----- control_img → latents (only for RGB) -----
                        if control_img.dim() == 4:  # (B,C,H,W) RGB tensor
                            control_values = control_img.to(dtype=weight_dtype).to(accelerator.device).unsqueeze(2)  # (B,C,1,H,W)
                            control_latents = vae.encode(control_values).latent_dist.sample()                         # (B,1,z,H',W')
                        else:
                            # If already latent
                            control_latents = control_img.to(dtype=weight_dtype).to(accelerator.device)

                    else:
                        # img is already latent cache
                        pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)
                        # control could be latent cache or RGB → check dimensions
                        if control_img.dim() == 4:  # If RGB, convert with VAE
                            control_values = control_img.to(dtype=weight_dtype).to(accelerator.device).unsqueeze(2)
                            control_latents = vae.encode(control_values).latent_dist.sample()
                        else:
                            control_latents = control_img.to(dtype=weight_dtype).to(accelerator.device)

                    pixel_latents  = pixel_latents.permute(0, 2, 1, 3, 4)   # (B,z,1,H',W')
                    control_latents = control_latents.permute(0, 2, 1, 3, 4)
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents  = (pixel_latents  - latents_mean) * latents_std
                    control_latents = (control_latents - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                # Concatenate across channels.
                # pack the latents.
                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control_img = QwenImageEditPipeline._pack_latents(
                    control_latents,
                    bsz, 
                    control_latents.shape[2],
                    control_latents.shape[3],
                    control_latents.shape[4],
                )
                # latent image ids for RoPE.
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2)]] * bsz
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
                with torch.no_grad():
                    if not args.precompute_text_embeddings:
                        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                            prompt=prompts,
                            device=packed_noisy_model_input.device,
                            num_images_per_prompt=1,
                            max_sequence_length=256,
                        )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # --- 1) Prepare mask at latent scale ---
                H_lat, W_lat = noisy_model_input.shape[-2], noisy_model_input.shape[-1]
                mask_soft_b = F.interpolate(
                    control_mask_img.to(device=pixel_latents.device, dtype=pixel_latents.dtype),
                    size=(H_lat, W_lat),
                    mode="area",
                )  # (B,1,H',W')

                alpha_fg  = float(getattr(args, "mask_fg_boost", 2.0))
                lambda_bg = float(getattr(args, "mask_bg_scale", 1.0))
                id_lambda = float(getattr(args, "lambda_identity_bg", 0.0))

                # --- 2) Calculate diffusion weighting ---
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

                # --- 3) Calculate target (needed for expand_as) ---
                target = (noise - pixel_latents).permute(0, 2, 1, 3, 4)  # (B,z,T?,H',W')

                # --- 4) Create and apply spatial weights ---
                spatial_weights = mask_soft_b * alpha_fg + (1.0 - mask_soft_b) * lambda_bg  # (B,1,H',W')
                spatial_weights = spatial_weights.unsqueeze(2).expand_as(target)            # (B,z,T?,H',W')
                weighting = weighting * spatial_weights

                # --- 5) Final loss ---
                diff = (model_pred.float() - target.float())
                loss = torch.mean((weighting.float() * diff**2).reshape(target.shape[0], -1), dim=1).mean()

                # --- 6) (Optional) Background identity regularization ---
                if id_lambda > 0:
                    bg_mask = (1.0 - mask_soft_b).unsqueeze(2).expand_as(target)
                    bg_loss = torch.mean((diff**2) * bg_mask.float())
                    loss = loss + id_lambda * bg_loss
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Log basic metrics every step (more frequently)
                if accelerator.is_main_process and global_step % 10 == 0:  # Every 10 steps
                    accelerator.log({
                        "training/step_loss": loss.detach().item(),
                        "training/learning_rate": lr_scheduler.get_last_lr()[0],
                        "training/global_step": global_step
                    }, step=global_step)

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # TensorBoard logging
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    # Additional metrics logging
                    log_comprehensive_metrics(accelerator, global_step, flux_transformer)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    #accelerator.save_state(save_path)
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")
                    
                    # Run inference after checkpoint save (OOM prevention)
                    run_inference_after_checkpoint(args, accelerator, global_step, weight_dtype, flux_transformer, save_path)
                    
                    # Reload model only if inference is enabled
                    test_configs = getattr(args, 'inference_config', {})
                    enable_inference = test_configs.get('enable_inference', True)
                    
                    if enable_inference:
                        # Reload model after inference to resume training
                        flux_transformer, lora_layers_model, optimizer, lr_scheduler = reload_model_after_inference(
                            args, accelerator, weight_dtype, lora_config, optimizer, lr_scheduler, lora_layers_model, save_path
                        )
                        
                        if flux_transformer is None:
                            logger.error("Model reload failed, stopping training")
                            break
                    else:
                        logger.info("Inference disabled, skipping model reload")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
