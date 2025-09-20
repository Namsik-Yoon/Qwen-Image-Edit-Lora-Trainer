import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
import gc
import math
import json
from datetime import datetime
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import psutil
import GPUtil
from PIL import Image

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    QwenImageInpaintPipeline as QIPipe,   # [CHANGED] inpaint 파이프라인
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import AttnProcsLayers

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb

from omegaconf import OmegaConf
from utils.dataset import loader   # [ASSUME] 배치에 precomputed 텐서들을 반환하도록 구현됨

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetune with precomputed embeddings (inpaint).")
    p.add_argument("--config", type=str, required=True, help="path to config yaml")
    args = p.parse_args()
    return args.config

# -------------------------
# Utils
# -------------------------
def lora_processors(model):
    processors = {}
    def fn(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
        for sub_name, child in module.named_children():
            fn(f"{name}.{sub_name}", child, processors)
        return processors
    for name, module in model.named_children():
        fn(name, module, processors)
    return processors

def get_gpu_memory_usage():
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
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }

def calculate_lora_statistics(model):
    lora_params = []
    lora_grad_norms = []
    for name, p in model.named_parameters():
        if 'lora' in name and p.requires_grad:
            lora_params.append(p.data.flatten())
            if p.grad is not None:
                lora_grad_norms.append(p.grad.data.norm().item())
    if lora_params:
        all_params = torch.cat(lora_params)
        return {
            'lora_param_count': sum(p.numel() for p in lora_params),
            'lora_param_mean': all_params.mean().item(),
            'lora_param_std': all_params.std().item(),
            'lora_param_max': all_params.max().item(),
            'lora_param_min': all_params.min().item(),
            'lora_grad_norm_mean': float(np.mean(lora_grad_norms)) if lora_grad_norms else 0.0,
            'lora_grad_norm_max': float(np.max(lora_grad_norms)) if lora_grad_norms else 0.0
        }
    return {}

def log_comprehensive_metrics(accelerator, step, model):
    try:
        metrics = {}
        metrics.update({f"lora/{k}": v for k, v in calculate_lora_statistics(model).items()})
        metrics.update({f"system/gpu_{k}": v for k, v in get_gpu_memory_usage().items()})
        metrics.update({f"system/{k}": v for k, v in get_system_metrics().items()})
        if metrics:
            accelerator.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Metrics logging failed: {e}")

# -------------------------
# Main
# -------------------------
def main():
    args = OmegaConf.load(parse_args())

    # Accelerator
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

    # -------------------------
    # 모델 & LoRA
    # -------------------------
    weight_dtype = torch.bfloat16 if args.mixed_precision in ("bf16", "fp16") else torch.float32
    offload_dir = os.path.join(args.output_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        variant=getattr(args, "variant", None),
        offload_folder=offload_dir,
    )

    # (선택) 마지막 K개 블록 양자화
    if args.quantize:
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        K = int(getattr(args, "quantize_last_k_blocks", 0))
        selected_blocks = all_blocks[-K:] if K > 0 else []
        for block in tqdm(selected_blocks, desc=f"Quantizing last {K} blocks"):
            block.to(device, dtype=weight_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to(device)
        flux_transformer.to(device, dtype=weight_dtype)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)

    flux_transformer.add_adapter(lora_config)

    # 스케줄러
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
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

    # 학습 세팅
    flux_transformer.requires_grad_(False)
    flux_transformer.train()
    for n, p in flux_transformer.named_parameters():
        p.requires_grad = ('lora' in n)
    logger.info(f"Trainable params (LoRA only): {sum(p.numel() for p in flux_transformer.parameters() if p.requires_grad)/1e6:.2f} M")

    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()

    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(lora_layers, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # -------------------------
    # DataLoader (캐시 기반)
    # loader 가 반드시 다음 텐서들을 배치로 반환하도록 해주세요:
    # pixel_latents:           (B, z, T, H', W')   # [NOTE] 이미 μ/σ 정규화된 라텐트
    # prompt_embeds:           (B, L, D)
    # prompt_embeds_mask:      (B, L) (int/bool)
    # control_latents:         (B, z, T, H', W')   # masked_image_latents
    # mask_latent (soft mask): (B, 1, H', W')
    # -------------------------
    train_dataloader = loader(output_dir=args.output_dir, **args.data_config)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    lora_layers_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training (precomputed embeddings) *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps", disable=not accelerator.is_local_main_process)

    global_step = 0
    train_loss = 0.0

    for epoch in range(1):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # -------------------------
                # 배치 언팩 (모두 캐시 텐서)
                # -------------------------
                pixel_latents, prompt_embeds, prompt_embeds_mask, control_latents, mask_latent = batch
                del batch

                # dtype/device 정리
                pixel_latents      = pixel_latents.to(device=accelerator.device, dtype=weight_dtype)       # (B,z,T,H',W')
                control_latents    = control_latents.to(device=accelerator.device, dtype=weight_dtype)     # (B,z,T,H',W')
                mask_latent        = mask_latent.to(device=accelerator.device, dtype=weight_dtype)         # (B,1,H',W')
                prompt_embeds      = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)       # (B,L,D)
                prompt_embeds_mask = prompt_embeds_mask.to(device=accelerator.device)                      # (B,L) int/bool

                # [IMPORTANT] precompute 단계에서 이미 μ/σ 정규화 완료된 latents 가정 → 추가 정규화 없음

                # 노이즈 샘플/타임스텝
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
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise  # (B,z,T,H',W')

                # pack
                packed_noisy = QIPipe._pack_latents(
                    noisy_model_input, bsz,
                    noisy_model_input.shape[2], noisy_model_input.shape[3], noisy_model_input.shape[4]
                )
                packed_control = QIPipe._pack_latents(
                    control_latents, bsz,
                    control_latents.shape[2], control_latents.shape[3], control_latents.shape[4]
                )

                # latent image ids for RoPE.
                img_shapes = [[
                    (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                    (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
                ]] * bsz

                packed_input = torch.cat([packed_noisy, packed_control], dim=1)

                # 텍스트 길이
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                # 전방향
                model_pred = flux_transformer(
                    hidden_states=packed_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy.size(1)]

                latent_h, latent_w = noisy_model_input.shape[-2], noisy_model_input.shape[-1]
                model_pred = QIPipe._unpack_latents(
                    model_pred,
                    height=latent_h,
                    width=latent_w,
                    vae_scale_factor=1,
                )

                # 손실 가중(인페인트)
                # mask_latent: (B,1,H',W') 이미 라텐트 해상도
                alpha_fg  = float(getattr(args, "mask_fg_boost", 2.0))
                lambda_bg = float(getattr(args, "mask_bg_scale", 1.0))
                id_lambda = float(getattr(args, "lambda_identity_bg", 0.0))

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = (noise - pixel_latents)  # (B,z,T,H',W')

                spatial = mask_latent * alpha_fg + (1.0 - mask_latent) * lambda_bg  # (B,1,H',W')
                spatial = spatial.unsqueeze(2).expand_as(target)                    # (B,z,T,H',W')
                weighting = weighting * spatial

                diff = (model_pred.float() - target.float())
                loss = torch.mean((weighting.float() * diff**2).reshape(target.shape[0], -1), dim=1).mean()

                if id_lambda > 0:
                    bg_mask = (1.0 - mask_latent).unsqueeze(2).expand_as(target)
                    bg_loss = torch.mean((diff**2) * bg_mask.float())
                    loss = loss + id_lambda * bg_loss

                # logging(간단)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                if accelerator.is_main_process and global_step % 10 == 0:
                    accelerator.log({
                        "training/step_loss": loss.detach().item(),
                        "training/learning_rate": lr_scheduler.get_last_lr()[0],
                        "training/global_step": global_step
                    }, step=global_step)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": avg_loss.item()}, step=global_step)
                    log_comprehensive_metrics(accelerator, global_step, flux_transformer)

                # 저장 & (선택)인퍼런스
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for c in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, c), ignore_errors=True)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped = unwrap_model(flux_transformer)
                    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
                    # [CHANGED] Inpaint 파이프라인의 save_lora_weights 사용
                    QIPipe.save_lora_weights(save_path, lora_state, safe_serialization=True)
                    logger.info(f"Saved LoRA weights to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
