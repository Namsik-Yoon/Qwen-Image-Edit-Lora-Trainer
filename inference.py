from diffusers import QwenImageEditPipeline
import torch
from PIL import Image
import argparse

import os
import torch
from PIL import Image
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration

def main(args):
    """
    Main function to load the model, apply quantization, and generate an image.
    """
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Using device: {device} with dtype: {torch_dtype}")

    # GPU memory optimization settings
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # Model settings
    model_id = "Qwen/Qwen-Image-Edit"
    torch_dtype = torch.bfloat16
    device = "cuda"

    # Transformer quantization settings
    transformer_quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )

    # Load transformer model
    print("Loading transformer model...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=transformer_quantization_config,
        torch_dtype=torch_dtype,
    )
    transformer = transformer.to("cuda")

    # Text encoder quantization settings
    text_encoder_quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load text encoder model
    print("Loading text encoder model...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=text_encoder_quantization_config,
        torch_dtype=torch_dtype,
    )
    text_encoder = text_encoder.to("cpu")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = QwenImageEditPipeline.from_pretrained(
        model_id, 
        transformer=transformer, 
        text_encoder=text_encoder, 
        torch_dtype=torch_dtype
    )

    # Load LoRA weights
    if args.lora_weights:
        print(f"Loading LoRA weights from: {args.lora_weights}")
        pipeline.load_lora_weights(args.lora_weights)

    # Offload model to CPU to save VRAM, parts will be moved to GPU as needed
    pipeline.enable_model_cpu_offload()

    # Set up the generator for reproducibility
    
    input_images = [i for i in os.listdir("/workspace/flymyai-lora-trainer/datasets/test") if i.endswith(".png") or i.endswith(".jpg")]
    for i, input_image in enumerate(input_images):
        generator = torch.Generator(device=device).manual_seed(i)
        img = Image.open(os.path.join("/workspace/flymyai-lora-trainer/datasets/test", input_image)).convert("RGB")
        print("Generating image...")
        # Generate the image
        image = pipeline(
            image=img,
            prompt=args.prompt,
            negative_prompt=[" "],
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
        ).images[0]

        # Save the output image
        image.save(os.path.join(args.output_image, "test_" + input_image))
        print(f"Image successfully saved to {os.path.join(args.output_image, 'test_' + input_image)}")

if __name__ == "__main__":
    negative_prompt = """cartoon, anime, CGI, 3D render, illustration, painting, unreal engine,
                        lowres, blurry, out of focus, motion blur, jpeg artifacts,
                        watermark, text overlay,
                        duplicate objects, warped geometry, distorted perspective, melted metal,
                        pattern tiling, repetitive textures,
                        graffiti,
                        background changes, lighting changes, color cast"""
    parser = argparse.ArgumentParser(description="Generate images with a quantized Qwen-Image model using LoRA.")

    # Model and Weights Arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image-Edit", help="Path or name of the base model.")
    parser.add_argument("--lora_weights", type=str, default="", help="Path to the LoRA weights.")
    parser.add_argument("--output_image", type=str, default="generated_image.png", help="Filename for the output image.")

    # Generation Arguments
    parser.add_argument("--prompt", type=str, default='''man in the city''', help="The prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default=negative_prompt, help="The negative prompt for image generation.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--true_cfg_scale", type=float, default=5.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--seed", type=int, default=655, help="Random seed for the generator.")

    # Quantization Arguments
    parser.add_argument("--quantization", type=str, default="qfloat8", choices=["qfloat8"], help="The quantization type to apply.")

    args = parser.parse_args()
    main(args)