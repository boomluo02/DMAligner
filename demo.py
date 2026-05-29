#!/usr/bin/env python3
"""
DMAligner Demo: Body Correction with Diffusion Model

Usage:
    python demo.py --image1 path/to/img1.png --image2 path/to/img2.png [--output path/to/output.png]

The model takes two images (e.g., two views of a person) and outputs a corrected/aligned prediction.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Set this to the GPU you want to use
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bcd_pipeline import AlignmentPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="DMAligner Demo: align/correct body distortion")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first input image (img1)")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second input image (img2)")
    parser.add_argument(
        "--model_path", type=str,
        default="checkpoints/final_model",
        help="Path to the pretrained model checkpoint"
    )
    parser.add_argument("--output", type=str, default="results/demo_output.png", help="Path to save the output image")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on: 'cuda' or 'cpu'")
    parser.add_argument("--height", type=int, default=512, help="Resize input height (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=960, help="Resize input width (must be divisible by 8)")
    return parser.parse_args()


def load_image(image_path: str, target_size: tuple, device: torch.device) -> torch.Tensor:
    """Load and preprocess an image to a normalized tensor in [-1, 1]."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    print(f"  Original image size: {img.size}")

    # Resize to target size
    img = img.resize(target_size, Image.BILINEAR)

    # Convert to tensor [0, 1] and then scale to [-1, 1]
    img_tensor = torch.from_numpy(
        __import__("numpy").array(img).transpose(2, 0, 1)
    ).float() / 127.5 - 1.0

    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def save_image(tensor: torch.Tensor, output_path: str):
    """Save a tensor as an image file.  The tensor is expected to be in [0,1]."""
    # tensor shape: (1, 3, H, W), values already in [0, 1] (pipeline output)
    img = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(img).save(output_path)
    print(f"  Output saved to: {output_path}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate target size
    assert args.height % 8 == 0 and args.width % 8 == 0, \
        f"Height ({args.height}) and width ({args.width}) must be divisible by 8."
    target_size = (args.width, args.height)

    # ============ 1. Load model ============
    print(f"\n[1/4] Loading model from: {args.model_path}")
    pipeline = AlignmentPipeline.from_pretrained(
        args.model_path,
        safety_checker=None,
        torch_dtype=torch.float32,
    )
    pipeline.to(device)
    print("  Model loaded successfully.")

    # ============ 2. Load input images ============
    print(f"\n[2/4] Loading input images...")
    img1 = load_image(args.image1, target_size, device)
    img2 = load_image(args.image2, target_size, device)
    print(f"  img1 shape: {img1.shape}, img2 shape: {img2.shape}")

    # ============ 3. Run inference ============
    print(f"\n[3/4] Running inference (steps={args.steps})...")
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    with torch.no_grad():
        result = pipeline(
            img1, img2,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="ndarray",
        )

    pred_image = result.pred_images[0]  # numpy array (H, W, 3) in [0, 1]
    pred_tensor = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0)
    print(f"  Prediction shape: {pred_tensor.shape}")

    # ============ 4. Save output ============
    print(f"\n[4/4] Saving output...")
    save_image(pred_tensor, args.output)

    del pipeline
    torch.cuda.empty_cache()
    print("\nDone! ✅")


if __name__ == "__main__":
    main()
