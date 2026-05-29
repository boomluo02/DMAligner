# DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis

<div align="center">

**CVPR 2026 (🌟 Highlight)**

Xinglong Luo<sup>1</sup>, Ao Luo<sup>2</sup>, Zhengning Wang<sup>1†</sup>, Yueqi Yang<sup>3</sup>, Chaoyu Feng<sup>3</sup>, Lei Lei<sup>3</sup>, Bing Zeng<sup>1</sup>, Shuaicheng Liu<sup>1†</sup>

<sup>1</sup>University of Electronic Science and Technology of China &emsp; <sup>2</sup>Southwest Jiaotong University &emsp; <sup>3</sup>Independent Researcher

[![arXiv](https://img.shields.io/badge/arXiv-2602.23022-b31b1b.svg)](https://arxiv.org/abs/2602.23022)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-Highlight)]()
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue)](https://boomluo02.github.io/DMAligner/)

</div>

## 📝 Abstract

Image alignment is a fundamental task in computer vision. Existing methods predominantly employ optical flow-based image warping. However, this is susceptible to occlusions and illumination variations, leading to ghosting artifacts. We present **DMAligner**, a diffusion-based framework that achieves image alignment through alignment-oriented view synthesis — directly generating the aligned image instead of warping.

## 🔥 News

- **2026.03** 🎉 DMAligner accepted to **CVPR 2026** as a 🌟**Highlight** paper! | [Project Page](https://boomluo02.github.io/DMAligner/)

## 🛠️ Installation

```bash
git clone https://github.com/boomluo02/DMAligner.git
cd DMAligner
conda env create -f environment.yml
conda activate torch211
```

**Requirements**: PyTorch 2.1.1, CUDA 12.1, diffusers 0.30.3, xformers 0.0.23. See [`environment.yml`](environment.yml).

## 🎮 Quick Demo

```bash
# Place your image pairs in inputs/pair{1..N}/ as img1.png and img2.png
python demo.py
```

Outputs per pair (`results/pairN/`): `input_img1.png`, `input_img2.png`, `pred.png`, and animated GIF comparisons.

```bash
# Custom paths
python demo.py --input_dir ./my_images --output ./my_results --steps 50
```

## 📦 Pretrained Model

Download the checkpoint and place under `checkpoints/final_model/`.

## 📊 DSIA Dataset

The **Dynamic Scene Image Alignment (DSIA)** dataset is built with Blender, simulating real-world dynamics including camera motion, moving characters/objects, and varying illumination.

| Property | Value |
|----------|-------|
| Scenes | 1,033 (indoor & outdoor) |
| Image Pairs | 30,000+ |
| Resolution | 960 × 540 |
| Characters | 25 |
| Objects | 100 |

## 🏋️ Training

```bash
accelerate launch --num_processes 2 --main_process_port 29501 --num_machines 1 \
    code/train_fov_trans.py --use_wandb --base_config config/train_align.yaml
```

See [`config/train_align.yaml`](config/train_align.yaml) for training configuration.

## 📂 Project Structure

```
DMAligner/
├── demo.py                  # Inference demo (batch processing + GIF output)
├── environment.yml          # Conda environment
├── code/
│   └── train_fov_trans.py   # Training entry point
├── config/
│   ├── train_align.yaml     # Training configuration
│   └── train_config.py      # Config parser
├── models/
│   ├── bcd_pipeline.py      # AlignmentPipeline (inference)
│   ├── attention_processor.py
│   ├── image_encoder.py
│   └── refinenet.py
├── tools/
│   ├── init_1.py            # Training initialization
│   ├── loss.py              # Loss functions
│   ├── utils_1.py           # Utility functions
│   └── wandb.py             # W&B logging
├── data/
│   ├── dataset.py           # Data loader
│   └── csv_files/           # Dataset split CSVs
└── inputs/                  # (User-provided) input image pairs
```

## 🎓 Citation

```bibtex
@inproceedings{luo2026dmaligner,
  title={DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis},
  author={Luo, Xinglong and Luo, Ao and Wang, Zhengning and Yang, Yueqi and
          Feng, Chaoyu and Lei, Lei and Zeng, Bing and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026},
  note={Highlight}
}
```

## 🙏 Acknowledgments

We thank the authors of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [Diffusers](https://github.com/huggingface/diffusers), [PointOdyssey](https://pointodyssey.com/), and [Kubric](https://github.com/google-research/kubric) for their open-source contributions.
