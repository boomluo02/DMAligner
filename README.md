# BCD
Body Correction with Diffusion model

pytorch==2.1.1
torchvision==0.16.1 
torchaudio==2.1.1 
pytorch-cuda=12.1

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch
xformers==0.0.23
accelerate==0.31.0 # 0.33.0 has a bug with resume_from_checkpoint

Trouble shooting:
1. RuntimeError: Input type (c10::Half) and bias type (float) should be the same
     - Solution 1: disable mixed precision training, disable 'with amp.autocast(cuda)'
     - Solution 2: change input to float16, condition = condition.half().to(accelerator.device), this may cause black image while generating
