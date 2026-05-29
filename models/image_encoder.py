
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

class ImageEncoder(nn.Module):
    '''Feature wise Image Encoder
    The feature-wise image encoder is a convolutional neural network that takes an image as input and outputs a feature vector.
    The feature vector is used in Unet2DWithConditon, replace the text embedding.
    The feature vector shape is same as the text embedding shape.
    '''
    def __init__(self, input_dim, output_dim):
        super(ImageEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, self.output_dim)
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc(x))
        return x

class OpenClipEncoder:
    '''Clip based Image Encoder
    Clip based image encoder using OpenClip model as a feature extractor.
    Then we project the feature vector to the same shape as the text embedding.
    As IP-adpater do, to save VRAM, we use OpenClipEncoder preprocess the images and save the image embeddings in the disk.
    '''
    def __init__(self, image_encoder_path, cross_attention_dim, device, num_tokens=4):
        super(OpenClipEncoder, self).__init__()
        self.device = device
        self.open_clip_processor = OpenClipProcessor(image_encoder_path, device)
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=self.open_clip_processor.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_tokens,
        ).to(self.device, dtype=torch.float16)
        

    def get_image_embeds(self, image):
        image_tokens = self.open_clip_processor.get_image_tokens(image)
        image_embeds = self.image_proj_model(image_tokens)
        return image_embeds
    
class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class OpenClipProcessor:
    '''OpenClip Processor
    OpenClip Processor using pretrained OpenClip model to tokenize the images.
    '''
    def __init__(self, image_encoder_path, device):
        super(OpenClipProcessor, self).__init__()
        self.device = device
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

    @torch.inference_mode()
    def get_image_tokens(self, image):
        if isinstance(image, torch.Tensor):
            # to PIL image
            image = transforms.ToPILImage()(image)
        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        image_tokens = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        return image_tokens