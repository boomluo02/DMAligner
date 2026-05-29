from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_up_block,
)
from diffusers.utils.import_utils import is_torch_version

def extract_encoder_features(encoder, input_tensor):
    features = {}
    # Hook function to save features
    def hook_fn(module, input, output, name):
        features[name] = output
    # Register hooks
    hooks = []
    # conv_in
    hook = encoder.conv_in.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'encoder_conv_in'))
    hooks.append(hook)
    # down_blocks
    for idx, down_block in enumerate(encoder.down_blocks):
        hook = down_block.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, f'down_block_{idx}'))
        hooks.append(hook)
    # mid_block
    hook = encoder.mid_block.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'encoder_mid_block'))
    hooks.append(hook)
    # forward pass
    _ = encoder(input_tensor)
    # Remove hooks
    for hook in hooks:
        hook.remove()
    return features

def extract_decoder_features(decoder, input_tensor):
    features = {}
    # Hook function to save features
    def hook_fn(module, input, output, name):
        features[name] = output
    # Register hooks
    hooks = []
    # conv_in
    hook = decoder.conv_in.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'decoder_conv_in'))
    hooks.append(hook)
    # up_blocks
    for idx, up_block in enumerate(decoder.up_blocks):
        hook = up_block.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, f'up_block_{idx}'))
        hooks.append(hook)
    # mid_block
    hook = decoder.mid_block.register_forward_hook(lambda module, input, output: hook_fn(module, input, output, 'decoder_mid_block'))
    hooks.append(hook)
    # forward pass
    _ = decoder(input_tensor)
    # Remove hooks
    for hook in hooks:
        hook.remove()
    return features


class RefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            norm_type=norm_type,
            mid_block_add_attention=mid_block_add_attention,
        )

        # for each layer, inchannels should be doubled
        self.decoder.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.decoder.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
        )

        # up
        up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels)) # [512, 512, 256, 128]

        # # refine net v5
        # concat_channels = [512, 256, 128, 128]

        # refine net v4
        # concat_channels = [1024, 768, 640, 384]

        # refine net v3
        # concat_channels = [512, 256, 128, 128]
        
        # refine net v2
        # surface level from encoder, deep level from decoder
        # concat_channels = [512, 512, 128, 128]

        # refine net v1 fix
        # deep level from encoder, surface level from decoder
        # concat_channels = [512, 256, 128, 256]

        # refine net v1
        # deep level from encoder, surface level from decoder
        concat_channels = [512, 256, 128, 128]
        
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel + concat_channels[i]
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.decoder.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            up_blocks.append(up_block)
        
        self.decoder.up_blocks = up_blocks

        # out
        if norm_type == "spatial":
            self.decoder.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.decoder.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.decoder.conv_act = nn.SiLU()
        self.decoder.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.decoder.gradient_checkpointing = False

        # print(f"RefineNet: {self.decoder}")

    def forward(
        self,
        sample: torch.Tensor,
        feature_maps: Dict[str, torch.Tensor],
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The modified forward method of the `Decoder` class."""

        sample = self.decoder.conv_in(sample)

        # # refine net v5
        # used_feature_maps = [
        #     feature_maps['down_block_2'], # 512
        #     feature_maps['down_block_1'], # 256
        #     feature_maps['down_block_0'], # 128
        #     feature_maps['encoder_conv_in'], # 128
        # ]

        # # refine net v4
        # used_feature_maps = [
        #     torch.cat([feature_maps['decoder_mid_block'], feature_maps['down_block_2']], dim=1), # 512 + 512 = 1024
        #     torch.cat([feature_maps['up_block_0'], feature_maps['down_block_1']], dim=1), # 512 + 256 = 768
        #     torch.cat([feature_maps['up_block_1'], feature_maps['down_block_0']], dim=1), # 512 + 128 = 640
        #     torch.cat([feature_maps['up_block_2'], feature_maps['encoder_conv_in']], dim=1), # 256 + 128 = 384
        # ]

        # # refine net v3
        # used_feature_maps = [
        #     feature_maps['decoder_mid_block'], # 512
        #     feature_maps['down_block_1'], # 256 
        #     feature_maps['down_block_0'], # 128
        #     feature_maps['encoder_conv_in'], # 128
        # ]

        # refine net v2
        # surface level from encoder, deep level from decoder
        # used_feature_maps = [
        #     feature_maps['decoder_mid_block'], # 512
        #     feature_maps['up_block_0'], # 512 
        #     feature_maps['down_block_0'], # 128
        #     feature_maps['encoder_conv_in'], # 128
        # ]

        # # refine net v1 fix
        # # deep level from encoder, surface level from decoder
        # used_feature_maps = [
        #     feature_maps['down_block_2'], # 512  up block 0
        #     feature_maps['down_block_1'], # 256 up block 1
        #     feature_maps['down_block_0'], # 128 up block 2
        #     feature_maps['up_block_2'], # 256 up block 3
        # ]
        
        # refine net v1
        # deep level from encoder, surface level from decoder
        used_feature_maps = [
            feature_maps['down_block_2'], # 512  up block 0
            feature_maps['down_block_1'], # 256 up block 1
            feature_maps['down_block_0'], # 128 up block 2
            feature_maps['up_block_3'], # 128 up block 3
        ]

        upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
        if self.decoder.training and self.decoder.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.decoder.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.decoder.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        torch.cat([sample, used_feature_maps[i]], dim=1),
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.decoder.mid_block),
                    sample,
                    latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.decoder.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), 
                        torch.cat([sample, used_feature_maps[i]], dim=1),
                        latent_embeds
                )
        else:
            # middle
            sample = self.decoder.mid_block(
                sample,
                latent_embeds
            )
            sample = sample.to(upscale_dtype)

            # up
            for i, up_block in enumerate(self.decoder.up_blocks):
                sample = up_block(
                    torch.cat([sample, used_feature_maps[i]], dim=1),
                    latent_embeds
                )

        # post-process
        if latent_embeds is None:
            sample = self.decoder.conv_norm_out(sample)
        else:
            sample = self.decoder.conv_norm_out(sample, latent_embeds)
        sample = self.decoder.conv_act(sample)
        sample = self.decoder.conv_out(sample)

        return sample
    
