from typing import Any

import torch

from .BaseModel import BaseModel
from .Swin.SwinUnet import SwinUNet

class SwinUnetLightning(BaseModel):
    """_summary_ Swin backbone with Unet head for binary segmentation.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        input_height: int, 
        input_width: int, 
        input_channels: int, 
        base_channels: int, 
        num_class: int,
        num_blocks: int,
        patch_size: int,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=False,
            *args,
            **kwargs
        )
        
        self.model = SwinUNet(H=input_height,
                              W=input_width, 
                              ch=input_channels, 
                              C=base_channels, 
                              num_class=num_class, 
                              num_blocks=num_blocks, 
                              patch_size = patch_size)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x)