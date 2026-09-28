"""
specimen_invariance_framework/models/backbones.py
=================================================
Backbone feature extractors using timm:
- ConvNeXt-Tiny (primary workhorse)
- ResNet-50 (generalization baseline)
- EfficientNetV2-S (generalization baseline)
- Swin-T (vision transformer generalization baseline)
"""

from typing import Tuple
import torch
import torch.nn as nn
import timm


def build_backbone(backbone_name: str = "convnext_tiny", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Instantiates the visual backbone with pooling, removing the classification head (num_classes=0).
    
    Returns:
        (backbone_module, feature_dim)
    """
    name_map = {
        "convnext_tiny": "convnext_tiny",
        "resnet50": "resnet50",
        "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
        "tf_efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
        "swin_t": "swin_tiny_patch4_window7_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }
    timm_name = name_map.get(backbone_name.lower(), backbone_name)
    
    try:
        backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    except Exception as e:
        # Fallback to base name if specific pretrained tag is unavailable
        fallback_name = timm_name.split(".")[0]
        backbone = timm.create_model(fallback_name, pretrained=pretrained, num_classes=0)
        
    feature_dim = backbone.num_features
    return backbone, feature_dim
