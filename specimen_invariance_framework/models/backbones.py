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
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    import torchvision.models as tv_models


def _build_torchvision_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Fallback backbone builder using native torchvision.models."""
    b_name = backbone_name.lower()
    weights_arg = "DEFAULT" if pretrained else None
    
    if "convnext" in b_name:
        model = tv_models.convnext_tiny(weights=weights_arg)
        feat_dim = model.classifier[0].in_features if hasattr(model.classifier[0], "in_features") else 768
        model.classifier = nn.Identity()
        return model, feat_dim
    elif "resnet50" in b_name:
        model = tv_models.resnet50(weights=weights_arg)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim
    elif "efficientnet" in b_name:
        model = tv_models.efficientnet_v2_s(weights=weights_arg)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feat_dim
    elif "swin" in b_name:
        model = tv_models.swin_t(weights=weights_arg)
        feat_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feat_dim
    else:
        # Default fallback to convnext_tiny
        model = tv_models.convnext_tiny(weights=weights_arg)
        model.classifier = nn.Identity()
        return model, 768


def build_backbone(backbone_name: str = "convnext_tiny", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Instantiates the visual backbone with pooling, removing the classification head.
    Supports timm with automatic fallback to native torchvision models.
    
    Returns:
        (backbone_module, feature_dim)
    """
    if not HAS_TIMM:
        return _build_torchvision_backbone(backbone_name, pretrained=pretrained)

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
        # Fallback to base name or native torchvision
        try:
            fallback_name = timm_name.split(".")[0]
            backbone = timm.create_model(fallback_name, pretrained=pretrained, num_classes=0)
        except Exception:
            try:
                fallback_name = timm_name.split(".")[0]
                backbone = timm.create_model(fallback_name, pretrained=False, num_classes=0)
            except Exception:
                return _build_torchvision_backbone(backbone_name, pretrained=pretrained)
        
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        if "convnext" in backbone_name:
            feature_dim = 768
        elif "resnet50" in backbone_name:
            feature_dim = 2048
        elif "efficientnet" in backbone_name:
            feature_dim = 1280
        elif "swin" in backbone_name:
            feature_dim = 768
        else:
            feature_dim = 768

    return backbone, feature_dim
