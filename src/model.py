import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 8, dropout: float = 0.4) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.

    Head replaced with:
        Linear(1280 → 256) → BatchNorm → ReLU → Dropout → Linear(256 → 8)

    Phase 1 (epochs 1–5):  backbone frozen, only head trains.
    Phase 2 (epoch 6+):    unfreeze last MBConv block + head.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # ── Replace classifier head ───────────────────────────────────────────────
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    # ── Phase-1: freeze entire backbone ──────────────────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False

    # Head is trainable by default (newly initialised params)
    return model


def unfreeze_last_block(model: nn.Module):
    """
    Phase-2: unfreeze the last MBConv block (features[8]) + the Conv head (features[8]).
    Called after warm-up epochs are done.
    """
    # EfficientNet-B0 backbone: model.features is a Sequential of 9 sub-modules (0-8)
    # Unfreeze features[7] (last MBConv stage) and features[8] (top conv)
    for layer_idx in [7, 8]:
        for param in model.features[layer_idx].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Unfreeze] Phase-2 active — trainable params: {trainable:,}")


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Total params    : {total:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Frozen params   : {frozen:>10,}")