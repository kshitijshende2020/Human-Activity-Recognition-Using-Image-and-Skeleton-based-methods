import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=12, freeze_backbone=True):
    """
    Builds a ResNet50 model pretrained on ImageNet for action classification.
    Replaces the final classification head (model.fc) to output `num_classes` logits.
    
    If freeze_backbone=True, freezes all backbone parameters and keeps only model.fc trainable.
    """
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.resnet50(pretrained=True)

    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully-connected classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Ensure the new final layer is explicitly trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def unfreeze_last_blocks(model, num_blocks=1):
    """
    Unfreezes the last `num_blocks` residual blocks within model.layer4 (counting from the end).
    For num_blocks=1, only model.layer4[-1] parameters become trainable.
    All earlier layers and model.fc remain unchanged.
    """
    blocks = list(model.layer4)
    if num_blocks > len(blocks):
        raise ValueError(f"Requested num_blocks ({num_blocks}) exceeds available blocks in layer4 ({len(blocks)}).")

    blocks_to_unfreeze = blocks[-num_blocks:]
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True

    return model

if __name__ == "__main__":
    print("=" * 65)
    print("STAGE 5a — RESNET50 MODEL DEFINITION SANITY CHECK")
    print("=" * 65)

    # 1. Build initial frozen-backbone model
    print("\n--- 1. Building Model with Frozen Backbone ---")
    model = build_model(num_classes=12, freeze_backbone=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_phase1 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params_phase1:,} (Phase 1: Head Only)")

    # Test forward pass shape
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Dummy Input Shape    : {tuple(dummy_input.shape)}")
    print(f"Dummy Output Shape   : {tuple(dummy_output.shape)} (Expected: (1, 12))")
    assert dummy_output.shape == (1, 12), f"Expected output shape (1, 12), got {dummy_output.shape}"

    # 2. Unfreeze last residual block in layer4
    print("\n--- 2. Unfreezing Last Residual Block (num_blocks=1) ---")
    model = unfreeze_last_blocks(model, num_blocks=1)

    trainable_params_phase2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters : {trainable_params_phase2:,} (Phase 2: Last Block + Head)")

    # 3. Layer4 Block Trainability Verification
    print("\n--- 3. Detailed layer4 Block Trainability Check ---")
    for idx, block in enumerate(model.layer4):
        is_trainable = any(p.requires_grad for p in block.parameters())
        print(f"  model.layer4[{idx}] trainable: {is_trainable}")

    fc_trainable = any(p.requires_grad for p in model.fc.parameters())
    print(f"  model.fc       trainable: {fc_trainable}")

    print("\n" + "=" * 65)
    print("SANITY CHECK SUCCESSFUL: Model Definition Verified!")
    print("=" * 65)