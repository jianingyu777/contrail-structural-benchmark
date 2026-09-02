# Checkpoint record

## Production segmentation checkpoint

- Filename: `stageA_semantic_fp_fold5_best_iou.pth`
- Size: 336,765,728 bytes
- SHA-256: `4FEC1A57D650CA3D10C0B4343BDF2A1295A5895A112E2FED589B9A2ACB9CBD51`
- Architecture: MaxViT-Base encoder, U-Net-like decoder, one-channel
  segmentation head; the training checkpoint also contains a centerline head.
- Production probability threshold: 0.5
- Validation-selected reporting threshold in the archived manuscript table:
  0.90. This is not the production mask threshold.

The recovered training source scans validation thresholds from 0.20 to 0.95
when selecting each training checkpoint. That per-epoch training threshold is
distinct from both the frozen 0.5 production threshold and the separately
selected held-out reporting threshold.

The binary is excluded from Git history. Attach it to a versioned GitHub
release or a DOI-bearing archive, preserve the filename and digest, and add
the permanent URL to this record after deposit.
