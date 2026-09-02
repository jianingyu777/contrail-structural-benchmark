# StageA training-source audit

## Conclusion

The supplied archive contains the matching historical StageA training source,
not merely an architecture reconstruction. It establishes the original
five-fold entry point, loss curriculum, EMA, hard-negative weighting,
validation-threshold scan and checkpoint naming convention.

The evidence supports the wording:

> Original model, inference and recovered training source, with a maintained
> validation and test workflow for the public release environment.

It does not support claiming that the original command line or software
environment is known exactly.

## Evidence

| Check | Result |
|---|---|
| Source archive | `stageA_semantic_training_code_20260902.zip` |
| Archive SHA-256 | `8AFB514EB0D3414546A335D3C17550705B649F7E0D57F42C2D0DC332F2A6EC27` |
| Historical entry point | `main_cv_fp.py -> trainer_cv_fp()` |
| Checkpoint filename emitted | `stageA_semantic_fp_fold5_best_iou.pth` |
| Checkpoint SHA-256 | `4FEC1A57D650CA3D10C0B4343BDF2A1295A5895A112E2FED589B9A2ACB9CBD51` |
| Strict state load | 0 missing, 0 unexpected |
| Trainable parameters | 83,948,118 |
| Fold-5 split | 12,864 training; 3,216 validation |
| Fold-5 training-stem SHA-256 | `56C9223B160000465D6ADCDC5EA1A81207803FBFBB8C2C151730C9B3AFF311B3` |
| Fold-5 validation-stem SHA-256 | `67BB9EE5C1007B792767CBA6215B013EC893102CCE2A8845523B736B9D5DF066` |

The complete train-plus-validation pool contains 16,080 patches with the
following recovered strata: 8,328 negative, 1,907 small-positive, 2,745
medium-positive and 3,100 large-positive patches.

## Recovered recipe

- MaxViT-Base encoder: `maxvit_base_tf_512.in21k_ft_in1k`.
- Four-stage U-Net-like decoder: 384, 192, 96 and 64 channels.
- Segmentation and auxiliary centerline heads.
- Five-fold `StratifiedKFold`, shuffle enabled, seed 3407.
- Positive/negative sampler weights 1.0/1.2.
- Adam, learning rate `3e-5`, zero explicit weight decay.
- Five warm-up epochs followed by cosine decay; 500-epoch cap.
- BCE positive weight 5.0 and background-batch multiplier 2.5.
- Tversky alpha/beta 0.3/0.7.
- clDice and centerline losses begin after 20% of training and ramp to 0.5
  and 0.3.
- EMA decay 0.999.
- Hard-negative weighting begins after 35% of training, with gamma 5 and
  maximum weight 5.
- Early-stopping patience 20 epochs.

## Boundaries

The checkpoint is a plain `OrderedDict`. It does not preserve the optimizer,
epoch, selected validation threshold, random-number states, package versions
or launch arguments. A complete 500-epoch retraining was not rerun during this
source audit.

The historical validation implementation counts each negative patch once per
candidate threshold in its denominator. The maintained trainer fixes that
metric defect and labels its output accordingly. The historical file is kept
unchanged, apart from the documented default-path sanitation in `args.py`, so
reviewers can inspect the exact recovered logic.
