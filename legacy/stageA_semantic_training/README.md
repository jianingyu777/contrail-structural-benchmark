# Recovered StageA training source

This directory preserves the historical source bundle that matches the
`stageA_semantic_fp_fold*_*.pth` checkpoint family. The recovered call chain
is:

```text
main_cv_fp.py
  -> cv_trainer_fp.py::trainer_cv_fp()
  -> models/timm_model.py::Model
  -> models/unet_decoder.py::UnetDecoder
```

The source writes the production checkpoint name
`stageA_semantic_fp_fold5_best_iou.pth` when launched with:

```bash
python main_cv_fp.py --model_name stageA_semantic --root DATA_ROOT
```

The original command line was not stored in the plain PyTorch state
dictionary, so command-line overrides cannot be recovered with certainty.
The source predates the checkpoint except for `models/unet_decoder.py`, whose
filesystem timestamp is later; its parameter names and tensor shapes still
match the checkpoint architecture.

## Release sanitation

The source is retained as historical evidence and is not the package's
supported entry point. One portability-only edit was made before release:
the private absolute default data path in `args.py` was replaced with `data`.
`source_manifest.json` records the SHA-256 hashes of the unmodified files and
the supplied archive.

The historical imports include packages that are not required by the active
trainer, and its decoder uses an older
`segmentation-models-pytorch.Conv2dReLU` API. Use the maintained
`contrail-train` command for a current environment. No exact historical lock
file was recovered.

## Metric compatibility note

The historical validation loop increments its negative-patch denominator
inside the threshold loop. This produces a nonstandard negative-patch rate
when several thresholds are scanned. The file is retained unchanged so that
the historical behavior remains inspectable. The maintained trainer uses one
denominator per negative patch and records the corrected rate explicitly.

The archived per-fold positive-patch IoU values were:

```text
0.836990, 0.810684, 0.835078, 0.838409, 0.833904
```

