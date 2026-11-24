# Table 7 manual reference masks

This directory contains the 60 hand-annotated reference regions used for the
Table 7 in-domain annotation-reference check.

## Contents

- `masks/<ID>_mask.png`: 512 x 512 binary reference mask (0/255).
- `valid/<ID>_valid.png`: 512 x 512 valid-area mask (255 = evaluated).
- `ignore/<ID>_ignore.png`: 512 x 512 ignore-area mask (255 = excluded).

The identifiers (`FT001`-`FT030`, `VA001`-`VA010`, `EV001`-`EV020`) and their
dataset record IDs are listed in
`results/six_model_seed42/table7_manual_rois.csv`.

## Provenance and processing

- Masks were drawn from scratch by human annotators on the full 512 x 512
  displays, without reference to model predictions.
- For Table 7, each mask, valid mask and ignore mask is centre-cropped to
  256 x 256 (`128:384`). The matching image input is the same centre crop of the
  three-band uint16 patch, processed with the dataset's per-patch enhancement
  (see `code/prepare_manual_mask_inputs.py`).
- The original images and uint16 source values are part of the public
  ContrailStruct30 Figshare release.

## Licence

These annotation masks are released under the Creative Commons Attribution 4.0
International (CC BY 4.0) licence, consistent with the dataset release.
