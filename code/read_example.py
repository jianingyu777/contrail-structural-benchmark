"""Read one distributed record using declared path roots and array conventions."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Folder containing train, validation and test")
    p.add_argument("--metadata-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--record-id")
    args = p.parse_args()
    frame = pd.read_csv(args.metadata_root / "metadata/records.csv", keep_default_na=False)
    row = frame.set_index("record_id").loc[args.record_id] if args.record_id else frame.iloc[0]
    with Image.open(args.data_root / row.image_8bit_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    raw = tifffile.imread(args.data_root / row.image_16bit_path)
    if raw.shape == (3, 256, 256):
        raw = np.moveaxis(raw, 0, -1)
    with Image.open(args.data_root / row.mask_path) as im:
        mask = np.asarray(im)
    assert rgb.shape == raw.shape == (256, 256, 3) and mask.shape == (256, 256)
    assert raw.dtype == np.uint16 and set(np.unique(mask)).issubset({0, 255})
    assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest() == row.image_8bit_pixel_sha256
    assert hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest() == row.mask_pixel_sha256
    print(json.dumps({"rgb_shape": list(rgb.shape), "uint16_shape": list(raw.shape),
                      "mask_shape": list(mask.shape), "foreground_pixels": int((mask > 0).sum()),
                      "split": row["split"], "pixel_hashes_match": True}))

if __name__ == "__main__":
    main()
