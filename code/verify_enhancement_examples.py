"""Check the archived end-to-end examples without modifying expected images."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import rasterio
from convert_uint16_to_8bit import convert_array

def main():
    root = Path(__file__).resolve().parents[1] / "enhancement/examples"
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        specification = json.loads((folder / "example.json").read_text())
        with rasterio.open(folder / "input.tif") as source:
            data = source.read(out_dtype="float32")
        result = convert_array(data)
        with Image.open(folder / "expected.png") as im:
            expected = np.asarray(im.convert("RGB"))
        assert np.array_equal(result, expected), folder.name
        digest = hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest()
        assert digest == specification["expected_rgb_pixel_sha256"], folder.name
        print(f"{folder.name}: pixel-exact; {digest}")

if __name__ == "__main__":
    main()
