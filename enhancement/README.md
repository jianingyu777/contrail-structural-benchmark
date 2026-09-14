# Patch-local image enhancement

Input: a three-band uint16 TIFF covering 256 by 256 pixels. Rasterio reads
bands as `(3, 256, 256)`; TIFF storage may decode as HWC in other readers.
Bands 1, 2 and 3 become R, G and B. Processing neither resamples nor shifts
the patch. Every mean and percentile below is computed on the complete input
patch, not the full scene or a 1024-pixel candidate neighbourhood.

1. Convert source values to float32. For each display band, replace zeros by
   the mean of nonzero values; an all-zero band stays zero.
2. Stretch the filled band from its 0th to 95th percentile, clip to [0,255]
   and truncate to uint8. Constant bands yield zero display values.
3. Use the original, unfilled source values for the radiance/temperature
   transform and ratios B2/(B3+1e-6) and B2/(B1+1e-6).
4. Select pixels with T2 below its 90th percentile, B2/B3 above its 40th
   percentile, and B2/B1 below its 50th percentile. NumPy's linear percentile
   interpolation is used.
5. Convert RGB to OpenCV HSV, multiply selected saturation values by 1.2,
   clip/truncate to uint8 and convert back to RGB. Save a PNG with Pillow.

The fixed gains, offsets and Planck-transform constants are in
`conversion_parameters.json` and the source script. The same coefficient set
is used for L4A, L4B and `NOT_ENCODED` entries by this display implementation.
These are algorithm parameters; physical calibration of the underlying source
values must be taken from the corresponding SDGSAT-1 product information.

Run from the package root:

```sh
python code/convert_uint16_to_8bit.py enhancement/examples/L4B/input.tif reconstructed.png
python code/verify_enhancement_examples.py
```

The converter refuses to overwrite an existing destination. Each example
contains its final dataset record ID, input, expected image, array shapes and
decoded-RGB SHA-256. Verification compares pixel values, not PNG compression
bytes. Real examples are supplied for all three processing-level categories.
The additional `conversion_sample.csv` records a deterministic verification
sample: the first 25 records in each encoded level and the one unencoded
record, 51 patches in total. All 51 matched exactly. This sample and the three
executable examples are not represented as a new full-library regeneration
test. The fixed distributed image files remain the benchmark inputs.
