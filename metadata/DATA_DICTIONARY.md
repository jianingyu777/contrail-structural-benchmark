# ContrailStruct30 data dictionary

## Record-level attributes

`records.csv` contains 19,455 records. `record_id` is its unique key.
Read source product identifiers as strings. Empty CSV fields indicate a
missing value; `NOT_ENCODED` is an explicit source-level category, not a
numeric sentinel. The exact observed missing counts are in `fields.csv`.

| Field | Type | Unit | Definition |
|---|---|---|---|
| `record_id` | string |  | Unique record key and shared image/mask filename stem |
| `scene_id` | string |  | Processing-level-specific source-scene key; joins source_scenes.csv |
| `acquisition_date` | date | UTC calendar date | ISO 8601 YYYY-MM-DD; not a time of day |
| `year` | integer | year | Calendar year of acquisition |
| `month` | integer | month | Calendar month, 1 through 12 |
| `season` | string |  | Meteorological month grouping, used consistently in both hemispheres |
| `longitude_from_filename` | number | degrees east | Nominal source-scene longitude; not a patch-centre coordinate |
| `latitude_from_filename` | number | degrees north | Nominal source-scene latitude; not a patch-centre coordinate |
| `source_product_id` | string |  | Source-product identifier; read as string rather than a numerical measurement |
| `source_product_level` | string |  | Processing-level token encoded in the source name |
| `split` | string |  | Fixed scene/date-grouped partition |
| `sample_class` | string |  | positive for nonempty reference; difficult_negative for empty reference |
| `foreground_pixels` | integer | pixels | Number of reference-mask pixels equal to 255 |
| `foreground_fraction` | number | fraction | foreground_pixels divided by 65536; not a percentage |
| `connected_component_count` | integer | components | Number of eight-connected foreground components; zero for empty reference |
| `image_8bit_path` | string |  | Forward-slash path relative to data root; RGB PNG |
| `image_16bit_path` | string |  | Forward-slash path relative to data root; three-band TIFF |
| `mask_path` | string |  | Forward-slash path relative to data root; binary PNG |
| `image_8bit_dtype` | string |  | Decoded RGB array dtype |
| `image_8bit_shape` | string | pixels | Decoded RGB H / W / C array dimensions |
| `image_16bit_dtype` | string |  | Decoded source-value array dtype |
| `image_16bit_shape` | string | pixels | Recorded TIFF array dimensions: H / W / C or C / H / W; inspect before use |
| `mask_dtype` | string |  | Decoded reference array dtype |
| `mask_shape` | string | pixels | Decoded reference H / W array dimensions |
| `image_8bit_pixel_sha256` | string |  | 64 lowercase hexadecimal characters; C-contiguous decoded RGB uint8 bytes |
| `mask_pixel_sha256` | string |  | 64 lowercase hexadecimal characters; C-contiguous decoded mask uint8 bytes |

## Encoding and joins

- `season`: DJF = Dec-Feb, MAM = Mar-May, JJA = Jun-Aug, SON = Sep-Nov.
- `split`: train, validation or test. Join `benchmark_splits.csv` on `record_id`.
- `source_product_level`: L4A, L4B or NOT_ENCODED. Join scene summaries on `scene_id`.
- RGB arrays are HWC; rasterio reads TIFFs as CHW. The read example standardizes them to HWC.
- Masks are single-channel uint8 with 0 = background and 255 = contrail support.
- Foreground fractions are dimensionless fractions in [0,1].
- Hashes use decoded contiguous pixel bytes, not compressed image-file bytes.

## Path roots

Image and mask paths are relative to the directory containing `train`,
`validation` and `test`. The metadata root contains the `metadata` directory.
They can be the same directory or supplied separately to the reading and training scripts.

## Source windows

The scene longitude and latitude locate the source observation, not each patch.
`review/source_windows_100.csv` supplies verified parent-raster windows for the
100 reassessment examples. Join using `record_id` or `case_id`. Offsets are
zero-based integer pixels from the upper-left corner, with columns increasing
rightward and rows downward. Width and height define a half-open window.
`source_width_px` and `source_height_px` describe the complete parent raster.
`crs` uses the available CRS identifier/string; an empty value denotes an
unreferenced grid. Six affine coefficients map pixel-corner coordinates
to source-CRS coordinates; use col+0.5 and row+0.5 for pixel centres.
Pixel-verified windows and scene-level nominal positions are separate products.

## Other tables

`source_scenes.csv` summarizes counts by source-scene entry, not physical flight.
`benchmark_splits.csv` is the compact fixed partition table. `dataset_summary.json`
summarizes composition. `fields.csv` and `records_schema.json` are the canonical
machine-readable field definitions for the record table.
