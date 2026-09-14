# Contextual reassessment evidence

The sample contains 100 patches from 100 processing-level-specific scene
entries and 94 source products. Coverage and challenge groups contain 60 and
40 cases. Two additional experts followed the annotation handbook.
The separate contextual reassessment package provides the author-supplied
`annotation_handbook_zh.docx` and full image evidence. Table 3 in the article
gives the principal rules in English, including the conditional treatment of
confidently linked segments across occlusion. This code repository contains
the decision tables and source windows, not the larger image package.

## Stages

1. Expert 1 interpreted imagery with the reference mask hidden and recorded
   `PRESENT`, `ABSENT` or `UNCERTAIN`.
2. Expert 1 inspected reference masks for 75 cases. The other 25 cases had
   concordant absent decisions and empty references.
3. Expert 2 inspected all 23 discordant/uncertain cases and 16 randomly sampled
   concordant cases. Masks were visible, but Expert 1's decisions were hidden.

Original decisions are preserved in `all_100_decisions.csv`. All 75 and 39
mask-visible inspections accepted the existing reference. Initial agreement
was 77/90 among determinate decisions; the remaining 10 initial decisions were
uncertain. These are sample descriptions, not population accuracy estimates.
The experts did not independently redraw pixel boundaries.

## Explanatory provenance

The review form permitted blank comments for accepted masks. Original note
columns remain unchanged. The author later explained the seven present/empty
cases as reconsidered interpretations and the six absent/non-empty cases as
overlooked small targets. These group-level accounts are retained in separate
columns and are not presented as contemporaneous, independently confirmed
case-specific notes. No explanatory reason is inferred for the ten uncertain
cases. `later_explanation_individually_confirmed=not_recorded` describes the
available documentation, not a further expert decision.

## Inspect a case

Join tables using `case_id` and `record_id`. In the separate contextual
reassessment package, each focus-case folder contains:

- `enhanced.png`: the distributed enhanced patch.
- `source_values.tif`: the corresponding distributed uint16 patch.
- `reference_mask.png`: the unchanged reference mask.
- `full_scene_locator.png`: the scene overview and marked patch position used
  in the review material.
- `expert_context.png`: the supplied contextual review display.
- `context_source_values.tif`: source-raster neighbourhood for direct inspection.
- `context_rgb.png`: a declared per-band 2nd-98th percentile RGB rendering of
  that neighbourhood, prepared for the article's case presentation.
- `case.json`: recorded decisions, spatial window and display-transform details.

The source and enhanced displays derive from the same observation; they are
not independent measurements. Files show actual recorded evidence and do not
add inferred flight matches or new reviewer decisions.

## Source localization

`source_windows_100.csv` provides verified zero-based column and row offsets
in an identified parent raster, with width and height in pixels. Windows are
half-open: columns `[col_offset_px, col_offset_px + width_px)` and rows
`[row_offset_px, row_offset_px + height_px)`. Source dimensions and available
CRS/affine coefficients describe the parent grid. An identity affine with an
empty CRS is not a geolocation. Case locators use the corresponding overview
dimensions and preserve the parent-scene coordinate relationship.
