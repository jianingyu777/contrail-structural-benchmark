# Data-access boundary

## Included in this repository

- source code and tests under `src/` and `tests/`;
- configuration files and software-environment records;
- the author-generated frozen libRadtran LUT and height-prior model artifacts;
- synthetic example tables that contain no observational records;
- cryptographic manifests and provenance documentation.

## Not redistributed here

- provider-controlled SDGSAT-1 source imagery;
- the internal 16-bit provenance mirror;
- raw ABI, SEVIRI or ERA5 source products;
- raw third-party ADS-B archives;
- the enhanced 8-bit patch corpus;
- the 337-MB production checkpoint in Git history.

The segmentation checkpoint should be attached to a versioned GitHub release
or deposited in a DOI-bearing archive. The 30-m segmentation dataset is not
bundled with this software. Its DOI will be made publicly available after
acceptance of the associated article. Provider-controlled source imagery and
the internal 16-bit provenance mirror remain outside this software release.

## Metadata still to complete before public deposit

- add the software release DOI after an archival repository mints it;
- add the associated article DOI after publication;
- add the segmentation-dataset DOI after article acceptance;
- publish repository metadata even if a restricted dataset cannot be released.
