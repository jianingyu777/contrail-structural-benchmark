# Data-access boundary

## Included in this repository

- source code and tests under `src/` and `tests/`;
- configuration files and software-environment records;
- the author-generated frozen libRadtran LUT and height-prior model artifacts;
- synthetic example tables that contain no observational records;
- cryptographic manifests and provenance documentation.

## Not redistributed here

- original SDGSAT-1 full-scene imagery, whose distribution is subject to
  authorization by the data provider;
- raw ABI, SEVIRI or ERA5 source products;
- raw ADS-B archives obtained from ADSB.lol Globe History;
- the enhanced 8-bit patch corpus;
- the 337-MB production checkpoint, which is excluded from Git history.

The segmentation checkpoint should be attached to a versioned GitHub release
or deposited in a DOI-bearing archive. The 30-m segmentation dataset is not
bundled with this software. Its DOI will be made publicly available after
acceptance of the associated article. Distribution of the original SDGSAT-1
full-scene imagery is subject to authorization by the data provider and is not
granted by this software release.

## Third-party ADS-B source and licence

ADS-B trajectories used in the geometric audit were obtained from the
[ADSB.lol Globe History](https://github.com/adsblol/globe_history_2024) daily
archives, distributed under the
[Open Database License v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
We thank ADSB.lol, its contributing feeders and partner networks for
maintaining and openly sharing the historical aircraft-trace archive. The raw
ADS-B archives are not redistributed with this software.

## Metadata still to complete before public deposit

- add the software release DOI after an archival repository mints it;
- add the associated article DOI after publication;
- add the segmentation-dataset DOI after article acceptance;
- publish repository metadata even if a restricted dataset cannot be released.
