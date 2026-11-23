# Attribution and scope

Author-developed code in this package implements the completed ContrailStruct30
experiments and is offered under the MIT licence in `LICENSE`.

The segmentation networks are the complete public architectures supplied by
segmentation-models-pytorch 0.5.0 (MIT):
https://github.com/qubvel-org/segmentation_models.pytorch

The implementations depend on PyTorch (BSD-style), torchvision (BSD-style),
timm (Apache-2.0), Albumentations 1.4.24 (MIT) and other packages identified in
the environment files. The dependencies and ImageNet initialization weights
retain their own licences and distribution conditions. The package does not
relicense third-party software or original full-scene source observations.

The dataset's enhanced images, binary masks, corresponding three-band uint16
source-value TIFF patches, and author-generated metadata are released under
CC BY 4.0. Permission to redistribute the derived 256 x 256 source-value
patches was obtained from the SDGSAT-1 Open Science Program. The release does
not include original full-scene observations; those remain subject to the
applicable source-data terms and are not redistributed here. The data portal is
https://data.sdgsat.ac.cn/. Review context is linked by source product and pixel
window rather than by redistributing full scenes.

Historical flight trajectories used in candidate search were accessed through
ADSB.lol (https://adsb.lol/). The flight archive itself is not redistributed here.
