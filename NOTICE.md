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
relicense third-party software or source observations.

The dataset's enhanced images, binary masks and author-generated metadata use
CC BY 4.0. SDGSAT-1 source-value TIFFs and full-scene/context material remain
subject to the applicable source-data use and redistribution terms. The data
portal is https://data.sdgsat.ac.cn/. Original full-scene files are not included
in the compact reproducibility package; review context is linked by source
product and pixel window.

Historical flight trajectories used in candidate search were accessed through
ADSB.lol (https://adsb.lol/). The flight archive itself is not redistributed here.
