# Frozen height and LUT resources

This directory contains author-generated frozen artifacts needed to reproduce
the model-assisted height constraint and the thermal-infrared lookup-table
step.

Serialized `joblib` models must only be loaded from this trusted release;
Python model serialization is not a safe exchange format for untrusted files.
The model records do not constitute independent cloud-height truth.

`artifact_manifest.json` records file sizes, SHA-256 hashes, roles, training
runtime versions and the libRadtran version stored by the LUT. Verify those
hashes before loading either serialized model.

The CALIOP-calibrated joblib contains a NumPy 2 random-state serialization and
does not load correctly under NumPy 1.26. Use the supplied NumPy 2.2 release
environment; this is an artifact-format requirement rather than a retrieval
assumption.

The LUT contains 12 effective optical-depth nodes, cloud-top heights from
6.5 to 12.5 km, five layer thicknesses, eight effective radii and five view
angles. Its global attributes identify `libRadtran-2.0.4`.
