from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from contrail_benchmark.segmentation.inference import (  # noqa: E402
    hann_weight,
    infer_raster,
    sliding_origins,
)


def test_sliding_origins_cover_final_edge() -> None:
    assert sliding_origins(100, 256, 128) == [0]
    assert sliding_origins(600, 256, 128) == [0, 128, 256, 384, 512]


def test_hann_weights_are_positive_and_fixed_size() -> None:
    weight = hann_weight(32)
    assert weight.shape == (32, 32)
    assert np.all(weight > 0)
    assert weight[16, 16] > weight[0, 0]


def test_whole_scene_overlap_blending_on_synthetic_raster(tmp_path) -> None:
    rasterio = pytest.importorskip("rasterio")
    torch = pytest.importorskip("torch")

    class ZeroLogitModel(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(
                (batch.shape[0], 1, batch.shape[2], batch.shape[3]),
                dtype=batch.dtype,
                device=batch.device,
            )

    source = tmp_path / "synthetic_scene.tif"
    data = np.arange(3 * 61 * 79, dtype=np.uint16).reshape(3, 61, 79) % 255
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=79,
        height=61,
        count=3,
        dtype="uint16",
        transform=rasterio.transform.from_origin(0.0, 61.0, 1.0, 1.0),
    ) as destination:
        destination.write(data)

    probability, profile = infer_raster(
        ZeroLogitModel(),
        source,
        tile=32,
        overlap=0.5,
        batch_size=3,
        input_mode="enhanced-8bit",
        device=torch.device("cpu"),
    )
    assert probability.shape == (61, 79)
    assert np.allclose(probability, 0.5, atol=1e-6)
    assert profile["width"] == 79
    assert profile["height"] == 61
