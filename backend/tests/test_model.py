import numpy as np
import torch

from app.ml.preprocess import build_input_stack, compute_non_redundant_indices


def training_reference(raw_11: np.ndarray) -> np.ndarray:
    """Literal copy of MARIDAVisualizationDataset._load_patch input path from the training notebook."""
    raw_11 = np.nan_to_num(raw_11, nan=0.0, posinf=10000.0, neginf=0.0)
    scaled_11 = np.clip(raw_11 / 10000.0, 0.0, 2.0)
    physics_5 = compute_non_redundant_indices(scaled_11)
    return np.concatenate([scaled_11, physics_5], axis=0)


def test_weights_load_strict(engine):
    assert engine.n_params == 2_552_213          # + 2 126 BatchNorm buffers = 2.55 M in the state dict
    assert len(engine.sha256) == 64


def test_checkpoint_was_trained_on_reflectance_div_10000(engine):
    """BN running variance ~1e-11 proves the training inputs were tiny (reflectance / 10 000)."""
    rv = engine.model.pst.raw_stem[1].running_var
    assert float(rv.max()) < 1e-8


def test_preprocess_matches_training():
    rng = np.random.default_rng(1)
    x = (rng.random((11, 64, 64)) * 0.3).astype(np.float32)
    x[:, :3, :3] = np.nan
    ours = build_input_stack(x)
    ref = training_reference(x)
    assert ours.shape == (16, 64, 64) and ours.dtype == np.float32
    np.testing.assert_array_equal(ours, ref)


def test_single_tile_parity_with_plain_forward(engine):
    rng = np.random.default_rng(2)
    refl = (rng.random((11, 256, 256)) * 0.1).astype(np.float32)
    res = engine.predict(refl)
    with torch.inference_mode():
        p = torch.softmax(engine.model(torch.from_numpy(build_input_stack(refl))[None]), 1)[0]
    np.testing.assert_allclose(res.debris_prob, p[0].numpy(), atol=1e-5)
    assert (res.class_map == p.argmax(0).numpy()).mean() > 0.999


def test_tiled_inference_arbitrary_size(engine):
    rng = np.random.default_rng(3)
    refl = (rng.random((11, 300, 517)) * 0.1).astype(np.float32)
    calls = []
    res = engine.predict(refl, progress=lambda f, m: calls.append(f))
    assert res.debris_prob.shape == (300, 517)
    assert np.isfinite(res.debris_prob).all() and (0 <= res.debris_prob).all() and (res.debris_prob <= 1).all()
    assert calls and abs(calls[-1] - 1.0) < 1e-6
    res2 = engine.predict(refl)
    np.testing.assert_array_equal(res.debris_prob, res2.debris_prob)   # deterministic


def test_tiny_image(engine):
    refl = np.full((11, 20, 30), 0.03, np.float32)
    assert engine.predict(refl).debris_prob.shape == (20, 30)
