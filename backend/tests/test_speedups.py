"""Speed-ups must not change results: tile skipping, scene cache, instant re-threshold."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import synth
from app.imagery.cache import SceneCache
from app.imagery.geo import aoi_mask, parse_geometry
from app.imagery.masks import build_mask
from test_masks_and_providers import _FakeClient, _stac_item


def test_tile_skipping_is_exact_for_analysed_pixels(engine):
    refl, scl, _ = synth.scene(520, 700)
    valid = build_mask(refl, scl, np.ones(scl.shape, bool)).valid
    valid[:, :300] = False                      # pretend a large land/cloud block on the west
    full = engine.predict(refl)
    fast = engine.predict(refl, needed=valid)
    assert fast.tiles_skipped > 0 and fast.tiles + fast.tiles_skipped == full.tiles
    # same tiles, same blending; only float rounding from different batch composition (~1e-9 here)
    np.testing.assert_allclose(fast.debris_prob[valid], full.debris_prob[valid], atol=1e-5)
    np.testing.assert_array_equal(fast.class_map[valid], full.class_map[valid])


def test_scene_cache_roundtrip_is_lossless(tmp_path):
    from app.imagery.base import SceneData, SceneInfo
    from app.imagery.geo import grid_for
    aoi = parse_geometry(synth.bbox_lonlat(64, 80, 0))
    g = grid_for(aoi)
    refl, scl, _ = synth.scene(g.height, g.width)
    refl[0, 0, 0] = np.nan
    data = SceneData(refl, scl, g, SceneInfo("x,y", "stac", "2026-10-05T00:00:00+00:00", 3.0))
    c = SceneCache(tmp_path / "c", max_gb=1)
    k = c.key("stac", "col", "x,y", aoi)
    assert c.get(k) is None
    c.put(k, data)
    back = c.get(k)
    np.testing.assert_array_equal(back.reflectance, data.reflectance)
    np.testing.assert_array_equal(back.scl, data.scl)
    assert back.grid == g and back.scene.id == "x,y" and back.scene.cloud_cover == 3.0
    assert c.key("stac", "col", "x,z", aoi) != k


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _wait(client, rid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_second_run_uses_cache_and_gives_identical_result(client, tmp_path):
    from app.imagery.registry import providers
    refl, scl, _ = synth.scene(300, 400)
    item = _stac_item(tmp_path, refl, scl, "S2B_16PCC_20261005_0_L2A")
    prov = providers()["stac"]
    prov._client = _FakeClient([item])
    prov._items_seen.clear()
    body = {"geometry": synth.bbox_lonlat(300, 400, 10), "provider": "stac", "scene_id": item.id}
    a = _wait(client, client.post("/api/runs", json=body).json()["id"])
    b = _wait(client, client.post("/api/runs", json=body).json()["id"])
    assert a["status"] == b["status"] == "done", (a["error"], b["error"])
    assert a["stats"]["from_cache"] is False and b["stats"]["from_cache"] is True
    for k in ("max_prob", "mean_prob", "debris_pixels", "clusters", "analysed_km2", "fractions"):
        assert a["stats"][k] == b["stats"][k], k
    da = client.get(a["layers"]["detections"]["url"]).json()
    db_ = client.get(b["layers"]["detections"]["url"]).json()
    assert da["features"] == db_["features"]
    assert client.get("/api/info").json()["cache"]["entries"] >= 1


def test_reprocess_matches_fresh_run(client, tmp_path):
    refl, scl, _ = synth.scene()
    names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
    p = synth.write_tif(tmp_path / "s.tif", np.concatenate([refl, scl[None].astype(np.float32)]), descriptions=names)
    with open(p, "rb") as f:
        sid = client.post("/api/uploads", files={"file": ("s.tif", f, "image/tiff")}).json()["scene_id"]
    g = synth.bbox_lonlat()
    base = _wait(client, client.post("/api/runs", json={"geometry": g, "provider": "upload", "scene_id": sid,
                                                         "threshold": 0.5}).json()["id"])
    fresh9 = _wait(client, client.post("/api/runs", json={"geometry": g, "provider": "upload", "scene_id": sid,
                                                           "threshold": 0.9, "min_cluster_pixels": 5}).json()["id"])
    t = time.time()
    r = client.post(f"/api/runs/{base['id']}/reprocess", json={"threshold": 0.9, "min_cluster_pixels": 5})
    assert r.status_code == 200, r.text
    assert time.time() - t < 5
    re9 = r.json()
    for k in ("clusters", "debris_pixels", "debris_area_m2", "threshold", "min_cluster_pixels"):
        assert re9["stats"][k] == fresh9["stats"][k], k
    f1 = client.get(re9["layers"]["detections"]["url"]).json()["features"]
    f2 = client.get(fresh9["layers"]["detections"]["url"]).json()["features"]
    assert [x["properties"] for x in f1] == [x["properties"] for x in f2]
    assert client.post(f"/api/runs/{base['id']}/reprocess", json={"threshold": 2}).status_code == 422
