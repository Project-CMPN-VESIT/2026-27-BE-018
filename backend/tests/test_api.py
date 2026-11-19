import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import synth


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _wait(client, run_id, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.3)
    raise AssertionError("run did not finish")


def test_health_and_info(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_loaded"]
    info = client.get("/api/info").json()
    assert info["model"]["classes"][0] == "Marine Debris"
    names = {p["name"]: p for p in info["providers"]}
    assert names["stac"]["available"] and names["upload"]["available"] and not names["gee"]["available"]


def test_validation_errors(client):
    big = {"type": "Polygon", "coordinates": [[[-90, 10], [-80, 10], [-80, 20], [-90, 20], [-90, 10]]]}
    assert client.post("/api/runs", json={"geometry": big}).status_code == 422
    pt = {"type": "Point", "coordinates": [0, 0]}
    assert client.post("/api/runs", json={"geometry": pt}).status_code == 422
    r = client.post("/api/runs", json={"geometry": synth.bbox_lonlat(), "provider": "gee"})
    assert r.status_code == 400 and "unavailable" in r.json()["detail"]
    assert client.post("/api/runs", json={"geometry": synth.bbox_lonlat(), "provider": "nope"}).status_code == 400
    assert client.get("/api/runs/doesnotexist").status_code == 404
    assert client.get("/api/runs/x/files/../../etc/passwd").status_code == 404
    assert client.post("/api/uploads", files={"file": ("a.txt", b"hello")}).status_code == 422


def test_full_flow_upload_aoi_run(client, tmp_path):
    refl, scl, debris = synth.scene()
    arr = np.concatenate([refl, scl[None].astype(np.float32)])
    names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
    p = synth.write_tif(tmp_path / "scene.tif", arr, descriptions=names)
    with open(p, "rb") as f:
        up = client.post("/api/uploads", files={"file": ("scene.tif", f, "image/tiff")},
                         data={"acquired": "2026-10-05T16:20:00Z"})
    assert up.status_code == 201, up.text
    scene_id = up.json()["scene_id"]

    aoi = client.post("/api/aois", json={"name": "Test bay", "geometry": synth.bbox_lonlat()}).json()
    assert 9 < aoi["area_km2"] < 12

    found = client.post("/api/scenes/search", json={"provider": "upload", "geometry": aoi["geometry"],
                                                     "start_date": "2026-01-01", "end_date": "2026-12-31"}).json()
    assert found["count"] >= 1 and found["scenes"][0]["id"] == scene_id

    r = client.post("/api/runs", json={"aoi_id": aoi["id"], "provider": "upload", "scene_id": scene_id,
                                       "threshold": 0.5})
    assert r.status_code == 202, r.text
    run = _wait(client, r.json()["id"])
    assert run["status"] == "done", run["error"]
    st = run["stats"]
    assert st["mask_method"] == "scl"
    assert st["fractions"]["land"] > 0.1 and st["fractions"]["cloud"] > 0.02
    assert st["analysed_km2"] > 5 and 0 <= st["max_prob"] <= 1
    for name in ("heatmap", "rgb", "classes"):
        lay = run["layers"][name]
        assert len(lay["coordinates"]) == 4
        img = client.get(lay["url"])
        assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"
    gj = client.get(run["layers"]["detections"]["url"]).json()
    assert gj["type"] == "FeatureCollection"
    for f in gj["features"]:
        assert f["properties"]["max_prob"] >= 0.5 and f["geometry"]["type"] in ("Polygon", "MultiPolygon")
    tif = client.get(f"/api/runs/{run['id']}/files/probability.tif")
    assert tif.status_code == 200 and "attachment" in tif.headers["content-disposition"]
    rep = client.get(f"/api/runs/{run['id']}/files/report.json").json()
    assert rep["stats"]["model_sha256"] == st["model_sha256"]

    lst = client.get("/api/aois").json()
    assert lst[0]["last_run"]["id"] == run["id"]
    assert client.get(f"/api/runs?aoi_id={aoi['id']}").json()[0]["id"] == run["id"]

    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert client.delete(f"/api/aois/{aoi['id']}").status_code == 204
    assert client.delete(f"/api/uploads/{scene_id.split(':')[1]}").status_code == 204


def test_failed_run_reports_reason(client):
    r = client.post("/api/runs", json={"geometry": synth.bbox_lonlat(), "provider": "upload",
                                       "scene_id": "upload:missing"})
    run = _wait(client, r.json()["id"])
    assert run["status"] == "failed" and "not found" in run["error"]
