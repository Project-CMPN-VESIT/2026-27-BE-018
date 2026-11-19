"""GEE adapter against a fake `ee` module (no Google credentials or network needed)."""
import sys
import types
from datetime import date

import numpy as np
import pytest

import synth
from app.imagery.gee import EE_BANDS, GeeProvider
from app.imagery.geo import grid_for, parse_geometry

REFL, SCL, _ = synth.scene(300, 400)
FULL_GRID = None


def _fake_ee(calls):
    ee = types.ModuleType("ee")

    class Chain:
        def __init__(self, *a, **k): pass
        def __getattr__(self, name): return lambda *a, **k: self
        def getInfo(self):
            return {"features": [
                {"geometry": synth.bbox_lonlat(), "properties": {"idx": "20261005T162021_20261005T162544_T16PCC",
                 "t": 1791217221000, "cloud": 7.5, "sat": "Sentinel-2B", "tile": "16PCC"}}]}

    ee.ImageCollection = Chain
    ee.Image = Chain
    ee.Geometry = lambda g: g
    ee.Feature = Chain
    ee.Filter = types.SimpleNamespace(lte=lambda *a: None)
    ee.ServiceAccountCredentials = lambda email, key: ("creds", email, key)
    ee.Initialize = lambda creds=None, project=None: calls.append(("init", project))

    def compute_pixels(req):
        g = req["grid"]
        w, h = g["dimensions"]["width"], g["dimensions"]["height"]
        x0, y0 = g["affineTransform"]["translateX"], g["affineTransform"]["translateY"]
        c0 = int(round((x0 - FULL_GRID.transform.c) / 10)) + OFF[1]
        r0 = int(round((FULL_GRID.transform.f - y0) / 10)) + OFF[0]
        calls.append(("px", r0, c0, h, w))
        arr = np.zeros((h, w), dtype=[(b, "<u2") for b in EE_BANDS + ["SCL"]])
        for k, b in enumerate(EE_BANDS):
            arr[b] = (REFL[k, r0:r0 + h, c0:c0 + w] * 10000).astype(np.uint16)
        arr["SCL"] = SCL[r0:r0 + h, c0:c0 + w]
        return arr

    ee.data = types.SimpleNamespace(computePixels=compute_pixels)
    return ee


OFF = (0, 0)


def test_gee_search_and_load(tmp_path, monkeypatch):
    global FULL_GRID, OFF
    calls = []
    monkeypatch.setitem(sys.modules, "ee", _fake_ee(calls))
    key = tmp_path / "k.json"
    key.write_text("{}")
    prov = GeeProvider(True, "my-proj", "sa@my-proj.iam.gserviceaccount.com", key, "COPERNICUS/S2_SR_HARMONIZED")
    assert prov.status()[0]
    aoi = parse_geometry(synth.bbox_lonlat(300, 400, 20))
    FULL_GRID = grid_for(aoi)
    # offset of the AOI grid inside the synthetic scene (scene origin = synth.ORIGIN)
    OFF = (int(round((synth.ORIGIN[1] - FULL_GRID.transform.f) / 10)), int(round((FULL_GRID.transform.c - synth.ORIGIN[0]) / 10)))
    monkeypatch.setattr("app.imagery.gee.BLOCK", 100)    # force several computePixels blocks

    scenes = prov.search(aoi, date(2026, 10, 1), date(2026, 10, 31), 30, 10)
    assert scenes[0].cloud_cover == 7.5 and scenes[0].tiles == ["16PCC"] and scenes[0].coverage > 0.99
    data = prov.load(scenes[0].id, aoi)
    assert ("init", "my-proj") in calls
    assert sum(1 for c in calls if c[0] == "px") >= 6
    assert data.reflectance.shape == (11, *FULL_GRID.shape)
    exp = REFL[:, OFF[0]:OFF[0] + FULL_GRID.height, OFF[1]:OFF[1] + FULL_GRID.width]
    np.testing.assert_allclose(data.reflectance, exp, atol=1.1e-4)    # DN/10000 quantisation
    assert (data.scl == SCL[OFF[0]:OFF[0] + FULL_GRID.height, OFF[1]:OFF[1] + FULL_GRID.width]).all()


def test_gee_disabled_status():
    ok, why = GeeProvider(False, "", "", None, "C").status()
    assert not ok and "disabled" in why


def test_gee_missing_key(tmp_path):
    ok, why = GeeProvider(True, "p", "", tmp_path / "nope.json", "C").status()
    assert not ok and "not found" in why
