from datetime import date, datetime, timezone

import numpy as np
import pystac
import pytest
from shapely.geometry import shape

import synth
from app.imagery.base import ProviderError
from app.imagery.geo import aoi_mask, area_km2, grid_for, parse_geometry, GeometryError
from app.imagery.masks import build_mask
from app.imagery.stac import StacProvider
from app.imagery.upload import UploadProvider
from app.pipeline.runner import pick_best_scene
from app.imagery.base import SceneInfo


def test_mask_keeps_debris_labelled_as_vegetation():
    refl, scl, debris = synth.scene()
    inside = np.ones(scl.shape, bool)
    m = build_mask(refl, scl, inside)
    assert m.valid[debris].all(), "floating debris inside water must stay analysable"
    assert not m.valid[:, :40].any(), "coastal land strip must be masked"
    assert m.cloud[-30:, -30:].all() and not m.valid[-30:, -30:].any()
    fr = m.fractions(inside)
    assert abs(sum(fr[k] for k in ("valid", "cloud", "land", "nodata")) - 1) < 0.02


def test_spectral_fallback_without_scl():
    refl, _, debris = synth.scene()
    m = build_mask(refl, None, np.ones(debris.shape, bool))
    assert m.method == "spectral"
    assert m.valid[debris].mean() > 0.9 and not m.valid[:, :40].any()


def test_geometry_validation():
    with pytest.raises(GeometryError):
        parse_geometry({"type": "Point", "coordinates": [0, 0]})
    g = parse_geometry({"type": "Feature", "geometry": synth.bbox_lonlat()})
    assert 9 < area_km2(g) < 12
    grid = grid_for(g)
    assert grid.crs == "EPSG:32616" and grid.transform.a == 10
    assert aoi_mask(g, grid).mean() > 0.95


def test_scene_picker_prefers_coverage_then_cloud():
    a = SceneInfo("a", "stac", "2026-10-01T10:00:00+00:00", 5.0, coverage=0.3)
    b = SceneInfo("b", "stac", "2026-10-02T10:00:00+00:00", 20.0, coverage=1.0)
    c = SceneInfo("c", "stac", "2026-10-03T10:00:00+00:00", 2.0, coverage=1.0)
    assert pick_best_scene([a, b, c]).id == "c"
    assert pick_best_scene([a, b]).id == "b"
    assert pick_best_scene([]) is None


# ---------------------------------------------------------------- upload provider
def test_upload_11band_reflectance(tmp_path):
    refl, _, _ = synth.scene()
    p = synth.write_tif(tmp_path / "s.tif", refl)
    up = UploadProvider(tmp_path / "uploads")
    meta = up.register(p, "s.tif", "2026-10-05")
    assert meta["value_type"] == "reflectance"
    aoi = parse_geometry(synth.bbox_lonlat())
    data = up.load(f"upload:{meta['id']}", aoi)
    assert data.reflectance.shape[0] == 11
    np.testing.assert_allclose(np.nanmedian(data.reflectance[3]), synth.WATER[3], rtol=0.1)
    assert up.search(aoi, date(2026, 1, 1), date(2026, 12, 31), 100, 10)[0].id == f"upload:{meta['id']}"


def test_upload_13band_dn_geographic(tmp_path):
    """L1C-style 13 bands as DN with offset, stored in EPSG:4326 -> reprojected to UTM, B9/B10 dropped."""
    refl, _, _ = synth.scene(120, 160)
    full = np.insert(refl, [9, 9], 0.01, axis=0)          # add B9, B10 after B8A
    dn = (full * 10000 + 1000).astype(np.uint16)          # baseline >= 04.00 style +1000 offset
    from rasterio.warp import calculate_default_transform, reproject
    import rasterio
    src_tr = synth.transform()
    tr, w, h = calculate_default_transform(synth.CRS, "EPSG:4326", 160, 120, *rasterio.transform.array_bounds(120, 160, src_tr))
    out = np.zeros((13, h, w), np.uint16)
    for i in range(13):
        reproject(dn[i], out[i], src_transform=src_tr, src_crs=synth.CRS, dst_transform=tr, dst_crs="EPSG:4326")
    p = synth.write_tif(tmp_path / "l1c.tif", out, crs="EPSG:4326", tr=tr, nodata=0)
    up = UploadProvider(tmp_path / "uploads")
    meta = up.register(p, "l1c.tif", dn_offset=-1000)
    assert meta["value_type"] == "DN" and meta["bands"]["B11"] == 12
    data = up.load(f"upload:{meta['id']}", shape(meta["footprint"]))
    assert data.grid.crs == "EPSG:32616"
    np.testing.assert_allclose(np.nanmedian(data.reflectance[3]), synth.WATER[3], atol=0.004)


def test_upload_named_bands_any_order(tmp_path):
    refl, scl, _ = synth.scene(64, 64)
    names = ["B12", "B11", "B8A", "B8", "B7", "B6", "B5", "B4", "B3", "B2", "B1"]
    arr = np.concatenate([refl[::-1], scl[None].astype(np.float32)])
    p = synth.write_tif(tmp_path / "n.tif", arr, descriptions=names + ["SCL"])
    up = UploadProvider(tmp_path / "uploads")
    meta = up.register(p, "n.tif")
    assert meta["has_scl"] and meta["bands"]["B01"] == 11
    data = up.load(f"upload:{meta['id']}", shape(meta["footprint"]))
    np.testing.assert_allclose(np.nanmedian(data.reflectance[0]), synth.WATER[0], rtol=0.1)
    assert data.scl is not None and (data.scl == 6).mean() > 0.5


def test_upload_rejects_bad_layout(tmp_path):
    p = synth.write_tif(tmp_path / "x.tif", np.zeros((4, 10, 10), np.float32))
    with pytest.raises(ProviderError):
        UploadProvider(tmp_path / "uploads").register(p, "x.tif")


# ---------------------------------------------------------------- STAC provider (local COGs, no network)
class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeClient:
    def __init__(self, items):
        self.items = items

    def search(self, ids=None, **kw):
        return _FakeSearch([i for i in self.items if ids is None or i.id in ids])


def _stac_item(tmp_path, refl, scl, item_id, baseline_offset=True, boa_applied=None):
    keys = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
    res_of = {"coastal": 60, "rededge1": 20, "rededge2": 20, "rededge3": 20, "nir08": 20, "swir16": 20, "swir22": 20}
    item = pystac.Item(item_id, geometry=None, bbox=None, datetime=datetime(2026, 10, 5, 16, 20, tzinfo=timezone.utc),
                       properties={"eo:cloud_cover": 12.5, "platform": "sentinel-2b", "proj:epsg": 32616,
                                   "s2:mgrs_tile": "16PCC", "s2:processing_baseline": "05.11"})
    if boa_applied is not None:
        item.properties["earthsearch:boa_offset_applied"] = boa_applied
    h, w = scl.shape
    for k, key in enumerate(keys):
        r = res_of.get(key, 10)
        f = r // 10
        band = refl[k][: h - h % f, : w - w % f].reshape(h // f, f, w // f, f).mean((1, 3))
        dn = (band * 10000 + (1000 if baseline_offset and not boa_applied else 0)).astype(np.uint16)
        p = synth.write_tif(tmp_path / f"{item_id}_{key}.tif", dn, tr=synth.transform(r), nodata=0)
        a = pystac.Asset(str(p), media_type=pystac.MediaType.COG)
        # Earth Search quirk: offset is still advertised when it has already been applied to the DNs
        a.extra_fields["raster:bands"] = [{"scale": 0.0001, "offset": -0.1 if baseline_offset else 0.0}]
        item.add_asset(key, a)
    item.add_asset("scl", pystac.Asset(str(synth.write_tif(tmp_path / f"{item_id}_scl.tif",
                   scl[::2, ::2].copy(), tr=synth.transform(20))), media_type=pystac.MediaType.COG))
    foot = synth.bbox_lonlat(h, w, 0)
    item.geometry, item.bbox = foot, list(shape(foot).bounds)
    return item


def test_stac_provider_reads_resamples_and_scales(tmp_path):
    refl, scl, _ = synth.scene(240, 320)
    item = _stac_item(tmp_path, refl, scl, "S2B_16PCC_20261005_0_L2A")
    prov = StacProvider("http://fake", "sentinel-2-l2a")
    prov._client = _FakeClient([item])
    aoi = parse_geometry(synth.bbox_lonlat(240, 320, 10))
    scenes = prov.search(aoi, date(2026, 10, 1), date(2026, 10, 31), 50, 10)
    assert len(scenes) == 1 and scenes[0].coverage > 0.99 and scenes[0].cloud_cover == 12.5
    data = prov.load(scenes[0].id, aoi)
    assert data.grid.crs == "EPSG:32616"
    # offset -0.1 must be removed; 20 m / 60 m bands resampled onto the 10 m grid
    for b in (3, 8, 9, 0):
        np.testing.assert_allclose(np.nanmedian(data.reflectance[b]), synth.WATER[b], atol=0.003)
    assert data.scl is not None and set(np.unique(data.scl)) <= {4, 6, 9}


def test_stac_missing_scene(tmp_path):
    prov = StacProvider("http://fake", "sentinel-2-l2a")
    prov._client = _FakeClient([])
    with pytest.raises(ProviderError):
        prov.load("nope", parse_geometry(synth.bbox_lonlat()))


def test_stac_earthsearch_offset_already_applied(tmp_path):
    """boa_offset_applied=True + stale raster:bands offset -0.1 must NOT be subtracted twice."""
    refl, scl, _ = synth.scene(120, 160)
    item = _stac_item(tmp_path, refl, scl, "S2A_X", baseline_offset=True, boa_applied=True)
    prov = StacProvider("http://fake", "sentinel-2-l2a")
    prov._client = _FakeClient([item])
    aoi = parse_geometry(synth.bbox_lonlat(120, 160, 5))
    data = prov.load("S2A_X", aoi)
    assert not np.isnan(data.reflectance).all()
    np.testing.assert_allclose(np.nanmedian(data.reflectance[3]), synth.WATER[3], atol=0.003)


def test_read_to_grid_other_utm_zone_matches_native(tmp_path):
    """Regression for 'Chunk and warp failed': a granule stored in a neighbouring UTM zone is read natively
    and warped in memory; values must match reading the same data in the grid's own CRS."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    from app.imagery.base import read_to_grid
    refl, _, _ = synth.scene(200, 260)
    band = (refl[3] * 10000).astype(np.float32)
    native = synth.write_tif(tmp_path / "native.tif", band)
    b = rasterio.transform.array_bounds(200, 260, synth.transform())
    tr, w, h = calculate_default_transform(synth.CRS, "EPSG:32617", 260, 200, *b, resolution=10)
    other = np.zeros((h, w), np.float32)
    reproject(band, other, src_transform=synth.transform(), src_crs=synth.CRS, dst_transform=tr, dst_crs="EPSG:32617",
              resampling=Resampling.bilinear)
    foreign = synth.write_tif(tmp_path / "zone17.tif", other, crs="EPSG:32617", tr=tr)
    aoi = parse_geometry(synth.bbox_lonlat(200, 260, 30))
    grid = grid_for(aoi, "EPSG:32616")
    a = read_to_grid(str(native), grid, Resampling.bilinear)
    c = read_to_grid(str(foreign), grid, Resampling.bilinear)
    assert c.shape == a.shape and (c > 0).mean() > 0.99
    assert abs(float(np.median(c)) - float(np.median(a))) < 5          # DN units (reflectance x 1e4)
