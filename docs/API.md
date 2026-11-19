# REST API (v1)

Base URL `http://localhost:8000`. Interactive OpenAPI docs: `/docs`. If `GLN_API_KEY` is set, send `X-API-Key: <key>`
on every request except `/api/health`. Geometries are GeoJSON in EPSG:4326 (Polygon, MultiPolygon, Feature or
FeatureCollection).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness: `{status, model_loaded, device}` |
| GET | `/api/info` | model metadata, providers and availability, defaults, limits, queue |
| POST | `/api/scenes/search` | `{provider?, geometry, start_date, end_date, max_cloud?, limit?}` → scenes (id, datetime, cloud_cover, coverage, tiles, thumbnail) |
| POST | `/api/uploads` | multipart `file` (.tif) + `acquired?` + `dn_offset?` → upload metadata incl. `scene_id` |
| GET / DELETE | `/api/uploads`, `/api/uploads/{id}` | list / remove uploads |
| POST | `/api/runs` | start a detection (202). Body: `geometry` or `aoi_id`, `provider?`, `scene_id?` (else auto-pick within `start_date`..`end_date`), `max_cloud`, `threshold?`, `min_cluster_pixels?`, `mask_land?`, `mask_clouds?` |
| GET | `/api/runs?aoi_id=&limit=` | run history (newest first) |
| GET | `/api/runs/{id}` | status (`queued → searching → fetching → inferring → rendering → done / failed`), `progress`, `message`, `error`, `scene`, `stats`, `layers`, `queue_position` |
| DELETE | `/api/runs/{id}` | delete a run and its files |
| GET | `/api/runs/{id}/files/{name}` | `heatmap.png`, `rgb.png`, `classes.png` (Web Mercator overlays; corners in `layers.*.coordinates`), `detections.geojson`, `probability.tif`, `classes.tif`, `report.json` |
| GET / POST | `/api/aois` | list (with last run) / create `{name, geometry, notes?}` |
| GET / PATCH / DELETE | `/api/aois/{id}` | read / rename or notes / delete (`?delete_runs=true` also removes its runs) |
| POST | `/api/aois/{id}/runs` | run on the best scene of the last `days_back` days: `{provider?, days_back=30, max_cloud=30, threshold?}` |

### `stats` in a finished run

`aoi_km2`, `analysed_km2`, `fractions{valid,cloud,land,nodata}`, `mask_method` (`scl`/`spectral`), `threshold`,
`debris_pixels`, `debris_area_m2`, `clusters`, `max_prob`, `mean_prob`, `class_fractions{class: share of analysed
water}`, `grid{crs,width,height,res_m}`, `timing_s{fetch,inference,render,total}`, `tiles`, `model_sha256`, `warnings[]`.

### Detection features (`detections.geojson`)

Polygon per cluster with properties `rank, pixels, area_m2, max_prob, mean_prob, lon, lat, dominant_class`, ranked
by `max_prob × log(1 + pixels)`, capped at 500.

### Errors

`400` provider problem (unavailable, no scene, no coverage) · `401` API key · `404` not found ·
`409` deleting a run in progress · `413` upload too large · `422` invalid geometry, dates or area limit.
Failed runs keep a readable reason in `error`.
