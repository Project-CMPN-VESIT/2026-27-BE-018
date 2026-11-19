import { useEffect, useRef, useState } from "react";
import { api, type Info, type Polygon, type Run, type Scene, type Upload } from "./api";
import { areaKm2, fmt, geometryFromGeoJSON } from "./geo";

interface Props {
  info: Info;
  aoi: Polygon | null;
  aoiId: string | null;
  aoiName?: string;
  setAoi(p: Polygon | null): void;
  drawing: boolean;
  startDraw(): void;
  useView(): void;
  onRun(r: Run): void;
  onSaveArea(name: string): Promise<void>;
  toast(msg: string, kind?: "error" | "ok"): void;
}

const daysAgo = (n: number) => fmt.isoDay(new Date(Date.now() - n * 864e5));

export function DetectPanel({ info, aoi, aoiId, aoiName, setAoi, drawing, startDraw, useView, onRun, onSaveArea, toast }: Props) {
  const firstAvailable = info.providers.find((p) => p.name === info.default_provider && p.available)?.name
    ?? info.providers.find((p) => p.available)?.name ?? "stac";
  const [provider, setProvider] = useState(firstAvailable);
  const [start, setStart] = useState(daysAgo(info.defaults.days_back));
  const [end, setEnd] = useState(daysAgo(0));
  const [maxCloud, setMaxCloud] = useState(info.defaults.max_cloud);
  const [scenes, setScenes] = useState<Scene[] | null>(null);
  const [sceneId, setSceneId] = useState<string>("auto");
  const [busy, setBusy] = useState<"" | "search" | "run" | "upload" | "save">("");
  const [adv, setAdv] = useState(false);
  const [threshold, setThreshold] = useState(info.defaults.threshold);
  const [minPx, setMinPx] = useState(info.defaults.min_cluster_pixels);
  const [maskLand, setMaskLand] = useState(info.defaults.mask_land);
  const [maskClouds, setMaskClouds] = useState(info.defaults.mask_clouds);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [acquired, setAcquired] = useState("");
  const [dnOffset, setDnOffset] = useState(0);
  const [areaName, setAreaName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const gjRef = useRef<HTMLInputElement>(null);

  const area = aoi ? areaKm2(aoi) : 0;
  const tooBig = area > info.limits.max_aoi_km2;
  const isUpload = provider === "upload";

  useEffect(() => { setScenes(null); setSceneId("auto"); }, [provider]);
  // a new area / date window invalidates searched satellite scenes (an uploaded image stays selected)
  useEffect(() => { setScenes(null); if (provider !== "upload") setSceneId("auto"); }, [aoi, start, end, maxCloud]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (isUpload) api.uploads().then(setUploads).catch((e) => toast(e.message, "error")); }, [isUpload]);

  async function search() {
    if (!aoi) return;
    setBusy("search");
    try {
      const r = await api.searchScenes({ provider, geometry: aoi, start_date: start, end_date: end, max_cloud: maxCloud, limit: 30 });
      setScenes(r.scenes);
      if (!r.scenes.length) toast("No scenes match. Widen the dates or raise the cloud limit.");
    } catch (e: any) { toast(e.message, "error"); } finally { setBusy(""); }
  }

  async function run() {
    setBusy("run");
    try {
      const body: Record<string, unknown> = {
        geometry: aoi, aoi_id: aoiId ?? undefined, provider, max_cloud: maxCloud, threshold, min_cluster_pixels: minPx,
        mask_land: maskLand, mask_clouds: maskClouds,
      };
      if (isUpload) {
        if (sceneId === "auto") throw new Error("Choose an uploaded image first.");
        body.scene_id = sceneId;
      } else if (sceneId !== "auto") body.scene_id = sceneId;
      else { body.start_date = start; body.end_date = end; }
      onRun(await api.createRun(body));
    } catch (e: any) { toast(e.message, "error"); } finally { setBusy(""); }
  }

  async function doUpload(f: File) {
    setBusy("upload");
    try {
      const u = await api.upload(f, acquired, dnOffset);
      setUploads((x) => [u, ...x]);
      setSceneId(u.scene_id);
      if (!aoi) setAoi(u.footprint);
      toast(`Uploaded ${u.filename} (${u.value_type}${u.has_scl ? ", with SCL" : ""})`, "ok");
    } catch (e: any) { toast(e.message, "error"); } finally {
      setBusy("");
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function importGeoJSON(f: File) {
    try { setAoi(geometryFromGeoJSON(JSON.parse(await f.text()))); }
    catch (e: any) { toast(`GeoJSON: ${e.message}`, "error"); }
    finally { if (gjRef.current) gjRef.current.value = ""; }
  }

  const canRun = !!aoi && !tooBig && !busy && (!isUpload || sceneId !== "auto");

  return (
    <div className="panel">
      <section>
        <h3><span className="step">1</span> Area of interest</h3>
        <div className="btn-row">
          <button className={drawing ? "primary" : ""} onClick={startDraw}>{drawing ? "Drag on the map…" : "Draw rectangle"}</button>
          <button onClick={useView}>Use map view</button>
          <button onClick={() => gjRef.current?.click()}>Import GeoJSON</button>
          <input ref={gjRef} type="file" accept=".geojson,.json,application/geo+json" hidden onChange={(e) => e.target.files?.[0] && importGeoJSON(e.target.files[0])} />
        </div>
        {aoi ? (
          <div className={`aoi-info ${tooBig ? "bad" : ""}`}>
            <span>{fmt.km2(area)}</span>
            {tooBig ? <span>exceeds the {info.limits.max_aoi_km2} km² limit; draw a smaller area</span> : <span>≈ {Math.max(1, Math.round(area / 3))} s to analyse</span>}
            <button className="link" onClick={() => setAoi(null)}>clear</button>
          </div>
        ) : <p className="hint">Draw a box over coastal water, river mouths or known accumulation zones.</p>}
        {aoi && aoiName && <p className="hint">Saved area: <b>{aoiName}</b> (runs are added to its history)</p>}
        {aoi && !tooBig && !aoiId && (
          <div className="save-row">
            <input placeholder="Name this area to save it" value={areaName} maxLength={120} onChange={(e) => setAreaName(e.target.value)} />
            <button disabled={!areaName.trim() || busy === "save"} onClick={async () => {
              setBusy("save");
              try { await onSaveArea(areaName.trim()); setAreaName(""); } finally { setBusy(""); }
            }}>Save</button>
          </div>
        )}
      </section>

      <section>
        <h3><span className="step">2</span> Imagery</h3>
        <label className="field">Source
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            {info.providers.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.available}>{p.label}{p.available ? "" : " (not configured)"}</option>
            ))}
          </select>
        </label>
        {!info.providers.find((p) => p.name === provider)?.available && <p className="warn">{info.providers.find((p) => p.name === provider)?.detail}</p>}

        {isUpload ? (
          <>
            <div className="grid2">
              <label className="field">Acquired (optional)<input type="date" value={acquired} onChange={(e) => setAcquired(e.target.value)} /></label>
              <label className="field" title="Only for raw DN files: -1000 for L2A/L1C processed with baseline 04.00 or later">DN offset
                <input type="number" step={100} value={dnOffset} onChange={(e) => setDnOffset(Number(e.target.value))} /></label>
            </div>
            <button className="wide" disabled={busy === "upload"} onClick={() => fileRef.current?.click()}>
              {busy === "upload" ? "Uploading…" : "Upload Sentinel-2 GeoTIFF"}
            </button>
            <input ref={fileRef} type="file" accept=".tif,.tiff" hidden onChange={(e) => e.target.files?.[0] && doUpload(e.target.files[0])} />
            <p className="hint">11 bands (B1–B8, B8A, B11, B12), 12-band L2A or 13-band L1C stacks, or named bands. Reflectance or DN.</p>
            <div className="scene-list">
              {uploads.map((u) => (
                <label key={u.id} className={`scene ${sceneId === u.scene_id ? "sel" : ""}`}>
                  <input type="radio" checked={sceneId === u.scene_id} onChange={() => { setSceneId(u.scene_id); if (!aoi) setAoi(u.footprint); }} />
                  <div><b>{u.filename}</b><small>{u.acquired ? fmt.date(u.acquired) : "date unknown"} · {u.value_type}{u.has_scl ? " · SCL" : ""}</small></div>
                  <button className="link" onClick={(e) => { e.preventDefault(); setAoi(u.footprint); }}>extent</button>
                </label>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="grid2">
              <label className="field">From<input type="date" value={start} max={end} onChange={(e) => setStart(e.target.value)} /></label>
              <label className="field">To<input type="date" value={end} min={start} max={daysAgo(0)} onChange={(e) => setEnd(e.target.value)} /></label>
            </div>
            <label className="field">Max scene cloud cover <b>{maxCloud}%</b>
              <input type="range" min={0} max={100} step={5} value={maxCloud} onChange={(e) => setMaxCloud(Number(e.target.value))} />
            </label>
            <button className="wide" disabled={!aoi || tooBig || busy === "search"} onClick={search}>
              {busy === "search" ? "Searching…" : "Find scenes"}
            </button>
            <div className="scene-list">
              <label className={`scene ${sceneId === "auto" ? "sel" : ""}`}>
                <input type="radio" checked={sceneId === "auto"} onChange={() => setSceneId("auto")} />
                <div><b>Best available scene</b><small>least cloudy pass that covers the area in the date range</small></div>
              </label>
              {scenes?.map((s) => (
                <label key={s.id} className={`scene ${sceneId === s.id ? "sel" : ""}`}>
                  <input type="radio" checked={sceneId === s.id} onChange={() => setSceneId(s.id)} />
                  {s.thumbnail && <img src={s.thumbnail} alt="" loading="lazy" />}
                  <div>
                    <b>{fmt.date(s.datetime)}</b>
                    <small>{s.platform.replace("sentinel-", "S")} · cloud {s.cloud_cover ?? "?"}% · covers {fmt.pct(s.coverage)}{s.tiles.length ? ` · ${s.tiles.join(", ")}` : ""}</small>
                  </div>
                </label>
              ))}
            </div>
          </>
        )}
      </section>

      <section>
        <h3 className="clickable" onClick={() => setAdv(!adv)}><span className="step">3</span> Detection settings <span className="chev">{adv ? "▾" : "▸"}</span></h3>
        {adv && (
          <>
            <label className="field">Debris probability threshold <b>{threshold.toFixed(2)}</b>
              <input type="range" min={0.1} max={0.95} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            </label>
            <label className="field">Minimum cluster size <b>{minPx} px ({minPx * 100} m²)</b>
              <input type="range" min={1} max={30} step={1} value={minPx} onChange={(e) => setMinPx(Number(e.target.value))} />
            </label>
            <label className="check"><input type="checkbox" checked={maskLand} onChange={(e) => setMaskLand(e.target.checked)} /> Ignore land</label>
            <label className="check"><input type="checkbox" checked={maskClouds} onChange={(e) => setMaskClouds(e.target.checked)} /> Ignore clouds and cloud shadow</label>
          </>
        )}
      </section>

      <button className="primary run" disabled={!canRun} onClick={run}>
        {busy === "run" ? "Submitting…" : "Run debris detection"}
      </button>
    </div>
  );
}
