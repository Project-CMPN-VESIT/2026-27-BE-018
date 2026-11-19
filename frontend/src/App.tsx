import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, apiKey, fileUrl, type AOI, type Detection, type Info, type Polygon, type Run } from "./api";
import { DetectPanel } from "./DetectPanel";
import { bboxOf, rectPolygon } from "./geo";
import { AreasPanel, HistoryPanel } from "./ListsPanels";
import { MapView, type Basemap, type LayerState, type MapHandle } from "./MapView";
import { ResultPanel } from "./ResultPanel";

type Tab = "detect" | "areas" | "history";
type Toast = { id: number; msg: string; kind: "error" | "ok" | "info" };

const DEFAULT_LAYERS: LayerState = {
  density: true,
  heatmap: { on: true, opacity: 0.9 },
  rgb: { on: false, opacity: 1 },
  classes: { on: false, opacity: 0.7 },
  outlines: true,
};

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("detect");
  const [aoi, setAoiRaw] = useState<Polygon | null>(null);
  const [aoiId, setAoiId] = useState<string | null>(null);   // set when the current AOI is a saved area
  const setAoi = useCallback((p: Polygon | null, savedId: string | null = null) => { setAoiRaw(p); setAoiId(savedId); }, []);
  const [drawing, setDrawing] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [detections, setDetections] = useState<Detection[] | null>(null);
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [basemap, setBasemap] = useState<Basemap>("satellite");
  const [aois, setAois] = useState<AOI[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [filterAoi, setFilterAoi] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [panelOpen, setPanelOpen] = useState(true);
  const mapRef = useRef<MapHandle>(null);
  const fittedRun = useRef<string | null>(null);

  const toast = useCallback((msg: string, kind: Toast["kind"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-3), { id, msg, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 9000 : 4500);
  }, []);

  // ---------------------------------------------------------------- bootstrap
  const loadInfo = useCallback(async () => {
    try { setInfo(await api.info()); setFatal(null); }
    catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        const k = prompt("This server requires an API key:");
        if (k) { apiKey.set(k); return loadInfo(); }
      }
      setFatal(e instanceof Error ? e.message : String(e));
    }
  }, []);
  const refreshAois = useCallback(() => api.aois().then(setAois).catch((e) => toast(e.message, "error")), [toast]);
  const refreshRuns = useCallback(() => api.runs(filterAoi ?? undefined).then(setRuns).catch((e) => toast(e.message, "error")), [filterAoi, toast]);
  useEffect(() => { loadInfo(); }, [loadInfo]);
  useEffect(() => { if (info) { refreshAois(); } }, [info, refreshAois]);
  useEffect(() => { if (info) refreshRuns(); }, [info, refreshRuns]);

  // ---------------------------------------------------------------- active run polling
  useEffect(() => {
    if (!runId) { setRun(null); setDetections(null); return; }
    let stop = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const r = await api.run(runId);
        if (stop) return;
        setRun(r);
        if (r.status === "done" || r.status === "failed") {
          refreshRuns(); refreshAois();
          if (r.status === "done" && r.layers) {
            const gj = await (await fetch(await fileUrl(r.layers.detections.url))).json();
            if (!stop) setDetections(gj.features.map((f: any) => f.properties as Detection));
            if (fittedRun.current !== r.id) { fittedRun.current = r.id; mapRef.current?.fitBounds(bboxOf(r.geometry)); }
            if (r.stats) toast(`${r.stats.clusters} debris cluster${r.stats.clusters === 1 ? "" : "s"} found`, "ok");
          }
          return;
        }
        timer = window.setTimeout(tick, 1500);
      } catch (e: any) {
        if (!stop) { toast(e.message, "error"); timer = window.setTimeout(tick, 5000); }
      }
    };
    setDetections(null);
    tick();
    return () => { stop = true; window.clearTimeout(timer); };
  }, [runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const openRun = (r: Run) => {
    setRunId(r.id); setRun(r); setAoi(r.geometry, r.aoi_id);
    if (r.status === "done") { fittedRun.current = null; }
    setPanelOpen(true);
  };

  // ---------------------------------------------------------------- actions
  const onRun = (r: Run) => { fittedRun.current = null; openRun(r); refreshRuns(); toast("Detection queued", "info"); };
  const saveArea = async (name: string) => {
    if (!aoi) return;
    try { const a = await api.createAoi({ name, geometry: aoi }); setAoiId(a.id); toast(`Saved “${name}”`, "ok"); refreshAois(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const useView = () => {
    const b = mapRef.current?.viewBounds();
    if (b) setAoi(rectPolygon([b[0], b[1]], [b[2], b[3]]));
  };

  if (fatal) return (
    <div className="fatal">
      <h1>GeoLitterNet Watch</h1>
      <p>Cannot reach the API: {fatal}</p>
      <p className="muted">Start the backend (see README) and reload.</p>
      <button className="primary" onClick={loadInfo}>Retry</button>
    </div>
  );

  return (
    <div className={`app ${panelOpen ? "" : "collapsed"}`}>
      <aside className="sidebar">
        <header className="brand">
          <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden><circle cx="16" cy="16" r="15" fill="#0e2a3d" /><path d="M4 19c4-3 8 3 12 0s8 3 12 0" stroke="#38bdf8" strokeWidth="2.5" fill="none" /><circle cx="20" cy="12" r="3.5" fill="#f43f5e" /></svg>
          <div>
            <h1>GeoLitterNet Watch</h1>
            <p>Marine debris from Sentinel-2</p>
          </div>
          <button className="icon collapse" title="Hide panel" onClick={() => setPanelOpen(false)}>‹</button>
        </header>
        <nav className="tabs">
          {(["detect", "areas", "history"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "on" : ""} onClick={() => { setTab(t); if (t === "history") refreshRuns(); if (t === "areas") refreshAois(); }}>
              {t === "detect" ? "Detect" : t === "areas" ? `Saved areas${aois.length ? ` (${aois.length})` : ""}` : "History"}
            </button>
          ))}
        </nav>
        <div className="scroll">
          {!info ? <p className="muted pad">Loading model…</p> : tab === "detect" ? (
            <DetectPanel info={info} aoi={aoi} aoiId={aoiId} aoiName={aois.find((a) => a.id === aoiId)?.name} setAoi={(p) => setAoi(p)} drawing={drawing} startDraw={() => setDrawing(true)} useView={useView}
              onRun={onRun} onSaveArea={saveArea} toast={toast} />
          ) : tab === "areas" ? (
            <AreasPanel aois={aois}
              onShow={(a) => { setAoi(a.geometry, a.id); setTab("detect"); mapRef.current?.fitBounds(bboxOf(a.geometry)); }}
              onRunLatest={async (a, days, cloud) => {
                try { onRun(await api.runAoi(a.id, { days_back: days, max_cloud: cloud })); setAoi(a.geometry, a.id); }
                catch (e: any) { toast(e.message, "error"); }
              }}
              onHistory={(a) => { setFilterAoi(a.id); setTab("history"); }}
              onRename={async (a, name) => { try { await api.patchAoi(a.id, { name }); refreshAois(); } catch (e: any) { toast(e.message, "error"); } }}
              onDelete={async (a) => { try { await api.deleteAoi(a.id); if (aoiId === a.id) setAoiId(null); refreshAois(); toast("Area deleted"); } catch (e: any) { toast(e.message, "error"); } }} />
          ) : (
            <HistoryPanel runs={runs} aois={aois} filterAoi={filterAoi} setFilterAoi={setFilterAoi} activeId={runId}
              onOpen={openRun}
              onDelete={async (r) => {
                try { await api.deleteRun(r.id); if (runId === r.id) setRunId(null); refreshRuns(); }
                catch (e: any) { toast(e.message, "error"); }
              }} />
          )}
        </div>
        {info && (
          <footer className="foot muted tiny">
            {info.model.name} · {(info.model.parameters / 1e6).toFixed(2)}M params · {info.model.device.toUpperCase()}
            {info.queue.running ? " · busy" : ""}
          </footer>
        )}
      </aside>

      <main className="mapwrap">
        <MapView ref={mapRef} aoi={aoi} onAoi={(p) => setAoi(p)} drawing={drawing} onDrawEnd={() => setDrawing(false)} run={run}
          detections={detections} layers={layers} basemap={basemap} savedAois={aois}
          onPickDetection={() => { /* popup handled in map */ }} />
        {!panelOpen && <button className="reopen" onClick={() => setPanelOpen(true)}>☰ Panel</button>}
        <div className="basemap-toggle">
          <button className={basemap === "satellite" ? "on" : ""} onClick={() => setBasemap("satellite")}>Satellite</button>
          <button className={basemap === "dark" ? "on" : ""} onClick={() => setBasemap("dark")}>Dark</button>
        </div>
        {drawing && <div className="draw-hint">Click and drag to draw the area · <button className="link" onClick={() => setDrawing(false)}>cancel</button></div>}
        {run && (
          <ResultPanel run={run} detections={detections} layers={layers} setLayers={setLayers}
            onFly={(d) => mapRef.current?.flyTo(d.lon, d.lat, 15)} onClose={() => setRunId(null)} />
        )}
        <div className="toasts">
          {toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>)}
        </div>
      </main>
    </div>
  );
}
