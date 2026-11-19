import { apiKey, fileUrl, type Detection, type Run } from "./api";
import { fmt } from "./geo";
import type { LayerState } from "./MapView";
import { StatusChip } from "./ListsPanels";

const CLASS_COLORS: Record<string, string> = {
  "Marine Debris": "#ff00c8", "Dense Sargassum": "#008000", "Sparse Sargassum": "#82c850", "Natural Organic": "#8c6428",
  Ship: "#ffffff", Clouds: "#c8c8c8", "Marine Water": "#143ca0", "Sediment Water": "#aa823c", Foam: "#e6e6b4",
  "Turbid Water": "#bea064", "Shallow Water": "#3caac8", Waves: "#96c8ff", "Cloud Shadow": "#3c3c3c", Wakes: "#ffdc78",
  "Mixed Water": "#5a78c8",
};

interface Props {
  run: Run;
  detections: Detection[] | null;
  layers: LayerState;
  setLayers(l: LayerState): void;
  onFly(d: Detection): void;
  onClose(): void;
}

async function download(url: string, name: string) {
  const u = await fileUrl(url);
  const a = document.createElement("a");
  a.href = u; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

export function ResultPanel({ run, detections, layers, setLayers, onFly, onClose }: Props) {
  const st = run.stats;
  const running = run.status !== "done" && run.status !== "failed";
  const L = layers;
  const set = (patch: Partial<LayerState>) => setLayers({ ...L, ...patch });

  return (
    <div className="result">
      <div className="result-head">
        <div>
          <div className="eyebrow">Detection run <span className="mono">{run.id}</span></div>
          <h2>{!run.scene ? "Preparing…" : run.provider === "upload"
            ? `Uploaded · ${run.scene.extras?.filename ?? "image"}`
            : `Sentinel-2 · ${fmt.date(run.scene.datetime)}`}</h2>
        </div>
        <StatusChip s={run.status} />
        <button className="icon" aria-label="Close" onClick={onClose}>×</button>
      </div>

      {running && (
        <div className="progress-wrap">
          <div className="progress"><div style={{ width: `${Math.max(3, run.progress * 100)}%` }} /></div>
          <div className="muted small">{run.status === "queued" && run.queue_position ? `waiting in queue (position ${run.queue_position})` : run.message || "starting…"}</div>
        </div>
      )}
      {run.status === "failed" && <div className="alert err">{run.error}</div>}

      {run.status === "done" && st && (
        <>
          <div className="stats">
            <div className="stat hero"><span>{st.clusters}</span><label>debris clusters</label></div>
            <div className="stat"><span>{fmt.area(st.debris_area_m2)}</span><label>flagged area</label></div>
            <div className="stat"><span>{fmt.pct(st.max_prob)}</span><label>peak probability</label></div>
            <div className="stat"><span>{fmt.km2(st.analysed_km2)}</span><label>water analysed</label></div>
          </div>
          <div className="coverage">
            <div className="bar">
              <i className="v" style={{ width: fmt.pct(st.fractions.valid, 1) }} title="analysed water" />
              <i className="c" style={{ width: fmt.pct(st.fractions.cloud, 1) }} title="cloud" />
              <i className="l" style={{ width: fmt.pct(st.fractions.land, 1) }} title="land" />
              <i className="n" style={{ width: fmt.pct(st.fractions.nodata, 1) }} title="no data" />
            </div>
            <div className="muted small">
              water {fmt.pct(st.fractions.valid)} · cloud {fmt.pct(st.fractions.cloud)} · land {fmt.pct(st.fractions.land)}
              {st.fractions.nodata > 0.005 && <> · no data {fmt.pct(st.fractions.nodata)}</>}
              {run.scene?.cloud_cover != null && <> · scene cloud {run.scene.cloud_cover}%</>}
            </div>
          </div>
          {st.warnings.map((w) => <div key={w} className="alert warn">{w}</div>)}

          <h4>Layers</h4>
          <div className="layers">
            <label className="check"><input type="checkbox" checked={L.density} onChange={(e) => set({ density: e.target.checked })} /> Debris density heatmap</label>
            <LayerRow label="Probability (pixel)" v={L.heatmap} onChange={(heatmap) => set({ heatmap })} />
            <label className="check"><input type="checkbox" checked={L.outlines} onChange={(e) => set({ outlines: e.target.checked })} /> Detection outlines</label>
            <LayerRow label="True colour (scene)" v={L.rgb} onChange={(rgb) => set({ rgb })} />
            <LayerRow label="MARIDA classes" v={L.classes} onChange={(classes) => set({ classes })} />
          </div>
          <div className="legend">
            <div className="ramp" /><div className="ramp-labels"><span>low</span><span>P(debris)</span><span>high</span></div>
          </div>
          {L.classes.on && (
            <div className="class-legend">
              {Object.entries(st.class_fractions).sort((a, b) => b[1] - a[1]).map(([c, f]) => (
                <span key={c}><i style={{ background: CLASS_COLORS[c] }} />{c} {fmt.pct(f, f < 0.01 ? 2 : 0)}</span>
              ))}
            </div>
          )}

          <h4>Top detections {detections && detections.length > 0 && <span className="muted">({detections.length})</span>}</h4>
          {detections?.length ? (
            <table className="dets">
              <thead><tr><th>#</th><th>P max</th><th>Area</th><th>Location</th></tr></thead>
              <tbody>
                {detections.slice(0, 50).map((d) => (
                  <tr key={d.rank} onClick={() => onFly(d)}>
                    <td>{d.rank}</td><td><b>{(d.max_prob * 100).toFixed(0)}%</b></td><td>{fmt.area(d.area_m2)}</td>
                    <td className="mono" title="click to zoom">{d.lat.toFixed(4)}, {d.lon.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted small">No pixel cluster exceeded the {st.threshold.toFixed(2)} threshold.</p>}

          <h4>Downloads</h4>
          <div className="btn-row">
            <button onClick={() => download(`/api/runs/${run.id}/files/detections.geojson`, `gln_${run.id}_detections.geojson`)}>Detections GeoJSON</button>
            <button onClick={() => download(`/api/runs/${run.id}/files/probability.tif`, `gln_${run.id}_probability.tif`)}>Probability GeoTIFF</button>
            <button onClick={() => download(`/api/runs/${run.id}/files/classes.tif`, `gln_${run.id}_classes.tif`)}>Classes GeoTIFF</button>
            <button onClick={() => download(`/api/runs/${run.id}/files/report.json`, `gln_${run.id}_report.json`)}>Report JSON</button>
          </div>
          <p className="muted tiny">
            {run.provider.toUpperCase()} · {run.scene?.tiles?.join(", ")} · {st.tiles} tiles in {st.timing_s.total}s
            (download {st.timing_s.fetch}s, model {st.timing_s.inference}s) · mask: {st.mask_method}
            {apiKey.get() ? " · authenticated" : ""}
          </p>
        </>
      )}
    </div>
  );
}

function LayerRow({ label, v, onChange }: { label: string; v: { on: boolean; opacity: number }; onChange(v: { on: boolean; opacity: number }): void }) {
  return (
    <div className="layer-row">
      <label className="check"><input type="checkbox" checked={v.on} onChange={(e) => onChange({ ...v, on: e.target.checked })} /> {label}</label>
      <input type="range" min={0.1} max={1} step={0.05} value={v.opacity} disabled={!v.on} onChange={(e) => onChange({ ...v, opacity: Number(e.target.value) })} />
    </div>
  );
}
