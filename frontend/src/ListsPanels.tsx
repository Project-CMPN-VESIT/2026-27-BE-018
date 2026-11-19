import { useState } from "react";
import type { AOI, Run } from "./api";
import { fmt } from "./geo";

export function StatusChip({ s }: { s: Run["status"] }) {
  const label: Record<Run["status"], string> = {
    queued: "queued", searching: "finding scene", fetching: "downloading", inferring: "analysing",
    rendering: "rendering", done: "done", failed: "failed",
  };
  return <span className={`chip ${s}`}>{label[s]}</span>;
}

export function AreasPanel(props: {
  aois: AOI[];
  onShow(a: AOI): void;
  onRunLatest(a: AOI, daysBack: number, maxCloud: number): void;
  onHistory(a: AOI): void;
  onRename(a: AOI, name: string): void;
  onDelete(a: AOI): void;
}) {
  const [days, setDays] = useState(30);
  const [cloud, setCloud] = useState(30);
  if (!props.aois.length)
    return <div className="panel"><p className="empty">No saved areas yet. Draw an area on the Detect tab and give it a name to save it.</p></div>;
  return (
    <div className="panel">
      <div className="grid2 compact">
        <label className="field">Look back
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d} days</option>)}
          </select>
        </label>
        <label className="field">Max cloud
          <select value={cloud} onChange={(e) => setCloud(Number(e.target.value))}>
            {[10, 20, 30, 50, 80].map((d) => <option key={d} value={d}>{d}%</option>)}
          </select>
        </label>
      </div>
      <ul className="cards">
        {props.aois.map((a) => (
          <li key={a.id} className="card">
            <div className="card-head">
              <b title={a.notes}>{a.name}</b>
              <span className="muted">{fmt.km2(a.area_km2)}</span>
            </div>
            {a.last_run ? (
              <div className="muted small">
                Last run {fmt.date(a.last_run.created_at)} <StatusChip s={a.last_run.status} />
                {a.last_run.status === "done" && <> · {a.last_run.clusters ?? 0} clusters · {fmt.area(a.last_run.debris_area_m2)}</>}
              </div>
            ) : <div className="muted small">Never analysed</div>}
            <div className="btn-row">
              <button className="primary" onClick={() => props.onRunLatest(a, days, cloud)}>Run latest</button>
              <button onClick={() => props.onShow(a)}>Show</button>
              <button onClick={() => props.onHistory(a)}>History</button>
              <button onClick={() => { const n = prompt("Rename area", a.name); if (n && n.trim()) props.onRename(a, n.trim()); }}>Rename</button>
              <button className="danger" onClick={() => confirm(`Delete "${a.name}"? Its past runs are kept in History.`) && props.onDelete(a)}>Delete</button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function HistoryPanel(props: {
  runs: Run[];
  aois: AOI[];
  filterAoi: string | null;
  setFilterAoi(id: string | null): void;
  activeId: string | null;
  onOpen(r: Run): void;
  onDelete(r: Run): void;
}) {
  const name = (id: string | null) => props.aois.find((a) => a.id === id)?.name;
  return (
    <div className="panel">
      <label className="field">Filter
        <select value={props.filterAoi ?? ""} onChange={(e) => props.setFilterAoi(e.target.value || null)}>
          <option value="">All runs</option>
          {props.aois.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
      </label>
      {!props.runs.length && <p className="empty">No runs yet.</p>}
      <ul className="cards">
        {props.runs.map((r) => (
          <li key={r.id} className={`card clickable ${props.activeId === r.id ? "active" : ""}`} onClick={() => props.onOpen(r)}>
            <div className="card-head">
              <b>{name(r.aoi_id) ?? "Ad-hoc area"}</b>
              <StatusChip s={r.status} />
            </div>
            <div className="muted small">
              Imagery {fmt.date(r.scene?.datetime)} · {r.provider.toUpperCase()} · run {fmt.dateTime(r.created_at)}
            </div>
            {r.status === "done" && r.stats && (
              <div className="small">
                <b>{r.stats.clusters}</b> clusters · {fmt.area(r.stats.debris_area_m2)} · max {fmt.pct(r.stats.max_prob)}
              </div>
            )}
            {r.status === "failed" && <div className="small err">{r.error}</div>}
            <button className="link danger right" onClick={(e) => { e.stopPropagation(); confirm("Delete this run and its files?") && props.onDelete(r); }}>delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
