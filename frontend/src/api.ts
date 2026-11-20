// Typed client for the GeoLitterNet Watch API (same origin; Vite proxies /api in development).

export type Polygon = { type: "Polygon" | "MultiPolygon"; coordinates: any };

export interface ProviderInfo { name: string; label: string; available: boolean; detail: string }
export interface Info {
  model: { name: string; checkpoint: string; sha256: string; parameters: number; classes: string[]; device: string };
  providers: ProviderInfo[];
  default_provider: string;
  defaults: { threshold: number; max_cloud: number; days_back: number; min_cluster_pixels: number; mask_land: boolean; mask_clouds: boolean };
  limits: { max_aoi_km2: number; max_upload_mb: number };
  queue: { running: string | null; waiting: number };
}
export interface Scene {
  id: string; provider: string; datetime: string; cloud_cover: number | null; platform: string;
  tiles: string[]; coverage: number | null; thumbnail: string | null; extras: Record<string, any>;
}
export interface Layer { url: string; coordinates?: [number, number][] }
export interface RunStats {
  aoi_km2: number; analysed_km2: number; fractions: { valid: number; nodata: number; cloud: number; land: number };
  mask_method: string; threshold: number; min_cluster_pixels: number; debris_pixels: number; debris_area_m2: number;
  clusters: number; max_prob: number; mean_prob: number; class_fractions: Record<string, number>;
  timing_s: { fetch: number; inference: number; render: number; total: number }; tiles: number; warnings: string[];
  tiles_skipped?: number; from_cache?: boolean; reprocess_s?: number;
}
export type RunStatus = "queued" | "searching" | "fetching" | "inferring" | "rendering" | "done" | "failed";
export interface Run {
  id: string; aoi_id: string | null; geometry: Polygon; provider: string; scene_id: string | null; scene: Scene | null;
  params: Record<string, any>; status: RunStatus; progress: number; message: string; error: string | null;
  stats: RunStats | null; layers: Record<"heatmap" | "rgb" | "classes" | "detections", Layer> | null;
  created_at: string; started_at: string | null; finished_at: string | null; queue_position?: number;
}
export interface RunSummary { id: string; status: RunStatus; created_at: string; scene_datetime: string | null; debris_area_m2: number | null; clusters: number | null; max_prob: number | null }
export interface AOI { id: string; name: string; geometry: Polygon; area_km2: number; notes: string; created_at: string; last_run: RunSummary | null }
export interface Upload { id: string; scene_id: string; filename: string; acquired: string; uploaded: string; value_type: string; has_scl: boolean; footprint: Polygon }
export interface Detection {
  rank: number; pixels: number; area_m2: number; max_prob: number; mean_prob: number; lon: number; lat: number; dominant_class: string;
}

const KEY_STORE = "gln.apiKey";
export const apiKey = {
  get: () => { try { return localStorage.getItem(KEY_STORE) || ""; } catch { return ""; } },
  set: (k: string) => { try { localStorage.setItem(KEY_STORE, k); } catch { /* private mode */ } },
};

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function req<T>(method: string, path: string, body?: unknown, isForm = false): Promise<T> {
  const headers: Record<string, string> = {};
  const k = apiKey.get();
  if (k) headers["X-API-Key"] = k;
  if (body !== undefined && !isForm) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    method, headers,
    body: body === undefined ? undefined : isForm ? (body as FormData) : JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (typeof j.detail === "string") msg = j.detail;
      else if (Array.isArray(j.detail)) msg = j.detail.map((d: any) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    } catch { /* not JSON */ }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/** Adds the API key as a query-less fetch for images is not possible, so files are fetched as blobs when a key is set. */
export async function fileUrl(url: string): Promise<string> {
  const k = apiKey.get();
  if (!k) return url;
  const res = await fetch(url, { headers: { "X-API-Key": k } });
  if (!res.ok) throw new ApiError(res.status, "file download failed");
  return URL.createObjectURL(await res.blob());
}

export const api = {
  info: () => req<Info>("GET", "/api/info"),
  searchScenes: (b: { provider: string; geometry: Polygon; start_date: string; end_date: string; max_cloud: number; limit?: number }) =>
    req<{ provider: string; count: number; scenes: Scene[] }>("POST", "/api/scenes/search", b),
  createRun: (b: Record<string, unknown>) => req<Run>("POST", "/api/runs", b),
  run: (id: string) => req<Run>("GET", `/api/runs/${id}`),
  runs: (aoiId?: string, limit = 50) => req<Run[]>("GET", `/api/runs?limit=${limit}${aoiId ? `&aoi_id=${aoiId}` : ""}`),
  deleteRun: (id: string) => req<void>("DELETE", `/api/runs/${id}`),
  reprocess: (id: string, b: { threshold: number; min_cluster_pixels: number }) => req<Run>("POST", `/api/runs/${id}/reprocess`, b),
  aois: () => req<AOI[]>("GET", "/api/aois"),
  createAoi: (b: { name: string; geometry: Polygon; notes?: string }) => req<AOI>("POST", "/api/aois", b),
  patchAoi: (id: string, b: { name?: string; notes?: string }) => req<AOI>("PATCH", `/api/aois/${id}`, b),
  deleteAoi: (id: string, deleteRuns = false) => req<void>("DELETE", `/api/aois/${id}?delete_runs=${deleteRuns}`),
  runAoi: (id: string, b: { provider?: string; days_back: number; max_cloud: number; threshold?: number }) =>
    req<Run>("POST", `/api/aois/${id}/runs`, b),
  uploads: () => req<Upload[]>("GET", "/api/uploads"),
  upload: (file: File, acquired: string, dnOffset: number) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("acquired", acquired);
    fd.append("dn_offset", String(dnOffset));
    return req<Upload>("POST", "/api/uploads", fd, true);
  },
  deleteUpload: (id: string) => req<void>("DELETE", `/api/uploads/${id}`),
};

/** Cache-busting version for files that change when a run is re-analysed (detections.geojson). */
export const detVersion = (r: Run) => `?v=${r.stats?.threshold ?? ""}-${r.stats?.min_cluster_pixels ?? ""}`;
