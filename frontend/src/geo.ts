import type { Polygon } from "./api";

export type LngLat = [number, number];
export type BBox = [number, number, number, number];

export function rectPolygon(a: LngLat, b: LngLat): Polygon {
  const [x0, x1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [y0, y1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  return { type: "Polygon", coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] };
}

function rings(p: Polygon): number[][][] {
  return p.type === "Polygon" ? [p.coordinates[0]] : p.coordinates.map((poly: number[][][]) => poly[0]);
}

export function bboxOf(p: Polygon): BBox {
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const r of rings(p)) for (const [x, y] of r) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  return [x0, y0, x1, y1];
}

/** Spherical polygon area (km²), good to <0.5% for areas of this size. */
export function areaKm2(p: Polygon): number {
  const R = 6371.0088;
  const rad = Math.PI / 180;
  let total = 0;
  for (const r of rings(p)) {
    let s = 0;
    for (let i = 0; i < r.length - 1; i++) {
      const [x1, y1] = r[i], [x2, y2] = r[i + 1];
      s += (x2 - x1) * rad * (2 + Math.sin(y1 * rad) + Math.sin(y2 * rad));
    }
    total += Math.abs((s * R * R) / 2);
  }
  return total;
}

export function geometryFromGeoJSON(obj: any): Polygon {
  if (!obj || typeof obj !== "object") throw new Error("not GeoJSON");
  if (obj.type === "FeatureCollection") {
    const polys = (obj.features || []).map((f: any) => f.geometry).filter((g: any) => g && /Polygon/.test(g.type));
    if (!polys.length) throw new Error("no polygon in the FeatureCollection");
    if (polys.length === 1) return polys[0];
    return { type: "MultiPolygon", coordinates: polys.flatMap((g: any) => (g.type === "Polygon" ? [g.coordinates] : g.coordinates)) };
  }
  if (obj.type === "Feature") return geometryFromGeoJSON(obj.geometry);
  if (obj.type === "Polygon" || obj.type === "MultiPolygon") return obj;
  throw new Error("GeoJSON must contain a Polygon or MultiPolygon");
}

export const fmt = {
  date: (iso?: string | null) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—"),
  dateTime: (iso?: string | null) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—"),
  pct: (v?: number | null, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`),
  area: (m2?: number | null) => (m2 == null ? "—" : m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} km²` : m2 >= 1e4 ? `${(m2 / 1e4).toFixed(2)} ha` : `${Math.round(m2)} m²`),
  km2: (v?: number | null) => (v == null ? "—" : v < 10 ? `${v.toFixed(2)} km²` : `${v.toFixed(0)} km²`),
  coord: (lat: number, lon: number) => `${Math.abs(lat).toFixed(5)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(5)}°${lon >= 0 ? "E" : "W"}`,
  isoDay: (d: Date) => d.toISOString().slice(0, 10),
};
