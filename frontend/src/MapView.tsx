import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import maplibregl, { GeoJSONSource, LngLatBoundsLike, Map as MLMap, StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { detVersion, fileUrl, type AOI, type Detection, type Polygon, type Run } from "./api";
import { fmt, rectPolygon, type BBox, type LngLat } from "./geo";

export type Basemap = "satellite" | "dark";
export interface LayerState {
  density: boolean;
  heatmap: { on: boolean; opacity: number };
  rgb: { on: boolean; opacity: number };
  classes: { on: boolean; opacity: number };
  outlines: boolean;
}
export interface MapHandle {
  fitBounds(b: BBox): void;
  flyTo(lon: number, lat: number, zoom?: number): void;
  viewBounds(): BBox | null;
}
interface Props {
  aoi: Polygon | null;
  onAoi(p: Polygon): void;
  drawing: boolean;
  onDrawEnd(): void;
  run: Run | null;
  detections: Detection[] | null;
  layers: LayerState;
  basemap: Basemap;
  savedAois: AOI[];
  onPickDetection(d: Detection): void;
}

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    satellite: {
      type: "raster", tileSize: 256, maxzoom: 19,
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    },
    dark: {
      type: "raster", tileSize: 256, maxzoom: 19,
      tiles: ["a", "b", "c"].map((s) => `https://${s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png`),
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#071521" } },
    { id: "satellite", type: "raster", source: "satellite", paint: { "raster-saturation": -0.2, "raster-brightness-max": 0.85 } },
    { id: "dark", type: "raster", source: "dark", layout: { visibility: "none" } },
  ],
};

const IMAGE_LAYERS = ["rgb", "classes", "heatmap"] as const;

export const MapView = forwardRef<MapHandle, Props>(function MapView(p, ref) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const drawStart = useRef<LngLat | null>(null);
  const blobUrls = useRef<string[]>([]);
  const props = useRef(p);
  props.current = p;

  useImperativeHandle(ref, () => ({
    fitBounds: (b) => map.current?.fitBounds(b as LngLatBoundsLike, { padding: 60, duration: 900, maxZoom: 14 }),
    flyTo: (lon, lat, zoom = 15) => map.current?.flyTo({ center: [lon, lat], zoom, duration: 1100 }),
    viewBounds: () => {
      const b = map.current?.getBounds();
      return b ? [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] : null;
    },
  }));

  // ---------------------------------------------------------------- init
  useEffect(() => {
    if (!el.current) return;
    const m = new maplibregl.Map({
      container: el.current, style: STYLE, center: [-88.2, 16.1], zoom: 7, maxPitch: 0, dragRotate: false,
      attributionControl: { compact: true },
    });
    m.touchZoomRotate.disableRotation();
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    m.on("load", () => {
      m.addSource("saved", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "saved-line", type: "line", source: "saved", paint: { "line-color": "#94a3b8", "line-width": 1.2, "line-dasharray": [2, 2] } });
      m.addSource("aoi", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": "#38bdf8", "fill-opacity": 0.06 } });
      m.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#38bdf8", "line-width": 2 } });
      m.addSource("det", { type: "geojson", data: EMPTY });
      m.addSource("det-pts", { type: "geojson", data: EMPTY });
      m.addLayer({
        id: "density", type: "heatmap", source: "det-pts", maxzoom: 16,
        paint: {
          "heatmap-weight": ["*", ["get", "max_prob"], ["min", 1, ["/", ["ln", ["+", 1, ["get", "pixels"]]], 3]]],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 5, 0.8, 12, 1.6],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 5, 10, 9, 22, 13, 40, 16, 60],
          "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.95, 13, 0.6, 16, 0.15],
          "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
            0, "rgba(255,255,178,0)", 0.15, "rgba(254,204,92,0.55)", 0.35, "#fd8d3c", 0.6, "#f03b20", 0.85, "#bd0026", 1, "#7a0177"],
        },
      });
      m.addLayer({ id: "det-fill", type: "fill", source: "det", paint: { "fill-color": "#ff00c8", "fill-opacity": 0.18 } });
      m.addLayer({ id: "det-line", type: "line", source: "det", paint: { "line-color": "#ff00c8", "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1, 15, 2.5] } });
      m.addLayer({
        id: "det-pt", type: "circle", source: "det-pts", minzoom: 9,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 3, 14, 6], "circle-color": "#ff00c8",
          "circle-stroke-color": "#fff", "circle-stroke-width": 1.2, "circle-opacity": ["interpolate", ["linear"], ["zoom"], 13, 1, 15, 0],
          "circle-stroke-opacity": ["interpolate", ["linear"], ["zoom"], 13, 1, 15, 0],
        },
      });
      const pick = (e: maplibregl.MapLayerMouseEvent) => {
        const f = e.features?.[0];
        if (!f) return;
        const d = f.properties as unknown as Detection;
        props.current.onPickDetection(d);
        new maplibregl.Popup({ closeButton: true, maxWidth: "260px" }).setLngLat(e.lngLat).setHTML(
          `<div class="pop"><b>Detection #${d.rank}</b><br/>Max probability <b>${(Number(d.max_prob) * 100).toFixed(0)}%</b><br/>` +
          `Area ${fmt.area(Number(d.area_m2))} · ${d.pixels} px<br/><span class="mono">${fmt.coord(Number(d.lat), Number(d.lon))}</span></div>`,
        ).addTo(m);
      };
      for (const id of ["det-fill", "det-pt"]) {
        m.on("click", id, pick);
        m.on("mouseenter", id, () => (m.getCanvas().style.cursor = "pointer"));
        m.on("mouseleave", id, () => (m.getCanvas().style.cursor = ""));
      }
      setReady(true);
    });
    map.current = m;
    return () => { m.remove(); map.current = null; };
  }, []);

  // ---------------------------------------------------------------- basemap
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    m.setLayoutProperty("satellite", "visibility", p.basemap === "satellite" ? "visible" : "none");
    m.setLayoutProperty("dark", "visibility", p.basemap === "dark" ? "visible" : "none");
  }, [p.basemap, ready]);

  // ---------------------------------------------------------------- AOI + saved areas
  useEffect(() => {
    if (!ready) return;
    (map.current!.getSource("aoi") as GeoJSONSource).setData(p.aoi ? { type: "Feature", properties: {}, geometry: p.aoi as any } : EMPTY);
  }, [p.aoi, ready]);
  useEffect(() => {
    if (!ready) return;
    (map.current!.getSource("saved") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: p.savedAois.map((a) => ({ type: "Feature", properties: { name: a.name }, geometry: a.geometry as any })),
    });
  }, [p.savedAois, ready]);

  // ---------------------------------------------------------------- rectangle drawing
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    if (!p.drawing) { m.getCanvas().style.cursor = ""; return; }
    m.getCanvas().style.cursor = "crosshair";
    m.dragPan.disable();
    const down = (e: maplibregl.MapMouseEvent | maplibregl.MapTouchEvent) => {
      e.preventDefault();
      drawStart.current = [e.lngLat.lng, e.lngLat.lat];
    };
    const move = (e: maplibregl.MapMouseEvent | maplibregl.MapTouchEvent) => {
      if (!drawStart.current) return;
      props.current.onAoi(rectPolygon(drawStart.current, [e.lngLat.lng, e.lngLat.lat]));
    };
    const up = (e: maplibregl.MapMouseEvent | maplibregl.MapTouchEvent) => {
      if (!drawStart.current) return;
      const a = drawStart.current;
      drawStart.current = null;
      const b: LngLat = [e.lngLat.lng, e.lngLat.lat];
      if (Math.abs(a[0] - b[0]) > 1e-5 && Math.abs(a[1] - b[1]) > 1e-5) props.current.onAoi(rectPolygon(a, b));
      props.current.onDrawEnd();
    };
    m.on("mousedown", down); m.on("mousemove", move); m.on("mouseup", up);
    m.on("touchstart", down); m.on("touchmove", move); m.on("touchend", up);
    return () => {
      m.off("mousedown", down); m.off("mousemove", move); m.off("mouseup", up);
      m.off("touchstart", down); m.off("touchmove", move); m.off("touchend", up);
      m.dragPan.enable();
      m.getCanvas().style.cursor = "";
    };
  }, [p.drawing, ready]);

  // ---------------------------------------------------------------- run image overlays
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    let cancelled = false;
    for (const id of IMAGE_LAYERS) {
      if (m.getLayer(`img-${id}`)) m.removeLayer(`img-${id}`);
      if (m.getSource(`img-${id}`)) m.removeSource(`img-${id}`);
    }
    blobUrls.current.forEach((u) => URL.revokeObjectURL(u));
    blobUrls.current = [];
    const layers = p.run?.status === "done" ? p.run.layers : null;
    if (!layers) return;
    (async () => {
      for (const id of IMAGE_LAYERS) {
        const lay = layers[id];
        if (!lay?.coordinates) continue;
        const url = await fileUrl(lay.url);
        if (url.startsWith("blob:")) blobUrls.current.push(url);
        if (cancelled || !map.current) return;
        m.addSource(`img-${id}`, { type: "image", url, coordinates: lay.coordinates as any });
        m.addLayer({
          id: `img-${id}`, type: "raster", source: `img-${id}`,
          paint: { "raster-opacity": 0, "raster-resampling": id === "rgb" ? "linear" : "nearest", "raster-fade-duration": 0 },
        }, "aoi-fill");
      }
      applyLayerState(m, props.current.layers);
    })();
    return () => { cancelled = true; };
  }, [p.run?.id, p.run?.status, ready]);

  // ---------------------------------------------------------------- detections
  useEffect(() => {
    if (!ready) return;
    const dets = p.detections ?? [];
    (map.current!.getSource("det-pts") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: dets.map((d) => ({ type: "Feature", properties: d as any, geometry: { type: "Point", coordinates: [d.lon, d.lat] } })),
    });
  }, [p.detections, ready]);
  useEffect(() => {
    if (!ready) return;
    const src = map.current!.getSource("det") as GeoJSONSource;
    const lay = p.run?.status === "done" ? p.run.layers?.detections : null;
    if (!lay) { src.setData(EMPTY); return; }
    let cancelled = false;
    fileUrl(lay.url + detVersion(p.run!)).then((u) => { if (!cancelled) src.setData(u); }).catch(() => src.setData(EMPTY));
    return () => { cancelled = true; };
  }, [p.run?.id, p.run?.status, p.run?.stats?.threshold, p.run?.stats?.min_cluster_pixels, ready]);

  // ---------------------------------------------------------------- layer visibility / opacity
  useEffect(() => {
    if (ready && map.current) applyLayerState(map.current, p.layers);
  }, [p.layers, ready]);

  return <div ref={el} className="map" />;
});

function applyLayerState(m: MLMap, s: LayerState) {
  const img = (id: string, on: boolean, op: number) => {
    if (m.getLayer(`img-${id}`)) m.setPaintProperty(`img-${id}`, "raster-opacity", on ? op : 0);
  };
  img("rgb", s.rgb.on, s.rgb.opacity);
  img("classes", s.classes.on, s.classes.opacity);
  img("heatmap", s.heatmap.on, s.heatmap.opacity);
  m.setLayoutProperty("density", "visibility", s.density ? "visible" : "none");
  for (const id of ["det-fill", "det-line", "det-pt"]) m.setLayoutProperty(id, "visibility", s.outlines ? "visible" : "none");
}
