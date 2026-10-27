#!/usr/bin/env python3
"""
row_watch.py — pipeline right-of-way vegetation-stress watcher (proof of concept)

Free data only: Sentinel-2 L2A (10 m, ~5-day revisit) from Microsoft Planetary
Computer's open STAC catalog. No account or API key needed. Sentinel data is
free for commercial use.

Method (the same idea behind PipeWatch's leak detection, at coarser resolution):
  1. Split the pipeline route into segments (default 250 m).
  2. For each segment, build two zones:
       - Corridor: the strip over and beside the pipe that is scored
         (default 100 m wide; spills run off the line, and at 10 m pixels a
         narrower strip is only a few pixels across)
       - Reference ring: nearby land outside the corridor (60–250 m away)
  3. For every Sentinel-2 scene, compute NDVI. For each corridor pixel take
         diff = NDVI(pixel) - median NDVI(reference ring of its segment)
     Comparing to adjacent land cancels out weather, drought and green-up,
     which affect both; a leak affects only the ground around the pipe.
  4. Score each pixel's diff against its own baseline from the same time of
     year (+/- 30 days) in the baseline years -> a per-pixel z-score.
  5. A pixel is anomalous when z <= -3.5 in two consecutive clear scenes
     (one-off noise rarely persists; dead vegetation does).
     A segment is FLAGGED when a cluster of >= 5 adjacent anomalous pixels
     (>= 500 m2) touches it, and raises an ALERT when it is flagged on two
     consecutive clear scenes.

Usage:
  pip install pystac-client planetary-computer odc-stac geopandas rasterio pandas numpy scipy folium
  python row_watch.py --route my_pipeline.geojson \
      --baseline-start 2019-04-01 --monitor-start 2023-04-01 --end 2023-10-31 \
      --out results

Route file: any line format geopandas can read (GeoJSON, KML, shapefile, GPKG).
Keep the PoC section to roughly 10–30 km of pipe. Give the baseline at least
3 years (4 is better): the per-pixel "normal range" comes from it.

Outputs (in --out):
  map.html           self-contained web map: segments coloured ok / watch / ALERT,
                     anomaly patches outlined; click a segment for its chart
  segments.geojson   one feature per segment with status, first flag/alert dates
  anomalies.geojson  outlines of the anomalous-pixel clusters behind each flag
  timeseries.csv     every segment x date: NDVI, diff, anomaly cluster size, flags
  ndvi_cube.npz      cached NDVI for the corridor. Re-running with the same route
                     and dates reuses it, so changing --z etc. needs no new
                     downloads. Delete it (or pass --refresh) to re-download.
"""
import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, shape as to_shape
from shapely.ops import linemerge, substring, unary_union

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Sentinel-2 scene classification: 4 = vegetation, 5 = bare soil.
# Clouds, shadows, water, snow and saturated pixels are dropped.
VALID_SCL = [4, 5]
NODATA = -32768   # NDVI is cached as int16 (NDVI x 10000)
CACHE_BUFFER = 300  # m around the route kept in the cache (>= --ref-outer)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def read_route(path):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    merged = unary_union(list(gdf.geometry))
    if merged.geom_type == "MultiLineString":
        merged = linemerge(merged)
    if merged.geom_type == "LineString":
        lines = [merged]
    elif isinstance(merged, MultiLineString):
        lines = list(merged.geoms)
    else:
        sys.exit("Route file must contain line geometry (the pipeline centerline).")
    return lines, utm


def make_segments(lines, seg_len):
    segs = []
    for line in lines:
        n = max(1, int(np.ceil(line.length / seg_len)))
        step = line.length / n
        for i in range(n):
            segs.append(substring(line, i * step, (i + 1) * step))
    return segs


def build_zones(segs, row_width, ref_inner, ref_outer):
    route = unary_union(segs)
    exclusion = route.buffer(ref_inner)
    rows = [s.buffer(row_width / 2, cap_style=2) for s in segs]
    refs = [s.buffer(ref_outer, cap_style=2).difference(exclusion) for s in segs]
    return rows, refs


def rasterize_mask(geom, shape, transform, all_touched=False):
    from rasterio.features import rasterize

    if geom.is_empty:
        return np.zeros(shape, bool)
    return rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0,
                     dtype="uint8", all_touched=all_touched).astype(bool)


# --------------------------------------------------------------------------
# Imagery
# --------------------------------------------------------------------------
def stac_client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(bbox_ll, start, end, max_cloud):
    items = stac_client().search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox_ll),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    return list(items)


def item_offset(item):
    """Processing baseline 04.00+ adds a +1000 DN offset to reflectance."""
    pb = item.properties.get("s2:processing_baseline")
    try:
        return 1000 if float(pb) >= 4.0 else 0
    except (TypeError, ValueError):
        return 1000 if item.datetime.date() >= pd.Timestamp("2022-01-25").date() else 0


def dedupe_items(items):
    """Planetary Computer holds the same acquisition more than once (original
    processing plus ESA's reprocessing, often with a different baseline and
    therefore a different offset). Keep the newest version of each tile-date,
    then make sure every scene date uses a single offset."""
    newest = {}
    for it in items:
        key = (it.datetime.date(), it.properties.get("s2:mgrs_tile"))
        ptime = it.properties.get("s2:generation_time") or it.id.rsplit("_", 1)[-1]
        if key not in newest or ptime > newest[key][0]:
            newest[key] = (ptime, it)
    by_day = {}
    for (day, _), (_, it) in newest.items():
        by_day.setdefault(day, []).append(it)
    out = []
    for its in by_day.values():
        common = Counter(item_offset(i) for i in its).most_common(1)[0][0]
        out += [i for i in its if item_offset(i) == common]
    return sorted(out, key=lambda i: i.datetime)


def load_ndvi(items, utm, bbox_ll, keep, batch=12):
    """Download B04/B08/SCL for the bbox; return dates, NDVI (int16, dates x kept
    pixels), grid shape and transform. Only pixels where `keep` is True are stored."""
    from odc.stac import configure_rio, load as stac_load

    configure_rio(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
    ds = stac_load(items, bands=["B04", "B08", "SCL"], crs=utm, resolution=10,
                   bbox=list(bbox_ll), groupby="solar_day", chunks={})
    gb = ds.odc.geobox
    shape, transform = tuple(gb.shape), gb.transform
    idx = np.flatnonzero(keep(shape, transform).ravel())
    days = pd.DatetimeIndex(ds.time.values).normalize()
    offsets = {pd.Timestamp(it.datetime.date()): item_offset(it) for it in items}

    n = len(days)
    ndvi_out = np.full((n, idx.size), NODATA, dtype="int16")
    for b0 in range(0, n, batch):
        sl = ds.isel(time=slice(b0, b0 + batch)).compute()
        for k in range(sl.sizes["time"]):
            t = b0 + k
            off = offsets.get(days[t], 0)
            red = (sl.B04.values[k].ravel()[idx].astype("float32") - off) / 10000.0
            nir = (sl.B08.values[k].ravel()[idx].astype("float32") - off) / 10000.0
            valid = np.isin(sl.SCL.values[k].ravel()[idx], VALID_SCL) & (red > 0) & (nir > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = np.clip((nir - red) / (nir + red), -1, 1)
            ndvi_out[t][valid] = np.round(ndvi[valid] * 10000).astype("int16")
            print(f"  [{t + 1}/{n}] {days[t].date()}  clear: {valid.mean():.0%}")
    return days, ndvi_out, idx, shape, transform


def get_imagery(a, find_items, utm, bbox_ll, keep):
    """NDVI for the corridor. Reuses the cache when it covers this route and
    date range (a shorter range is cut out of a longer cached one)."""
    from affine import Affine

    key = json.dumps([np.round(bbox_ll, 5).tolist(), str(utm), a.max_cloud])
    key = hashlib.md5(key.encode()).hexdigest()[:12]
    path = a.cache or os.path.join(a.out, "ndvi_cube.npz")
    start, end = pd.Timestamp(a.baseline_start), pd.Timestamp(a.end)
    if os.path.exists(path) and not a.refresh:
        z = np.load(path, allow_pickle=False)
        if "idx" in z and str(z["key"]) == key and \
                pd.Timestamp(str(z["start"])) <= start and pd.Timestamp(str(z["end"])) >= end:
            print(f"Using cached imagery in {path} (pass --refresh to re-download)")
            days = pd.DatetimeIndex(z["dates"])
            sel = (days >= start) & (days <= end)
            return days[sel], z["ndvi"][sel], z["idx"], tuple(z["shape"]), Affine(*z["transform"])
    days, ndvi, idx, shape, transform = load_ndvi(find_items(), utm, bbox_ll, keep)
    np.savez_compressed(path, key=key, start=str(start.date()), end=str(end.date()),
                        dates=days.values.astype("datetime64[ns]"), ndvi=ndvi,
                        idx=idx, shape=np.array(shape), transform=np.array(list(transform)[:6]))
    return days, ndvi, idx, shape, transform


def landcover_mask(shape, transform, utm):
    """True where land is cultivated, built-up or water, from free land-cover maps:
    ESA WorldCover 10 m (global) plus USDA Cropland Data Layer 'cultivated' (US)."""
    from odc.geo.geobox import GeoBox
    from odc.stac import load as stac_load

    gb = GeoBox(shape, transform, utm)
    aoi = gb.extent.to_crs(4326).geom.__geo_interface__
    cat = stac_client()
    bad = np.zeros(shape, bool)
    wc = sorted(cat.search(collections=["esa-worldcover"], intersects=aoi).item_collection(), key=lambda i: i.id)
    if wc:
        latest = [i for i in wc if i.id.split("_")[3] == wc[-1].id.split("_")[3]]
        m = stac_load(latest, bands=["map"], geobox=gb, resampling="nearest")["map"].values[0]
        bad |= np.isin(m, [40, 50, 80])  # cropland, built-up, water
    cdl = [i for i in cat.search(collections=["usda-cdl"], intersects=aoi).item_collection()
           if i.id.startswith("cultivated")]
    if cdl:
        year = max(i.id.split("_")[1] for i in cdl)
        m = stac_load([i for i in cdl if i.id.split("_")[1] == year], bands=["cultivated"],
                      geobox=gb, resampling="nearest")["cultivated"].values[0]
        bad |= m == 2
    print(f"Land-cover mask: excluding {bad.mean():.0%} of the area (cropland / built-up / water)")
    return bad


# --------------------------------------------------------------------------
# Rainfall (CHIRPS daily, ~5 km, 50°S–50°N, free, no account)
# --------------------------------------------------------------------------
CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/"
              "{d:%Y}/chirps-v2.0.{d:%Y.%m.%d}.cog")


def get_rainfall(points_ll, start, end, path, workers=3):
    """Daily rainfall (mm) averaged over a few points along the route, read
    straight from the CHIRPS files (one small web request per day). Cached in
    `path`; only days not yet in the cache are fetched. Days not published yet
    (CHIRPS lags ~3 weeks) come back empty and are retried next run."""
    import rasterio
    from concurrent.futures import ThreadPoolExecutor

    days = pd.date_range(start, end, freq="D")
    have = (pd.read_csv(path, parse_dates=["date"]).set_index("date")["rain_mm"]
            if os.path.exists(path) else pd.Series(dtype=float))
    todo = [d for d in days if d not in have.index]

    def one(d):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                              GDAL_HTTP_RETRY_DELAY="1"):
                with rasterio.open("/vsicurl/" + CHIRPS_URL.format(d=d)) as src:
                    v = [float(x[0]) for x in src.sample(points_ll)]
            v = [x for x in v if x >= 0]  # -9999 = no data (sea, outside coverage)
            return d, (sum(v) / len(v)) if v else np.nan
        except Exception:
            return d, np.nan

    got = {}
    if todo:
        # Probe one day first: the CHIRPS server sometimes refuses requests
        # (HTTP 403) after heavy use; don't hammer it, just carry on without rain.
        import requests
        try:
            requests.head(CHIRPS_URL.format(d=todo[0]), timeout=30,
                          headers={"User-Agent": "row_watch (research)"}).raise_for_status()
        except Exception as e:
            print(f"CHIRPS rainfall unavailable right now ({e}); continuing without new rainfall days.")
            todo = []
    if todo:
        print(f"Fetching CHIRPS rainfall for {len(todo)} days...")
        with ThreadPoolExecutor(workers) as ex:
            got = dict(ex.map(one, todo))
    rain = pd.concat([have, pd.Series(got, dtype=float)]).sort_index()
    rain = rain[~rain.index.duplicated(keep="last")]
    rain.dropna().rename_axis("date").rename("rain_mm").to_csv(path)
    return rain.reindex(days)


def rain_context(rain, monitor_start):
    """Daily rain plus what is normal for each day of the year, from the
    baseline years (smoothed over a month so one storm doesn't define normal)."""
    base = rain[rain.index < pd.Timestamp(monitor_start)].dropna()
    clim = base.groupby(base.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    k = 31  # circular running mean over ±15 days
    smooth = np.convolve(np.r_[clim.values[-k:], clim.values, clim.values[:k]], np.ones(k) / k, "same")[k:-k]
    normal = pd.Series(smooth[rain.index.dayofyear - 1], index=rain.index)
    return pd.DataFrame({"rain": rain, "normal": normal})


def weather_note(ctx, date):
    """A short hint about the weather before a patch appeared, or ''.
    Rules of thumb only; they point the reviewer at possible weather effects."""
    if ctx is None:
        return ""
    d = pd.Timestamp(date)
    recent = ctx.loc[d - pd.Timedelta(days=10):d, "rain"]
    if recent.notna().sum() < 8:
        return ""
    heavy = recent.rolling(3, min_periods=1).sum().max()
    last = ctx.loc[d - pd.Timedelta(days=30):d]
    prev = ctx.loc[d - pd.Timedelta(days=60):d - pd.Timedelta(days=31)]
    r_last, n_last = last["rain"].sum(), last["normal"].sum()
    r_prev, n_prev = prev["rain"].sum(), prev["normal"].sum()
    if heavy >= 50:
        return f"{heavy:.0f} mm rain in 3 days just before: standing water?"
    if n_prev > 0 and r_prev < 0.5 * n_prev and r_last >= max(25, n_last):
        return f"first rain after a dry spell ({r_last:.0f} mm in 30 days): neighbours greening faster?"
    if n_last > 0 and r_last < 0.3 * n_last:
        return f"very dry ({r_last:.0f} mm in 30 days, {r_last / n_last:.0%} of normal)"
    return ""


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
def detect(days, ndvi, idx, shape, transform, segs, a, exclude=None):
    """Per-pixel seasonal z-scores, persistence, clustering. Returns
    (per segment x date table, anomaly cluster polygons)."""
    from scipy import ndimage
    from rasterio.features import shapes as poly_shapes

    rows, refs = build_zones(segs, a.row_width, a.ref_inner, a.ref_outer)
    pos = np.full(shape[0] * shape[1], -1)
    pos[idx] = np.arange(idx.size)  # grid cell -> column in `ndvi`
    ok = np.ones(shape, bool) if exclude is None else ~exclude

    segmap = np.full(shape, -1, "int32")  # corridor pixel -> segment
    for i, g in enumerate(rows):
        segmap[rasterize_mask(g, shape, transform, all_touched=True) & (segmap < 0) & ok] = i
    corr = np.flatnonzero(segmap.ravel() >= 0)
    sid = segmap.ravel()[corr]

    vals = ndvi.astype("float32")
    vals[ndvi == NODATA] = np.nan
    vals /= 10000.0
    nseg, ndays = len(segs), len(days)
    ref_med = np.full((ndays, nseg), np.nan, "float32")
    for i, g in enumerate(refs):
        cells = np.flatnonzero(rasterize_mask(g, shape, transform).ravel() & ok.ravel())
        v = vals[:, pos[cells[pos[cells] >= 0]]]
        if v.shape[1] < 10:
            continue
        clear = np.isfinite(v).mean(1) >= a.min_valid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_med[clear, i] = np.nanmedian(v[clear], 1)
    pix = vals[:, pos[corr]]
    diff = pix - ref_med[:, sid]  # dates x corridor pixels

    doy = days.dayofyear.values
    is_base = days < pd.Timestamp(a.monitor_start)
    base_t = np.flatnonzero(is_base)
    months = set(range(a.months[0], a.months[1] + 1))
    clear_frac = pd.DataFrame(np.isfinite(diff).T).groupby(sid).mean().T.values  # dates x seg (present segs)
    seg_present = np.unique(sid)
    z_prev = np.full(corr.size, np.nan, "float32")
    records, polys = [], []
    for t in range(ndays):
        seg_diff = pd.Series(diff[t]).groupby(sid).median()
        row = {"date": days[t], "period": "baseline" if is_base[t] else "monitor"}
        if is_base[t]:
            for j, sd in seg_diff.items():
                if clear_frac[t, np.searchsorted(seg_present, j)] >= a.min_valid:
                    records.append({**row, "seg_id": j, "diff": sd})
            continue
        d = np.abs(doy[base_t] - doy[t])
        b = diff[base_t[np.minimum(d, 365 - d) <= a.season_window]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            n = np.isfinite(b).sum(0)
            mu = np.nanmean(b, 0)
            sd = np.maximum(np.nanstd(b, 0, ddof=1), a.min_std)
        z = (diff[t] - mu) / sd
        z[n < a.min_base_obs] = np.nan
        anom = (z <= -a.z) & (z_prev <= -a.z)  # anomalous now and at this pixel's previous clear look
        seen = np.isfinite(z)
        z_prev[seen] = z[seen]

        grid = np.zeros(shape, bool)
        grid.ravel()[corr[anom]] = True
        lab, nlab = ndimage.label(grid, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        labs = lab.ravel()[corr]
        z_s = pd.Series(z).groupby(sid)
        for j, sdiff in seg_diff.items():
            if clear_frac[t, np.searchsorted(seg_present, j)] < a.min_valid:
                continue
            l = labs[(sid == j) & (labs > 0)]
            records.append({**row, "seg_id": j, "diff": sdiff,
                            "z05": z_s.get_group(j).quantile(0.05),
                            "anom_px": int(anom[sid == j].sum()),
                            "cluster_px": int(sizes[l].max()) if l.size else 0,
                            "in_season": days[t].month in months})
        if nlab and days[t].month in months:
            big = np.isin(lab, np.flatnonzero(sizes >= a.min_cluster)) & (lab > 0)
            for geom, _ in poly_shapes(big.astype("uint8"), mask=big, transform=transform, connectivity=8):
                polys.append({"date": days[t].strftime("%Y-%m-%d"), "geometry": to_shape(geom)})

    df = pd.DataFrame(records).sort_values(["seg_id", "date"]).reset_index(drop=True)
    mon = df["period"] == "monitor"
    df["flag"] = mon & df["in_season"].fillna(False).astype(bool) & (df["cluster_px"].fillna(0) >= a.min_cluster)
    scored = df[mon & df["in_season"].fillna(False).astype(bool)]
    prev = scored.groupby("seg_id")["flag"].shift(1, fill_value=False).astype(bool)
    df["alert"] = False
    df.loc[scored.index, "alert"] = scored["flag"] & prev
    return df, polys


def summarize(segs, utm, df):
    gdf = gpd.GeoDataFrame({"seg_id": range(len(segs))}, geometry=segs, crs=utm)
    gdf["start_m"] = np.cumsum([0] + [s.length for s in segs[:-1]]).round()
    mon = df[(df["period"] == "monitor") & df["in_season"].fillna(False).astype(bool)]
    agg = mon.groupby("seg_id").agg(n_obs=("date", "count"), worst_z=("z05", "min"),
                                    peak_cluster_px=("cluster_px", "max"),
                                    n_flags=("flag", "sum"), n_alerts=("alert", "sum"))
    fmt = lambda s: s.dt.strftime("%Y-%m-%d")
    first_flag = fmt(mon[mon["flag"]].groupby("seg_id")["date"].min()).rename("first_flag")
    first_alert = fmt(mon[mon["alert"]].groupby("seg_id")["date"].min()).rename("first_alert")
    gdf = gdf.join(agg, on="seg_id").join(first_flag, on="seg_id").join(first_alert, on="seg_id")
    gdf["peak_cluster_m2"] = gdf.pop("peak_cluster_px") * 100
    gdf["status"] = np.where(gdf["n_alerts"].fillna(0) > 0, "ALERT",
                     np.where(gdf["n_flags"].fillna(0) > 0, "watch", "ok"))
    gdf.loc[gdf["n_obs"].fillna(0) == 0, "status"] = "no data"
    return gdf.to_crs(4326)


# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------
STATUS_COLORS = {"ALERT": "#d62728", "watch": "#ff9f1c", "ok": "#2ca02c", "no data": "#9e9e9e"}


def sparkline_svg(g, w=330, h=130):
    """Inline SVG of the segment's NDVI difference over time (the last baseline
    year plus the monitoring period). Grey = baseline, blue = monitoring,
    red = flagged scenes."""
    g = g.dropna(subset=["diff"])
    mon_start = g.loc[g["period"] == "monitor", "date"].min()
    if pd.notna(mon_start):
        g = g[g["date"] >= mon_start - pd.Timedelta(days=365)]
    if g.empty:
        return ""
    t0, t1 = g["date"].min().value, g["date"].max().value
    lo, hi = g["diff"].min() - 0.02, g["diff"].max() + 0.02
    px = lambda d: 34 + (d.value - t0) / max(t1 - t0, 1) * (w - 44)
    py = lambda v: 8 + (hi - v) / max(hi - lo, 1e-6) * (h - 28)
    out = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="font:10px sans-serif">',
           f'<rect x="34" y="8" width="{w - 44}" height="{h - 28}" fill="#fafafa" stroke="#ccc"/>']
    if lo < 0 < hi:
        out.append(f'<line x1="34" x2="{w - 10}" y1="{py(0):.1f}" y2="{py(0):.1f}" stroke="#bbb"/>')
    for v in (lo + 0.02, hi - 0.02):
        out.append(f'<text x="30" y="{py(v) + 3:.1f}" text-anchor="end">{v:+.2f}</text>')
    mon = g[g["period"] == "monitor"]
    if len(mon):
        x = px(mon["date"].min())
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{h - 20}" stroke="#555" stroke-dasharray="2,2"/>')
    for d, v, per, f in zip(g["date"], g["diff"], g["period"], g["flag"]):
        col, r = ("#999", 1.5) if per == "baseline" else (("#d62728", 3) if f else ("#1f77b4", 2.2))
        out.append(f'<circle cx="{px(d):.1f}" cy="{py(v):.1f}" r="{r}" fill="{col}"/>')
    out.append(f'<text x="34" y="{h - 6}">{g["date"].min():%Y-%m}</text>')
    out.append(f'<text x="{w - 10}" y="{h - 6}" text-anchor="end">{g["date"].max():%Y-%m}</text></svg>')
    return "".join(out)


def merge_anomalies(polys, utm):
    """Merge per-scene anomaly patches that overlap into one patch, keeping the
    dates on which it was seen."""
    if not polys:
        return gpd.GeoDataFrame({"dates": []}, geometry=[], crs=utm)
    g = gpd.GeoDataFrame(polys, crs=utm)
    # Pixels touching at a corner, or one pixel apart, belong to the same patch.
    groups = gpd.GeoSeries([unary_union(g.geometry.buffer(15))], crs=utm).explode(index_parts=False)
    out = []
    for grp in groups:
        members = g[g.intersects(grp)]
        out.append({"dates": sorted(set(members["date"])), "geometry": unary_union(members.geometry)})
    return gpd.GeoDataFrame(out, crs=utm)


def describe_patches(patches, segs, df, a):
    """Give each anomaly patch its own status, dates and position along the route.
    A patch is an ALERT when it was present on two consecutive scored scenes of
    the segments it touches, otherwise watch."""
    starts = np.cumsum([0] + [s.length for s in segs[:-1]])
    scored = df[(df["period"] == "monitor") & df["in_season"].fillna(False).astype(bool)]
    rows = []
    for geom, dates in zip(patches.geometry, patches["dates"]):
        touch = [i for i, s in enumerate(segs) if s.distance(geom) <= a.row_width / 2 + 10]
        seg_dates = sorted(scored.loc[scored["seg_id"].isin(touch), "date"].dt.strftime("%Y-%m-%d").unique())
        seen = [d for d in seg_dates if d in set(dates)]
        if not seen:
            continue
        first_alert = next((d2 for d1, d2 in zip(seg_dates, seg_dates[1:]) if d1 in dates and d2 in dates), None)
        near = min(touch or range(len(segs)), key=lambda i: segs[i].distance(geom.centroid))
        km = (starts[near] + segs[near].project(geom.centroid)) / 1000
        rows.append({"status": "ALERT" if first_alert else "watch", "area_m2": round(geom.area),
                     "first_seen": seen[0], "first_alert": first_alert, "last_seen": seen[-1],
                     "n_scenes": len(seen), "seen": ",".join(seen), "km": round(km, 2),
                     "seg_ids": ",".join(map(str, touch)),
                     "geometry": geom})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=patches.crs) if rows else \
        gpd.GeoDataFrame({"status": []}, geometry=[], crs=patches.crs)
    if len(out):
        out = out.sort_values(["status", "area_m2"], ascending=[True, False]).reset_index(drop=True)
        out.insert(0, "patch_id", range(1, len(out) + 1))
    return out


def patch_imagery(patch, segs, days, ndvi, idx, shape, transform, margin=15, min_size=40, max_dates=16):
    """Cut the NDVI history around a patch for the map's image viewer: clear
    scenes from 90 days before it appeared to 30 days after it was last seen
    (default before/after dates included), the patch's pixels and the route
    line, all in the chip's pixel grid. NDVI is packed one byte per pixel."""
    import base64
    from shapely.geometry import box

    (x0, y0, x1, y1) = patch.geometry.bounds
    c0, r1 = ~transform * (x0, y0)
    c1, r0 = ~transform * (x1, y1)
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    half = max(min_size, (r1 - r0) + 2 * margin, (c1 - c0) + 2 * margin) / 2
    R0, R1 = int(max(0, cr - half)), int(min(shape[0], cr + half))
    C0, C1 = int(max(0, cc - half)), int(min(shape[1], cc + half))
    rr, cols = np.mgrid[R0:R1, C0:C1]
    pos = np.full(shape[0] * shape[1], -1)
    pos[idx] = np.arange(idx.size)
    col = pos[(rr * shape[1] + cols).ravel()]
    stack = np.full((len(days), col.size), NODATA, "int16")
    stack[:, col >= 0] = ndvi[:, col[col >= 0]]
    clear = (stack != NODATA).mean(1)

    def best(cands):
        for t in cands:
            if clear[t] >= 0.8:
                return t
        return max(cands, key=lambda t: clear[t]) if cands else None

    first, last = pd.Timestamp(patch["first_seen"]), pd.Timestamp(patch["last_seen"])
    key = pd.Timestamp(patch["first_alert"] if isinstance(patch["first_alert"], str) else patch["first_seen"])
    before = best([t for t in np.flatnonzero(days < first)[::-1] if days[t] >= first - pd.Timedelta(days=120)])
    after = best(list(np.flatnonzero(days >= key)[:6]))
    if before is None or after is None:
        return None

    seen = set(patch["seen"].split(","))
    window = [t for t in range(len(days))
              if first - pd.Timedelta(days=90) <= days[t] <= last + pd.Timedelta(days=30) and clear[t] >= 0.7]
    seen_t = [t for t in window if days[t].strftime("%Y-%m-%d") in seen]
    must = {int(before), int(after)} | set(seen_t[:6]) | set(seen_t[-2:])
    rest = [t for t in window if t not in must]
    room = max(0, max_dates - len(must))
    if len(rest) > room:
        rest = [rest[int(i)] for i in np.linspace(0, len(rest) - 1, room)] if room else []
    ts = sorted(must | set(int(t) for t in rest))

    sub = stack[ts].astype("float32")
    packed = np.where(sub == NODATA, 255, np.clip(np.round((sub / 10000 + 1) * 127), 0, 254)).astype("uint8")
    inside = rasterize_mask(patch.geometry, shape, transform)[R0:R1, C0:C1].astype("uint8")
    xa, ya = transform * (C0, R0)
    xb, yb = transform * (C1, R1)
    near = unary_union(segs).intersection(box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
    lines = list(getattr(near, "geoms", [near])) if not near.is_empty else []
    route = [[[round(c - C0, 2), round(r - R0, 2)] for c, r in (~transform * xy for xy in ln.coords)]
             for ln in lines if ln.geom_type == "LineString"]
    b64 = lambda arr: base64.b64encode(arr.tobytes()).decode()
    return {"w": int(C1 - C0), "h": int(R1 - R0), "pixel_m": abs(transform.a),
            "dates": [days[t].strftime("%Y-%m-%d") for t in ts],
            "seen": [days[t].strftime("%Y-%m-%d") in seen for t in ts],
            "before": ts.index(before), "after": ts.index(after),
            "ndvi": b64(packed), "mask": b64(inside), "route": route}


def copernicus_link(lat, lon, date, layer="1_TRUE_COLOR"):
    """Copernicus Browser (free) at this spot on this date. Layers: 1_TRUE_COLOR, 3_NDVI."""
    return ("https://browser.dataspace.copernicus.eu/?zoom=16"
            f"&lat={lat:.5f}&lng={lon:.5f}&themeId=DEFAULT-THEME&datasetId=S2_L2A_CDAS"
            f"&fromTime={date}T00%3A00%3A00.000Z&toTime={date}T23%3A59%3A59.999Z"
            f"&layerId={layer}&dateMode=SINGLE")


PANEL_CSS = """
<style>
#rw-panel{position:fixed;top:10px;left:10px;bottom:10px;width:330px;z-index:1001;background:#fff;
  border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.3);font:13px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
  color:#222;display:flex;flex-direction:column;overflow:hidden}
#rw-panel header{padding:12px 14px 8px;border-bottom:1px solid #eee}
#rw-panel h1{font-size:13px;font-weight:600;color:#555;margin:0 28px 2px 0}
#rw-panel .verdict{font-size:21px;font-weight:700;line-height:1.2;margin:0 0 10px}
#rw-panel .verdict .also{color:#b86e00;font-size:16px;font-weight:600}
#rw-panel .sec{margin:0 0 8px}
#rw-panel .lab{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#333;margin-bottom:2px}
#rw-panel .lab span{font-weight:400;text-transform:none;letter-spacing:0;color:#888}
#rw-panel svg{display:block;font:10px -apple-system,Segoe UI,sans-serif;fill:#555;overflow:visible}
#rw-panel .rw-seg{cursor:pointer}
#rw-panel .rw-seg:hover{opacity:.7}
#rw-panel .key{display:flex;flex-wrap:wrap;gap:2px 10px;font-size:12px;color:#333}
#rw-panel .key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#rw-panel .key .tri{font-size:9px;margin-right:3px}
#rw-panel .key .unit{color:#888}
#rw-panel dl.facts{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:4px 0 6px;font-size:12px}
#rw-panel dl.facts dt{color:#777}
#rw-panel dl.facts dd{margin:0;color:#222}
#rw-panel .warn{color:#b86e00}
#rw-panel details.how{font-size:12px;color:#444;margin:2px 0 4px}
#rw-panel details.how summary{cursor:pointer;color:#1f6feb}
#rw-panel details.how p{margin:4px 0 0}
#rw-panel .gen{color:#888}
#rw-close{position:absolute;top:8px;right:10px;border:0;background:none;font-size:20px;cursor:pointer;color:#666}
#rw-panel .note{background:#fff8e1;border-left:3px solid #ff9f1c;padding:6px 8px;margin-top:8px;font-size:12px}
#rw-list{overflow-y:auto;flex:1;padding:4px 0}
#rw-list h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#777;margin:8px 14px 4px}
.rw-item{display:block;width:100%;text-align:left;border:0;background:none;padding:7px 14px;cursor:pointer;
  border-bottom:1px solid #f2f2f2;font:inherit;color:inherit}
.rw-item:hover,.rw-item:focus{background:#f3f6fb;outline:none}
.rw-item b.st{display:inline-block;min-width:46px;padding:0 5px;border-radius:3px;color:#fff;font-size:11px;text-align:center;margin-right:6px}
.rw-item .sub{color:#666;font-size:12px;margin-left:58px}
.wx{color:#6b5a00;font-size:12px;font-style:italic}
#rw-open{position:fixed;top:10px;left:10px;z-index:1000;display:none;border:0;border-radius:6px;
  background:#fff;box-shadow:0 2px 8px rgba(0,0,0,.3);padding:8px 12px;font:600 13px sans-serif;cursor:pointer}
@media (max-width:700px){#rw-panel{width:auto;right:10px;bottom:auto;max-height:70vh}}
</style>
"""

PANEL_JS = """
document.addEventListener("DOMContentLoaded", function(){  // after folium has built the map
  var map = %(map)s, layers = {%(layers)s};
  window.RW_MAP = map;
  var panel = document.getElementById('rw-panel'), openBtn = document.getElementById('rw-open');
  function show(v){ panel.style.display = v ? 'flex' : 'none'; openBtn.style.display = v ? 'none' : 'block'; }
  document.getElementById('rw-close').onclick = function(){ show(false); };
  openBtn.onclick = function(){ show(true); };
  var wide = window.innerWidth > 700;
  if (!wide) show(false);
  L.control.zoom({position: 'bottomright'}).addTo(map);
  map.fitBounds(%(bounds)s, wide ? {paddingTopLeft: [350, 20], paddingBottomRight: [20, 20]} : {padding: [20, 20]});
  document.querySelectorAll('.rw-seg').forEach(function(r){
    r.addEventListener('click', function(){
      var b = r.dataset.b.split(',').map(Number);
      if (!wide) show(false);
      map.fitBounds([[b[0], b[1]], [b[2], b[3]]], {maxZoom: 16, paddingTopLeft: wide ? [350, 20] : [20, 20], paddingBottomRight: [20, 20]});
    });
  });
  document.querySelectorAll('.rw-item').forEach(function(btn){
    btn.onclick = function(){
      var lyr = layers[btn.dataset.id], b = lyr.getBounds();
      if (!wide) show(false);
      map.fitBounds(b, {maxZoom: 17, paddingTopLeft: wide ? [350, 0] : [0, 0]});
      setTimeout(function(){ lyr.openPopup(b.getCenter()); }, 350);
    };
  });
});
"""


VIEWER_CSS = """
<style>
.rw-thumbs{display:flex;gap:8px;margin:6px 0;cursor:zoom-in}
.rw-thumbs figure{margin:0;text-align:center;font-size:11px;color:#555}
.rw-thumbs canvas{width:150px;height:150px;border:1px solid #ccc;display:block}
.rw-inspect{display:block;width:100%;margin:2px 0 6px;padding:6px;border:1px solid #1f6feb;border-radius:5px;
  background:#1f6feb;color:#fff;font:600 12px sans-serif;cursor:pointer}
.rw-inspect:hover{background:#1a5fd0}
#rw-viewer{position:fixed;inset:0;z-index:3000;background:rgba(10,14,20,.72);display:flex;align-items:center;
  justify-content:center;font:13px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222}
#rw-viewer[hidden]{display:none}
.rw-v-box{background:#fff;border-radius:10px;width:min(1200px,96vw);height:min(860px,94vh);display:flex;
  flex-direction:column;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,.4)}
.rw-v-box>header{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid #eee}
.rw-v-box>header h2{font-size:15px;margin:0;flex:1}
#rw-v-close{border:0;background:none;font-size:24px;line-height:1;cursor:pointer;color:#555;padding:0 4px}
.rw-v-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;padding:8px 14px;border-bottom:1px solid #eee;background:#fafbfc}
.rw-v-modes{display:inline-flex;border:1px solid #ccd;border-radius:6px;overflow:hidden}
.rw-v-modes button{border:0;background:#fff;padding:5px 11px;cursor:pointer;font:inherit;border-right:1px solid #ccd}
.rw-v-modes button:last-child{border-right:0}
.rw-v-modes button[aria-pressed=true]{background:#1f6feb;color:#fff}
.rw-v-date{display:inline-flex;align-items:center;gap:4px}
.rw-v-date b{font-size:12px;color:#444;margin-right:2px}
.rw-v-date select{font:inherit;padding:3px 4px;max-width:210px}
.rw-v-toolbar button.sm{border:1px solid #ccd;background:#fff;border-radius:4px;padding:3px 8px;cursor:pointer;font:inherit}
.rw-v-toolbar label{display:inline-flex;align-items:center;gap:4px;font-size:12px;cursor:pointer}
#rw-v-stage{flex:1;min-height:0;display:flex;gap:6px;padding:6px;background:#2b2f36}
#rw-v-stage .pane{flex:1;min-width:0;min-height:0;position:relative}
#rw-v-stage canvas{width:100%;height:100%;display:block;touch-action:none;cursor:grab;border-radius:4px;background:#3a3f47}
.rw-v-box>footer{padding:8px 14px;border-top:1px solid #eee;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;font-size:12px}
#rw-v-readout{flex:1 1 360px;font-variant-numeric:tabular-nums;color:#222}
#rw-v-legend{display:flex;align-items:center;gap:6px;color:#555}
#rw-v-legend .bar{width:140px;height:10px;border-radius:2px;border:1px solid #ccc}
#rw-v-links a{color:#1f6feb}
.rw-v-hint{flex-basis:100%;color:#777;font-size:11px}
#rw-v-rain{flex-basis:100%;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
#rw-v-rain svg{font:10px sans-serif;fill:#555;overflow:visible}
@media (max-width:700px){
  .rw-v-box{width:100vw;height:100vh;border-radius:0}
  #rw-v-stage{flex-direction:column}
  .rw-v-hint{display:none}
}
</style>
"""

VIEWER_HTML = """
<div id="rw-viewer" role="dialog" aria-modal="true" aria-labelledby="rw-v-title" hidden>
 <div class="rw-v-box">
  <header><h2 id="rw-v-title"></h2><button id="rw-v-close" aria-label="Close viewer">&times;</button></header>
  <div class="rw-v-toolbar">
   <div class="rw-v-modes" role="group" aria-label="Comparison mode">
    <button data-mode="side" title="Side by side (1)">Side by side</button><button data-mode="swipe" title="Drag the divider (2)">Swipe</button><button data-mode="blink" title="Flip between dates (3)">Blink</button><button data-mode="change" title="After minus before (4)">Change</button>
   </div>
   <span class="rw-v-date"><b>Before</b><button class="sm" data-step="b,-1" aria-label="Earlier before date">&#9664;</button><select id="rw-v-before" aria-label="Before date"></select><button class="sm" data-step="b,1" aria-label="Later before date">&#9654;</button></span>
   <span class="rw-v-date"><b>After</b><button class="sm" data-step="a,-1" aria-label="Earlier after date">&#9664;</button><select id="rw-v-after" aria-label="After date"></select><button class="sm" data-step="a,1" aria-label="Later after date">&#9654;</button></span>
   <span><button class="sm" data-zoom="in" aria-label="Zoom in">+</button> <button class="sm" data-zoom="out" aria-label="Zoom out">&minus;</button> <button class="sm" data-zoom="fit">Fit</button></span>
   <label><input type="checkbox" id="rw-v-outline" checked> Patch outline</label>
   <label><input type="checkbox" id="rw-v-route" checked> Pipeline</label>
   <label id="rw-v-rel-wrap" title="Subtract the typical change across the whole image, so area-wide drying or green-up cancels out (the same comparison the detector makes)"><input type="checkbox" id="rw-v-rel" checked> Change vs. surroundings</label>
  </div>
  <div id="rw-v-stage"></div>
  <footer>
   <div id="rw-v-readout"></div><div id="rw-v-legend"></div><div id="rw-v-links"></div>
   <div id="rw-v-rain"></div>
   <div class="rw-v-hint">Scroll or double-click to zoom, drag to pan. Keys: 1&ndash;4 modes &middot; &larr;/&rarr; after date &middot; Shift+&larr;/&rarr; before date &middot; +/&minus;/0 zoom &middot; Space pause blink &middot; Esc close. Dates marked &#9679; are scenes where the patch was flagged.</div>
  </footer>
 </div>
</div>
"""

VIEWER_JS = r"""
document.addEventListener("DOMContentLoaded", function(){
  var DATA = JSON.parse(document.getElementById('rw-data').textContent);
  var map = window.RW_MAP;
  // NDVI ramp: brown (bare) -> pale -> green (vigorous); change ramp: red (lost green) -> white -> blue (gained)
  var RAMP = [[-0.1,140,90,40],[0.2,235,225,160],[0.45,120,180,80],[0.75,20,100,30]];
  var DIV = [[-0.3,178,24,43],[-0.1,239,138,98],[0,247,247,247],[0.1,103,169,207],[0.3,33,102,172]];
  function interp(stops, v){
    if (v <= stops[0][0]) return stops[0].slice(1);
    for (var k = 1; k < stops.length; k++) if (v <= stops[k][0]) {
      var a = stops[k-1], b = stops[k], f = (v - a[0]) / (b[0] - a[0]);
      return [1,2,3].map(function(j){ return Math.round(a[j] + f * (b[j] - a[j])); });
    }
    return stops[stops.length-1].slice(1);
  }
  function gradient(stops){
    var lo = stops[0][0], hi = stops[stops.length-1][0];
    return 'linear-gradient(to right,' + stops.map(function(s){
      return 'rgb(' + s.slice(1).join(',') + ') ' + Math.round((s[0]-lo)/(hi-lo)*100) + '%'; }).join(',') + ')';
  }
  function decode(s){ var b = atob(s), a = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }

  var patches = {};
  function getPatch(pid){
    var p = patches[pid];
    if (!p) {
      var d = DATA[pid];
      p = patches[pid] = Object.assign({id: pid}, d, {px: decode(d.ndvi), inside: decode(d.mask), imgs: {}, base: {}});
      p.edges = edgesOf(p);
    }
    return p;
  }
  function value(p, t, i){ var q = p.px[t * p.w * p.h + i]; return q === 255 ? null : q / 127 - 1; }
  function edgesOf(p){  // pixel borders around the patch, drawn as crisp lines at any zoom
    var e = [], w = p.w, h = p.h;
    function inn(x, y){ return x >= 0 && y >= 0 && x < w && y < h && p.inside[y*w+x] === 1; }
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) if (inn(x, y)) {
      if (!inn(x, y-1)) e.push([x, y, x+1, y]);
      if (!inn(x, y+1)) e.push([x, y+1, x+1, y+1]);
      if (!inn(x-1, y)) e.push([x, y, x, y+1]);
      if (!inn(x+1, y)) e.push([x+1, y, x+1, y+1]);
    }
    return e;
  }
  function baseline(p, t, t2){  // median change outside the patch = area-wide drying or green-up
    var k = t + ':' + t2;
    if (p.base[k] !== undefined) return p.base[k];
    var d = [];
    for (var i = 0; i < p.w * p.h; i++) { var a = value(p, t, i), b = value(p, t2, i); if (!p.inside[i] && a !== null && b !== null) d.push(b - a); }
    d.sort(function(x, y){ return x - y; });
    return p.base[k] = d.length ? d[Math.floor(d.length / 2)] : 0;
  }
  function image(p, kind, t, t2){  // one canvas pixel per 10 m pixel; cached
    var rel = kind === 'change' && V.rel, key = kind + ':' + t + ':' + t2 + (rel ? ':rel' : '');
    var off = rel ? baseline(p, t, t2) : 0;
    if (p.imgs[key]) return p.imgs[key];
    var c = document.createElement('canvas'); c.width = p.w; c.height = p.h;
    var ctx = c.getContext('2d'), im = ctx.createImageData(p.w, p.h);
    for (var i = 0; i < p.w * p.h; i++) {
      var a = value(p, t, i), rgb;
      if (kind === 'ndvi') rgb = a === null ? [205,205,205] : interp(RAMP, a);
      else { var b = value(p, t2, i); rgb = (a === null || b === null) ? [205,205,205] : interp(DIV, b - a - off); }
      im.data[i*4] = rgb[0]; im.data[i*4+1] = rgb[1]; im.data[i*4+2] = rgb[2]; im.data[i*4+3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return p.imgs[key] = c;
  }

  // ---- drawing helpers (view = {s: CSS px per image pixel, ox, oy}) ----
  function prep(cv){
    var r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    var W = Math.max(1, Math.round(r.width * d)), H = Math.max(1, Math.round(r.height * d));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    var ctx = cv.getContext('2d'); ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
    return {ctx: ctx, w: r.width, h: r.height};
  }
  function fitView(p, w, h){ var s = Math.min(w / p.w, h / p.h) * 0.96; return {s: s, ox: (w - p.w * s) / 2, oy: (h - p.h * s) / 2}; }
  function paint(ctx, img, v, clip){
    ctx.save();
    if (clip) { ctx.beginPath(); ctx.rect(clip[0], 0, clip[1] - clip[0], 1e5); ctx.clip(); }
    ctx.translate(v.ox, v.oy); ctx.scale(v.s, v.s); ctx.imageSmoothingEnabled = false; ctx.drawImage(img, 0, 0);
    ctx.restore();
  }
  function overlays(ctx, p, v, o){
    ctx.save(); ctx.translate(v.ox, v.oy); ctx.scale(v.s, v.s); ctx.lineJoin = 'round';
    if (o.route && p.route.length) {
      [['rgba(0,0,0,.6)', 3.5, []], ['#fff', 1.6, [6, 4]]].forEach(function(st){
        ctx.beginPath();
        p.route.forEach(function(line){ line.forEach(function(pt, k){ k ? ctx.lineTo(pt[0], pt[1]) : ctx.moveTo(pt[0], pt[1]); }); });
        ctx.strokeStyle = st[0]; ctx.lineWidth = st[1] / v.s; ctx.setLineDash(st[2].map(function(x){ return x / v.s; })); ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    if (o.outline) {
      ctx.beginPath(); p.edges.forEach(function(e){ ctx.moveTo(e[0], e[1]); ctx.lineTo(e[2], e[3]); });
      ctx.strokeStyle = '#ff00ff'; ctx.lineWidth = 2 / v.s; ctx.stroke();
    }
    if (o.hover) { ctx.strokeStyle = '#ffeb3b'; ctx.lineWidth = 2.5 / v.s; ctx.strokeRect(o.hover[0], o.hover[1], 1, 1); }
    ctx.restore();
  }
  function tag(ctx, text, x, y, right){
    ctx.font = '600 12px -apple-system,Segoe UI,sans-serif';
    var w = ctx.measureText(text).width + 14, x0 = right ? x - w : x;
    ctx.fillStyle = 'rgba(0,0,0,.7)'; ctx.fillRect(x0, y, w, 22); ctx.fillStyle = '#fff'; ctx.fillText(text, x0 + 7, y + 15);
  }
  function scaleBar(ctx, p, v, H){
    var perM = v.s / p.pixel_m, len = [10,20,50,100,200,500,1000].filter(function(L){ return L * perM >= 50; })[0] || 1000, L = len * perM;
    ctx.fillStyle = 'rgba(255,255,255,.88)'; ctx.fillRect(8, H - 30, L + 10, 22);
    ctx.fillStyle = '#111'; ctx.fillRect(13, H - 14, L, 3); ctx.font = '11px sans-serif'; ctx.fillText(len + ' m', 13, H - 18);
  }

  // ---- rainfall between the chosen dates ----
  var RAIN = DATA._rain || null;
  function dayIdx(date){ return Math.round((Date.parse(date) - Date.parse(RAIN.start)) / 864e5); }
  function addDays(date, k){ return new Date(Date.parse(date) + k * 864e5).toISOString().slice(0, 10); }
  function total(d0, d1, field){  // days after d0 up to and including d1
    var s = 0, n = 0, a = dayIdx(d0) + 1, b = dayIdx(d1);
    for (var i = Math.max(0, a); i <= Math.min(b, RAIN.mm.length - 1); i++) { var v = RAIN[field][i]; if (v !== null) { s += v; n++; } }
    return {mm: s, n: n, days: Math.max(0, b - a + 1)};
  }
  function shortDate(d){ return new Date(Date.parse(d)).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'}); }
  function vsNormal(r, nrm){
    if (!r.days) return '';
    if (r.n < r.days * 0.8) return '<b>' + Math.round(r.mm) + ' mm</b> <span style="color:#b86e00">(data incomplete)</span>';
    var pct = nrm.mm > 0 ? r.mm / nrm.mm : 1, col = pct < 0.5 ? '#9a6a00' : pct > 1.5 ? '#2f6db5' : '#333';
    return '<b style="color:' + col + '">' + Math.round(r.mm) + ' mm</b> (normal ' + Math.round(nrm.mm) + ')';
  }
  function rainHtml(dB, dA){
    if (!RAIN) return '';
    var lo = dB < dA ? dB : dA, hi = dB < dA ? dA : dB;
    var txt = 'Rain between the dates: ' + vsNormal(total(lo, hi, 'mm'), total(lo, hi, 'normal')) +
      ' &middot; 30 days before ' + shortDate(hi) + ': ' + vsNormal(total(addDays(hi, -30), hi, 'mm'), total(addDays(hi, -30), hi, 'normal'));
    var s0 = addDays(lo, -30), s1 = addDays(hi, 3), n = dayIdx(s1) - dayIdx(s0), W = 300, H = 34;
    var x = function(d){ return (dayIdx(d) - dayIdx(s0)) / n * W; }, bw = Math.max(1, W / n), svg = '';
    for (var i = dayIdx(s0); i <= dayIdx(s1); i++) {
      var v = i >= 0 && i < RAIN.mm.length ? RAIN.mm[i] : null;
      if (v !== null && v >= 0.5) { var h = Math.min(v, 40) / 40 * 22; svg += '<rect x="' + ((i - dayIdx(s0)) / n * W).toFixed(1) + '" y="' + (26 - h).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + h.toFixed(1) + '" fill="#2f6db5"><title>' + addDays(RAIN.start, i) + ': ' + Math.round(v) + ' mm</title></rect>'; }
    }
    [[dB, 'B'], [dA, 'A']].forEach(function(m){ var X = x(m[0]).toFixed(1); svg += '<line x1="' + X + '" x2="' + X + '" y1="0" y2="26" stroke="#d62728" stroke-width="1.5"/><text x="' + X + '" y="34" text-anchor="middle" fill="#d62728">' + m[1] + '</text>'; });
    return '<span>' + txt + '</span><svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Daily rain around the chosen dates">' +
      '<line x1="0" x2="' + W + '" y1="26" y2="26" stroke="#b9c3cf"/>' + svg + '</svg>';
  }

  // ---- viewer ----
  var V = {mode: 'side', swipe: 0.5, showB: true, paused: false, rel: true};
  var dlg = document.getElementById('rw-viewer'), stage = document.getElementById('rw-v-stage');
  var selB = document.getElementById('rw-v-before'), selA = document.getElementById('rw-v-after');
  var chkO = document.getElementById('rw-v-outline'), chkR = document.getElementById('rw-v-route');
  function fmt(x){ return x === null ? 'no data (cloud)' : x.toFixed(2); }
  function sgn(x){ return (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(2); }
  function copernicus(date){
    return 'https://browser.dataspace.copernicus.eu/?zoom=16&lat=' + V.p.lat.toFixed(5) + '&lng=' + V.p.lon.toFixed(5) +
      '&themeId=DEFAULT-THEME&datasetId=S2_L2A_CDAS&fromTime=' + date + 'T00%3A00%3A00.000Z&toTime=' + date +
      'T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR&dateMode=SINGLE';
  }

  function buildStage(){
    stage.innerHTML = '';
    var n = V.mode === 'side' ? 2 : 1;
    for (var k = 0; k < n; k++) {
      var pane = document.createElement('div'), cv = document.createElement('canvas');
      pane.className = 'pane'; cv.setAttribute('role', 'img'); pane.appendChild(cv); stage.appendChild(pane); attach(cv);
    }
  }
  function fit(){ var r = stage.querySelector('canvas').getBoundingClientRect(); V.view = fitView(V.p, r.width, r.height); V.fitS = V.view.s; }
  var raf = 0;
  function redraw(){ if (!raf) raf = requestAnimationFrame(function(){ raf = 0; render(); }); }
  function render(){
    var p = V.p, cvs = stage.querySelectorAll('canvas'), v = V.view, dB = p.dates[V.b], dA = p.dates[V.a];
    var o = {outline: chkO.checked, route: chkR.checked, hover: V.hover};
    if (V.mode === 'side') {
      [[V.b, 'Before'], [V.a, 'After']].forEach(function(x, k){
        var c = prep(cvs[k]); paint(c.ctx, image(p, 'ndvi', x[0]), v); overlays(c.ctx, p, v, o);
        tag(c.ctx, x[1] + ' · ' + p.dates[x[0]], 8, 8); scaleBar(c.ctx, p, v, c.h);
        cvs[k].setAttribute('aria-label', x[1] + ' greenness image, ' + p.dates[x[0]]);
      });
    } else if (V.mode === 'swipe') {
      var c = prep(cvs[0]), X = V.swipe * c.w;
      paint(c.ctx, image(p, 'ndvi', V.a), v); paint(c.ctx, image(p, 'ndvi', V.b), v, [0, X]); overlays(c.ctx, p, v, o);
      c.ctx.fillStyle = '#fff'; c.ctx.fillRect(X - 1, 0, 2, c.h);
      c.ctx.beginPath(); c.ctx.arc(X, c.h / 2, 13, 0, 7); c.ctx.fill();
      c.ctx.fillStyle = '#333'; c.ctx.font = 'bold 13px sans-serif'; c.ctx.fillText('↔', X - 7, c.h / 2 + 5);
      tag(c.ctx, 'Before · ' + dB, 8, 8); tag(c.ctx, 'After · ' + dA, c.w - 8, 8, true); scaleBar(c.ctx, p, v, c.h);
      cvs[0].setAttribute('aria-label', 'Swipe comparison, before ' + dB + ', after ' + dA);
    } else if (V.mode === 'blink') {
      var c = prep(cvs[0]), t = V.showB ? V.b : V.a;
      paint(c.ctx, image(p, 'ndvi', t), v); overlays(c.ctx, p, v, o);
      tag(c.ctx, (V.showB ? 'Before · ' : 'After · ') + p.dates[t] + (V.paused ? '  (paused)' : ''), 8, 8); scaleBar(c.ctx, p, v, c.h);
    } else {
      var c = prep(cvs[0]);
      paint(c.ctx, image(p, 'change', V.b, V.a), v); overlays(c.ctx, p, v, o);
      tag(c.ctx, 'Change · ' + dB + ' → ' + dA + (V.rel ? ' · vs. surroundings' : ''), 8, 8); scaleBar(c.ctx, p, v, c.h);
      cvs[0].setAttribute('aria-label', 'Change in greenness from ' + dB + ' to ' + dA);
    }
    // readout, legend, links
    var ro = document.getElementById('rw-v-readout');
    if (V.hover) {
      var i = V.hover[1] * p.w + V.hover[0], b = value(p, V.b, i), a = value(p, V.a, i);
      var rel = a !== null && b !== null ? a - b - baseline(p, V.b, V.a) : null;
      ro.innerHTML = '<b>Pixel</b> ' + (p.inside[i] ? '<span style="color:#c0c">inside patch</span>' : 'outside patch') +
        ' &middot; greenness before <b>' + fmt(b) + '</b> &rarr; after <b>' + fmt(a) + '</b>' +
        (rel !== null ? ' &middot; change <b>' + sgn(a - b) + '</b>, vs. surroundings <b style="color:' +
          (rel < -0.05 ? '#b2182b' : rel > 0.05 ? '#2166ac' : '#333') + '">' + sgn(rel) + '</b>' : '');
    } else ro.textContent = 'Hover over the image to read greenness (NDVI) values.';
    var lg = document.getElementById('rw-v-legend'), chg = V.mode === 'change';
    lg.innerHTML = (chg ? 'less green' : 'bare') + '<span class="bar" style="background:' + gradient(chg ? DIV : RAMP) + '"></span>' +
      (chg ? 'greener' : 'vigorous') + (chg ? ' <span style="color:#888">(&minus;0.3 &hellip; +0.3)</span>' : ' <span style="color:#888">(NDVI &minus;0.1 &hellip; 0.75)</span>');
    document.getElementById('rw-v-rain').innerHTML = rainHtml(dB, dA);
    document.getElementById('rw-v-links').innerHTML = 'Satellite photo: <a target="_blank" rel="noopener" href="' + copernicus(dB) +
      '">before</a> &middot; <a target="_blank" rel="noopener" href="' + copernicus(dA) + '">after</a>';
  }

  function zoomAt(mx, my, f){
    var v = V.view, s = Math.min(V.fitS * 24, Math.max(V.fitS * 0.5, v.s * f)); f = s / v.s;
    v.ox = mx - (mx - v.ox) * f; v.oy = my - (my - v.oy) * f; v.s = s; redraw();
  }
  function attach(cv){
    var drag = null;
    function local(e){ var r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top, r.width]; }
    function nearDivider(m){ return V.mode === 'swipe' && Math.abs(m[0] - V.swipe * m[2]) < 16; }
    cv.addEventListener('wheel', function(e){ e.preventDefault(); var m = local(e); zoomAt(m[0], m[1], Math.pow(1.0015, -e.deltaY)); }, {passive: false});
    cv.addEventListener('dblclick', function(e){ var m = local(e); zoomAt(m[0], m[1], e.shiftKey ? 0.5 : 2); });
    cv.addEventListener('pointerdown', function(e){
      cv.setPointerCapture(e.pointerId); var m = local(e);
      drag = nearDivider(m) ? {swipe: true} : {x: e.clientX, y: e.clientY, ox: V.view.ox, oy: V.view.oy};
      if (!drag.swipe) cv.style.cursor = 'grabbing';
    });
    cv.addEventListener('pointermove', function(e){
      var m = local(e);
      if (drag && drag.swipe) V.swipe = Math.min(1, Math.max(0, m[0] / m[2]));
      else if (drag) { V.view.ox = drag.ox + e.clientX - drag.x; V.view.oy = drag.oy + e.clientY - drag.y; }
      var px = Math.floor((m[0] - V.view.ox) / V.view.s), py = Math.floor((m[1] - V.view.oy) / V.view.s);
      V.hover = (px >= 0 && py >= 0 && px < V.p.w && py < V.p.h) ? [px, py] : null;
      if (!drag) cv.style.cursor = nearDivider(m) ? 'ew-resize' : 'grab';
      redraw();
    });
    cv.addEventListener('pointerup', function(){ drag = null; cv.style.cursor = 'grab'; });
    cv.addEventListener('pointerleave', function(){ if (!drag) { V.hover = null; redraw(); } });
  }

  function setMode(mode){
    // keep the zoom when switching between single-pane modes (same canvas size)
    var keep = V.view && V.mode !== 'side' && mode !== 'side' ? V.view : null;
    V.mode = mode;
    document.querySelectorAll('.rw-v-modes button').forEach(function(b){ b.setAttribute('aria-pressed', b.dataset.mode === mode); });
    clearInterval(V.timer); V.showB = true;
    document.getElementById('rw-v-rel-wrap').style.display = mode === 'change' ? '' : 'none';
    if (mode === 'blink') V.timer = setInterval(function(){ if (!V.paused) { V.showB = !V.showB; redraw(); } }, 700);
    buildStage(); fit();
    if (keep) V.view = keep;
    render();
  }
  function fillSelects(){
    [selB, selA].forEach(function(sel){
      sel.innerHTML = V.p.dates.map(function(d, k){ return '<option value="' + k + '">' + d + (V.p.seen[k] ? '  ●' : '') + '</option>'; }).join('');
    });
    selB.value = V.b; selA.value = V.a;
  }
  function step(which, dir){
    var k = Math.min(V.p.dates.length - 1, Math.max(0, V[which] + dir)); V[which] = k;
    (which === 'b' ? selB : selA).value = k; redraw();
  }
  function open(pid){
    V.p = getPatch(pid); V.b = V.p.before; V.a = V.p.after; V.hover = null; V.view = null;
    document.getElementById('rw-v-title').textContent = V.p.title;
    fillSelects();
    V.lastFocus = document.activeElement; dlg.hidden = false; document.getElementById('rw-v-close').focus();
    setMode(V.mode);
  }
  function close(){ dlg.hidden = true; clearInterval(V.timer); if (V.lastFocus) V.lastFocus.focus(); }
  window.RW_OPEN_VIEWER = open;

  document.getElementById('rw-v-close').onclick = close;
  dlg.addEventListener('click', function(e){ if (e.target === dlg) close(); });
  document.querySelectorAll('.rw-v-modes button').forEach(function(b){ b.onclick = function(){ setMode(b.dataset.mode); }; });
  document.querySelectorAll('[data-step]').forEach(function(b){ b.onclick = function(){ var s = b.dataset.step.split(','); step(s[0], +s[1]); }; });
  selB.onchange = function(){ V.b = +selB.value; redraw(); };
  selA.onchange = function(){ V.a = +selA.value; redraw(); };
  chkO.onchange = chkR.onchange = redraw;
  document.getElementById('rw-v-rel').onchange = function(e){ V.rel = e.target.checked; redraw(); };
  document.querySelectorAll('[data-zoom]').forEach(function(b){ b.onclick = function(){
    if (b.dataset.zoom === 'fit') { fit(); redraw(); return; }
    var r = stage.querySelector('canvas').getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, b.dataset.zoom === 'in' ? 1.5 : 1 / 1.5);
  }; });
  document.addEventListener('keydown', function(e){
    if (dlg.hidden) return;
    var tagName = (e.target.tagName || '').toLowerCase();
    if (e.key === 'Escape') { close(); e.preventDefault(); return; }
    if (tagName === 'select') return;
    if ('1234'.indexOf(e.key) >= 0 && e.key) { setMode(['side','swipe','blink','change'][+e.key - 1]); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { step(e.shiftKey ? 'b' : 'a', e.key === 'ArrowLeft' ? -1 : 1); e.preventDefault(); }
    else if (e.key === '+' || e.key === '=') document.querySelector('[data-zoom=in]').click();
    else if (e.key === '-') document.querySelector('[data-zoom=out]').click();
    else if (e.key === '0') document.querySelector('[data-zoom=fit]').click();
    else if (e.key === ' ' && V.mode === 'blink') { V.paused = !V.paused; redraw(); e.preventDefault(); }
    else if (e.key === 'Tab') {  // keep focus inside the dialog
      var f = dlg.querySelectorAll('button, select, input'), first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { last.focus(); e.preventDefault(); }
      else if (!e.shiftKey && document.activeElement === last) { first.focus(); e.preventDefault(); }
    }
  });
  window.addEventListener('resize', function(){ if (!dlg.hidden) { fit(); redraw(); } });

  // Popup thumbnails, drawn when a patch popup opens; clicking them opens the viewer.
  map.on('popupopen', function(e){
    var el = e.popup.getElement();
    el.querySelectorAll('.rw-thumbs[data-pid]').forEach(function(box){
      var p = getPatch(box.dataset.pid);
      if (!box.childElementCount) [[p.before, 'Before'], [p.after, 'After']].forEach(function(x){
        var fig = document.createElement('figure'), cv = document.createElement('canvas'), cap = document.createElement('figcaption');
        cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', x[1] + ' greenness, ' + p.dates[x[0]]);
        cap.textContent = x[1] + ': ' + p.dates[x[0]]; fig.appendChild(cv); fig.appendChild(cap); box.appendChild(fig);
      });
      box.querySelectorAll('canvas').forEach(function(cv, k){
        var c = prep(cv), v = fitView(p, c.w, c.h); paint(c.ctx, image(p, 'ndvi', k ? p.after : p.before), v);
        overlays(c.ctx, p, v, {outline: true, route: false});
      });
      box.onclick = function(){ open(box.dataset.pid); };
    });
    el.querySelectorAll('.rw-inspect[data-pid]').forEach(function(b){ b.onclick = function(){ open(b.dataset.pid); }; });
  });
});
"""


def nice_date(d, year=True):
    """2023-05-01 -> '1 May 2023' (or '1 May')."""
    d = pd.Timestamp(d)
    return f"{d.day} {d:%b}" + (f" {d.year}" if year else "")


def date_span(a, b):
    """'1 May – 30 Nov 2023', or with both years when they differ."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return f"{nice_date(a, a.year != b.year)} – {nice_date(b)}"


def route_strip_svg(seg_gdf, marks_km, w=300):
    """Where along the route: one coloured block per segment (click to zoom), with
    km labels and a black triangle for each --mark."""
    lengths = list(seg_gdf.to_crs(seg_gdf.estimate_utm_crs()).length / 1000)
    total = seg_gdf["start_m"].iloc[-1] / 1000 + lengths[-1]
    x = lambda km: 1 + km / total * (w - 2)
    out = [f"<svg class='rw-strip' width='{w}' height='38' viewBox='0 0 {w} 38' role='img' "
           f"aria-label='Route status from km 0 to km {total:.1f}'>"]
    for (_, r), L in zip(seg_gdf.iterrows(), lengths):
        km0 = r["start_m"] / 1000
        b = r.geometry.bounds
        out.append(f"<rect class='rw-seg' data-b='{b[1]:.6f},{b[0]:.6f},{b[3]:.6f},{b[2]:.6f}' x='{x(km0):.1f}' y='10' "
                   f"width='{max(1.2, x(km0 + L) - x(km0) + 0.4):.1f}' height='14' fill='{STATUS_COLORS[r['status']]}'>"
                   f"<title>km {km0:.2f}–{km0 + L:.2f}: {r['status']} (click to zoom)</title></rect>")
    for km, label in marks_km:
        out.append(f"<path d='M{x(km) - 5:.1f},1 h10 l-5,7 z' fill='#111'><title>{label}</title></path>")
    step = next(s for s in (0.5, 1, 2, 5, 10, 20, 50, 100) if total / s <= 6)
    for k in np.arange(0, total - step * 0.5, step):  # leave room for the end label
        anchor = "start" if k == 0 else "middle"
        out.append(f"<text x='{x(k):.1f}' y='35' text-anchor='{anchor}'>{k:g}</text>")
    out.append(f"<text x='{w - 1}' y='35' text-anchor='end'>km {total:.1f}</text></svg>")
    return "".join(out)


def timeline_svg(info, patches, w=300):
    """When: the monitoring period month by month. Top: red / orange dots on the
    dates patches first alerted / appeared. Middle: a tick per clear image.
    Bottom: daily rain, on a tan / blue background where the past 30 days were
    much drier / wetter than normal. Unscored months are greyed out."""
    t0, t1 = pd.Timestamp(info["monitor_start"]), pd.Timestamp(info["end"])
    span = max((t1 - t0).days, 1)
    x = lambda d: 1 + (pd.Timestamp(d) - t0).days / span * (w - 2)
    ctx = info.get("rain")
    H = 66 if ctx is not None else 46
    lab_y = H - 5
    out = [f"<svg class='rw-time' width='{w}' height='{H}' viewBox='0 0 {w} {H}' role='img' "
           f"aria-label='Monitoring timeline {date_span(t0, t1)}'>",
           f"<rect x='1' y='12' width='{w - 2}' height='16' fill='#eef3fb' stroke='#c9d6ea'/>"]
    if ctx is not None:
        c = ctx.loc[t0:t1]
        r30 = ctx["rain"].rolling(30, min_periods=25).sum().loc[t0:t1]
        n30 = ctx["normal"].rolling(30, min_periods=25).sum().loc[t0:t1]
        for d0 in pd.date_range(t0, t1, freq="7D"):  # weekly background: 30-day rain vs normal
            ratio = r30.get(min(d0 + pd.Timedelta(days=6), t1), np.nan) / max(n30.get(min(d0 + pd.Timedelta(days=6), t1), np.nan), 1e-6)
            fill = "#f1dfb8" if ratio < 0.5 else "#cfe3f5" if ratio > 1.5 else None
            if fill:
                a, b = x(d0), x(min(d0 + pd.Timedelta(days=7), t1))
                out.append(f"<rect x='{a:.1f}' y='30' width='{b - a:.1f}' height='20' fill='{fill}'/>")
        bw = max(1.0, (w - 2) / span)
        for d, mm in c["rain"].dropna().items():
            if mm >= 0.5:
                h = min(mm, 40) / 40 * 19
                out.append(f"<rect x='{x(d):.1f}' y='{50 - h:.1f}' width='{bw:.1f}' height='{h:.1f}' fill='#2f6db5'>"
                           f"<title>{nice_date(d)}: {mm:.0f} mm</title></rect>")
        out.append(f"<line x1='1' x2='{w - 1}' y1='50' y2='50' stroke='#b9c3cf'/>")
    months = pd.date_range(t0.to_period("M").to_timestamp(), t1, freq="MS")
    step = max(1, int(np.ceil(len(months) / 8)))
    m_lo, m_hi = info["months"]
    labelled = False
    for k, m0 in enumerate(months):
        m1 = m0 + pd.offsets.MonthBegin(1)
        a, b = x(max(m0, t0)), x(min(m1, t1))
        if not (m_lo <= m0.month <= m_hi):
            out.append(f"<rect x='{a:.1f}' y='12' width='{b - a:.1f}' height='16' fill='#e2e2e2'>"
                       f"<title>{m0:%b %Y}: not scored (vegetation dormant)</title></rect>")
        if m0 >= t0:
            out.append(f"<line x1='{a:.1f}' x2='{a:.1f}' y1='12' y2='{H - 14}' stroke='#c9d6ea'/>")
        if k % step == 0 and b - a >= 24:  # skip a sliver of a month at either end
            lab = f"{m0:%b}" + (f" \u2019{m0:%y}" if m0.month == 1 or not labelled else "")
            out.append(f"<text x='{a + 2:.1f}' y='{lab_y}'>{lab}</text>")
            labelled = True
    for d in info["image_dates"]:
        out.append(f"<line x1='{x(d):.1f}' x2='{x(d):.1f}' y1='15' y2='25' stroke='#5b7db1' stroke-width='1'/>")
    if len(patches):
        for _, p in patches.sort_values("status", ascending=False).iterrows():
            d = p["first_alert"] if p["status"] == "ALERT" else p["first_seen"]
            what = "alert" if p["status"] == "ALERT" else "first seen"
            out.append(f"<circle cx='{x(d):.1f}' cy='6' r='4' fill='{STATUS_COLORS[p['status']]}' stroke='#fff'>"
                       f"<title>Patch #{int(p['patch_id'])} {what} {nice_date(d)}</title></circle>")
    out.append("</svg>")
    return "".join(out)


def overview_html(seg_gdf, patches, info):
    """Top of the side panel: verdict first, then where (route strip), when
    (timeline) and the key facts; detection settings folded away."""
    import html as htmlmod

    esc = htmlmod.escape
    n_alert = int((patches["status"] == "ALERT").sum()) if len(patches) else 0
    n_watch = int((patches["status"] == "watch").sum()) if len(patches) else 0
    if n_alert:
        verdict = (f"<span style='color:{STATUS_COLORS['ALERT']}'>{n_alert} alert{'s' if n_alert > 1 else ''} to check</span>"
                   + (f"<span class='also'> · {n_watch} to watch</span>" if n_watch else ""))
    elif n_watch:
        verdict = f"<span style='color:#b86e00'>No alerts</span><span class='also'> · {n_watch} to watch</span>"
    else:
        verdict = f"<span style='color:{STATUS_COLORS['ok']}'>No alerts</span>"
    present = set(seg_gdf["status"])
    key = "".join(f"<span><i style='background:{STATUS_COLORS[s]}'></i>{'alert' if s == 'ALERT' else s}</span>"
                  for s in ("ALERT", "watch", "ok", "no data") if s in present or s in ("ALERT", "watch", "ok"))
    if info["marks_km"]:
        key += "<span><b class='tri'>&#9660;</b>marker</span>"
    b0, b1 = pd.Timestamp(info["baseline_start"]), pd.Timestamp(info["monitor_start"]) - pd.Timedelta(days=1)
    years = (b1 - b0).days / 365.25
    latest = pd.Timestamp(info["latest"]) if info["latest"] else None
    gap = (pd.Timestamp(info["end"]) - latest).days if latest is not None else None
    stale = f" <span class='warn'>({gap} days before period end)</span>" if gap is not None and gap > 20 else ""
    facts = [
        ("Watching", f"{date_span(info['monitor_start'], info['end'])}"),
        ("Images", f"{info['n_images']} clear &middot; latest {nice_date(latest) if latest is not None else '–'}{stale}"),
        ("Normal from", f"{b0:%b %Y} – {b1:%b %Y} &middot; {years:.0f} years"),
    ]
    rain_key = ""
    ctx = info.get("rain")
    if ctx is not None:
        per = ctx.loc[pd.Timestamp(info["monitor_start"]):pd.Timestamp(info["end"])].dropna()
        if len(per):
            pct = per["rain"].sum() / max(per["normal"].sum(), 1e-6)
            word = "drier than" if pct < 0.8 else "wetter than" if pct > 1.2 else "close to"
            facts.append(("Rain", f"{per['rain'].sum():.0f} mm &middot; {pct:.0%} of normal ({word} usual)"
                          + (f" <span class='warn'>(to {nice_date(per.index.max())})</span>"
                             if (pd.Timestamp(info['end']) - per.index.max()).days > 5 else "")))
        rain_key = ("<div class='key'><span><i style='background:#f1dfb8'></i>past 30 days dry</span>"
                    "<span><i style='background:#cfe3f5'></i>wet</span><span class='unit'>vs. normal &middot; "
                    "rain: CHIRPS</span></div>")
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts)
    m_lo, m_hi = info["months"]
    how = (f"Vegetation within {info['row_width'] / 2:.0f} m of the pipe is compared with land 60–250 m away, "
           f"and each 10 m pixel with the same time of year in the normal period. A pixel is unusual at "
           f"z ≤ −{info['z']} on two clear images in a row; a <b>watch</b> is an unusual patch of at least "
           f"{info['min_cluster'] * 100:,} m², an <b>alert</b> is one seen on two clear images in a row. Only "
           f"{pd.Timestamp(2000, m_lo, 1):%B}–{pd.Timestamp(2000, m_hi, 1):%B} is scored."
           + (" Cropland, built-up land and water are ignored." if info["mask_cropland"] else "")
           + f"<br><span class='gen'>Map generated {nice_date(info['generated'])}.</span>")
    return (f"<h1>{esc(info['title'])}</h1>"
            f"<div class='verdict'>{verdict}</div>"
            f"<div class='sec'><div class='lab'>Where <span>· {info['km']:.1f} km in {info['n_segs']} sections"
            f" of {info['seg_len']:.0f} m &middot; click to zoom</span></div>{route_strip_svg(seg_gdf, info['marks_km'])}"
            f"<div class='key'>{key}</div></div>"
            f"<div class='sec'><div class='lab'>When <span>· dots = patches, ticks = clear images"
            f"{', bars = daily rain' if info.get('rain') is not None else ''}</span></div>"
            f"{timeline_svg(info, patches)}{rain_key}</div>"
            f"<dl class='facts'>{dl}</dl>"
            f"<details class='how'><summary>How alerts are decided</summary><p>{how}</p></details>"
            f"<div class='note'><b>An alert is not proof of a leak.</b> Ploughing, clearing and repair work also "
            f"kill vegetation; check each one with the before/after images.</div>")


def write_map(seg_gdf, df, patches, views, info, path, markers=()):
    import html as htmlmod
    import folium

    esc = htmlmod.escape
    c = seg_gdf.geometry.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=13, tiles=None, control_scale=True, zoom_control=False)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", show=False).add_to(m)
    val = lambda v, f="{:.1f}": "–" if pd.isna(v) else f.format(v)
    txt = lambda v: v if isinstance(v, str) else "–"
    num = lambda v: int(v) if pd.notna(v) else 0

    # Segments: a visible line plus a wide, invisible line on top that is easy to click.
    seg_layer = folium.FeatureGroup(name="Pipeline segments").add_to(m)
    for _, r in seg_gdf.iterrows():
        sid, col = r["seg_id"], STATUS_COLORS[r["status"]]
        geo = r.geometry.__geo_interface__
        html = (f"<b>Segment {sid}</b> &nbsp;(km {val(r['start_m'] / 1000, '{:.2f}')})<br>"
                f"Status: <b style='color:{col}'>{r['status']}</b><br>"
                f"First flag: {txt(r['first_flag'])} &nbsp; First alert: {txt(r['first_alert'])}<br>"
                f"Largest anomaly patch: {val(r['peak_cluster_m2'], '{:,.0f}')} m² &nbsp; Worst z: {val(r['worst_z'])}<br>"
                f"Flags / alerts / scenes scored: {num(r['n_flags'])} / {num(r['n_alerts'])} / {num(r['n_obs'])}<br>"
                f"{sparkline_svg(df[df.seg_id == sid])}<br>"
                f"<span style='font-size:10px;color:#666'>NDVI over the pipe minus NDVI of nearby land. "
                f"Grey = baseline years, blue = monitoring, red = flagged scene; dashed line = monitoring start.</span>")
        folium.GeoJson(geo, style_function=lambda _, col=col: {"color": col, "weight": 6, "opacity": 0.9}
                       ).add_to(seg_layer)
        folium.GeoJson(geo, style_function=lambda _: {"color": "#000", "weight": 24, "opacity": 0.01},
                       tooltip=f"Segment {sid}: {r['status']} (click for details)",
                       popup=folium.Popup(html, max_width=370)).add_to(seg_layer)

    # Anomaly patches: outline + a status-coloured dot that stays visible when zoomed out.
    patch_layer = folium.FeatureGroup(name="Anomaly patches").add_to(m)
    layer_vars = []
    for _, p in patches.to_crs(4326).iterrows():
        pid, col = int(p["patch_id"]), STATUS_COLORS[p["status"]]
        pt = p.geometry.representative_point()
        view = views.get(pid)
        if view:
            view.update(lat=pt.y, lon=pt.x,
                        title=f"Patch #{pid} \u00b7 {p['status']} \u00b7 {p['area_m2']:,} m\u00b2 at km {p['km']:.2f}")
            d_before, d_after = view["dates"][view["before"]], view["dates"][view["after"]]
            th = (f"<div class='rw-thumbs' data-pid='{pid}' title='Click to inspect'></div>"
                  f"<button class='rw-inspect' data-pid='{pid}'>&#x2922; Inspect &amp; compare images</button>")
            links = (f"<a href='{esc(copernicus_link(pt.y, pt.x, d_before))}' target='_blank' rel='noopener'>"
                     f"before ({d_before})</a> · <a href='{esc(copernicus_link(pt.y, pt.x, d_after))}' "
                     f"target='_blank' rel='noopener'>after ({d_after})</a>")
        else:
            th, links = "<p style='color:#777'>No clear image of this spot.</p>", "–"
        html = (f"<b>Patch #{pid}</b> &nbsp;<b style='color:{col}'>{p['status']}</b><br>"
                f"{p['area_m2']:,} m² at km {p['km']:.2f} (segment {esc(p['seg_ids'])})<br>"
                f"First seen {p['first_seen']} &nbsp; First alert {txt(p['first_alert'])}<br>"
                f"Last seen {p['last_seen']} &nbsp; ({p['n_scenes']} scenes)"
                + (f"<div class='wx'>Weather: {esc(p['weather'])}</div>" if isinstance(p.get('weather'), str) and p['weather'] else "")
                + f"{th}"
                f"<span style='font-size:11px;color:#555'>Greenness (NDVI): brown = bare, green = vigorous; "
                f"magenta = the anomaly patch.</span><br>"
                f"<span style='font-size:12px'>Satellite photo (Copernicus Browser): {links}</span><br>"
                f"<a href='https://www.google.com/maps/search/?api=1&query={pt.y:.6f},{pt.x:.6f}' "
                f"target='_blank' rel='noopener' style='font-size:12px'>Open location in Google Maps</a>")
        feats = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"kind": "area"}, "geometry": p.geometry.__geo_interface__},
            {"type": "Feature", "properties": {"kind": "dot"}, "geometry": pt.__geo_interface__}]}
        gj = folium.GeoJson(
            feats,
            marker=folium.CircleMarker(radius=7, fill=True),
            style_function=lambda f, col=col: (
                {"color": "#ff00ff", "weight": 2, "fillOpacity": 0.15} if f["properties"]["kind"] == "area"
                else {"color": "#fff", "weight": 2, "fillColor": col, "fillOpacity": 1, "radius": 7}),
            tooltip=f"Patch #{pid}: {p['status']}, {p['area_m2']:,} m²",
            popup=folium.Popup(html, max_width=340))
        gj.add_to(patch_layer)
        layer_vars.append(f"{pid}: {gj.get_name()}")

    for lat, lon, label in markers:
        folium.Marker([lat, lon], tooltip=label, icon=folium.Icon(color="black", icon="info-sign")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    # Side panel: overview (verdict, where, when, facts) and the clickable list of patches.
    wx = lambda p: (f"<div class='sub wx'>Weather: {esc(p['weather'])}</div>"
                    if isinstance(p.get("weather"), str) and p["weather"] else "")
    items = ""
    for status in ("ALERT", "watch"):
        sub = patches[patches["status"] == status] if len(patches) else patches
        if not len(sub):
            continue
        items += f"<h2>{'Alerts' if status == 'ALERT' else 'Watch'} ({len(sub)})</h2>"
        for _, p in sub.iterrows():
            when = (f"alert {nice_date(p['first_alert'])}" if p["status"] == "ALERT"
                    else f"seen {nice_date(p['first_seen'])}")
            items += (f"<button class='rw-item' data-id='{int(p['patch_id'])}'>"
                      f"<b class='st' style='background:{STATUS_COLORS[p['status']]}'>#{int(p['patch_id'])}</b>"
                      f"km {p['km']:.2f} &middot; {p['area_m2']:,} m²<div class='sub'>{when}"
                      f"{' &middot; first seen ' + nice_date(p['first_seen'], False) if p['status'] == 'ALERT' else ''}"
                      f"</div>{wx(p)}</button>")
    if not items:
        items = "<p style='padding:10px 14px;color:#555'>No anomaly patches in the monitoring period.</p>"
    n_alert = int((patches["status"] == "ALERT").sum()) if len(patches) else 0
    panel = (f"<div id='rw-panel'><header><button id='rw-close' title='Hide panel' aria-label='Hide panel'>&times;</button>"
             f"{overview_html(seg_gdf, patches, info)}</header><div id='rw-list'>{items}</div></div>"
             f"<button id='rw-open'>&#9776; {n_alert} alert{'s' if n_alert != 1 else ''}</button>")
    raw = lambda x: folium.Element("{% raw %}" + x + "{% endraw %}")  # no template syntax in our own HTML
    root = m.get_root()
    root.header.add_child(raw(PANEL_CSS + VIEWER_CSS))
    root.html.add_child(raw(panel + VIEWER_HTML))
    data = {str(k): v for k, v in views.items()}
    ctx = info.get("rain")
    if ctx is not None and views:
        r0 = min(pd.Timestamp(v["dates"][0]) for v in views.values()) - pd.Timedelta(days=35)
        win = ctx.loc[r0:]
        data["_rain"] = {"start": f"{win.index[0]:%Y-%m-%d}",
                         "mm": [None if pd.isna(v) else round(float(v), 1) for v in win["rain"]],
                         "normal": [round(float(v), 2) for v in win["normal"]]}
    root.html.add_child(raw("<script id='rw-data' type='application/json'>"
                            + json.dumps(data).replace("</", "<\\/") + "</script>"))
    b = seg_gdf.total_bounds
    root.script.add_child(raw(PANEL_JS % {
        "map": m.get_name(), "layers": ", ".join(layer_vars),
        "bounds": json.dumps([[b[1], b[0]], [b[3], b[2]]])}))
    root.script.add_child(raw(VIEWER_JS))
    m.save(path)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--route", required=True, help="pipeline centerline (GeoJSON/KML/SHP/GPKG)")
    p.add_argument("--baseline-start", required=True, help="YYYY-MM-DD (3-4 years before monitor start)")
    p.add_argument("--monitor-start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--seg-len", type=float, default=250, help="segment length, m (default 250)")
    p.add_argument("--row-width", type=float, default=100, help="width of the scored corridor, m (default 100)")
    p.add_argument("--ref-inner", type=float, default=60, help="reference ring inner distance, m (default 60)")
    p.add_argument("--ref-outer", type=float, default=250, help=f"reference ring outer distance, m (default 250, max {CACHE_BUFFER})")
    p.add_argument("--z", type=float, default=3.5, help="per-pixel z-score threshold (default 3.5)")
    p.add_argument("--min-cluster", type=int, default=5, help="min adjacent anomalous pixels for a flag (default 5 = 500 m2)")
    p.add_argument("--months", default="4-10", help="months scored, e.g. 4-10 = Apr-Oct (default)")
    p.add_argument("--max-cloud", type=float, default=60, help="max scene cloud cover, %% (default 60)")
    p.add_argument("--min-valid", type=float, default=0.6, help="min clear-pixel fraction per zone (default 0.6)")
    p.add_argument("--mask-cropland", action="store_true",
                   help="ignore cultivated, built-up and water pixels (ESA WorldCover + USDA CDL)")
    p.add_argument("--mark", action="append", default=[], metavar="LAT,LON,LABEL",
                   help="add a marker to the map (e.g. a known incident); repeatable")
    p.add_argument("--refresh", action="store_true", help="ignore cached imagery and re-download")
    p.add_argument("--cache", help="imagery cache file (default: <out>/ndvi_cube.npz); share one between runs of the same route")
    p.add_argument("--title", help="name shown on the map (default: route file name)")
    p.add_argument("--no-rain", action="store_true",
                   help="skip the CHIRPS rainfall overlay (e.g. offline, or outside 50°S–50°N)")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    a.months = [int(x) for x in a.months.split("-")]
    a.season_window, a.min_base_obs, a.min_std = 30, 4, 0.03
    if a.ref_outer > CACHE_BUFFER:
        sys.exit(f"--ref-outer can be at most {CACHE_BUFFER} m")

    os.makedirs(a.out, exist_ok=True)
    lines, utm = read_route(a.route)
    segs = make_segments(lines, a.seg_len)
    total_km = sum(s.length for s in segs) / 1000
    print(f"Route: {total_km:.1f} km -> {len(segs)} segments ({utm})")

    corridor = unary_union(segs).buffer(CACHE_BUFFER)
    bbox_ll = gpd.GeoSeries([corridor], crs=utm).to_crs(4326).total_bounds

    def find_items():
        items = search_scenes(bbox_ll, a.baseline_start, a.end, a.max_cloud)
        print(f"Found {len(items)} Sentinel-2 items", end="")
        items = dedupe_items(items)
        print(f" -> {len(items)} after removing duplicate processings "
              f"({len({i.datetime.date() for i in items})} dates)")
        if not items:
            sys.exit("No scenes found; check dates and route location.")
        return items

    keep = lambda shape, transform: rasterize_mask(corridor, shape, transform, all_touched=True)
    days, ndvi, idx, shape, transform = get_imagery(a, find_items, utm, bbox_ll, keep)
    if (days < pd.Timestamp(a.monitor_start)).sum() < 20:
        print("WARNING: very few baseline scenes; results will be noisy. Use a longer baseline.")
    exclude = landcover_mask(shape, transform, utm) if a.mask_cropland else None

    df, polys = detect(days, ndvi, idx, shape, transform, segs, a, exclude)
    if df.empty:
        sys.exit("No usable observations (all cloudy?). Try a higher --max-cloud or lower --min-valid.")
    df.to_csv(os.path.join(a.out, "timeseries.csv"), index=False)
    seg_gdf = summarize(segs, utm, df)
    seg_gdf.to_file(os.path.join(a.out, "segments.geojson"), driver="GeoJSON")
    patches = describe_patches(merge_anomalies(polys, utm), segs, df, a)
    ctx = None
    if not a.no_rain:
        line_ll = gpd.GeoSeries([unary_union(segs)], crs=utm).to_crs(4326).iloc[0]
        line_ll = linemerge(line_ll) if line_ll.geom_type == "MultiLineString" else line_ll
        pts = [line_ll.interpolate(f, normalized=True) for f in (0, 0.5, 1)] if line_ll.geom_type == "LineString" \
            else [line_ll.centroid]
        if all(abs(p.y) <= 50 for p in pts):
            rain = get_rainfall([(p.x, p.y) for p in pts], a.baseline_start, a.end, os.path.join(a.out, "rainfall.csv"))
            if rain.notna().mean() > 0.5:
                ctx = rain_context(rain, a.monitor_start)
            else:
                print("Rainfall unavailable (network?); continuing without it.")
        else:
            print("Route is outside CHIRPS coverage (50°S–50°N); no rainfall overlay.")
    if len(patches):
        patches["weather"] = [weather_note(ctx, d) for d in patches["first_seen"]]
    views = {int(p["patch_id"]): patch_imagery(p, segs, days, ndvi, idx, shape, transform)
             for _, p in patches.iterrows()}
    views = {k: v for k, v in views.items() if v}
    patches.to_crs(4326).to_file(os.path.join(a.out, "anomalies.geojson"), driver="GeoJSON")

    markers = []
    for s in a.mark:
        lat, lon, *label = s.split(",")
        markers.append((float(lat), float(lon), ",".join(label) or "marker"))
    starts = np.cumsum([0] + [sg.length for sg in segs[:-1]])
    marks_km = []
    for lat, lon, label in markers:
        pt = gpd.GeoSeries.from_xy([lon], [lat], crs=4326).to_crs(utm).iloc[0]
        i = min(range(len(segs)), key=lambda k: segs[k].distance(pt))
        if segs[i].distance(pt) < 1000:
            marks_km.append(((starts[i] + segs[i].project(pt)) / 1000, label))
    scored = df[(df["period"] == "monitor")]["date"].drop_duplicates().sort_values()
    info = {"title": a.title or os.path.splitext(os.path.basename(a.route))[0].replace("_", " "),
            "km": total_km, "n_segs": len(segs), "seg_len": a.seg_len,
            "baseline_start": a.baseline_start, "monitor_start": a.monitor_start, "end": a.end,
            "months": a.months, "image_dates": list(scored), "n_images": len(scored),
            "latest": scored.max() if len(scored) else None, "marks_km": marks_km,
            "row_width": a.row_width, "z": a.z, "min_cluster": a.min_cluster,
            "mask_cropland": a.mask_cropland, "generated": pd.Timestamp.now(), "rain": ctx}
    write_map(seg_gdf, df, patches, views, info, os.path.join(a.out, "map.html"), markers)

    print("\nAnomaly patches to check (ALERT first, largest first):")
    cols = ["patch_id", "status", "area_m2", "km", "first_seen", "first_alert", "seg_ids"]
    if "weather" in patches and patches["weather"].astype(bool).any():
        cols.append("weather")
    print(patches[cols].fillna("–").to_string(index=False) if len(patches) else "  none")
    print(f"\nSegment status counts: {seg_gdf['status'].value_counts().to_dict()}")
    print(f"Wrote {a.out}/map.html, segments.geojson, anomalies.geojson, timeseries.csv")


if __name__ == "__main__":
    main()
