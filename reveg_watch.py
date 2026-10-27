#!/usr/bin/env python3
"""
reveg_watch.py — revegetation and restoration screening for a pipeline ROW
(proof of concept, free Sentinel-2 imagery only)

What it reports, per half-mile section between mileposts:
  1. Revegetation: how green the restored workspace is compared with similar
     nearby undisturbed land ("% of reference"), each year from pre-construction
     to now, with the 2026 trend and the latest (October) reading.
  2. Workspace check: ground that was vegetated before construction but bare
     afterwards OUTSIDE the approved workspace, and how the temporary yards are
     recovering.
  3. Erosion-risk screening: bare ground in the ROW now, combined with slope
     (USGS 3DEP elevation) and nearby water / wetland.

Inputs: a GeoPackage with layers
  route      pipeline centerline (LineStrings, column `segment`)
  mileposts  points with columns `mp` (number) and `segment`
  workspace  polygons with columns `name` and `kind` (row / yard / facility)

Usage:
  python reveg_watch.py --project ~/Merjent_local/<project>/data/project.gpkg \
      --data ~/Merjent_local/<project>/data --out ~/Merjent_local/<project>/outputs \
      --start 2023-04-01 --end 2026-10-19 --pre-years 2023,2024 --construction-year 2025

Everything is screening, not a verdict: 10 m pixels, greenness not measured
cover, and every flag needs a look on the ground or in sharper imagery.
"""
import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box, mapping
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import row_watch as rw  # shared helpers: scene search, de-duplication, rainfall, PNG/date formatting

NODATA = rw.NODATA
PEAK = ((6, 15), (9, 15))          # peak growing season (month, day) in the upper Midwest
CLASSES = [  # % of reference at peak season -> revegetation class
    (90, "similar", "#1a9850"), (70, "establishing", "#91cf60"),
    (50, "lagging", "#fc8d59"), (-1e9, "poor", "#d73027")]
# Colour-blind-safe alternative (blue-to-red), swapped in by the map's toggle
CB_COLORS = {"#1a9850": "#2c7bb6", "#91cf60": "#8fc1e3", "#fc8d59": "#fdae61", "#d73027": "#d7191c"}
NOCLASS = ("no data", "#9e9e9e")
RISK_COLORS = {"High": "#b2182b", "Medium": "#ef8a62", "Low": "#67a9cf", "n/a": "#bbbbbb"}
# ESA WorldCover classes -> land-use groups used to pick comparable reference land
LC_GROUP = {10: "forest", 20: "open", 30: "open", 40: "crop", 50: "built", 60: "open",
            80: "water", 90: "wetland", 95: "wetland", 100: "open"}
REF_FOR = {"crop": ["crop"], "open": ["open"], "wetland": ["wetland", "open"],
           "forest": ["open", "crop"], "built": ["open"], "water": ["wetland", "open"]}


def classify(pct):
    if pct is None or not np.isfinite(pct):
        return NOCLASS
    for lo, name, col in CLASSES:
        if pct >= lo:
            return name, col
    return NOCLASS


def in_peak(dates):
    md = dates.month * 100 + dates.day
    return (md >= PEAK[0][0] * 100 + PEAK[0][1]) & (md <= PEAK[1][0] * 100 + PEAK[1][1])


# --------------------------------------------------------------------------
# Sections and chunks
# --------------------------------------------------------------------------
def build_sections(mp, utm, section_mi):
    """Half-mile (default) sections between mileposts, drawn through the MP points."""
    rows = []
    for seg, g in mp.to_crs(utm).sort_values("mp").groupby("segment"):
        g = g.reset_index(drop=True)
        bins = np.floor(g["mp"] / section_mi + 1e-9) * section_mi
        for lo in np.unique(bins):
            inside = g[bins == lo]
            nxt = g[g["mp"] > inside["mp"].max()].head(1)
            pts = list(inside.geometry) + ([nxt.geometry.iloc[0]] if len(nxt) and nxt["mp"].iloc[0] - inside["mp"].max() < 0.2 else [])
            geom = LineString([(p.x, p.y) for p in pts]) if len(pts) > 1 else pts[0].buffer(20)
            rows.append({"segment": seg, "mp_from": round(float(lo), 2), "mp_to": round(float(lo + section_mi), 2),
                         "mp_first": float(inside["mp"].min()), "mp_last": float(inside["mp"].max()), "geometry": geom})
    sec = gpd.GeoDataFrame(rows, crs=utm).sort_values(["segment", "mp_from"]).reset_index(drop=True)
    sec.insert(0, "section_id", [f"{s} MP {a:.1f}–{b:.1f}" for s, a, b in zip(sec.segment, sec.mp_from, sec.mp_to)])
    return sec


def build_chunks(sec, per_chunk):
    chunks = []
    for seg, g in sec.groupby("segment"):
        idx = list(g.index)
        for k in range(0, len(idx), per_chunk):
            part = idx[k:k + per_chunk]
            chunks.append({"segment": seg, "sections": part,
                           "id": f"{seg}_{sec.loc[part[0], 'mp_from']:.1f}"})
    return chunks


# --------------------------------------------------------------------------
# Imagery
# --------------------------------------------------------------------------
def load_ndvi_quiet(items, utm, bbox_ll, keep, label, batch=16):
    """Like row_watch.load_ndvi, with one progress line per ~25% instead of per scene."""
    from odc.stac import configure_rio, load as stac_load

    configure_rio(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
    ds = stac_load(items, bands=["B04", "B08", "SCL"], crs=utm, resolution=10,
                   bbox=list(bbox_ll), groupby="solar_day", chunks={})
    gb = ds.odc.geobox
    shape, transform = tuple(gb.shape), gb.transform
    idx = np.flatnonzero(keep(shape, transform).ravel())
    days = pd.DatetimeIndex(ds.time.values).normalize()
    offsets = {pd.Timestamp(it.datetime.date()): rw.item_offset(it) for it in items}
    out = np.full((len(days), idx.size), NODATA, "int16")
    for b0 in range(0, len(days), batch):
        sl = ds.isel(time=slice(b0, b0 + batch)).compute()
        for k in range(sl.sizes["time"]):
            t = b0 + k
            off = offsets.get(days[t], 0)
            red = (sl.B04.values[k].ravel()[idx].astype("float32") - off) / 10000
            nir = (sl.B08.values[k].ravel()[idx].astype("float32") - off) / 10000
            ok = np.isin(sl.SCL.values[k].ravel()[idx], rw.VALID_SCL) & (red > 0) & (nir > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.clip((nir - red) / (nir + red), -1, 1)
            out[t][ok] = np.round(v[ok] * 10000).astype("int16")
        done = min(b0 + batch, len(days))
        if done == len(days) or (done // batch) % max(1, len(days) // batch // 4) == 0:
            print(f"    {label}: {done}/{len(days)} scenes")
    return days, out, idx, shape, transform


def get_chunk_imagery(cache_path, key, loader):
    from affine import Affine

    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return pd.DatetimeIndex(z["dates"]), z["ndvi"], z["idx"], tuple(z["shape"]), Affine(*z["transform"])
    days, ndvi, idx, shape, transform = loader()
    np.savez_compressed(cache_path, key=key, dates=days.values.astype("datetime64[ns]"), ndvi=ndvi, idx=idx,
                        shape=np.array(shape), transform=np.array(list(transform)[:6]))
    return days, ndvi, idx, shape, transform


def load_static(collection, band, shape, transform, utm, resampling="nearest", latest=True):
    """Land cover / elevation on the chunk's 10 m grid."""
    from odc.geo.geobox import GeoBox
    from odc.stac import load as stac_load

    gb = GeoBox(shape, transform, utm)
    aoi = gb.extent.to_crs(4326).geom.__geo_interface__
    items = list(rw.stac_client().search(collections=[collection], intersects=aoi).item_collection())
    if not items:
        return None
    if latest and collection == "esa-worldcover":
        year = max(i.id.split("_")[3] for i in items)
        items = [i for i in items if i.id.split("_")[3] == year]
    if collection == "usda-cdl":
        items = [i for i in items if i.id.startswith("cultivated")]
        if not items:
            return None
        year = max(i.id.split("_")[1] for i in items)
        items = [i for i in items if i.id.split("_")[1] == year]
    if collection == "3dep-seamless":
        items = [i for i in items if i.properties.get("gsd") == 10] or items
    ds = stac_load(items, bands=[band], geobox=gb, resampling=resampling, groupby="solar_day")
    return ds[band].max("time").values if "time" in ds.dims else ds[band].values


def slope_degrees(dem, pixel=10.0):
    gy, gx = np.gradient(dem.astype("float64"), pixel)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


# --------------------------------------------------------------------------
# Per-chunk analysis
# --------------------------------------------------------------------------
def analyse_chunk(ch, sec, ws, mp, items, utm, a):
    from rasterio.features import rasterize
    from scipy import ndimage
    from scipy.spatial import cKDTree

    secs = sec.loc[ch["sections"]]
    corridor = unary_union(list(secs.geometry)).buffer(a.ref_outer + 60)
    bbox_ll = gpd.GeoSeries([corridor.buffer(30)], crs=utm).to_crs(4326).total_bounds
    near_items = [i for i in items if box(*bbox_ll).intersects(_footprint(i))]
    keep = lambda shape, transform: rw.rasterize_mask(corridor, shape, transform, all_touched=True)
    key = hashlib.md5(json.dumps([np.round(bbox_ll, 5).tolist(), a.start, a.end, a.max_cloud]).encode()).hexdigest()[:12]
    days, ndvi, idx, shape, transform = get_chunk_imagery(
        os.path.join(a.data, "cache", f"{ch['id']}.npz"), key,
        lambda: load_ndvi_quiet(near_items, utm, bbox_ll, keep, ch["id"]))

    H, W = shape
    pos = np.full(H * W, -1)
    pos[idx] = np.arange(idx.size)
    rr, cc = np.divmod(idx, W)
    xs, ys = transform * (cc + 0.5, rr + 0.5)

    # zones from the workspace polygons
    ws_c = ws[ws.intersects(corridor.buffer(300))]
    all_ws = rw.rasterize_mask(unary_union(list(ws_c.geometry)) if len(ws_c) else corridor.buffer(-1e6), shape, transform)
    row_geom = unary_union(list(ws_c[ws_c.kind == "row"].geometry)) if (ws_c.kind == "row").any() else None
    row_any = rw.rasterize_mask(row_geom, shape, transform) if row_geom is not None else np.zeros(shape, bool)
    row_core = rw.rasterize_mask(row_geom.buffer(-a.edge_m), shape, transform) if row_geom is not None else row_any
    dist_out = ndimage.distance_transform_edt(~all_ws) * abs(transform.a)  # m outside any workspace

    # land cover before construction (WorldCover 2021) + USDA cultivated land
    wc = load_static("esa-worldcover", "map", shape, transform, utm)
    cdl = load_static("usda-cdl", "cultivated", shape, transform, utm)
    group = np.vectorize(lambda v: LC_GROUP.get(int(v), "open"))(wc) if wc is not None else np.full(shape, "open")
    if cdl is not None:
        group = np.where((cdl == 2) & (group != "water"), "crop", group)
    dem = load_static("3dep-seamless", "data", shape, transform, utm, resampling="bilinear")
    slope = slope_degrees(dem) if dem is not None else np.full(shape, np.nan)

    # pixel -> section via nearest milepost of the whole segment
    seg_mp = mp[mp.segment == ch["segment"]].to_crs(utm)
    tree = cKDTree(np.c_[seg_mp.geometry.x, seg_mp.geometry.y])
    dmp, k = tree.query(np.c_[xs, ys])
    mp_val = seg_mp["mp"].values[k]
    sec_of = np.full(idx.size, -1)
    for i, s in secs.iterrows():
        sec_of[(mp_val >= s.mp_from - 1e-9) & (mp_val < s.mp_to - 1e-9) & (dmp <= a.ref_outer + 120)] = i

    flat = lambda m: m.ravel()[idx]
    is_core, is_row, d_out, grp, slp = flat(row_core), flat(row_any), flat(dist_out), flat(group), flat(slope)
    vals = ndvi.astype("float32")
    vals[ndvi == NODATA] = np.nan
    vals /= 10000

    years = sorted(set(days.year))
    peak = in_peak(days)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp = {y: np.nanmedian(vals[peak & (days.year == y)], 0) if (peak & (days.year == y)).any()
                else np.full(idx.size, np.nan) for y in years}
        cur_sel = days >= pd.Timestamp(a.current_from)
        comp["current"] = np.nanmedian(vals[cur_sel], 0) if cur_sel.any() else np.full(idx.size, np.nan)
        recent_sel = days >= pd.Timestamp(a.end) - pd.Timedelta(days=45)
        comp["recent"] = np.nanmedian(vals[recent_sel], 0) if recent_sel.any() else np.full(idx.size, np.nan)
    pre = np.nanmean(np.vstack([comp[y] for y in a.pre_years if y in comp]), 0) if a.pre_years else np.full(idx.size, np.nan)

    # ---- disturbance outside the approved workspace (item 2) ----
    post_years = [y for y in years if y >= a.construction_year]
    post_min = np.nanmin(np.vstack([comp[y] for y in post_years]), 0) if post_years else np.full(idx.size, np.nan)
    cand = ((d_out >= a.outside_min) & (d_out <= a.outside_max) & (pre >= 0.5) & (post_min < 0.3)
            & ~np.isin(grp, ["crop", "built", "water"]))
    dist_polys = []
    if cand.any():
        grid = np.zeros(shape, bool)
        grid.ravel()[idx[cand]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        from rasterio.features import shapes as poly_shapes
        from shapely.geometry import shape as to_shape
        for geom, v in poly_shapes(lab.astype("int32"), mask=grid, transform=transform, connectivity=8):
            v = int(v)
            if sizes[v] < a.min_patch_px:
                continue
            g = to_shape(geom)
            pix = idx[lab.ravel()[idx] == v]
            pp = pos[pix]
            yrs = [y for y in post_years if np.nanmedian(comp[y][pp]) < 0.3]
            dist_polys.append({"geometry": g, "area_m2": round(g.area), "pixels": int(sizes[v]),
                               "years_bare": ",".join(map(str, yrs)),
                               "ndvi_pre": round(float(np.nanmedian(pre[pp])), 2),
                               "ndvi_after": round(float(np.nanmedian(post_min[pp])), 2),
                               "dist_m": round(float(np.nanmedian(d_out[pp])))})

    # ---- per-section statistics ----
    rows, ts_rows, thumbs = [], [], {}
    pix_pct = np.full(idx.size, np.nan, "float32")
    ref_ring = (d_out >= a.ref_inner) & (d_out <= a.ref_outer) & ~cand
    for i, s in secs.iterrows():
        mine = sec_of == i
        rp = np.flatnonzero(mine & is_core)
        core_used = True
        if rp.size < a.min_row_px:
            rp, core_used = np.flatnonzero(mine & is_row), False
        if rp.size == 0:
            rows.append({"sec_index": i, "row_px": 0})
            continue
        pre_lc = pd.Series(grp[rp]).value_counts()
        pre_lc = pre_lc.index[0] if len(pre_lc) else "open"
        ref_groups = REF_FOR.get(pre_lc, ["open"])
        fp, ref_note = np.array([], int), ""
        for gname in ref_groups:
            fp = np.flatnonzero(mine & ref_ring & (grp == gname))
            if fp.size >= a.min_ref_px:
                ref_note = gname
                break
        if fp.size < a.min_ref_px:
            fp, ref_note = np.flatnonzero(mine & ref_ring & ~np.isin(grp, ["built", "water"])), "mixed"
        # per-date ratio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rv, fv = vals[:, rp], vals[:, fp]
            rclear, fclear = np.isfinite(rv).mean(1), np.isfinite(fv).mean(1) if fp.size else np.zeros(len(days))
            rmed, fmed = np.nanmedian(rv, 1), np.nanmedian(fv, 1) if fp.size else np.full(len(days), np.nan)
        good = (rclear >= a.min_valid) & (fclear >= a.min_valid) & (fmed >= 0.25)
        ratio = np.where(good, rmed / fmed, np.nan)
        for t in np.flatnonzero(good):
            ts_rows.append({"sec_index": i, "date": days[t], "row_ndvi": round(float(rmed[t]), 3),
                            "ref_ndvi": round(float(fmed[t]), 3), "pct_of_ref": round(float(ratio[t] * 100), 1)})
        r = {"sec_index": i, "row_px": int(rp.size), "row_core": core_used, "ref_px": int(fp.size),
             "land_before": pre_lc, "ref_land": ref_note}
        for y in years:
            sel = good & peak & (days.year == y)
            r[f"peak_{y}"] = float(np.nanmedian(ratio[sel]) * 100) if sel.any() else np.nan
            r[f"n_{y}"] = int(sel.sum())
        sel = good & (days >= pd.Timestamp(a.current_from))
        r["current"] = float(np.nanmedian(ratio[sel]) * 100) if sel.any() else np.nan
        r["current_dates"] = ",".join(days[sel].strftime("%Y-%m-%d"))
        # 2026 season trend: change in % of reference per 30 days, May–Sep of the last year
        ly = max(years)
        sel = good & (days.year == ly) & (days.month >= 5) & (days.month <= 9)
        if sel.sum() >= 4:
            tt = (days[sel] - days[sel][0]).days.values
            r["trend_30d"] = float(np.polyfit(tt, ratio[sel] * 100, 1)[0] * 30)
        else:
            r["trend_30d"] = np.nan
        # erosion screen: bare ground now (median of the last 45 days), slope, water nearby
        rec, rref = comp["recent"][rp], comp["recent"][fp] if fp.size else np.array([np.nan])
        r["bare_now"] = float(np.mean(rec[np.isfinite(rec)] < a.bare_ndvi)) if np.isfinite(rec).any() else np.nan
        r["bare_ref"] = float(np.mean(rref[np.isfinite(rref)] < a.bare_ndvi)) if np.isfinite(rref).any() else np.nan
        pk, pkr = comp[ly][rp], comp[ly][fp] if fp.size else np.array([np.nan])
        r["bare_peak"] = float(np.mean(pk[np.isfinite(pk)] < a.bare_ndvi)) if np.isfinite(pk).any() else np.nan
        r["bare_peak_ref"] = float(np.mean(pkr[np.isfinite(pkr)] < a.bare_ndvi)) if np.isfinite(pkr).any() else np.nan
        r["slope_p90"] = float(np.nanpercentile(slp[rp], 90)) if np.isfinite(slp[rp]).any() else np.nan
        near = mine & (d_out <= 60)
        r["near_water"] = bool(np.isin(grp[near], ["water", "wetland"]).sum() >= 3)
        rows.append(r)
        thumbs[i] = section_thumbs(rp, fp, idx, shape, transform, comp, pre, a, row_any, dist_out)
        # pixel-level: each ROW pixel's peak-season greenness as % of this section's reference
        if fp.size and pre_lc != "built":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                refmed = np.nanmedian(comp[ly][fp])
            allrow = np.flatnonzero(mine & is_row)
            if np.isfinite(refmed) and refmed >= 0.25:
                pix_pct[allrow] = comp[ly][allrow] / refmed * 100
    row_px = np.flatnonzero((sec_of >= 0) & is_row)
    clear = pd.Series(np.isfinite(vals[:, row_px]).mean(1) if row_px.size else np.zeros(len(days)), index=days)
    pixel = pixel_overlay(pix_pct, idx, shape, transform, utm)
    return rows, ts_rows, dist_polys, thumbs, {"days": days, "clear": clear, "n_row": row_px.size, "pixel": pixel}


def pixel_overlay(pct, idx, shape, transform, utm, ground_m=5.0):
    """Class-coloured PNG of pixel % of reference, warped to EPSG:3857 so a Leaflet image
    overlay lines up exactly. Colours are the exact class colours, so the map can swap
    them for the colour-blind set in the browser."""
    from pyproj import Transformer
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    if not np.isfinite(pct).any():
        return None
    H, W = shape
    grid = np.full(H * W, np.nan, "float32")
    grid[idx] = pct
    grid = grid.reshape(H, W)
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (W, H)
    lat = Transformer.from_crs(utm, 4326, always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)[1]
    res = ground_m / np.cos(np.radians(lat))
    dt, dw, dh = calculate_default_transform(utm, "EPSG:3857", W, H, left=min(x0, x1), bottom=min(y0, y1),
                                             right=max(x0, x1), top=max(y0, y1), resolution=res)
    out = np.full((dh, dw), np.nan, "float32")
    reproject(grid, out, src_transform=transform, src_crs=utm, dst_transform=dt, dst_crs="EPSG:3857",
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    rows_any = np.flatnonzero(np.isfinite(out).any(1))
    cols_any = np.flatnonzero(np.isfinite(out).any(0))
    if not rows_any.size:
        return None
    r0, r1, c0, c1 = rows_any[0], rows_any[-1] + 1, cols_any[0], cols_any[-1] + 1
    out = out[r0:r1, c0:c1]
    rgba = np.zeros(out.shape + (4,), "uint8")
    for lo, _, col in CLASSES:
        sel = np.isfinite(out) & (out >= lo) & ~(rgba[..., 3] > 0)
        rgba[sel] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [235]
    mx0, my0 = dt * (c0, r0)
    mx1, my1 = dt * (c1, r1)
    to_ll = Transformer.from_crs("EPSG:3857", 4326, always_xy=True)
    (w_, n_), (e_, s_) = to_ll.transform(mx0, my0), to_ll.transform(mx1, my1)
    return png_rgba_uri(rgba), [[s_, w_], [n_, e_]]


def png_rgba_uri(rgba):
    import base64
    import struct
    import zlib

    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _footprint(item):
    from shapely.geometry import shape as to_shape
    return to_shape(item.geometry)


# --------------------------------------------------------------------------
# Thumbnails (NDVI composites around a section)
# --------------------------------------------------------------------------
# Greenness ramp, brown (bare) -> teal (vigorous); same BrBG scheme as the comparison view
RAMP = np.array([[-0.1, 140, 81, 10], [0.08, 216, 179, 101], [0.26, 246, 232, 195], [0.44, 199, 234, 229],
                 [0.62, 90, 180, 172], [0.8, 1, 102, 94]])


def png_data_uri(rgb):
    import base64
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def section_thumbs(rp, fp, idx, shape, transform, comp, pre, a, row_any, dist_out, pad=14, up=3):
    from scipy import ndimage

    H, W = shape
    allp = np.r_[idx[rp], idx[fp]] if fp.size else idx[rp]
    r_, c_ = np.divmod(allp, W)
    R0, R1 = max(0, r_.min() - pad), min(H, r_.max() + pad + 1)
    C0, C1 = max(0, c_.min() - pad), min(W, c_.max() + pad + 1)
    pos = np.full(H * W, -1)
    pos[idx] = np.arange(idx.size)
    rr, cc = np.mgrid[R0:R1, C0:C1]
    col = pos[(rr * W + cc).ravel()]
    ws_edge = row_any[R0:R1, C0:C1]
    ws_edge = np.kron(ws_edge, np.ones((up, up), bool))
    edge = ndimage.binary_dilation(ws_edge) & ~ws_edge

    def render(v):
        img = np.full(col.size, np.nan, "float32")
        img[col >= 0] = v[col[col >= 0]]
        img = img.reshape(rr.shape)
        rgb = np.stack([np.interp(img, RAMP[:, 0], RAMP[:, k]) for k in (1, 2, 3)], -1)
        rgb[~np.isfinite(img)] = 205
        rgb = np.kron(rgb, np.ones((up, up, 1)))
        rgb[edge] = (255, 0, 255)
        return png_data_uri(rgb.astype("uint8"))

    inrow = row_any[R0:R1, C0:C1]

    def render_pct(v):
        """Pixel detail: ROW pixels coloured by % of this section's reference for the same period
        (exact class colours, so the colour-blind switch can swap them); land outside the ROW in grey."""
        img = np.full(col.size, np.nan, "float32")
        img[col >= 0] = v[col[col >= 0]]
        img = img.reshape(rr.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            refmed = np.nanmedian(v[fp]) if fp.size else np.nan
        grey = np.interp(img, [-0.1, 0.8], [238, 120])
        rgb = np.stack([grey, grey, grey], -1)
        rgb[~np.isfinite(img)] = 215
        if np.isfinite(refmed) and refmed >= 0.25:
            pct = img / refmed * 100
            done = np.zeros(img.shape, bool)
            for lo, _, colr in CLASSES:
                m = inrow & np.isfinite(pct) & (pct >= lo) & ~done
                rgb[m] = [int(colr[k:k + 2], 16) for k in (1, 3, 5)]
                done |= m
        rgb = np.kron(rgb, np.ones((up, up, 1)))
        rgb[edge] = (255, 0, 255)
        return png_data_uri(rgb.astype("uint8"))

    out = [("Before (" + "/".join(map(str, a.pre_years)) + " peak)", render(pre), render_pct(pre))]
    for y in sorted(k for k in comp if isinstance(k, (int, np.integer)) and k >= a.construction_year):
        out.append((f"{y} peak", render(comp[y]), render_pct(comp[y])))
    out.append(("Latest (since " + rw.nice_date(a.current_from, False) + ")", render(comp["current"]), render_pct(comp["current"])))
    return out


# --------------------------------------------------------------------------
# Yards (temporary workspace away from the ROW)
# --------------------------------------------------------------------------
def analyse_yard(name, geom, items, utm, a):
    ring_outer = a.ref_outer
    area = geom.buffer(ring_outer + 40)
    bbox_ll = gpd.GeoSeries([area], crs=utm).to_crs(4326).total_bounds
    near = [i for i in items if box(*bbox_ll).intersects(_footprint(i))]
    if not near:
        return None
    keep = lambda shape, transform: rw.rasterize_mask(area, shape, transform, all_touched=True)
    key = hashlib.md5(json.dumps([np.round(bbox_ll, 5).tolist(), a.start, a.end, a.max_cloud]).encode()).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    days, ndvi, idx, shape, transform = get_chunk_imagery(
        os.path.join(a.data, "cache", f"yard_{safe}.npz"), key, lambda: load_ndvi_quiet(near, utm, bbox_ll, keep, name))
    from scipy import ndimage
    inside = rw.rasterize_mask(geom.buffer(-a.edge_m), shape, transform).ravel()[idx]
    if inside.sum() < 5:
        inside = rw.rasterize_mask(geom, shape, transform).ravel()[idx]
    d_out = (ndimage.distance_transform_edt(~rw.rasterize_mask(geom, shape, transform)) * 10).ravel()[idx]
    wc = load_static("esa-worldcover", "map", shape, transform, utm)
    grp = np.vectorize(lambda v: LC_GROUP.get(int(v), "open"))(wc).ravel()[idx] if wc is not None else np.full(idx.size, "open")
    lc = pd.Series(grp[inside]).value_counts().index[0]
    ring = (d_out >= a.ref_inner) & (d_out <= ring_outer)
    ref = ring & np.isin(grp, REF_FOR.get(lc, ["open"])[:1])
    if ref.sum() < a.min_ref_px:
        ref = ring & ~np.isin(grp, ["built", "water"])
    vals = ndvi.astype("float32")
    vals[ndvi == NODATA] = np.nan
    vals /= 10000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rmed, fmed = np.nanmedian(vals[:, inside], 1), np.nanmedian(vals[:, ref], 1)
        rcl, fcl = np.isfinite(vals[:, inside]).mean(1), np.isfinite(vals[:, ref]).mean(1)
    good = (rcl >= a.min_valid) & (fcl >= a.min_valid) & (fmed >= 0.25)
    ratio = np.where(good, rmed / fmed, np.nan)
    out = {"name": name, "land_before": lc, "acres": round(geom.area / 4046.86, 1)}
    for y in sorted(set(days.year)):
        sel = good & in_peak(days) & (days.year == y)
        out[f"peak_{y}"] = float(np.nanmedian(ratio[sel]) * 100) if sel.any() else np.nan
    sel = good & (days >= pd.Timestamp(a.current_from))
    out["current"] = float(np.nanmedian(ratio[sel]) * 100) if sel.any() else np.nan
    out["_clear"] = pd.Series(rcl, index=days)                     # for the before/after date lists
    out["_bbox"] = gpd.GeoSeries([geom.buffer(400)], crs=utm).to_crs(4326).total_bounds
    return out


# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------
def pc_tile_url(body, render):
    """Register a Planetary Computer mosaic search and return an XYZ tile URL."""
    import requests

    api = "https://planetarycomputer.microsoft.com/api/data/v1"
    try:
        sid = requests.post(f"{api}/mosaic/register", json=body, timeout=60).json()["searchid"]
        tj = requests.get(f"{api}/mosaic/{sid}/tilejson.json", params=render, timeout=60).json()
        return tj["tiles"][0]
    except Exception as e:  # map still works without these layers
        print(f"  (imagery layer unavailable: {e})")
        return None


def year_bars_svg(r, years, pre_years, constr, w=300, h=96):
    """% of reference at peak season, one bar per year; dashed lines at 70% and 100%."""
    vals = [(y, r.get(f"peak_{y}", np.nan)) for y in years] + [("Oct", r.get("current", np.nan))]
    top = 130
    bw = (w - 40) / len(vals)
    y_ = lambda v: 8 + (top - min(max(v, 0), top)) / top * (h - 30)
    out = [f"<svg width='{w}' height='{h}' style='font:10px sans-serif' role='img' aria-label='% of reference by year'>",
           f"<line x1='30' x2='{w - 4}' y1='{y_(100):.1f}' y2='{y_(100):.1f}' stroke='#888' stroke-dasharray='3,2'/>",
           f"<text x='26' y='{y_(100) + 3:.1f}' text-anchor='end' fill='#555'>100%</text>",
           f"<line x1='30' x2='{w - 4}' y1='{y_(70):.1f}' y2='{y_(70):.1f}' stroke='#d73027' stroke-dasharray='2,3'/>",
           f"<text x='26' y='{y_(70) + 3:.1f}' text-anchor='end' fill='#d73027'>70%</text>"]
    for k, (y, v) in enumerate(vals):
        x0 = 34 + k * bw
        tag = "pre" if y in pre_years else "build" if y == constr else ("now" if y == "Oct" else "after")
        col = {"pre": "#9e9e9e", "build": "#c9a26b", "after": classify(v)[1], "now": classify(v)[1]}[tag]
        if np.isfinite(v):
            out.append(f"<rect x='{x0:.1f}' y='{y_(v):.1f}' width='{bw - 6:.1f}' height='{h - 22 - y_(v):.1f}' fill='{col}'>"
                       f"<title>{y}: {v:.0f}% of reference</title></rect>"
                       f"<text x='{x0 + (bw - 6) / 2:.1f}' y='{y_(v) - 2:.1f}' text-anchor='middle' fill='#333'>{v:.0f}</text>")
        else:
            out.append(f"<text x='{x0 + (bw - 6) / 2:.1f}' y='{h - 26}' text-anchor='middle' fill='#aaa'>–</text>")
        lab = f"{y}" if y != "Oct" else "Oct 26"
        sub = {"pre": "before", "build": "construction", "after": "", "now": "latest"}[tag]
        out.append(f"<text x='{x0 + (bw - 6) / 2:.1f}' y='{h - 11}' text-anchor='middle' fill='#333'>{lab}</text>"
                   f"<text x='{x0 + (bw - 6) / 2:.1f}' y='{h - 1}' text-anchor='middle' fill='#888'>{sub}</text>")
    out.append("</svg>")
    return "".join(out)


def curve_svg(ts, r, years, pre_years, constr, end, w=330, h=128):
    """% of reference on every clear date, with each peak season's median as a bar
    across the season, the construction year shaded, and 100% / 70% guides."""
    t0, t1 = pd.Timestamp(f"{min(years)}-04-01"), pd.Timestamp(end)
    span = max((t1 - t0).days, 1)
    x = lambda d: 30 + (pd.Timestamp(d) - t0).days / span * (w - 36)
    top = 140
    y = lambda v: 6 + (top - min(max(v, 0), top)) / top * (h - 24)
    out = [f"<svg width='{w}' height='{h}' style='font:10px sans-serif;display:block' role='img' "
           f"aria-label='% of reference over time'>",
           f"<rect x='{x(f'{constr}-01-01'):.1f}' y='6' width='{x(f'{constr}-12-31') - x(f'{constr}-01-01'):.1f}' "
           f"height='{h - 24}' fill='#f3e3c3'><title>{constr}: construction</title></rect>",
           f"<text x='{(x(f'{constr}-01-01') + x(f'{constr}-12-31')) / 2:.1f}' y='16' text-anchor='middle' fill='#9a6a2a'>construction</text>"]
    for v, colr, dash in ((100, "#888", "3,2"), (70, "#d7191c", "2,3")):
        out.append(f"<line x1='30' x2='{w - 6}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{colr}' stroke-dasharray='{dash}'/>"
                   f"<text x='26' y='{y(v) + 3:.1f}' text-anchor='end' fill='{colr}'>{v}%</text>")
    for d, v in ts:
        out.append(f"<circle cx='{x(d):.1f}' cy='{y(v):.1f}' r='1.4' fill='#4a6fa5' fill-opacity='.55'/>")
    for yr in years:
        v = r.get(f"peak_{yr}")
        if v is None or not np.isfinite(v):
            continue
        a, b = x(f"{yr}-06-15"), x(f"{yr}-09-15")
        colr = "#7a7a7a" if yr in pre_years else "#c9a26b" if yr == constr else classify(v)[1]
        out.append(f"<line x1='{a:.1f}' x2='{b:.1f}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{colr}' stroke-width='4'>"
                   f"<title>{yr} peak season: {v:.0f}%</title></line>"
                   f"<text x='{(a + b) / 2:.1f}' y='{y(v) - 5:.1f}' text-anchor='middle' fill='#222' font-weight='600'>{v:.0f}</text>")
    for yr in years:
        out.append(f"<text x='{x(f'{yr}-07-01'):.1f}' y='{h - 6}' text-anchor='middle' fill='#555'>{yr}</text>")
    out.append("</svg><div style='font-size:10px;color:#666'>Dots: each clear image. Bars: peak-season median "
               "(grey = before, tan = construction). Cropland swings with planting and harvest.</div>")
    return "".join(out)


def timeline_svg(ts, r, years, pre_years, constr, end):
    """Large, readable '% of reference over time' chart for the section details panel."""
    W, H, L, R, T, B = 760, 300, 46, 150, 34, 34
    t0, t1 = pd.Timestamp(f"{min(years)}-01-01"), pd.Timestamp(end) + pd.Timedelta(days=10)
    span = max((t1 - t0).days, 1)
    x = lambda d: L + (pd.Timestamp(d) - t0).days / span * (W - L - R)
    top = 140
    y = lambda v: T + (top - min(max(v, 0), top)) / top * (H - T - B)
    o = [f"<svg class='d-chart' viewBox='0 0 {W} {H}' role='img' aria-label='Percent of reference over time'>"]
    # peak-season windows and construction year
    for yr in years:
        a, b = x(f"{yr}-06-15"), x(f"{yr}-09-15")
        o.append(f"<rect x='{a:.1f}' y='{T}' width='{b - a:.1f}' height='{H - T - B}' fill='#eaf2fb'/>")
    ca, cb = x(f"{constr}-01-01"), x(f"{constr}-12-31")
    o.append(f"<rect x='{ca:.1f}' y='{T}' width='{cb - ca:.1f}' height='{H - T - B}' fill='#f3e3c3' fill-opacity='.75'/>"
             f"<text x='{(ca + cb) / 2:.1f}' y='{T + 14}' text-anchor='middle' class='c-con'>Construction {constr}</text>")
    # grid and axes
    for v in range(0, top + 1, 25):
        o.append(f"<line x1='{L}' x2='{W - R}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='#e3e3e3'/>"
                 f"<text x='{L - 6}' y='{y(v) + 4:.1f}' text-anchor='end' class='c-ax'>{v}%</text>")
    for yr in years + [max(years) + 1]:
        xx = x(f"{yr}-01-01")
        if L <= xx <= W - R:
            o.append(f"<line x1='{xx:.1f}' x2='{xx:.1f}' y1='{T}' y2='{H - B + 4}' stroke='#bbb'/>")
    for yr in years:
        o.append(f"<text x='{x(f'{yr}-07-01'):.1f}' y='{H - 12}' text-anchor='middle' class='c-yr'>{yr}</text>")
    # guides
    o.append(f"<line x1='{L}' x2='{W - R}' y1='{y(100):.1f}' y2='{y(100):.1f}' stroke='#555' stroke-dasharray='5,3' stroke-width='1.5'/>"
             f"<text x='{W - R + 6}' y='{y(100) + 4:.1f}' class='c-g'>100% = like nearby land</text>"
             f"<line x1='{L}' x2='{W - R}' y1='{y(70):.1f}' y2='{y(70):.1f}' stroke='#d7191c' stroke-dasharray='5,3' stroke-width='1.5'/>"
             f"<text x='{W - R + 6}' y='{y(70) + 4:.1f}' class='c-g' fill='#d7191c'>70% target</text>")
    # dots for every clear image
    def period_col(d):
        yr = pd.Timestamp(d).year
        return "#8a8a8a" if yr in pre_years else "#b8894d" if yr == constr else "#2b6cb0"
    for d, v in ts:
        o.append(f"<circle cx='{x(d):.1f}' cy='{y(v):.1f}' r='3' fill='{period_col(d)}' fill-opacity='.45'>"
                 f"<title>{rw.nice_date(d)}: {v:.0f}%</title></circle>")
    # monthly median line (Apr–Oct), one piece per year
    if ts:
        df = pd.DataFrame(ts, columns=["d", "v"])
        df = df[df.d.dt.month.between(4, 10)]
        med = df.groupby(df.d.dt.to_period("M"))["v"].median()
        by_year = {}
        for per, v in med.items():
            by_year.setdefault(per.year, []).append((per.to_timestamp() + pd.Timedelta(days=14), v))
        for pts in by_year.values():
            o.append("<polyline points='" + " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in pts) +
                     "' fill='none' stroke='#1f3b63' stroke-width='2.2' stroke-linejoin='round'/>")
    # peak-season medians, labelled
    for yr in years:
        v = r.get(f"peak_{yr}")
        if v is None or not np.isfinite(v):
            continue
        a, b = x(f"{yr}-06-15"), x(f"{yr}-09-15")
        col = "#6f6f6f" if yr in pre_years else "#a8783c" if yr == constr else classify(v)[1]
        lab = f"{v:.0f}%"
        ly = y(v) - 12 if y(v) - 12 > T + 20 else y(v) + 22
        o.append(f"<line x1='{a:.1f}' x2='{b:.1f}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{col}' stroke-width='7' stroke-linecap='round'>"
                 f"<title>{yr} peak season median: {v:.0f}%</title></line>"
                 f"<rect x='{(a + b) / 2 - 22:.1f}' y='{ly - 13:.1f}' width='44' height='18' rx='4' fill='#fff' stroke='{col}'/>"
                 f"<text x='{(a + b) / 2:.1f}' y='{ly:.1f}' text-anchor='middle' class='c-val' fill='{col}'>{lab}</text>")
    # legend
    o.append(f"<g class='c-leg' transform='translate({L},14)'>"
             "<circle cx='4' cy='-4' r='3.5' fill='#2b6cb0' fill-opacity='.5'/><text x='12' y='0'>one clear image</text>"
             "<line x1='112' x2='134' y1='-4' y2='-4' stroke='#1f3b63' stroke-width='2.2'/><text x='140' y='0'>monthly median</text>"
             "<line x1='248' x2='270' y1='-4' y2='-4' stroke='#6f6f6f' stroke-width='7' stroke-linecap='round'/><text x='278' y='0'>peak-season median (Jun 15–Sep 15, shaded blue)</text></g>")
    o.append("</svg>")
    return "".join(o)


def segment_curves_svg(curves, constr, w=300, h=86):
    """Project view: monthly median % of reference per segment (May–Sep)."""
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    pts = [d for c in curves.values() for d, _ in c]
    if not pts:
        return ""
    t0, t1 = min(pts), max(pts)
    span = max((t1 - t0).days, 1)
    x = lambda d: 28 + (pd.Timestamp(d) - t0).days / span * (w - 32)
    y = lambda v: 4 + (130 - min(max(v, 0), 130)) / 130 * (h - 20)
    out = [f"<svg viewBox='0 0 {w} {h}' style='width:100%;height:auto' role='img' aria-label='Recovery by segment over time'>",
           f"<rect x='{x(f'{constr}-01-01'):.1f}' y='4' width='{x(f'{constr}-12-31') - x(f'{constr}-01-01'):.1f}' height='{h - 20}' fill='#f3e3c3'/>"]
    for v, colr, dash in ((100, "#888", "3,2"), (70, "#d7191c", "2,3")):
        out.append(f"<line x1='28' x2='{w - 4}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{colr}' stroke-dasharray='{dash}'/>"
                   f"<text x='24' y='{y(v) + 3:.1f}' text-anchor='end' fill='{colr}'>{v}</text>")
    key = []
    for k, (seg, c) in enumerate(sorted(curves.items())):
        col = colors[k % len(colors)]
        by_year = {}
        for d, v in c:
            by_year.setdefault(d.year, []).append((d, v))
        for yr_pts in by_year.values():
            pl = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in yr_pts)
            out.append(f"<polyline points='{pl}' fill='none' stroke='{col}' stroke-width='2'/>")
        key.append(f"<span><i style='background:{col}'></i>{seg}</span>")
    for yr in sorted({d.year for d in pts}):
        out.append(f"<text x='{x(f'{yr}-07-01'):.1f}' y='{h - 4}' text-anchor='middle'>{yr}</text>")
    out.append("</svg>")
    return "".join(out) + "<div class='key'>" + "".join(key) + "<span class='unit'>median % of reference, May–Sep</span></div>"


def cmp_button(seg, geom, label="&#8644; Compare before / after here", cls="cmp-here"):
    """Button that opens the before/after slider on this feature (geometry in EPSG:4326)."""
    import html as htmlmod

    b = geom.buffer(0.002).bounds
    return (f"<button class='{cls}' data-seg='{htmlmod.escape(seg)}' data-b='{b[1]:.5f},{b[0]:.5f},{b[3]:.5f},{b[2]:.5f}' "
            f"title='Compare before / after imagery here' aria-label='Compare before / after imagery here'>{label}</button>")


def write_map(sec, ws, dist, yards, info, path):
    import html as htmlmod
    import folium

    esc = htmlmod.escape
    S = sec.to_crs(4326)
    c = S.geometry.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=9, tiles=None, control_scale=True, zoom_control=False)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Esri imagery (date varies)").add_to(m)
    if info.get("naip_tiles"):
        folium.TileLayer(info["naip_tiles"], attr="USDA NAIP via Microsoft Planetary Computer",
                         name="NAIP 2022–23 aerial (pre-construction, 0.6 m)", show=False, max_zoom=20).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", show=False).add_to(m)
    if info.get("s2_tiles"):  # overlay, so the Esri basemap shows where/when it doesn't render (zoomed out)
        folium.TileLayer(info["s2_tiles"], attr="Sentinel-2 (Copernicus) via Microsoft Planetary Computer",
                         name=f"Sentinel-2 · latest clear since {rw.nice_date(info['current_from'], False)} (10 m)",
                         overlay=True, show=True, min_zoom=11, max_native_zoom=16, max_zoom=19).add_to(m)

    W = ws.to_crs(4326)
    folium.GeoJson(W[W.kind == "row"].dissolve().__geo_interface__, name="Approved workspace (ROW)",
                   style_function=lambda _: {"color": "#ffffff", "weight": 1, "fillOpacity": 0.05, "dashArray": "3,3"}).add_to(m)

    years, pre_years, constr = info["years"], info["pre_years"], info["construction_year"]
    last = max(years)
    reveg = folium.FeatureGroup(name="Revegetation (% of reference, peak 2026)").add_to(m)
    risk_layer = folium.FeatureGroup(name="Erosion-risk screen", show=False).add_to(m)
    layer_vars, risk_vars, sec_lines, foot_vars, details, sec_meta = [], [], [], [], {}, {}
    for i, r in S.iterrows():
        cls, col = classify(r.get(f"peak_{last}"))
        thumbs = "".join(f"<figure><img data-green='{g}' data-pct='{pc}' alt='{esc(lab)}'>"  # src set by the map type switch
                         f"<figcaption>{esc(lab)}</figcaption></figure>"
                         for lab, g, pc in (r["thumbs"] or []))
        fmt = lambda v, f="{:.0f}%": "–" if v is None or not np.isfinite(v) else f.format(v)
        trend = r.get("trend_30d")
        trend_txt = ("–" if not np.isfinite(trend) else
                     f"{'improving' if trend > 2 else 'declining' if trend < -2 else 'steady'} ({trend:+.0f} pts / month)")
        land = {"crop": "cropland", "open": "grass / open", "wetland": "wetland", "forest": "forest (cleared)",
                "built": "developed", "water": "water"}.get(r.get("land_before"), r.get("land_before") or "–")
        ref = r.get("ref_land") or "–"
        ref_txt = {"crop": "nearby cropland", "open": "nearby grass / open land", "wetland": "nearby wetland",
                   "mixed": "mixed nearby land (no close match)"}.get(ref, ref)
        note = ""
        if r.get("differed_before"):
            note = (f"<div class='d-note'>This section was already {fmt(r.get('pre'))} of its reference before construction "
                    f"(different soil, drainage or use), so judge it against that, not against 100%.</div>")
        if r.get("land_before") == "built":
            note = "<div class='d-note'>Built-up land before construction (road, pad or station): not scored.</div>"
        crop_note = "not shown in cropland (harvest timing)"
        facts = [("Before construction", land), ("Compared with", ref_txt),
                 ("vs. before construction", fmt(r.get("vs_pre"))),
                 (f"{last} trend", trend_txt if r.get("land_before") != "crop" else crop_note),
                 ("Latest (Oct)", fmt(r.get("current")) if r.get("land_before") != "crop" else crop_note),
                 ("Erosion screen", f"<b style='color:{RISK_COLORS[r['risk']]}'>{r['risk']}</b> &middot; bare ({r['bare_basis']}) "
                                    f"{fmt(r['bare_row'] * 100)} of ROW vs {fmt(r['bare_nearby'] * 100)} nearby &middot; "
                                    f"steepest 10% {fmt(r.get('slope_p90'), '{:.1f}°')}"
                                    f"{' &middot; water/wetland nearby' if r.get('near_water') else ''}")]
        bb = r.geometry.buffer(0.002).bounds
        detail = (f"<div class='d-top'><div class='d-score'><span class='d-big' style='color:{col}'>{fmt(r.get(f'peak_{last}'))}</span>"
                  f"<span class='d-cls' style='background:{col}'>{cls}</span>"
                  f"<div class='d-sub'>peak {last} greenness, as % of comparable nearby land</div></div>"
                  f"<dl class='d-facts'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl></div>"
                  f"{note}"
                  f"<div class='d-actions'><button class='cmp-here' data-seg='{esc(r['segment'])}' "
                  f"data-b='{bb[1]:.5f},{bb[0]:.5f},{bb[3]:.5f},{bb[2]:.5f}'>&#8644; Compare before / after imagery here</button></div>"
                  f"<h3>Greenness compared with nearby land, every clear image since {min(years)}</h3>"
                  f"{timeline_svg(r['ts'] or [], r, years, pre_years, constr, info['end'])}"
                  + ("<p class='d-cap'>Cropland swings with planting and harvest; read the peak-season bars, not single images.</p>"
                     if r.get("land_before") == "crop" else "")
                  + f"<div class='d-th-head'><h3>Maps of this section</h3>"
                  f"<span class='d-mode' role='group' aria-label='Map type'>"
                  f"<button data-mode='green' aria-pressed='true'>Greenness</button>"
                  f"<button data-mode='pct' aria-pressed='false'>% of reference (pixel detail)</button></span></div>"
                  f"<div class='d-thumbs'>{thumbs}</div>"
                  f"<p class='d-cap' data-cap='green'>Brown = bare, teal = vigorous; magenta = approved workspace edge. Click an image to enlarge.</p>"
                  f"<p class='d-cap' data-cap='pct' hidden>Each 10 m pixel inside the workspace coloured by its greenness as % of this "
                  f"section's reference land for the same period: <span class='d-swatches'>"
                  + "".join(f"<i style='background:{c}'></i>{n} " for _, n, c in CLASSES)
                  + f"</span>. Grey = outside the workspace. Edge pixels mix in neighbouring land.</p>")
        details[str(i)] = {"title": r["section_id"], "html": detail, "seg": r["segment"]}
        geo = r.geometry.__geo_interface__
        vis = folium.GeoJson(geo, style_function=lambda _, col=col: {"color": col, "weight": 7, "opacity": 0.95})
        vis.add_to(reveg)
        sec_lines.append(f"[{vis.get_name()}, '{i}']")
        sec_meta[str(i)] = {"land": r.get("land_before") or "other",
                            "cls": cls if cls in [c[1] for c in CLASSES] else "other",
                            "mi": round(float(r["mp_to"] - r["mp_from"]), 2), "seg": r["segment"]}
        gj = folium.GeoJson(geo, style_function=lambda _: {"color": "#000", "weight": 26, "opacity": 0.01},
                            tooltip=f"{r['section_id']}: {fmt(r.get(f'peak_{last}'))} of reference ({cls}) · click for details")
        gj.add_to(reveg)
        layer_vars.append(f"'{i}': {gj.get_name()}")
        if r.get("foot") is not None and not r["foot"].is_empty:
            fp = folium.GeoJson(r["foot"].__geo_interface__,
                                style_function=lambda _, col=col: {"color": col, "weight": 1, "opacity": 0.9,
                                                                   "fillColor": col, "fillOpacity": 0.5},
                                tooltip=f"{r['section_id']}: {fmt(r.get(f'peak_{last}'))} of reference ({cls})")
            fp.add_to(reveg)
            foot_vars.append(f"[{fp.get_name()}, '{i}']")
        rcol = RISK_COLORS[r["risk"]]
        folium.GeoJson(geo, style_function=lambda _, c=rcol: {"color": c, "weight": 7, "opacity": 0.95},
                       tooltip=f"{r['section_id']}: erosion screen {r['risk']}").add_to(risk_layer)

    pix_vars = []
    if info.get("pixel_layers"):
        pix_group = folium.FeatureGroup(name="Pixel-level recovery (10 m, peak 2026)", show=False).add_to(m)
        for uri, bounds in info["pixel_layers"]:
            ov = folium.raster_layers.ImageOverlay(image=uri, bounds=bounds, opacity=0.9, interactive=False, zindex=420)
            ov.add_to(pix_group)
            pix_vars.append(ov.get_name())
        pix_group_var = pix_group.get_name()
    else:
        pix_group_var = "null"

    if len(dist):
        D = dist.to_crs(4326)
        dl = folium.FeatureGroup(name="Bare ground outside approved workspace").add_to(m)
        for j, d in D.iterrows():
            gj = folium.GeoJson(d.geometry.__geo_interface__,
                                style_function=lambda _: {"color": "#ff00ff", "weight": 2, "fillOpacity": 0.25},
                                tooltip=f"Outside workspace: {d['area_m2']:,} m², bare in {d['years_bare']}",
                                popup=folium.Popup(
                                    f"<b>Bare ground outside the approved workspace</b><br>{d['area_m2']:,} m² "
                                    f"(~{d['area_m2'] / 4046.86:.2f} ac), about {d['dist_m']:.0f} m outside, near {esc(d['near'])}<br>"
                                    f"Greenness before construction {d['ndvi_pre']:.2f} → after {d['ndvi_after']:.2f}; bare at peak season in {d['years_bare']}.<br>"
                                    f"<span style='color:#666'>Could be extra workspace, an access route, or unrelated work (another project, "
                                    f"a new building, a drained pond). Check against variances and the NAIP layer.</span><br>"
                                    + cmp_button(d["segment"], d.geometry), max_width=330))
            gj.add_to(dl)
            layer_vars.append(f"'d{j}': {gj.get_name()}")

    Y = ws.to_crs(4326)
    yard_layer = folium.FeatureGroup(name="Yards and facilities").add_to(m)
    for j, y in Y[Y.kind != "row"].iterrows():
        yr = yards.get(y["name"]) if y.kind == "yard" else None
        if y.kind == "facility":
            style, tip = {"color": "#555", "weight": 1.5, "fillOpacity": 0.1, "dashArray": "4,3"}, f"{y['name']} (permanent facility, not scored)"
            pop = f"<b>{esc(y['name'])}</b><br>Permanent facility: not expected to revegetate; not scored."
        else:
            cls, colr = classify(yr.get(f"peak_{last}") if yr else np.nan)
            style, tip = {"color": colr, "weight": 2.5, "fillOpacity": 0.25}, f"{y['name']}: {cls}"
            pop = (f"<b>{esc(y['name'])}</b> (temporary yard, {yr['acres'] if yr else '–'} ac)<br>"
                   f"Before construction: {yr['land_before'] if yr else '–'}<br>"
                   + (year_bars_svg(yr, years, pre_years, constr) if yr else "No imagery.")
                   + (cmp_button(f"Yard: {y['name']}", y.geometry) if f"Yard: {y['name']}" in info["compare"]["segments"] else ""))
        gj = folium.GeoJson(y.geometry.__geo_interface__, style_function=lambda _, s=style: s, tooltip=tip,
                            popup=folium.Popup(pop, max_width=340))
        gj.add_to(yard_layer)
        layer_vars.append(f"'y{j}': {gj.get_name()}")
    folium.LayerControl(collapsed=True).add_to(m)

    root = m.get_root()
    raw = lambda x: folium.Element("{% raw %}" + x + "{% endraw %}")
    root.header.add_child(raw(PANEL_CSS + COMPARE_CSS + DRAWER_CSS))
    root.html.add_child(raw(panel_html(S, dist, yards, Y, info) + COMPARE_HTML + DRAWER_HTML))
    root.html.add_child(raw("<script id='rv-details' type='application/json'>"
                            + json.dumps(details).replace("</", "<\\/") + "</script>"))
    b = S.total_bounds
    root.script.add_child(raw(PANEL_JS.replace("__MAP__", m.get_name()).replace("__LAYERS__", ", ".join(layer_vars))
                              .replace("__SECLINES__", ", ".join(sec_lines)).replace("__FOOT__", ", ".join(foot_vars))
                              .replace("__MPS__", json.dumps(info["mileposts"]))
                              .replace("__SECMETA__", json.dumps(sec_meta)).replace("__CB__", json.dumps(CB_COLORS))
                              .replace("__PIX__", ", ".join(pix_vars)).replace("__PIXGROUP__", pix_group_var)
                              .replace("__BOUNDS__", json.dumps([[b[1], b[0]], [b[3], b[2]]]))))
    root.script.add_child(raw(COMPARE_JS.replace("__MAP__", m.get_name()).replace("__CMP__", json.dumps(info["compare"]))
                              .replace("__SECLINES__", ", ".join(sec_lines))))
    m.save(path)


def strip_svg(g, last, w=300):
    """One bar per segment from first to last milepost, coloured by section class."""
    lo, hi = g["mp_from"].min(), g["mp_to"].max()
    x = lambda v: 1 + (v - lo) / (hi - lo) * (w - 2)
    out = [f"<svg class='rv-strip' width='{w}' height='30' viewBox='0 0 {w} 30' role='img'>"]
    for i, r in g.iterrows():
        cls, col = classify(r.get(f"peak_{last}"))
        out.append(f"<rect class='rv-sec' data-id='{i}' x='{x(r.mp_from):.1f}' y='2' width='{max(1.5, x(r.mp_to) - x(r.mp_from) + 0.3):.1f}' "
                   f"height='14' fill='{col}'><title>{r['section_id']}: {cls}</title></rect>")
    step = next(s for s in (1, 2, 5, 10) if (hi - lo) / s <= 7)
    for v in np.arange(np.ceil(lo / step) * step, hi + 1e-6, step):
        out.append(f"<text x='{x(v):.1f}' y='27' text-anchor='middle'>{v:g}</text>")
    out.append("</svg>")
    return "".join(out)


def ladder_html(S, dist, last):
    """Vertical milepost ladder: one column per segment, one cell per section, coloured by status,
    with erosion (▲) and outside-workspace (●) flags beside the cell. Cells grow with the panel."""
    import html as htmlmod
    import re

    out_mp = {}
    for v in (dist["near"] if len(dist) else []):
        m = re.match(r"(\S+) MP ([\d.]+)", str(v))
        if m:
            out_mp.setdefault(m.group(1), []).append(float(m.group(2)))
    cols = []
    for seg, g in S.groupby("segment"):
        g = g.sort_values("mp_from")
        ok = sum(r.mp_to - r.mp_from for _, r in g.iterrows() if classify(r[f"peak_{last}"])[0] in ("similar", "establishing"))
        tot = sum(r.mp_to - r.mp_from for _, r in g.iterrows() if np.isfinite(r[f"peak_{last}"]))
        rows = []
        for i, r in g.iterrows():
            v = r[f"peak_{last}"]
            cls, col = classify(v)
            if r.get("land_before") == "built":
                cls = "not scored"
            flags = ("<span class='f-ero' title='Erosion flag'>&#9650;</span>" if r["risk"] in ("High", "Medium") else "") + \
                    ("<span class='f-out' title='Bare ground outside workspace'>&#9679;</span>"
                     if any(r.mp_from <= x < r.mp_to for x in out_mp.get(seg, [])) else "")
            label = f"{r.mp_from:g}" if abs(r.mp_from - round(r.mp_from)) < 1e-6 else ""
            tip = f"{r['section_id']}: {'' if not np.isfinite(v) else f'{v:.0f}% · '}{cls}"
            rows.append(f"<div class='lad-r'><span class='lad-mp'>{label}</span>"
                        f"<div class='rv-sec lad-c' data-id='{i}' role='button' tabindex='0' style='background:{col}' "
                        f"title='{htmlmod.escape(tip)}' aria-label='{htmlmod.escape(tip)}'>"
                        f"<span class='pct'>{'' if not np.isfinite(v) else f'{v:.0f}'}</span></div>"
                        f"<span class='lad-f'>{flags}</span></div>")
        cols.append(f"<div class='lad-col'><div class='lad-h'><b>{seg}</b>"
                    f"<span>MP {g.mp_from.min():g}–{g.mp_to.max():g}</span>"
                    f"<span class='lad-sum'>{ok:.1f} of {tot:.1f} mi OK</span></div>{''.join(rows)}</div>")
    return f"<div class='lad'>{''.join(cols)}</div>"


def filters_html(S, info):
    """Chips to filter sections by land use before construction and by status."""
    last = max(info["years"])
    lands = [("crop", "Cropland"), ("open", "Grass / open"), ("forest", "Cleared forest"), ("wetland", "Wetland")]
    present = set(S["land_before"].dropna())
    land_chips = "".join(f"<button data-v='{v}' aria-pressed='true'>{lab} "
                         f"<span class='n'>{int((S['land_before'] == v).sum())}</span></button>"
                         for v, lab in lands if v in present)
    cls_chips = "".join(f"<button data-v='{n}' aria-pressed='true'><i style='background:{c}'></i>{n.capitalize()} "
                        f"<span class='n'>{sum(classify(v)[0] == n for v in S[f'peak_{last}'])}</span></button>"
                        for _, n, c in CLASSES)
    return (f"<div class='filters'><div class='pop-h'>Show sections <button id='flt-reset' type='button'>Show all</button></div>"
            f"<div class='chips' data-group='land' role='group' aria-label='Land use before construction'>{land_chips}</div>"
            f"<div class='chips' data-group='cls' role='group' aria-label='Status'>{cls_chips}</div>"
            f"<div id='flt-sum' aria-live='polite'></div></div>")


def panel_html(S, dist, yards, Y, info):
    import html as htmlmod

    esc = htmlmod.escape
    last = max(info["years"])
    col = f"peak_{last}"
    seclen = (S["mp_to"] - S["mp_from"])
    scored = S[np.isfinite(S[col])]
    miles = {name: float(seclen[[classify(v)[0] == name for v in S[col]]].sum()) for _, name, _ in CLASSES}
    total_mi = float(seclen[np.isfinite(S[col])].sum())
    ok_mi = miles["similar"] + miles["establishing"]
    verdict = (f"<span style='color:{CLASSES[0][2]}'>{ok_mi:.1f} of {total_mi:.1f} mi</span> at 70%+ of reference"
               if total_mi else "No sections could be scored")
    bad = miles["lagging"] + miles["poor"]
    sub = f"<div class='also'>{bad:.1f} mi below 70% &middot; peak {last} season</div>"
    strips = "".join(f"<div class='seg'><b>{seg}</b> <span>MP {g.mp_from.min():.1f}–{g.mp_to.max():.1f}</span>{strip_svg(g, last)}</div>"
                     for seg, g in S.groupby("segment"))
    key = "".join(f"<span><i style='background:{c}'></i>{n} {'≥' if lo > 0 else '<'}{lo if lo > 0 else 50}%</span>"
                  for lo, n, c in CLASSES)
    facts = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in info["facts"])

    def lst(rows, empty):
        return "".join(rows) if rows else f"<p class='empty'>{empty}</p>"

    worst = S[np.isfinite(S[col])].sort_values(col).head(25)
    land_word = {"crop": "cropland", "open": "grass/open", "wetland": "wetland", "forest": "cleared forest"}
    pct = lambda v, f: "–" if not np.isfinite(v) else f.format(v)
    lag_rows = [f"<button class='rv-item' data-id='{i}'><b class='st' style='background:{classify(r[col])[1]}'>{r[col]:.0f}%</b>"
                f"{esc(r['section_id'])}<div class='sub'>{esc(land_word.get(r['land_before'], str(r['land_before'])))}"
                f"{' &middot; <span style=\'color:#b35806\'>was ' + format(r['pre'], '.0f') + '% before</span>' if r['differed_before'] else ''}"
                f" &middot; trend {pct(r['trend_30d'], '{:+.0f}/mo')} &middot; Oct {pct(r['current'], '{:.0f}%')}</div></button>"
                for i, r in worst.iterrows()]
    risk = S[S["risk"].isin(["High", "Medium"])].sort_values("risk_score", ascending=False)
    row = lambda inner, seg, geom: (f"<div class='rv-row'>{inner}"
                                    f"{cmp_button(seg, geom, '&#8644;', 'rv-cmp') if seg in info['compare']['segments'] else ''}</div>")
    risk_rows = [row(f"<button class='rv-item' data-id='{i}'><b class='st' style='background:{RISK_COLORS[r['risk']]}'>{r['risk']}</b>"
                     f"{esc(r['section_id'])}<div class='sub'>bare {r['bare_row'] * 100:.0f}% of ROW vs {r['bare_nearby'] * 100:.0f}% nearby ({r['bare_basis']}) &middot; steep 10% {r['slope_p90']:.1f}°"
                     f"{' &middot; water nearby' if r['near_water'] else ''}{' &middot; cropland' if r['land_before'] == 'crop' else ''}</div></button>",
                     r["segment"], r.geometry)
                 for i, r in risk.iterrows()]
    D = dist.sort_values("area_m2", ascending=False) if len(dist) else dist
    DD = D.to_crs(4326) if len(D) else D
    dist_rows = [row(f"<button class='rv-item' data-id='d{j}'><b class='st' style='background:#c000c0'>{d['area_m2'] / 4046.86:.2f} ac</b>"
                     f"near {esc(d['near'])}<div class='sub'>{d['dist_m']:.0f} m outside &middot; bare in {d['years_bare']}</div></button>",
                     d["segment"], d.geometry)
                 for j, d in DD.iterrows()]
    yard_rows = []
    for j, y in Y[Y.kind != "row"].iterrows():
        if y.kind == "facility":
            continue
        yr = yards.get(y["name"]) or {}
        v = yr.get(col, np.nan)
        cls, c = classify(v)
        yard_rows.append(row(f"<button class='rv-item' data-id='y{j}'><b class='st' style='background:{c}'>"
                             f"{'–' if not np.isfinite(v) else f'{v:.0f}%'}</b>{esc(y['name'])}<div class='sub'>{cls} &middot; "
                             f"before: {yr.get('land_before', '–')}</div></button>", f"Yard: {y['name']}", y.geometry))
    how = ("Greenness (NDVI) of the restored workspace is compared with similar undisturbed land 30–150 m away "
           "(cropland with cropland, grass with grass, wetland with wetland; cleared forest with nearby open land), "
           "using only 10 m pixels fully inside the approved workspace. <b>% of reference</b> is the median over "
           f"clear Sentinel-2 images in the peak season (15 Jun–15 Sep). It is relative greenness, not measured % cover; "
           "field plots are needed to convert it. <b>Bare ground outside workspace</b>: vegetated before construction, "
           "bare at peak season since, 20–200 m outside the approved polygons, at least 400 m², not cropland. "
           "<b>Erosion screen</b>: how much more of the ROW is bare than the nearby reference land (last 45 days; "
           "for cropland, peak season, because October harvest timing differs between the ROW and the field), "
           "the steepest 10% of slopes (USGS 3DEP 10 m), and water or wetland within 60 m. High = 25+ points more bare "
           "with slope 4°+ or water nearby; Medium = 10+ points more bare. In cropland, the October reading and the "
           "season trend follow harvest and crop growth, so they are not shown.")
    n_ero, n_out = len(risk_rows), len(dist_rows)
    key = "".join(f"<span><i style='background:{c}'></i>{n}</span>" for _, n, c in CLASSES)
    return (f"<div id='rv-panel' aria-label='Revegetation summary'>"
            f"<div class='p-resize' id='rv-resize' role='separator' aria-orientation='vertical' aria-label='Resize panel' tabindex='0'></div>"
            f"<header class='p-head'>"
            f"<div class='p-title'><h1>{esc(info['title'])}</h1>"
            f"<button id='rv-close' class='icon-btn' aria-label='Collapse panel' title='Collapse panel'>&#10094;</button></div>"
            f"<div class='p-meta'>Peak {last} season &middot; latest clear image {rw.nice_date(info['latest_clear']) if info.get('latest_clear') else '–'}</div>"
            f"<div class='kpis'>"
            f"<button class='kpi' data-kpi='ok' title='Show sections at 70%+'><b class='ok' style='color:{CLASSES[0][2]}'>{ok_mi:.1f} mi</b><span>at 70%+ of reference</span></button>"
            f"<button class='kpi' data-kpi='bad' title='Show sections below 70%'><b class='bad'>{bad:.1f} mi</b><span>below 70%</span></button>"
            f"<button class='kpi' data-kpi='ero' title='Show erosion list'><b>{n_ero}</b><span>erosion flags</span></button>"
            f"<button class='kpi' data-kpi='out' title='Show outside-workspace list'><b>{n_out}</b><span>outside workspace</span></button></div>"
            f"<div class='toolbar'>"
            f"<button id='cmp-open' class='tb-btn'>&#8644; Compare</button>"
            f"<form class='mpgo' id='mp-form' role='search'><input id='mp-in' inputmode='decimal' placeholder='Go to MP' "
            f"aria-label='Go to milepost' autocomplete='off'><button aria-label='Go'>&#10140;</button></form>"
            f"<button id='flt-btn' class='tb-btn' aria-expanded='false' aria-controls='flt-pop'>Filter<span id='flt-badge' hidden></span></button>"
            f"<button id='set-btn' class='tb-btn icon' aria-expanded='false' aria-controls='set-pop' aria-label='Display settings' title='Display settings'>&#9881;</button></div>"
            f"<div id='mp-msg' aria-live='polite'></div>"
            f"<div id='flt-pop' class='pop' hidden>{filters_html(S, info)}</div>"
            f"<div id='set-pop' class='pop' hidden><label><input type='checkbox' id='pix-toggle'> Pixel detail (10 m) on the map</label>"
            f"<label><input type='checkbox' id='cb-toggle'> Colour-blind-safe colours</label></div>"
            f"<nav class='views' role='tablist'><button role='tab' data-view='overview' aria-selected='true'>Overview</button>"
            f"<button role='tab' data-view='lists' aria-selected='false'>Lists</button></nav></header>"
            f"<div class='p-body'>"
            f"<section data-view='overview'>"
            f"<div class='sec-h'>Revegetation by milepost <span>click a section for details</span></div>"
            f"{ladder_html(S, dist, last)}"
            f"<div class='key'>{key}<span>&#9650; erosion</span><span>&#9679; outside ws</span></div>"
            f"<div class='sec-h'>Recovery over time <span>construction shaded</span></div>"
            f"{segment_curves_svg(info.get('seg_curves', {}), info['construction_year'])}</section>"
            f"<section data-view='lists' hidden>"
            f"<nav class='tabs' role='tablist'>"
            f"<button role='tab' data-tab='lag' aria-selected='true'>Lowest ({len(lag_rows)})</button>"
            f"<button role='tab' data-tab='risk' aria-selected='false'>Erosion ({n_ero})</button>"
            f"<button role='tab' data-tab='out' aria-selected='false'>Outside ws ({n_out})</button>"
            f"<button role='tab' data-tab='yard' aria-selected='false'>Yards ({len(yard_rows)})</button></nav>"
            f"<div id='rv-list'><div data-pane='lag'>{lst(lag_rows, 'No sections scored.')}</div>"
            f"<div data-pane='risk' hidden>{lst(risk_rows, 'No sections flagged.')}</div>"
            f"<div data-pane='out' hidden>{lst(dist_rows, 'Nothing found outside the approved workspace.')}</div>"
            f"<div data-pane='yard' hidden>{lst(yard_rows, 'No temporary yards.')}</div></div></section></div>"
            f"<footer class='p-foot'><details><summary>Data and method</summary><dl class='facts'>{facts}</dl><p>{how}</p></details>"
            f"<span>Screening, not a verdict: field crews or sharper imagery confirm.</span></footer></div>"
            f"<button id='rv-open' aria-label='Expand panel'>&#9776; Revegetation</button>")


PANEL_CSS = """
<style>
#rv-panel{--ink:#1d1f23;--ink2:#5b6270;--ink3:#8a909b;--line:#e6e8ec;--soft:#f5f6f8;--accent:#1f6feb;
  position:fixed;top:10px;left:10px;bottom:10px;width:var(--pw,360px);z-index:1001;background:#fff;border-radius:12px;
  box-shadow:0 1px 2px rgba(0,0,0,.08),0 6px 24px rgba(0,0,0,.14);font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  color:var(--ink);display:flex;flex-direction:column;container-type:inline-size;container-name:panel}
#rv-panel *{box-sizing:border-box}
.p-resize{position:absolute;top:12px;bottom:12px;right:-6px;width:12px;cursor:ew-resize;z-index:2;display:flex;align-items:center;justify-content:center}
.p-resize::after{content:'';width:4px;height:44px;border-radius:2px;background:#c9cdd4;opacity:0;transition:opacity .15s}
#rv-panel:hover .p-resize::after,.p-resize:focus::after,.p-resize.drag::after{opacity:1}
.p-head{padding:14px 16px 0;border-bottom:1px solid var(--line);flex:none}
.p-title{display:flex;align-items:flex-start;gap:8px}
.p-title h1{flex:1;font-size:15px;font-weight:600;margin:0;line-height:1.3}
.icon-btn{border:0;background:none;color:var(--ink3);font-size:14px;cursor:pointer;padding:2px 6px;border-radius:6px}
.icon-btn:hover{background:var(--soft);color:var(--ink)}
.p-meta{color:var(--ink3);font-size:12px;margin:2px 0 10px}
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px}
@container panel (min-width:560px){.kpis{grid-template-columns:repeat(4,1fr)}}
.kpi{text-align:left;border:1px solid var(--line);background:var(--soft);border-radius:10px;padding:8px 10px;cursor:pointer;font:inherit;color:inherit}
.kpi:hover{border-color:#c9cdd4;background:#eef0f3}
.kpi b{display:block;font-size:19px;font-weight:600;line-height:1.2}
.kpi b.bad{color:#c4561d}
.kpi span{font-size:11.5px;color:var(--ink2)}
.toolbar{display:flex;gap:6px;margin:12px 0 4px;align-items:center}
.tb-btn{height:30px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:0 10px;font-size:12.5px;font-weight:500;font-family:inherit;color:var(--ink);cursor:pointer;white-space:nowrap}
.tb-btn:hover{background:var(--soft)}
.tb-btn[aria-expanded=true]{background:#eef4ff;border-color:var(--accent);color:#0b3d91}
.tb-btn.icon{width:32px;padding:0;font-size:15px}
#flt-badge{display:inline-block;min-width:16px;height:16px;margin-left:5px;border-radius:8px;background:var(--accent);color:#fff;font-size:10.5px;line-height:16px;text-align:center;padding:0 4px}
form.mpgo{flex:1;display:flex;min-width:0;height:30px;border:1px solid var(--line);border-radius:8px;overflow:hidden}
form.mpgo input{flex:1;min-width:0;border:0;padding:0 8px;font:inherit;outline:none}
form.mpgo button{border:0;border-left:1px solid var(--line);background:#fff;width:30px;cursor:pointer;color:var(--ink2)}
form.mpgo:focus-within{border-color:var(--accent)}
#mp-msg{color:#b35806;font-size:12px;min-height:0}
.pop{border:1px solid var(--line);border-radius:10px;padding:10px;margin:6px 0 4px;background:#fff;font-size:12.5px}
.pop label{display:block;padding:3px 0;cursor:pointer}
.pop-h{display:flex;justify-content:space-between;font-weight:600;margin-bottom:4px}
.pop-h button{border:0;background:none;color:var(--accent);font:inherit;font-weight:400;cursor:pointer;padding:0}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}
.chips button{border:1px solid var(--line);background:#fff;border-radius:14px;padding:3px 10px;font-size:12px;font-family:inherit;cursor:pointer;color:var(--ink3)}
.chips button[aria-pressed=true]{background:#eef4ff;border-color:#9dbcf5;color:var(--ink)}
.chips button i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.chips .n{color:var(--ink3);font-size:11px}
#flt-sum{font-size:12px;color:var(--ink2);margin-top:4px}
nav.views{display:flex;gap:18px;margin-top:10px}
nav.views button{border:0;background:none;padding:8px 0;font-size:13px;font-weight:600;font-family:inherit;color:var(--ink3);cursor:pointer;border-bottom:2px solid transparent}
nav.views button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--ink)}
.p-body{flex:1;overflow-y:auto;min-height:0}
.p-body section{padding:4px 16px 14px}
.p-body section[data-view=lists]{padding:0}
.sec-h{font-size:12.5px;font-weight:600;margin:12px 0 6px;color:var(--ink)}
.sec-h span{font-weight:400;color:var(--ink3);margin-left:6px}
.lad{display:flex;gap:12px;align-items:flex-start}
.lad-col{flex:1;min-width:0}
.lad-h{position:sticky;top:0;background:#fff;padding:2px 0 6px;z-index:1}
.lad-h b{display:block;font-size:13px}
.lad-h span{display:block;font-size:11px;color:var(--ink3);line-height:1.3}
.lad-r{display:grid;grid-template-columns:22px 1fr 16px;align-items:center;height:12px;column-gap:4px}
.lad-mp{font-size:10.5px;color:var(--ink3);text-align:right;line-height:1}
.lad-c{height:10.5px;border-radius:2px;cursor:pointer;position:relative;outline-offset:1px}
.lad-c:hover{box-shadow:0 0 0 2px rgba(29,31,35,.55);z-index:1}
.lad-c:focus-visible{outline:2px solid var(--accent)}
.lad-c.sel{box-shadow:0 0 0 2px var(--ink);z-index:2}
.lad-c .pct{display:none;position:absolute;right:5px;top:50%;transform:translateY(-50%);font-size:10.5px;font-weight:600;color:#fff;text-shadow:0 0 2px rgba(0,0,0,.65)}
.lad-f{font-size:9px;line-height:1;white-space:nowrap;letter-spacing:1px}
.f-ero{color:#b2182b}.f-out{color:#c000c0}
@container panel (min-width:470px){.lad-r{height:17px;grid-template-columns:30px 1fr 22px}.lad-c{height:15px}.lad-c .pct{display:block}
  .lad-mp{font-size:11.5px}.lad-f{font-size:11px}}
@container panel (min-width:640px){.lad-r{height:21px}.lad-c{height:19px}.lad-c .pct{font-size:11.5px}.lad{gap:18px}}
.key{display:flex;flex-wrap:wrap;gap:2px 12px;font-size:11.5px;color:var(--ink2);margin:10px 0 2px}
.key i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
#rv-panel svg{display:block;font:10px -apple-system,Segoe UI,sans-serif;fill:#5b6270}
#rv-panel svg text{fill:#5b6270}
#rv-panel nav.tabs{display:flex;border-bottom:1px solid var(--line);position:sticky;top:0;background:#fff;z-index:2}
#rv-panel nav.tabs button{flex:1;border:0;background:none;padding:9px 2px;font-size:12px;font-weight:600;font-family:inherit;color:var(--ink3);cursor:pointer;border-bottom:2px solid transparent}
#rv-panel nav.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--ink)}
#rv-list .empty{padding:12px 16px;color:var(--ink2)}
.rv-item{display:block;width:100%;text-align:left;border:0;background:none;padding:8px 16px;cursor:pointer;border-bottom:1px solid var(--line);font:inherit;color:inherit}
.rv-item:hover,.rv-item:focus{background:var(--soft);outline:none}
.rv-row{display:flex;align-items:stretch;border-bottom:1px solid var(--line)}
.rv-row .rv-item{flex:1;border-bottom:0}
.rv-cmp{flex:none;width:42px;border:0;border-left:1px solid var(--line);background:none;color:var(--accent);font-size:15px;cursor:pointer}
.rv-cmp:hover,.rv-cmp:focus{background:#eef4ff;outline:none}
.rv-item b.st{display:inline-block;min-width:50px;padding:1px 6px;border-radius:5px;color:#fff;font-size:11px;font-weight:600;text-align:center;margin-right:8px}
.rv-item .sub{color:var(--ink2);font-size:12px;margin-left:64px}
.p-foot{flex:none;border-top:1px solid var(--line);padding:8px 16px 10px;font-size:11.5px;color:var(--ink3)}
.p-foot details{margin-bottom:2px}
.p-foot summary{cursor:pointer;color:var(--accent)}
.p-foot dl.facts{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:6px 0;color:var(--ink2)}
.p-foot dl.facts dd{margin:0}
.p-foot p{color:var(--ink2);max-height:30vh;overflow:auto}
.mp-label{color:#fff;font:600 11px sans-serif;text-shadow:0 0 3px #000,0 0 2px #000;white-space:nowrap}
#rv-open{position:fixed;top:10px;left:10px;z-index:1000;display:none;border:1px solid #e6e8ec;border-radius:10px;background:#fff;
  box-shadow:0 2px 10px rgba(0,0,0,.18);padding:9px 14px;font:600 13px -apple-system,Segoe UI,sans-serif;cursor:pointer}
@media (max-width:700px){#rv-panel{width:auto !important;right:10px;bottom:auto;max-height:80vh}.p-resize{display:none}}
</style>
"""

PANEL_JS = """
document.addEventListener('DOMContentLoaded', function(){
  var map = __MAP__, layers = {__LAYERS__};
  var panel = document.getElementById('rv-panel'), openBtn = document.getElementById('rv-open');
  function show(v){ panel.style.display = v ? 'flex' : 'none'; openBtn.style.display = v ? 'none' : 'block'; }
  document.getElementById('rv-close').onclick = function(){ show(false); };
  openBtn.onclick = function(){ show(true); };
  var wide = window.innerWidth > 700;
  if (!wide) show(false);
  // ---- resizable panel (width remembered in this browser) ----
  function setWidth(w){ w = Math.max(300, Math.min(760, Math.round(w))); panel.style.setProperty('--pw', w + 'px'); return w; }
  try { var sw = parseInt(localStorage.getItem('rv-w'), 10); if (sw) setWidth(sw); } catch (e) {}
  function RV_PW(){ return (wide && panel.style.display !== 'none') ? panel.getBoundingClientRect().right + 12 : 20; }
  window.RV_PW = RV_PW;
  function syncPad(){ L.Popup.prototype.options.autoPanPaddingTopLeft = L.point(RV_PW(), 10); }
  syncPad();
  var grip = document.getElementById('rv-resize'), dragging = false;
  grip.addEventListener('pointerdown', function(e){ dragging = true; grip.classList.add('drag'); grip.setPointerCapture(e.pointerId); e.preventDefault(); });
  grip.addEventListener('pointermove', function(e){ if (dragging) setWidth(e.clientX - panel.getBoundingClientRect().left); });
  grip.addEventListener('pointerup', function(){
    dragging = false; grip.classList.remove('drag'); syncPad(); map.invalidateSize();
    try { localStorage.setItem('rv-w', parseInt(getComputedStyle(panel).getPropertyValue('--pw'), 10)); } catch (e) {}
    if (window.RV_REFLOW) window.RV_REFLOW();
  });
  grip.addEventListener('keydown', function(e){
    var w = panel.getBoundingClientRect().width;
    if (e.key === 'ArrowRight') { setWidth(w + 40); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { setWidth(w - 40); e.preventDefault(); }
    syncPad();
  });
  // ---- toolbar menus, views, summary cards ----
  function toggleMenu(btn, pop, force){
    var open = force !== undefined ? force : pop.hidden;
    pop.hidden = !open; btn.setAttribute('aria-expanded', open);
  }
  var fltBtn = document.getElementById('flt-btn'), fltPop = document.getElementById('flt-pop'),
      setBtn = document.getElementById('set-btn'), setPop = document.getElementById('set-pop');
  fltBtn.onclick = function(){ toggleMenu(fltBtn, fltPop); toggleMenu(setBtn, setPop, false); };
  setBtn.onclick = function(){ toggleMenu(setBtn, setPop); toggleMenu(fltBtn, fltPop, false); };
  function setView(v){
    document.querySelectorAll('nav.views button').forEach(function(b){ b.setAttribute('aria-selected', b.dataset.view === v); });
    document.querySelectorAll('.p-body section').forEach(function(sct){ sct.hidden = sct.dataset.view !== v; });
  }
  document.querySelectorAll('nav.views button').forEach(function(b){ b.onclick = function(){ setView(b.dataset.view); }; });
  function setTab(t){
    document.querySelectorAll('nav.tabs button').forEach(function(x){ x.setAttribute('aria-selected', x.dataset.tab === t); });
    document.querySelectorAll('#rv-list [data-pane]').forEach(function(p){ p.hidden = p.dataset.pane !== t; });
  }
  L.control.zoom({position: 'bottomright'}).addTo(map);
  // Section lines are the overview; zoomed in, the coloured ROW footprints take over.
  var secLines = [__SECLINES__], foot = [__FOOT__], dimmed = false;
  function restyle(){
    var z = map.getZoom();
    secLines.forEach(function(p){ var l = p[0]; l.setStyle(dimmed ? {opacity: 0.25, weight: 3} : z >= 14 ? {opacity: 0.7, weight: 2} : {opacity: 0.95, weight: 7}); });
    var pix = document.getElementById('pix-toggle');
    foot.forEach(function(f){ f[0].setStyle(dimmed ? {fillOpacity: 0, opacity: 0.6} : {fillOpacity: pix && pix.checked ? 0 : 0.5, opacity: 0.9}); });
  }
  window.RV_DIM = function(d){ dimmed = d; restyle(); };

  // ---- filters ----
  var META = __SECMETA__, visible = {};
  function showLayer(l, v){ if (l) l.eachLayer(function(p){ if (p._path) p._path.style.display = v ? '' : 'none'; }); }
  function pressed(group){ return Array.prototype.map.call(document.querySelectorAll('.chips[data-group=' + group + '] button[aria-pressed=true]'), function(b){ return b.dataset.v; }); }
  function applyFilter(){
    var lands = pressed('land'), clss = pressed('cls'), allLand = lands.length === document.querySelectorAll('.chips[data-group=land] button').length,
        allCls = clss.length === document.querySelectorAll('.chips[data-group=cls] button').length, n = 0, mi = 0, ok = 0, okMi = 0;
    Object.keys(META).forEach(function(id){
      var m = META[id];
      var v = (allLand || lands.indexOf(m.land) >= 0) && (allCls || clss.indexOf(m.cls) >= 0);
      visible[id] = v;
      if (v) { n++; mi += m.mi; if (m.cls === 'similar' || m.cls === 'establishing') okMi += m.mi; }
    });
    secLines.forEach(function(p){ showLayer(p[0], visible[p[1]]); });
    foot.forEach(function(f){ showLayer(f[0], visible[f[1]]); });
    Object.keys(META).forEach(function(id){ showLayer(layers[id], visible[id]); });
    document.querySelectorAll('.rv-sec').forEach(function(r){ r.style.opacity = visible[r.dataset.id] === false ? 0.15 : 1; });
    document.querySelectorAll('.rv-item').forEach(function(b){ if (META[b.dataset.id]) (b.closest('.rv-row') || b).hidden = !visible[b.dataset.id]; });
    var off = document.querySelectorAll('.chips button[aria-pressed=false]').length, badge = document.getElementById('flt-badge');
    if (badge) { badge.hidden = !off; badge.textContent = off; }
    var all = Object.keys(META).length;
    document.getElementById('flt-sum').textContent = n === all ? '' :
      'Showing ' + n + ' of ' + all + ' sections (' + mi.toFixed(1) + ' mi; ' + okMi.toFixed(1) + ' mi at 70%+ of reference)';
  }
  document.querySelectorAll('.chips button').forEach(function(b){ b.onclick = function(){
    b.setAttribute('aria-pressed', b.getAttribute('aria-pressed') !== 'true'); applyFilter(); }; });
  document.getElementById('flt-reset').onclick = function(){
    document.querySelectorAll('.chips button').forEach(function(b){ b.setAttribute('aria-pressed', 'true'); }); applyFilter(); };
  map.on('zoomend overlayadd', applyFilter);
  window.RV_VISIBLE = function(id){ return visible[id] !== false; };
  window.RV_FILTER = applyFilter;

  // ---- colour-blind-safe toggle: swap the four class colours everywhere ----
  var CB = __CB__, REV = {}, cbOn = false;
  Object.keys(CB).forEach(function(k){ REV[CB[k]] = k; });
  function swap(str, m){ return str.replace(/#[0-9a-fA-F]{6}/g, function(h){ return m[h.toLowerCase()] || h; }); }
  function recolorDom(root, m){
    root.querySelectorAll('[fill],[stroke],[style]').forEach(function(el){
      ['fill', 'stroke', 'style'].forEach(function(a){ var v = el.getAttribute(a); if (v && v.indexOf('#') >= 0) { var nv = swap(v, m); if (nv !== v) el.setAttribute(a, nv); } });
    });
  }
  function recolorLayers(m){
    map.eachLayer(function(l){
      if (!(l instanceof L.Path)) return;
      var o = {}, c = (l.options.color || '').toLowerCase(), f = (l.options.fillColor || '').toLowerCase();
      if (m[c]) o.color = m[c]; if (m[f]) o.fillColor = m[f];
      if (o.color || o.fillColor) l.setStyle(o);
    });
  }
  function hexRgb(h){ return [1, 3, 5].map(function(k){ return parseInt(h.slice(k, k + 2), 16); }); }
  // recolour a class-coloured image for the colour-blind set and show it in an <img>
  window.RV_RECOLOR_IMG = function(imgEl, src){
    var im = new Image(); im.onload = function(){
      var c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
      var ctx = c.getContext('2d'); ctx.drawImage(im, 0, 0);
      var d = ctx.getImageData(0, 0, c.width, c.height), px = d.data, lut = Object.keys(CB).map(function(k){ return [hexRgb(k), hexRgb(CB[k])]; });
      for (var i = 0; i < px.length; i += 4) for (var j = 0; j < lut.length; j++) { var a = lut[j][0];
        if (px[i] === a[0] && px[i + 1] === a[1] && px[i + 2] === a[2]) { px[i] = lut[j][1][0]; px[i + 1] = lut[j][1][1]; px[i + 2] = lut[j][1][2]; break; } }
      ctx.putImageData(d, 0, 0); imgEl.src = c.toDataURL();
    }; im.src = src;
  };
  function recolorImage(ov, m){
    var src = ov._origUrl || ov._url; ov._origUrl = ov._origUrl || ov._url;
    if (!cbOn) { ov.setUrl(ov._origUrl); return; }
    var img = new Image(); img.onload = function(){
      var c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      var ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
      var d = ctx.getImageData(0, 0, c.width, c.height), px = d.data, lut = Object.keys(CB).map(function(k){ return [hexRgb(k), hexRgb(CB[k])]; });
      for (var i = 0; i < px.length; i += 4) { if (!px[i + 3]) continue;
        for (var j = 0; j < lut.length; j++) { var a = lut[j][0]; if (px[i] === a[0] && px[i + 1] === a[1] && px[i + 2] === a[2]) { px[i] = lut[j][1][0]; px[i + 1] = lut[j][1][1]; px[i + 2] = lut[j][1][2]; break; } } }
      ctx.putImageData(d, 0, 0); ov.setUrl(c.toDataURL());
    }; img.src = src;
  }
  var PIX = [__PIX__], PIXGROUP = __PIXGROUP__;
  function setCB(on){
    if (on === cbOn) return;
    var m = on ? CB : REV; cbOn = on;
    recolorDom(document.getElementById('rv-panel'), m);
    var body = document.getElementById('rv-d-body'); if (body) recolorDom(body, m);
    recolorLayers(m);
    PIX.forEach(function(ov){ recolorImage(ov, m); });
    if (window.RV_THUMBMODE) window.RV_THUMBMODE();
    try { localStorage.setItem('rv-cb', on ? '1' : '0'); } catch (e) {}
  }
  window.RV_CB = function(){ return cbOn ? CB : null; };
  window.RV_RECOLOR = function(root){ if (cbOn) recolorDom(root, CB); };
  var cbBox = document.getElementById('cb-toggle');
  cbBox.onchange = function(){ setCB(cbBox.checked); };
  try { if (localStorage.getItem('rv-cb') === '1') { cbBox.checked = true; setCB(true); } } catch (e) {}

  // ---- pixel detail toggle (also in the layer menu) ----
  var pixBox = document.getElementById('pix-toggle');
  if (!PIXGROUP) pixBox.parentNode.hidden = true;
  function pixOn(on){
    if (!PIXGROUP) return;
    if (on) { PIXGROUP.addTo(map); if (map.getZoom() < 13) map.setZoom(14); } else map.removeLayer(PIXGROUP);
    restyle();  // one place decides footprint styling (pixel detail, comparison mode)
  }
  pixBox.onchange = function(){ pixOn(pixBox.checked); };
  map.on('overlayadd overlayremove', function(e){ if (e.layer === PIXGROUP) { pixBox.checked = e.type === 'overlayadd';
    restyle(); } });
  map.on('zoomend', restyle);
  foot.forEach(function(f){ f[0].on('click', function(){ if (window.RV_DETAIL) window.RV_DETAIL(f[1]); }); });
  // Milepost search and whole-mile labels
  var MPS = __MPS__, labels = L.layerGroup();
  MPS.forEach(function(p){
    if (Math.abs(p[2] - Math.round(p[2])) < 0.03)
      L.marker([p[0], p[1]], {interactive: false, icon: L.divIcon({className: 'mp-label', html: 'MP ' + Math.round(p[2]), iconSize: null, iconAnchor: [-6, 6]})}).addTo(labels);
  });
  function mpLabels(){ if (map.getZoom() >= 12) labels.addTo(map); else labels.remove(); }
  map.on('zoomend', mpLabels); mpLabels();
  document.getElementById('mp-form').onsubmit = function(e){
    e.preventDefault();
    var v = parseFloat(document.getElementById('mp-in').value), msg = document.getElementById('mp-msg');
    if (!isFinite(v)) { msg.textContent = 'Enter a milepost number.'; return; }
    var best = null; MPS.forEach(function(p){ if (!best || Math.abs(p[2] - v) < Math.abs(best[2] - v)) best = p; });
    if (!best || Math.abs(best[2] - v) > 0.3) { msg.textContent = 'No milepost near ' + v + ' on this project.'; return; }
    msg.textContent = '';
    if (!wide) show(false);
    document.getElementById('mp-in').blur();
    map.setView([best[0], best[1]], 16);
    L.popup().setLatLng([best[0], best[1]]).setContent('<b>MP ' + best[2] + '</b> (' + best[3] + ')').openOn(map);
  };
  map.fitBounds(__BOUNDS__, wide ? {paddingTopLeft: [RV_PW(), 20], paddingBottomRight: [20, 20]} : {padding: [20, 20]});
  // ---- section details drawer ----
  var DETAILS = JSON.parse(document.getElementById('rv-details').textContent), ORDER = Object.keys(DETAILS);
  var thumbModeNow = 'green';  // the thumbnail type stays as chosen when stepping between sections
  var drawer = document.getElementById('rv-detail'), dBody = document.getElementById('rv-d-body'),
      current = null, highlight = null, panelWasOpen = null;
  function drawerWidth(){ return drawer.getBoundingClientRect().width; }
  function openDetail(id){
    var d = DETAILS[id]; if (!d) return;
    current = id;
    document.getElementById('rv-d-title').textContent = d.title;
    dBody.innerHTML = d.html; dBody.scrollTop = 0;
    if (window.RV_RECOLOR) window.RV_RECOLOR(dBody);
    document.getElementById('rv-d-prev').disabled = ORDER.indexOf(id) <= 0;
    document.getElementById('rv-d-next').disabled = ORDER.indexOf(id) >= ORDER.length - 1;
    var pw = RV_PW(), avail = window.innerWidth - pw - 10;
    var keepPanel = wide && panel.style.display !== 'none' && avail >= 760;
    if (!keepPanel && panel.style.display !== 'none') {
      if (panelWasOpen === null) panelWasOpen = true;
      show(false); openBtn.style.display = 'none'; pw = 20; avail = window.innerWidth - 30;
    }
    // leave a strip of map visible for location context whenever there's room
    if (window.innerWidth > 900) drawer.style.width = (avail >= 900 ? Math.min(1080, avail - 260) : avail) + 'px';
    drawer.hidden = false;
    document.querySelectorAll('.lad-c.sel').forEach(function(c){ c.classList.remove('sel'); });
    var cell = document.querySelector('.lad-c[data-id="' + id + '"]');
    if (cell) { cell.classList.add('sel'); cell.scrollIntoView({block: 'nearest'}); }
    map.closePopup();
    if (highlight) map.removeLayer(highlight);
    var lyr = layers[id];
    if (lyr) {
      highlight = L.geoJSON(lyr.toGeoJSON(), {interactive: false, style: {color: '#ffeb3b', weight: 12, opacity: 0.85}}).addTo(map);
      var wideDrawer = window.innerWidth > 900;
      map.fitBounds(lyr.getBounds(), {maxZoom: 16, paddingTopLeft: [pw + 10, 30],
        paddingBottomRight: [wideDrawer ? drawerWidth() + 40 : 30, 30]});
    }
    dBody.querySelectorAll('.d-thumbs img').forEach(function(img){ img.onclick = function(){ img.classList.toggle('big'); }; });
    function thumbMode(mode){
      thumbModeNow = mode;
      dBody.querySelectorAll('.d-mode button').forEach(function(b){ b.setAttribute('aria-pressed', b.dataset.mode === mode); });
      dBody.querySelectorAll('.d-cap[data-cap]').forEach(function(p){ p.hidden = p.dataset.cap !== mode; });
      dBody.querySelectorAll('.d-thumbs img').forEach(function(img){
        if (mode === 'pct' && window.RV_CB && window.RV_CB()) window.RV_RECOLOR_IMG(img, img.dataset.pct);
        else img.src = mode === 'pct' ? img.dataset.pct : img.dataset.green;
      });
    }
    dBody.querySelectorAll('.d-mode button').forEach(function(b){ b.onclick = function(){ thumbMode(b.dataset.mode); }; });
    window.RV_THUMBMODE = function(){ thumbMode(thumbModeNow); };
    thumbMode(thumbModeNow);
    dBody.querySelectorAll('.cmp-here').forEach(function(b){ b.onclick = function(){
      var v = b.dataset.b.split(',').map(Number); closeDetail(); if (window.RV_COMPARE) window.RV_COMPARE(b.dataset.seg, [[v[0], v[1]], [v[2], v[3]]]);
    }; });
    document.getElementById('rv-d-close').focus();
  }
  function closeDetail(){
    drawer.hidden = true; current = null;
    document.querySelectorAll('.lad-c.sel').forEach(function(c){ c.classList.remove('sel'); });
    if (highlight) { map.removeLayer(highlight); highlight = null; }
    if (panelWasOpen) show(true); else if (panel.style.display === 'none') openBtn.style.display = 'block';
    panelWasOpen = null;
  }
  document.getElementById('rv-d-close').onclick = closeDetail;
  function step(dir){  // next / previous section that passes the filters
    var k = ORDER.indexOf(current) + dir;
    while (k >= 0 && k < ORDER.length && window.RV_VISIBLE && !window.RV_VISIBLE(ORDER[k])) k += dir;
    if (k >= 0 && k < ORDER.length) openDetail(ORDER[k]);
  }
  document.getElementById('rv-d-prev').onclick = function(){ step(-1); };
  document.getElementById('rv-d-next').onclick = function(){ step(1); };
  Object.keys(DETAILS).forEach(function(id){ if (layers[id]) layers[id].on('click', function(){ openDetail(id); }); });
  document.addEventListener('keydown', function(e){
    if (drawer.hidden) return;
    var big = document.querySelector('.d-thumbs img.big');
    if (e.key === 'Escape') { if (big) big.classList.remove('big'); else closeDetail(); }
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select') return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { document.getElementById('rv-d-prev').click(); e.preventDefault(); }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { document.getElementById('rv-d-next').click(); e.preventDefault(); }
  });
  window.RV_DETAIL = openDetail;
  window.RV_REFLOW = function(){ if (current) openDetail(current); };
  function go(id){
    if (DETAILS[id]) { openDetail(id); return; }
    var lyr = layers[id]; if (!lyr) return;
    var b = lyr.getBounds(); if (!wide) show(false);
    map.fitBounds(b, {maxZoom: 16, paddingTopLeft: [RV_PW(), 20], paddingBottomRight: [20, 20]});
    setTimeout(function(){ lyr.openPopup(b.getCenter()); }, 400);
  }
  document.querySelectorAll('.rv-item').forEach(function(b){ b.onclick = function(){ go(b.dataset.id); }; });
  document.querySelectorAll('.rv-cmp').forEach(function(b){ b.onclick = function(){
    var v = b.dataset.b.split(',').map(Number);
    if (!wide) show(false);
    if (window.RV_COMPARE) window.RV_COMPARE(b.dataset.seg, [[v[0], v[1]], [v[2], v[3]]]);
  }; });
  document.querySelectorAll('.rv-sec').forEach(function(r){ r.addEventListener('click', function(){ go(r.dataset.id); }); });
  document.querySelectorAll('nav.tabs button').forEach(function(t){ t.onclick = function(){ setTab(t.dataset.tab); }; });
  document.querySelectorAll('.lad-c').forEach(function(c){ c.addEventListener('keydown', function(e){
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(c.dataset.id); }
  }); });
  document.querySelectorAll('.kpi').forEach(function(k){ k.onclick = function(){
    var want = {ok: ['similar', 'establishing'], bad: ['lagging', 'poor']}[k.dataset.kpi];
    if (want) {
      document.querySelectorAll('.chips[data-group=land] button').forEach(function(b){ b.setAttribute('aria-pressed', 'true'); });
      document.querySelectorAll('.chips[data-group=cls] button').forEach(function(b){ b.setAttribute('aria-pressed', want.indexOf(b.dataset.v) >= 0); });
      if (window.RV_FILTER) window.RV_FILTER();
      setView('overview');
    } else { setView('lists'); setTab(k.dataset.kpi === 'ero' ? 'risk' : 'out'); }
  }; });
  map.on('popupopen', function(e){
    e.popup.getElement().querySelectorAll('.rv-thumbs img').forEach(function(img){
      img.onclick = function(){ img.classList.toggle('big'); };
    });
  });
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape') document.querySelectorAll('.rv-thumbs img.big').forEach(function(i){ i.classList.remove('big'); }); });
});
"""


DRAWER_HTML = """
<aside id="rv-detail" role="dialog" aria-labelledby="rv-d-title" hidden>
 <header class="d-head">
  <button id="rv-d-prev" class="d-navbtn" aria-label="Previous section">&#9664;</button>
  <button id="rv-d-next" class="d-navbtn" aria-label="Next section">&#9654;</button>
  <h2 id="rv-d-title"></h2>
  <button id="rv-d-close" aria-label="Close details">&times;</button>
 </header>
 <div id="rv-d-body"></div>
</aside>
"""

DRAWER_CSS = """
<style>
#rv-detail{position:fixed;top:10px;right:10px;bottom:10px;width:min(1080px,66vw);z-index:1003;background:#fff;border-radius:10px;
  box-shadow:0 4px 24px rgba(0,0,0,.35);display:flex;flex-direction:column;font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222}
#rv-detail[hidden]{display:none}
#rv-detail .d-head{display:flex;align-items:center;gap:8px;padding:12px 16px;border-bottom:1px solid #eee}
#rv-detail .d-head h2{flex:1;margin:0 0 0 4px;font-size:18px}
#rv-detail .d-navbtn{border:1px solid #ccd;background:#fff;border-radius:6px;width:34px;height:30px;cursor:pointer;font-size:13px}
#rv-detail .d-navbtn:disabled{opacity:.35;cursor:default}
#rv-d-close{border:0;background:none;font-size:28px;line-height:1;cursor:pointer;color:#555;padding:0 4px}
#rv-d-body{overflow-y:auto;padding:14px 18px 20px}
#rv-detail .d-top{display:grid;grid-template-columns:minmax(180px,230px) 1fr;gap:18px;align-items:start}
#rv-detail .d-big{font-size:48px;font-weight:800;line-height:1}
#rv-detail .d-cls{display:inline-block;margin-left:10px;padding:3px 10px;border-radius:12px;color:#fff;font-weight:700;vertical-align:12px;text-transform:capitalize}
#rv-detail .d-sub{color:#666;font-size:12px;margin-top:6px}
#rv-detail dl.d-facts{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0;font-size:13px}
#rv-detail dl.d-facts dt{color:#777}
#rv-detail dl.d-facts dd{margin:0}
#rv-detail .d-note{background:#fff4e5;border-left:4px solid #e08214;padding:8px 10px;margin:12px 0 0;font-size:13px}
#rv-detail .d-actions{margin:12px 0 4px}
#rv-detail .cmp-here{padding:7px 12px;border:1px solid #1f6feb;border-radius:6px;background:#fff;color:#1f6feb;font:600 13px sans-serif;cursor:pointer}
#rv-detail h3{font-size:14px;margin:18px 0 6px;color:#333}
#rv-detail svg.d-chart{width:100%;height:auto;display:block;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}
#rv-detail svg.d-chart .c-ax{font-size:12px;fill:#666}
#rv-detail svg.d-chart .c-yr{font-size:14px;font-weight:700;fill:#333}
#rv-detail svg.d-chart .c-g{font-size:12px;fill:#444}
#rv-detail svg.d-chart .c-con{font-size:12px;font-weight:700;fill:#8a5a1c}
#rv-detail svg.d-chart .c-val{font-size:12px;font-weight:800}
#rv-detail svg.d-chart .c-leg text{font-size:12px;fill:#444}
#rv-detail .d-thumbs{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
#rv-detail .d-th-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0 6px}
#rv-detail .d-th-head h3{margin:0}
#rv-detail .d-mode{display:inline-flex;border:1px solid #d5d9e0;border-radius:8px;overflow:hidden}
#rv-detail .d-mode button{border:0;background:#fff;padding:4px 12px;font-size:12.5px;font-family:inherit;cursor:pointer;color:#5b6270}
#rv-detail .d-mode button+button{border-left:1px solid #d5d9e0}
#rv-detail .d-mode button[aria-pressed=true]{background:#1d1f23;color:#fff}
#rv-detail .d-swatches i{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 6px;vertical-align:-1px}
#rv-detail .d-thumbs figure{margin:0;text-align:center;font-size:12px;color:#444}
#rv-detail .d-thumbs img{width:100%;aspect-ratio:1;object-fit:contain;background:#e8e8e8;image-rendering:pixelated;border:1px solid #ccc;border-radius:4px;cursor:zoom-in}
#rv-detail .d-thumbs img.big{position:fixed;inset:4vh 4vw;width:92vw;height:92vh;aspect-ratio:auto;z-index:4000;background:#222;cursor:zoom-out}
#rv-detail .d-cap{font-size:12px;color:#666;margin:6px 0 0}
@media (max-width:900px){#rv-detail{left:0;right:0;top:0;bottom:0;width:auto;border-radius:0}
  #rv-detail .d-top{grid-template-columns:1fr}#rv-detail .d-thumbs{grid-template-columns:repeat(2,1fr)}}
</style>
"""


COMPARE_CSS = """
<style>
.cmp-btn{display:block;width:100%;margin:6px 0 8px;padding:7px;border:1px solid #1f6feb;border-radius:6px;background:#1f6feb;
  color:#fff;font:600 13px sans-serif;cursor:pointer}
.cmp-btn:hover{background:#1a5fd0}
#cmp-box{position:fixed;top:10px;right:60px;z-index:1002;width:340px;background:#fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.3);
  padding:10px 12px;font:13px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#222}
#cmp-box[hidden]{display:none}
#cmp-box h2{font-size:14px;margin:0 22px 6px 0}
#cmp-box .cmp-row{display:flex;align-items:center;gap:6px;margin:4px 0}
#cmp-box .cmp-row b{flex:none;width:48px;font-size:12px;color:#444}
#cmp-box select{flex:1;font:inherit;padding:3px;min-width:0;width:auto}
#cmp-box .cmp-row > *{width:auto;max-width:none;margin:0}
#cmp-box .cmp-sm{border:1px solid #ccd;background:#fff;border-radius:4px;padding:2px 7px;cursor:pointer;font:inherit}
#cmp-box .cmp-modes{display:inline-flex;border:1px solid #ccd;border-radius:6px;overflow:hidden}
#cmp-box .cmp-modes button{border:0;background:#fff;padding:4px 10px;cursor:pointer;font:inherit}
#cmp-box .cmp-modes button[aria-pressed=true]{background:#1f6feb;color:#fff}
#cmp-box .cmp-hint{font-size:11px;color:#777;margin-top:4px}
#cmp-box .cmp-legend{display:flex;align-items:center;gap:6px;font-size:11px;color:#555;margin-top:4px}
#cmp-box .cmp-legend span.bar{width:120px;height:9px;border:1px solid #ccc;border-radius:2px;
  background:linear-gradient(to right,#8c510a,#d8b365,#f6e8c3,#c7eae5,#5ab4ac,#01665e)}
#cmp-close{position:absolute;top:6px;right:8px;border:0;background:none;font-size:20px;cursor:pointer;color:#666}
#cmp-status{font-size:11px;color:#b86e00;min-height:14px}
.cmp-div{position:absolute;top:0;bottom:0;width:4px;margin-left:-2px;background:#fff;z-index:650;cursor:ew-resize;
  box-shadow:0 0 4px rgba(0,0,0,.6);touch-action:none}
.cmp-div i{position:absolute;top:50%;left:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;background:#fff;
  box-shadow:0 1px 4px rgba(0,0,0,.5);font:bold 14px/30px sans-serif;color:#333;text-align:center;font-style:normal}
.cmp-lab{position:absolute;bottom:34px;z-index:651;background:rgba(0,0,0,.72);color:#fff;padding:3px 8px;border-radius:4px;
  font:600 12px sans-serif;white-space:nowrap;pointer-events:none}
.cmp-here{margin-top:6px;padding:5px 8px;border:1px solid #1f6feb;border-radius:5px;background:#fff;color:#1f6feb;font:600 12px sans-serif;cursor:pointer}
@media (max-width:700px){#cmp-box{top:auto;bottom:10px;left:10px;right:10px;width:auto}}
</style>
"""

COMPARE_HTML = """
<div id="cmp-box" role="dialog" aria-label="Compare before and after construction" hidden>
 <button id="cmp-close" aria-label="Close comparison">&times;</button>
 <h2>Compare before / after</h2>
 <div class="cmp-row"><b>Segment</b><select id="cmp-seg" aria-label="Segment"></select></div>
 <div class="cmp-row"><b>Before</b><select id="cmp-before" aria-label="Before image"></select></div>
 <div class="cmp-row"><b>After</b><button class="cmp-sm" data-step="-1" aria-label="Earlier after date">&#9664;</button><select id="cmp-after" aria-label="After image"></select><button class="cmp-sm" data-step="1" aria-label="Later after date">&#9654;</button></div>
 <div class="cmp-row"><b>Show</b><span class="cmp-modes" role="group" aria-label="Display"><button data-show="photo" aria-pressed="true">Photo</button><button data-show="green" aria-pressed="false">Greenness</button></span>
  <label style="font-size:12px;margin-left:auto"><input type="checkbox" id="cmp-lines"> section colours</label></div>
 <div class="cmp-legend" id="cmp-legend" hidden>bare<span class="bar"></span>vigorous</div>
 <div id="cmp-status"></div>
 <div class="cmp-hint">Drag the white divider. Dates listed are at least 80% clear over this segment's ROW (% shown).
  Sentinel-2 is 10 m per pixel; NAIP aerial is 0.6 m.</div>
</div>
"""

COMPARE_JS = r"""
document.addEventListener('DOMContentLoaded', function(){
  var map = __MAP__, CMP = __CMP__, secLines = [__SECLINES__];
  var API = 'https://planetarycomputer.microsoft.com/api/data/v1';
  var box = document.getElementById('cmp-box'), selSeg = document.getElementById('cmp-seg'),
      selB = document.getElementById('cmp-before'), selA = document.getElementById('cmp-after'),
      status = document.getElementById('cmp-status'), lines = document.getElementById('cmp-lines');
  var S = {on: false, show: 'photo', x: null, layers: {}}, searches = {};
  ['cmpL', 'cmpR'].forEach(function(n){ map.createPane(n).style.zIndex = 250; });
  var paneL = map.getPane('cmpL'), paneR = map.getPane('cmpR'), container = map.getContainer();
  var div = L.DomUtil.create('div', 'cmp-div', container); div.innerHTML = '<i>&#8644;</i>'; div.style.display = 'none';
  var labL = L.DomUtil.create('div', 'cmp-lab', container), labR = L.DomUtil.create('div', 'cmp-lab', container);
  labL.style.display = labR.style.display = 'none';
  L.DomEvent.disableClickPropagation(div);

  function nice(d){ return new Date(Date.parse(d)).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC'}); }
  function poly(b){ return {type: 'Polygon', coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]}; }
  function register(key, body){
    if (!searches[key]) searches[key] = fetch(API + '/mosaic/register', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)}).then(function(r){ if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function(j){ return j.searchid; });
    return searches[key];
  }
  function source(seg, value){  // value: 'naip' or a date
    var b = CMP.segments[seg].bbox;
    if (value === 'naip') return register('naip|' + seg, {collections: ['naip'], 'filter-lang': 'cql2-json', filter: {op: 'and', args: [
      {op: 's_intersects', args: [{property: 'geometry'}, poly(b)]},
      {op: 'anyinteracts', args: [{property: 'datetime'}, {interval: ['2010-01-01T00:00:00Z', (CMP.construction_year - 1) + '-12-31T23:59:59Z']}]}]},
      sortby: [{field: 'datetime', direction: 'desc'}]});
    return register(seg + '|' + value, {collections: ['sentinel-2-l2a'], 'filter-lang': 'cql2-json', filter: {op: 'and', args: [
      {op: 's_intersects', args: [{property: 'geometry'}, poly(b)]},
      {op: 'anyinteracts', args: [{property: 'datetime'}, {interval: [value + 'T00:00:00Z', value + 'T23:59:59Z']}]}]},
      sortby: [{field: 'eo:cloud_cover', direction: 'asc'}]});
  }
  function tiles(sid, value){
    var base = API + '/mosaic/' + sid + '/tiles/WebMercatorQuad/{z}/{x}/{y}?';
    var green = S.show === 'green';
    if (value === 'naip') return base + 'collection=naip&nodata=0&' + (green
      ? 'assets=image&expression=' + encodeURIComponent('(image_b4-image_b1)/(image_b4+image_b1)') + '&rescale=-0.1,0.8&colormap_name=brbg&format=png'
      : 'assets=image&asset_bidx=' + encodeURIComponent('image|1,2,3'));
    // Sentinel-2 processing baseline 04.00+ (from 25 Jan 2022) adds +1000 to reflectance: subtract it in the NDVI
    var off = value >= '2022-01-25' ? '-2000' : '';
    return base + 'collection=sentinel-2-l2a&nodata=0&' + (green
      ? 'assets=B08&assets=B04&asset_as_band=true&expression=' + encodeURIComponent('(B08-B04)/(B08+B04' + off + ')') + '&rescale=-0.1,0.8&colormap_name=brbg&format=png'
      : 'assets=visual&asset_bidx=' + encodeURIComponent('visual|1,2,3'));
  }
  function label(value){
    if (value === 'naip') return 'NAIP aerial (before construction, 0.6 m)';
    return 'Sentinel-2 ' + nice(value);
  }
  function setSide(side, value){
    var seg = selSeg.value, pane = side === 'L' ? 'cmpL' : 'cmpR', lab = side === 'L' ? labL : labR;
    lab.textContent = (side === 'L' ? 'Before · ' : 'After · ') + label(value);
    status.textContent = 'Loading imagery...';
    source(seg, value).then(function(sid){
      if (S.layers[side]) map.removeLayer(S.layers[side]);
      S.layers[side] = L.tileLayer(tiles(sid, value), {pane: pane, maxZoom: 20, maxNativeZoom: value === 'naip' ? 18 : 16,
        attribution: value === 'naip' ? 'USDA NAIP via Microsoft Planetary Computer' : 'Sentinel-2 (Copernicus) via Microsoft Planetary Computer'}).addTo(map);
      S.layers[side].on('load', function(){ status.textContent = ''; });
      S.layers[side].on('tileerror', function(){ status.textContent = 'Some tiles could not load (zoom in, or the image may not cover this spot).'; });
      clip();
    }).catch(function(e){ status.textContent = 'Could not load imagery: ' + e.message; });
  }
  function clip(){
    if (!S.on) return;
    var size = map.getSize(), nw = map.containerPointToLayerPoint([0, 0]), se = map.containerPointToLayerPoint(size);
    var x = map.containerPointToLayerPoint([S.x, 0]).x;
    paneL.style.clip = 'rect(' + nw.y + 'px,' + x + 'px,' + se.y + 'px,' + nw.x + 'px)';
    paneR.style.clip = 'rect(' + nw.y + 'px,' + se.x + 'px,' + se.y + 'px,' + x + 'px)';
    div.style.left = S.x + 'px';
    labL.style.right = (size.x - S.x + 10) + 'px'; labL.style.left = 'auto';
    labR.style.left = (S.x + 10) + 'px';
  }
  function fill(seg){
    var s = CMP.segments[seg];
    selB.innerHTML = '<option value="naip">NAIP aerial photo (before construction)</option>' +
      s.pre.slice().reverse().map(function(d){ return '<option value="' + d[0] + '">' + nice(d[0]) + ' · ' + d[1] + '% clear</option>'; }).join('');
    selA.innerHTML = s.post.slice().reverse().map(function(d){ return '<option value="' + d[0] + '">' + nice(d[0]) + ' · ' + d[1] + '% clear</option>'; }).join('');
  }
  function sameSeasonBefore(seg, after){  // pre-construction clear date closest in time of year, latest year first
    var s = CMP.segments[seg], doy = function(d){ var t = new Date(Date.parse(d)); return (t - Date.UTC(t.getUTCFullYear(), 0, 1)) / 864e5; };
    var best = null, bestScore = 1e9;
    s.pre.forEach(function(p){
      var dd = Math.abs(doy(p[0]) - doy(after)); dd = Math.min(dd, 365 - dd);
      var score = dd + (CMP.construction_year - 1 - Number(p[0].slice(0, 4))) * 20;
      if (dd <= 30 && score < bestScore) { best = p[0]; bestScore = score; }
    });
    return best || 'naip';
  }
  function dimLines(dim){
    if (window.RV_DIM) window.RV_DIM(dim);
  }
  function open(seg, bounds){
    if (seg) selSeg.value = seg;
    fill(selSeg.value);
    var after = selA.options.length ? selA.options[0].value : null;
    if (!after) { status.textContent = 'No clear images after construction for this segment.'; }
    selA.value = after; selB.value = sameSeasonBefore(selSeg.value, after);
    box.hidden = false; S.on = true;
    div.style.display = labL.style.display = labR.style.display = '';
    var left = window.RV_PW ? window.RV_PW() - 12 : 0; if (left < 20) left = 0;
    S.x = Math.round(left + (map.getSize().x - left) / 2);
    dimLines(!lines.checked);
    if (bounds) {
      // put the divider through the middle of the feature so it opens half "before", half "after"
      map.once('moveend', function(){
        S.x = Math.round(map.latLngToContainerPoint(L.latLngBounds(bounds).getCenter()).x); clip();
      });
      map.fitBounds(bounds, {maxZoom: 16, paddingTopLeft: [left + 20, 20], paddingBottomRight: [20, 20]});
    }
    setSide('L', selB.value); if (after) setSide('R', after);
    clip();
  }
  function close(){
    S.on = false; box.hidden = true;
    ['L', 'R'].forEach(function(k){ if (S.layers[k]) { map.removeLayer(S.layers[k]); S.layers[k] = null; } });
    paneL.style.clip = paneR.style.clip = '';
    div.style.display = labL.style.display = labR.style.display = 'none';
    dimLines(false);
  }
  window.RV_COMPARE = open;

  selSeg.innerHTML = Object.keys(CMP.segments).sort(function(x, y){
      var a = x.indexOf('Yard: ') === 0, b = y.indexOf('Yard: ') === 0; return a === b ? x.localeCompare(y) : a ? 1 : -1; })
    .map(function(s){ return '<option value="' + s.replace(/"/g, '&quot;') + '">' + s + '</option>'; }).join('');
  selSeg.onchange = function(){ open(selSeg.value); };
  selB.onchange = function(){ setSide('L', selB.value); };
  selA.onchange = function(){ setSide('R', selA.value); };
  document.querySelectorAll('#cmp-box [data-step]').forEach(function(b){ b.onclick = function(){
    var i = selA.selectedIndex - Number(b.dataset.step);  // list is newest first
    if (i >= 0 && i < selA.options.length) { selA.selectedIndex = i; setSide('R', selA.value); }
  }; });
  document.querySelectorAll('#cmp-box [data-show]').forEach(function(b){ b.onclick = function(){
    S.show = b.dataset.show;
    document.querySelectorAll('#cmp-box [data-show]').forEach(function(x){ x.setAttribute('aria-pressed', x === b); });
    document.getElementById('cmp-legend').hidden = S.show !== 'green';
    setSide('L', selB.value); setSide('R', selA.value);
  }; });
  lines.onchange = function(){ dimLines(!lines.checked); };
  document.getElementById('cmp-close').onclick = close;
  var opener = document.getElementById('cmp-open');
  if (opener) opener.onclick = function(){
    var c = map.getCenter(), best = Object.keys(CMP.segments)[0], bd = 1e9;
    Object.keys(CMP.segments).forEach(function(k){ var b = CMP.segments[k].bbox, d = Math.hypot((b[0] + b[2]) / 2 - c.lng, (b[1] + b[3]) / 2 - c.lat); if (d < bd) { bd = d; best = k; } });
    var b = CMP.segments[best].bbox;
    open(best, map.getZoom() < 12 ? [[b[1], b[0]], [b[3], b[2]]] : null);
  };
  map.on('move zoom resize', clip);
  // drag the divider
  var dragging = false;
  div.addEventListener('pointerdown', function(e){ dragging = true; div.setPointerCapture(e.pointerId); map.dragging.disable(); e.preventDefault(); });
  div.addEventListener('pointermove', function(e){
    if (!dragging) return;
    var r = container.getBoundingClientRect(); S.x = Math.max(0, Math.min(r.width, e.clientX - r.left)); clip();
  });
  div.addEventListener('pointerup', function(){ dragging = false; map.dragging.enable(); });
  // popup buttons: "Compare before / after here"
  map.on('popupopen', function(e){
    e.popup.getElement().querySelectorAll('.cmp-here').forEach(function(b){
      b.onclick = function(){
        var v = b.dataset.b.split(',').map(Number);
        map.closePopup(); open(b.dataset.seg, [[v[0], v[1]], [v[2], v[3]]]);
      };
    });
  });
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape' && S.on) close(); });
});
"""


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="GeoPackage with route, mileposts, workspace layers")
    p.add_argument("--data", required=True, help="folder for the imagery cache and rainfall (local, not synced)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--title", default="Revegetation screening")
    p.add_argument("--start", default="2023-04-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--pre-years", default="2023,2024", help="growing seasons before construction")
    p.add_argument("--construction-year", type=int, default=2025)
    p.add_argument("--current-from", default="2026-10-01", help="start of the 'latest' window")
    p.add_argument("--section-mi", type=float, default=0.5)
    p.add_argument("--max-cloud", type=float, default=60)
    p.add_argument("--segments", help="comma list to limit the run, e.g. PL-2")
    p.add_argument("--no-rain", action="store_true")
    a = p.parse_args()
    a.pre_years = [int(y) for y in a.pre_years.split(",") if y]
    a.ref_inner, a.ref_outer, a.edge_m = 30.0, 150.0, 7.0
    a.outside_min, a.outside_max, a.min_patch_px = 20.0, 200.0, 4
    a.min_row_px, a.min_ref_px, a.min_valid, a.bare_ndvi = 6, 20, 0.6, 0.3
    for d in (a.data, a.out, os.path.join(a.data, "cache")):
        os.makedirs(d, exist_ok=True)

    route = gpd.read_file(a.project, layer="route")
    mp = gpd.read_file(a.project, layer="mileposts")
    ws = gpd.read_file(a.project, layer="workspace")
    utm = route.estimate_utm_crs()
    if a.segments:
        keep = a.segments.split(",")
        route, mp = route[route.segment.isin(keep)], mp[mp.segment.isin(keep)]
    ws = ws.to_crs(utm)
    sec = build_sections(mp, utm, a.section_mi)
    chunks = build_chunks(sec, per_chunk=6)
    print(f"{len(sec)} sections of {a.section_mi} mi in {len(chunks)} chunks; {len(ws)} workspace polygons")

    all_rows, all_ts, all_dist, all_thumbs, days_seen = [], [], [], {}, set()
    clear_by_seg, pixel_layers = {}, []
    for seg in sorted(sec.segment.unique()):
        seg_area = gpd.GeoSeries([sec[sec.segment == seg].geometry.union_all().buffer(400)], crs=utm).to_crs(4326)
        items = rw.dedupe_items(rw.search_scenes(seg_area.total_bounds, a.start, a.end, a.max_cloud))
        print(f"{seg}: {len(items)} Sentinel-2 scenes {a.start} – {a.end}")
        for ch in [c for c in chunks if c["segment"] == seg]:
            rows, ts, dist, thumbs, meta = analyse_chunk(ch, sec, ws, mp, items, utm, a)
            all_rows += rows
            all_ts += ts
            for d in dist:
                d["segment"] = seg
            all_dist += dist
            all_thumbs.update(thumbs)
            days_seen |= set(meta["days"])
            clear_by_seg.setdefault(seg, []).append((meta["clear"], meta["n_row"]))
            if meta.get("pixel"):
                pixel_layers.append(meta["pixel"])
            print(f"  {ch['id']}: done")
    stats = pd.DataFrame(all_rows).set_index("sec_index")
    sec = sec.join(stats)
    years = sorted({d.year for d in days_seen})
    last = max(years)
    pre_cols = [f"peak_{y}" for y in a.pre_years if f"peak_{y}" in sec]
    sec["pre"] = sec[pre_cols].mean(axis=1) if pre_cols else np.nan
    sec["vs_pre"] = sec[f"peak_{last}"] / sec["pre"] * 100
    sec["class"] = [classify(v)[0] for v in sec[f"peak_{last}"]]  # recomputed below after exclusions
    # erosion screen
    # Erosion screen on EXCESS bare ground: ROW bare share minus the reference's, so a harvested
    # field next to a harvested ROW (or late-season dormancy) does not count.
    # Cropland: in October the ROW strip and the field around it are harvested at different
    # times, so use peak season (Jun–Sep), when bare ground in a crop field means poor growth.
    crop = sec["land_before"] == "crop"
    sec["bare_basis"] = np.where(crop, f"peak {last}", "last 45 days")
    sec["bare_row"] = np.where(crop, sec["bare_peak"], sec["bare_now"])
    sec["bare_nearby"] = np.where(crop, sec["bare_peak_ref"], sec["bare_ref"])
    sec["bare_excess"] = (sec["bare_row"] - pd.Series(sec["bare_nearby"], index=sec.index).fillna(0)).clip(lower=0)
    water = sec["near_water"].fillna(False).astype(bool)
    score = sec["bare_excess"].fillna(0) * (1 + sec["slope_p90"].fillna(0) / 4) * np.where(water, 1.5, 1)
    sec["risk_score"] = score.round(3)
    high = (sec["bare_excess"] >= 0.25) & ((sec["slope_p90"] >= 4) | water)
    med = sec["bare_excess"] >= 0.10
    sec["risk"] = np.where(pd.isna(sec["bare_row"]), "n/a", np.where(high, "High", np.where(med, "Medium", "Low")))
    # In cropland the October reading and the season trend follow harvest and crop growth,
    # not restoration: keep them in the tables but don't show them as results.
    crop = sec["land_before"] == "crop"
    sec["current_raw"], sec["trend_raw"] = sec["current"], sec["trend_30d"]
    sec.loc[crop, ["current", "trend_30d"]] = np.nan
    # Built-up land (roads, station pads) isn't expected to revegetate: not scored.
    built = sec["land_before"] == "built"
    sec.loc[built, [f"peak_{last}", "current", "trend_30d", "vs_pre"]] = np.nan
    sec.loc[built, "risk"] = "n/a"
    # Sections that already looked different from their neighbours before construction
    sec["differed_before"] = sec["pre"] < 85
    sec["class"] = [classify(v)[0] for v in sec[f"peak_{last}"]]
    sec.loc[sec["land_before"] == "built", "class"] = "not scored (built-up)"
    sec["thumbs"] = [all_thumbs.get(i) for i in sec.index]
    # each section's piece of the approved ROW workspace (for colouring the footprint)
    row_ws = ws[ws.kind == "row"].geometry.union_all()
    feet, taken = [], None
    for i, r in sec.iterrows():
        g = r.geometry.buffer(150, cap_style=2).intersection(row_ws)
        if taken is not None:
            g = g.difference(taken)
        taken = g if taken is None else taken.union(g)
        feet.append(g)
    sec["foot"] = list(gpd.GeoSeries(feet, crs=utm).to_crs(4326))
    ts_df = pd.DataFrame(all_ts)
    sec["ts"] = [list(zip(g["date"], g["pct_of_ref"])) if len(g) else [] for g in
                 (ts_df[ts_df.sec_index == i] if len(ts_df) else pd.DataFrame() for i in sec.index)]

    dist = gpd.GeoDataFrame(all_dist, crs=utm) if all_dist else gpd.GeoDataFrame({"area_m2": []}, geometry=[], crs=utm)
    if len(dist):  # neighbouring chunks overlap slightly: merge patches found twice
        groups = gpd.GeoSeries([dist.geometry.union_all()], crs=utm).explode(index_parts=False)
        merged = []
        for g in groups:
            hit = dist[dist.intersects(g)]
            yrs = sorted({y for v in hit["years_bare"] for y in str(v).split(",") if y})
            merged.append({"geometry": g, "area_m2": round(g.area), "years_bare": ",".join(yrs),
                           "ndvi_pre": float(hit["ndvi_pre"].mean()), "ndvi_after": float(hit["ndvi_after"].mean()),
                           "dist_m": float(hit["dist_m"].min()), "segment": hit["segment"].iloc[0]})
        dist = gpd.GeoDataFrame(merged, crs=utm)
        mpu = mp.to_crs(utm)
        near = gpd.sjoin_nearest(dist, mpu[["mp", "segment", "geometry"]].rename(columns={"segment": "seg_mp"}), how="left")
        near = near[~near.index.duplicated()]
        dist["near"] = [f"{s} MP {v:.1f}" for s, v in zip(near["seg_mp"], near["mp"])]

    yards = {}
    for _, y in ws[ws.kind == "yard"].iterrows():
        area = gpd.GeoSeries([y.geometry.buffer(300)], crs=utm).to_crs(4326)
        items = rw.dedupe_items(rw.search_scenes(area.total_bounds, a.start, a.end, a.max_cloud))
        r = analyse_yard(y["name"], y.geometry, items, utm, a)
        if r:
            yards[y["name"]] = r
            print(f"  yard {y['name']}: done")

    # rainfall per segment (CHIRPS) for context
    facts = []
    if not a.no_rain:
        for seg, g in sec.groupby("segment"):
            pts = gpd.GeoSeries(list(g.geometry.centroid), crs=utm).to_crs(4326)
            pick = [pts.iloc[0], pts.iloc[len(pts) // 2], pts.iloc[-1]]
            rain = rw.get_rainfall([(q.x, q.y) for q in pick], a.start, a.end, os.path.join(a.data, f"rain_{seg}.csv"))
            if rain.notna().mean() > 0.5:
                ctx = rw.rain_context(rain, f"{a.construction_year}-01-01")
                lastday = ctx["rain"].last_valid_index()
                s0 = pd.Timestamp(f"{last}-04-01")
                season = ctx.loc[s0:lastday].dropna()
                pct = season["rain"].sum() / max(season["normal"].sum(), 1e-6)
                facts.append((f"Rain {seg}", f"{season['rain'].sum():.0f} mm Apr–{rw.nice_date(lastday, False)} {last} "
                                             f"&middot; {pct:.0%} of normal"))

    cur_dates = sorted({d for s in sec["current_dates"].dropna() for d in s.split(",") if d})
    facts = [("Imagery", f"Sentinel-2 (free, 10 m) &middot; {len(days_seen)} dates {a.start[:4]}–{a.end[:4]}"),
             ("Latest clear", ", ".join(rw.nice_date(d, False) for d in cur_dates[-4:]) + f" {last}" if cur_dates else "none since " + a.current_from),
             ("Before", "/".join(map(str, a.pre_years)) + f" peak seasons &middot; construction {a.construction_year}")] + facts

    # tile layers for the overview
    s2 = pc_tile_url({"collections": ["sentinel-2-l2a"], "filter-lang": "cql2-json", "filter": {"op": "and", "args": [
        {"op": "s_intersects", "args": [{"property": "geometry"}, mapping(box(*gpd.GeoSeries([sec.geometry.union_all()], crs=utm).to_crs(4326).total_bounds))]},
        {"op": "anyinteracts", "args": [{"property": "datetime"}, {"interval": [f"{a.current_from}T00:00:00Z", f"{a.end}T23:59:59Z"]}]},
        {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 20]}]},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}]},
        {"collection": "sentinel-2-l2a", "assets": "visual", "asset_bidx": "visual|1,2,3", "nodata": 0})
    naip = pc_tile_url({"collections": ["naip"], "filter-lang": "cql2-json", "filter": {"op": "anyinteracts", "args": [
        {"property": "datetime"}, {"interval": ["2022-01-01T00:00:00Z", "2023-12-31T23:59:59Z"]}]},
        "sortby": [{"field": "datetime", "direction": "desc"}]},
        {"collection": "naip", "assets": "image", "asset_bidx": "image|1,2,3", "nodata": 0})

    compare = {"construction_year": a.construction_year, "pre_years": a.pre_years, "segments": {}}
    for seg, parts in clear_by_seg.items():
        tot = sum(n for _, n in parts) or 1
        clear = sum(c.reindex(sorted(set().union(*[set(c.index) for c, _ in parts]))).fillna(0) * n for c, n in parts) / tot
        good = clear[(clear >= 0.8) & clear.index.month.isin(range(4, 12))]  # Apr–Nov: dormant winter scenes aren't comparable
        b = gpd.GeoSeries([sec[sec.segment == seg].geometry.union_all().buffer(500)], crs=utm).to_crs(4326).total_bounds
        compare["segments"][seg] = {
            "bbox": [round(float(v), 5) for v in b],
            "pre": [[d.strftime("%Y-%m-%d"), round(float(v) * 100)] for d, v in good.items() if d.year < a.construction_year],
            "post": [[d.strftime("%Y-%m-%d"), round(float(v) * 100)] for d, v in good.items() if d.year >= a.construction_year]}
    # Yards get their own entries (several lie outside the segments' imagery areas). "Before" dates
    # only come from pre-construction years when the yard was still undisturbed (some were used in 2024).
    for name, y in yards.items():
        c = y["_clear"]
        good = c[(c >= 0.8) & c.index.month.isin(range(4, 12))]
        peaks = {yr: y.get(f"peak_{yr}") for yr in a.pre_years}
        best = max([v for v in peaks.values() if v is not None and np.isfinite(v)] or [np.nan])
        undisturbed = [yr for yr, v in peaks.items() if v is not None and np.isfinite(v) and v >= 0.8 * best]
        compare["segments"][f"Yard: {name}"] = {
            "bbox": [round(float(v), 5) for v in y["_bbox"]],
            "pre": [[d.strftime("%Y-%m-%d"), round(float(v) * 100)] for d, v in good.items() if d.year in undisturbed],
            "post": [[d.strftime("%Y-%m-%d"), round(float(v) * 100)] for d, v in good.items() if d.year >= a.construction_year]}
    info_compare = compare
    seg_curves = {}
    if len(ts_df):
        t = ts_df.assign(segment=sec.loc[ts_df.sec_index, "segment"].values,
                         land=sec.loc[ts_df.sec_index, "land_before"].values)
        t = t[t["date"].dt.month.between(5, 9) & (t["land"] != "built")]
        t["month"] = t["date"].dt.to_period("M").dt.to_timestamp() + pd.Timedelta(days=14)
        for seg, g in t.groupby("segment"):
            med = g.groupby("month")["pct_of_ref"].median()
            seg_curves[seg] = list(med.items())
    mps = mp.to_crs(4326)
    info = {"title": a.title, "years": years, "compare": info_compare, "seg_curves": seg_curves, "end": a.end,
            "latest_clear": cur_dates[-1] if cur_dates else None,
            "pixel_layers": pixel_layers,
            "mileposts": [[round(g.y, 6), round(g.x, 6), round(float(v), 2), sgm] for g, v, sgm in zip(mps.geometry, mps["mp"], mps["segment"])], "pre_years": a.pre_years, "construction_year": a.construction_year,
            "current_from": a.current_from, "facts": facts, "s2_tiles": s2, "naip_tiles": naip}
    write_map(sec, ws, dist, yards, info, os.path.join(a.out, "map.html"))

    # tables
    keep_cols = [c for c in sec.columns if c not in ("geometry", "thumbs", "current_dates", "foot", "ts")]
    sec[keep_cols].round(2).to_csv(os.path.join(a.out, "sections.csv"), index=False)
    sec[keep_cols + ["geometry"]].to_crs(4326).to_file(os.path.join(a.out, "sections.geojson"), driver="GeoJSON")
    if len(dist):
        dist.to_crs(4326).to_file(os.path.join(a.out, "outside_workspace.geojson"), driver="GeoJSON")
    pd.DataFrame([{k: v for k, v in y.items() if not k.startswith("_")} for y in yards.values()]).round(1) \
        .to_csv(os.path.join(a.out, "yards.csv"), index=False)
    ts = pd.DataFrame(all_ts)
    if len(ts):
        ts["section_id"] = sec.loc[ts.sec_index, "section_id"].values
        ts.drop(columns="sec_index").to_csv(os.path.join(a.out, "timeseries.csv"), index=False)

    print(f"\nPeak {last}, % of reference (miles by class):")
    print(sec.assign(mi=sec.mp_to - sec.mp_from).groupby("class")["mi"].sum().round(1).to_string())
    print(f"Erosion screen: {sec['risk'].value_counts().to_dict()}")
    print(f"Bare ground outside workspace: {len(dist)} patches, {dist['area_m2'].sum() / 4046.86 if len(dist) else 0:.1f} ac")
    print(f"Wrote {a.out}/map.html, sections.csv/.geojson, yards.csv, timeseries.csv")


if __name__ == "__main__":
    main()
