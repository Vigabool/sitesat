# ROW Watch: satellite vegetation-stress screening for pipeline rights-of-way (proof of concept)

ROW Watch looks for vegetation dying along a pipeline route using **free Sentinel-2 imagery** (10 m pixels, a new picture roughly every 5 days). It colours each 250 m segment of pipe **ok / watch / ALERT** and writes a web map you can open in any browser.

It follows the same idea as Stantec's PipeWatch, but uses free, coarser imagery. No accounts, API keys or paid data are needed. Imagery comes from Microsoft Planetary Computer's open catalogue. Sentinel data is free for commercial use. Google Earth Engine is deliberately avoided because its free tier is non-commercial.

---

## Setup (once)

You need Python 3.10 or newer. Create the virtual environment **outside** a cloud-synced folder (Proton Drive, OneDrive and similar), because the libraries are about 500 MB.

```bash
python3 -m venv ~/.venvs/row_watch
~/.venvs/row_watch/bin/pip install -r requirements.txt
```

## Running it

```bash
~/.venvs/row_watch/bin/python row_watch.py \
    --route my_pipeline.geojson \
    --baseline-start 2019-04-01 --monitor-start 2023-04-01 --end 2023-10-31 \
    --out results/my_pipeline
```

Then open `results/my_pipeline/map.html`.

| Option | Meaning |
|---|---|
| `--route` | Pipeline centerline: GeoJSON, KML, shapefile or GeoPackage. Keep a PoC run to 10–30 km. |
| `--baseline-start` | Start of the "normal" period. **Use 3–4 years.** Each pixel's normal range is learned from these years, and 2 years gave far more false alarms in testing. |
| `--monitor-start`, `--end` | The period you want screened. |
| `--mark LAT,LON,LABEL` | Optional map pin, such as a known incident. Repeatable. |
| `--title` | Heading shown on the map. |
| `--no-rain` | Skip the rainfall overlay (offline, or outside 50°S–50°N). |

Everything else has tested defaults; see `python row_watch.py -h`.

**Run time:** a 10 km route with 4.5 years of imagery took 1–3 minutes to download (220–430 scenes). The NDVI is cached (`ndvi_cube.npz`, about 20 MB), so re-running with different settings or a shorter date range takes seconds. Use `--cache` to share one cache between runs of the same route.

To test against a known location without the real centerline, `make_test_route.py` draws a straight line through a point:

```bash
~/.venvs/row_watch/bin/python make_test_route.py --lat 33.86608 --lon -97.96573 --bearing 61 \
    --km-before 7 --km-after 3 --out routes/montague_tx.geojson
```

## Outputs

| File | What it is |
|---|---|
| `map.html` | Self-contained web map; see "Reading the map" below. |
| `anomalies.geojson` | One feature per anomaly patch: status, area, km along the route, first seen / first alert / last seen, and segment ids. These are the places to send someone. |
| `segments.geojson` | One feature per segment: status, first flag / alert date, largest patch (m²). Opens in QGIS, ArcGIS or geojson.io. |
| `timeseries.csv` | Every segment × scene: NDVI difference, anomaly patch size, flag / alert. |
| `rainfall.csv` | Daily rainfall (mm) along the route from CHIRPS, cached so later runs only fetch new days. |

The terminal prints the same patch list, ALERT patches first and largest first.

### Reading the map

- **Side panel (left).** It reads from top to bottom:
  - **Verdict**, such as "5 alerts to check · 6 to watch". These counts are patches (places to look), not route sections.
  - **Where**: the whole route as one bar from km 0 to the end, coloured by section status, with ▼ for any `--mark` pins. Click the bar to zoom the map to that section.
  - **When**: the monitoring period month by month. Each tick is a clear image, and each dot is the date a patch first alerted (red) or appeared (orange). Grey months are not scored. Few or no ticks means clouds left gaps. Blue bars are daily rain; the background turns tan where the past 30 days were much drier than normal (under half) and blue where they were much wetter (over 1.5×).
  - **Facts**: the period watched, how many clear images there were and the latest one (flagged if it is well before the period end), the years used as "normal", and total rain in the period compared with normal.
  - **How alerts are decided** (click to expand): the detection rules in plain words, and the date the map was made.
  - **Patch list**: alerts first. Click an entry to zoom to that patch and open its details.

  On a phone the panel folds into an "n alerts" button.
- **Patches.** These are areas where vegetation dropped unusually (magenta outline, plus a dot coloured by status so they stay visible when zoomed out). Each popup has:
  - **before/after greenness thumbnails** with the patch outlined (brown = bare, green = vigorous). Click them, or **Inspect & compare images**, to open the image viewer described below.
  - links to the **actual satellite photo** for the before and after dates in the free Copernicus Browser. The first visit shows a "Log in / Use anonymously" prompt; either works.
  - a Google Maps link for directions.
- **Segments.** Click anywhere near the line (there is a wide invisible click area) for the segment's dates and its NDVI-difference chart.
- **Status.** A patch or segment is **watch** when a patch of at least 500 m² appeared on one scene, and **ALERT** when it appeared on two consecutive clear scenes. **no data** means clouds or masking left too little to score.

### Rainfall: was it the weather?

Daily rainfall comes from **CHIRPS** (Climate Hazards Center, UC Santa Barbara). It is a free, roughly 5 km satellite-and-gauge rainfall dataset covering 50°S–50°N, published about 3 weeks behind real time, with no account needed. The script reads the route's start, middle and end points and averages them; the first run takes about a minute per 4–5 years of dates. "Normal" for each time of year is the average over the same baseline years the detector uses. **Check the CHIRPS licence terms before relying on it in client deliverables.**

Rain appears in three places:
- the panel timeline and the **Rain** fact line;
- the image viewer: rain between the chosen before and after dates, and in the 30 days before the later date, each against normal, plus a daily-rain strip with both dates marked (B, A);
- a **Weather** note on a patch when the days before it first appeared match one of three rules of thumb:
  - **"… mm rain in 3 days just before: standing water?"**: 50 mm or more within 3 days, in the 10 days before. Ponding in low spots can look like dead vegetation.
  - **"first rain after a dry spell … neighbours greening faster?"**: the previous month had under half its normal rain, and the last 30 days had at least normal (and at least 25 mm). Pasture greens up faster than disturbed or compacted ground, so a patch can appear without any new damage.
  - **"very dry …"**: under 30% of normal rain in the last 30 days. Drought stress is uneven across soils and slopes.

**How far to trust the weather notes.**
- One rainfall value covers the whole site, so rain can't, by itself, explain why the pipe corridor differs from the land beside it; the detector already cancels out weather that affects both.
- The notes describe the weather; they are not a verdict on the cause. In the test cases, the real Payne spill was tagged "very dry" (it happened in a drought), and the Wyoming repair dig was tagged "first rain after a dry spell".
- Use the notes to decide what to look at in the images, not to dismiss alerts.

### Inspecting a patch in the image viewer

The viewer opens over the map and shows the 10 m greenness images around the patch. It has up to 16 clear scenes, from about 3 months before the patch appeared to a month after it was last seen.

| Control | What it does |
|---|---|
| **Side by side / Swipe / Blink / Change** (keys 1–4) | Four ways to compare the *Before* and *After* dates:<br>• Side by side shows the two images with matching zoom and pan.<br>• Swipe lets you drag a divider across a single image.<br>• Blink flips between the dates every 0.7 s (Space pauses).<br>• Change shows after minus before: red = lost greenness, blue = gained. |
| **Change vs. surroundings** (Change mode) | On by default. Subtracts the typical change across the whole image, so drought or green-up across the area cancels out and only local damage stays red. This is the same comparison the detector makes. |
| **Before / After dates** (◀ ▶, or ←/→ for After and Shift+←/→ for Before) | Pick any of the stored scenes. Dates marked ● are scenes where this patch was flagged. |
| **Zoom** | Scroll or double-click to zoom (Shift+double-click zooms out), drag to pan. The +/− and Fit buttons (keys + − 0) do the same. |
| **Hover** | Reads one pixel: whether it is inside the patch, its greenness before and after, the raw change, and the change vs. surroundings. |
| **Patch outline / Pipeline** | Show or hide the magenta patch outline and the dashed route line. |
| **Satellite photo** links | Open the real photo for the selected dates in Copernicus Browser. |
| **Rain row** | Rain between the two dates and in the 30 days before the later one, against normal, with a daily-rain strip (B = before, A = after). |

What to look for:
- **Spills** tend to look like a strip along the line or a lobe running downhill from it, with a sharp start date.
- **Field work** shows as straight edges matching field boundaries, often across the whole field.
- **Drought recovery**: a spot that stays red while everything around it turns blue (Montague, October 2023).

The images are greenness only (no true colour), at 10 m per pixel; use the satellite photo links to see what is actually on the ground. Pinch-zoom isn't supported on phones; use the +/− buttons.

Use `--title "Name"` to set the heading shown on the map (the default is the route file name).

---

## How it works

1. **Segments and zones.** The route is cut into 250 m segments. Each segment has two zones:
   - a **100 m-wide corridor** centred on the pipe, which is what gets scored;
   - a **reference ring** of land 60–250 m away.
2. **NDVI.** For every Sentinel-2 scene, NDVI (greenness) is computed. Cloud, shadow, water and snow are removed using Sentinel-2's own scene classification.
3. **Compare to neighbours.** For every corridor pixel: *diff = NDVI(pixel) − median NDVI(reference ring)*. Weather, drought and spring green-up affect both zones, so they cancel. A leak affects only the ground near the pipe.
4. **Compare to its own history.** Each pixel's diff is z-scored against that same pixel's diffs from the same time of year (±30 days) in the baseline years.
5. **Decide.**
   - A pixel is anomalous when z ≤ −3.5 on two consecutive clear scenes. Random noise rarely repeats; dead vegetation does.
   - A segment is flagged when anomalous pixels form a patch of at least 5 adjacent pixels (≥ 500 m²).
   - A segment raises an ALERT when it is flagged on two consecutive scenes.
   - Only April–October is scored (`--months`), because dormant vegetation can't show stress.

### What changed from the starter script, and why

| Change | Why |
|---|---|
| **Removed duplicate scenes.** | Planetary Computer stores many acquisitions twice: the original processing plus ESA's 2024 reprocessing, often with different reflectance offsets. Over 40% of tile-dates at the test site were duplicated. The script now keeps the newest version and never mixes offsets within one date. |
| **Scores pixels instead of segment medians.** | This was the decisive change. In the Montague TX test, the spill covered only a few of the ~75 pixels in a 30 m × 250 m ROW zone, so the segment median barely moved: best z was −1.6, and the original method missed it. Scoring pixels and looking for patches finds it. |
| **Corridor widened from 30 m to 100 m; reference ring from 60–200 m to 60–250 m.** | At 10 m pixels a 30 m zone is only 3 pixels wide. Spills also run downhill away from the line: at Montague, most of the damage was 50–200 m south of the release point. |
| **z threshold changed from −2.5 to −3.5 per pixel, plus pixel persistence and a minimum patch size.** | Per-pixel scores are much noisier than segment medians. Without these rules, 20–30 of 38 segments alerted falsely. |
| **Baseline lengthened from 2 years to 3–4 years.** | Same-season samples per pixel roughly double, so the normal range is estimated much better. |
| **Scores only April–October.** | December runs with the original method produced most of their false alerts in winter-wheat country. |
| **Added an imagery cache and corridor-only storage.** | Re-runs need no downloads, and the cache stays small for long routes. |
| **Added an optional `--mask-cropland` (ESA WorldCover + USDA Cropland Data Layer).** | Ignores cultivated fields. It **did not reduce false alarms** at Montague, because the land-cover maps missed many of the fields, so it is off by default. |

To avoid fitting the method to one incident, settings were checked across all three incidents plus a no-incident control year (sensitivity table below). The defaults are the strictest setting that still caught all three releases.

---

## Validation against real PHMSA releases

**Source.** PHMSA "Hazardous Liquid Accident Data – January 2010 to present". PHMSA's site blocks scripted downloads, so the file was fetched through a web browser. The search kept:
- onshore line-pipe releases;
- May–September, 2019 or later;
- at least 50 bbl;
- narratives mentioning pasture, grass, field or ranch.

**Test routes.** Each test route is a **straight 10 km line through the reported release coordinates**, with the bearing taken from the ROW scar visible in pre-incident 1 m NAIP aerial imagery. The real centerlines weren't available, so treat segment positions far from the incident as approximate.

| Case | What happened | Result with default settings |
|---|---|---|
| **Montague Co., TX**, PHMSA 20230219 | 300 bbl crude into a pasture, 23 Aug 2023, during a severe drought (NDVI ~0.18, grass already brown). Found the same day by air patrol. | **Detected.** First flag 26 Sep, **ALERT 6 Oct 2023: 44 days after the release.** The signal was weak: a 600 m² patch, just over the 500 m² threshold. The first clear scenes after the release (1 and 6 Sep) showed nothing yet. |
| **Payne Co., OK**, PHMSA 20220172 (Osage pipeline) | 7,110 bbl crude rupture, 8 Jul 2022. Pasture and woodland. Found the same day. | **Detected.** First flag 18 Jul, **ALERT 23 Jul 2022: 15 days after the release.** Patches of up to 4,700 m² trace about 600 m of ROW. Detected at every setting tested. |
| **Johnson Co., WY**, PHMSA 20220195 | 1,075 bbl diesel from a cracked weld. The leak most likely started 11 Jun 2022 but **went unnoticed until 27 Jul**. Sparse sagebrush. | **Not detected before discovery.** ALERT 9 Aug 2022: 59 days after the leak began and 13 days *after* it was found. This is most likely the excavation and repair work, not the leak itself. No false alarms anywhere else on this route. |

### False alarms

"Elsewhere" means more than 250 m from the release point, out of about 37 segments on each 10 km route.

| Run | ALERT segments elsewhere | watch segments elsewhere | ALERTs before the incident date |
|---|---|---|---|
| Montague 2023 | 5 | 5 | 1 |
| Montague 2022 (control year, no incident reported) | 8 | 5 | n/a |
| Payne 2022 | 5 | 5 | 2 |
| Johnson WY 2022 | 0 | 0 | 0 |

Not every "false alarm" is noise. In the control year, the biggest one (2.4 ha at Montague segments 36–37) is a pasture that **was actually cleared and tilled between Oct 2020 and Jul 2022**, which NAIP aerial photos confirm. Most of the others sit in or along cultivated fields. The tool finds real ground changes, but it can't tell a leak from a plough, so **every alert needs a human look** (current aerial imagery, a landowner call, or a drive-by).

**Rule of thumb from these tests:** in mixed farm and range country, expect roughly 4–8 segments per 10 km per season to need triage. In uniform rangeland, expect close to none.

### Sensitivity

Each cell shows *days from release to first ALERT near the incident / number of ALERT segments elsewhere*. The bold row is the default.

| z | min patch | Montague 2023 | Montague 2022 control | Payne 2022 | Johnson WY 2022 |
|---|---|---|---|---|---|
| 3.0 | 5 px | +34 d / 16 | – / 17 | +15 d / 7 | +59 d / 0 |
| 3.0 | 10 px | missed / 10 | – / 8 | +15 d / 4 | +59 d / 0 |
| **3.5** | **5 px** | **+44 d / 5** | **– / 8** | **+15 d / 5** | **+59 d / 0** |
| 3.5 | 10 px | missed / 5 | – / 4 | +15 d / 4 | +72 d / 0 |
| 4.0 | 5 px | missed / 4 | – / 4 | +15 d / 4 | +79 d / 0 |
| 4.0 | 10 px | missed / 3 | – / 3 | +15 d / 3 | +109 d / 0 |

Large releases are caught at any setting. Small releases in poor conditions sit right at the edge of what 10 m imagery can see, so any stricter setting misses Montague.

### Bottom line

With free imagery this works as a **screening and documentation tool**. It reliably shows the footprint of a large release within about 2 weeks, and a small one within about 6 weeks in a bad (drought) year. In these three tests it **did not find a leak before the operator did**, including the one leak that went unnoticed for 6 weeks.

---

## Limitations (plain English)

- **10 m pixels vs. a 15–30 m right-of-way.** The ROW is only 2–3 pixels wide, and each pixel blends ROW, spill and neighbouring land. A spill has to damage roughly **500 m² or more** (several pixels) before it can be seen reliably. Small seeps, and leaks under roads, gravel pads or woodland, will be missed.
- **Clouds.** Sentinel-2 passes every ~5 days (2–3 days at higher latitudes), but cloudy scenes are lost and an ALERT needs two clear looks. In practice that adds 1–4 weeks of delay, longer in wet seasons and cloudy regions.
- **Season and drought.** Only living vegetation can show stress. Dormant or drought-browned grass (Montague, Aug 2023) gives a weak, late signal. Winter, snow and dormancy cannot be screened.
- **Cropland and land management.** Ploughing, planting, harvest, haying, brush clearing and grazing all change NDVI. The seasonal baseline absorbs regular cycles, but not a field planted 3 weeks later than usual or a pasture newly cleared. This is the main source of false alarms. Uniform pasture and rangeland work best.
- **Gas vs. liquid.**
  - Crude and refined products kill vegetation directly, which is what this detects.
  - Natural gas leaks only show up indirectly: the gas displaces oxygen in the soil and roots slowly die over weeks to months, so only large, long-running gas leaks would appear.
  - HVL/NGL releases mostly flash to vapour and leave little lasting mark unless they ignite. A burn scar would show.
  - None of these are seen **before** vegetation reacts. Methane itself needs dedicated gas-imaging satellites, which is a different technology.
- **Repair and cleanup look like leaks.** Excavation, spill response and ROW construction also remove vegetation. The Wyoming alert was almost certainly the repair dig.
- **Test-route caveat.** The validation routes are straight lines through the release point, not the real centerlines.

## What PlanetScope (3 m, daily) would change

- **Resolution: about 9× more pixels per area.** A 30 m ROW becomes about 10 pixels wide instead of 3, so ROW and neighbouring land separate cleanly. The smallest detectable damage would drop from roughly 500 m² to roughly 50–100 m², small enough for many seeps and small releases. The Montague spill would have covered dozens of pixels instead of 6.
- **Daily revisits.** Far fewer cloud gaps. Two consecutive clear looks could come days apart instead of weeks, so detection lag after visible damage could plausibly drop from 2–6 weeks to under a week.
- **Red-edge band (SuperDove sensors).** Plant stress shows up in red-edge before visible browning, which may give earlier warning than NDVI.
- **What it wouldn't fix.** Dormant or drought-brown vegetation, cropland noise, telling leaks from land management, or seeing gas directly. Planet imagery is also less consistent scene to scene than Sentinel-2, so it needs a calibration step (Planet sells a harmonised product).
- **Cost.** PlanetScope is a commercial licence priced by area and time; free programmes such as NICFI are tropics-only or non-commercial. The method in this script carries over almost unchanged: swap the data loader, set the resolution to 3 m, and scale pixel-based thresholds such as `--min-cluster` to the smaller pixels.

---

## Data and privacy notes

- Only public data is used: Sentinel-2 (Copernicus, via Planetary Computer), PHMSA incident records, and NAIP / ESA WorldCover / USDA CDL for checking.
- The validation routes are hypothetical straight lines. **Don't put client centerlines in this folder** (it is cloud-synced). Run client data from a local, non-synced directory and keep outputs there.
- `runs/` holds the validation outputs and imagery caches (about 70 MB). Delete them freely; they can be regenerated.
