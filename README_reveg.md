# reveg_watch: revegetation and restoration screening (proof of concept)

`reveg_watch.py` screens a restored pipeline right-of-way using **free Sentinel-2 imagery** (10 m, about every 5 days). It reports on half-mile sections between the project's own mileposts:

1. **Revegetation:** how green the restored workspace is compared with similar undisturbed land nearby ("% of reference"). It gives this for each peak growing season from before construction to now, plus the current-season trend and the latest (October) reading.
2. **Workspace check:**
   - ground that was vegetated before construction but bare afterwards **outside** the approved workspace polygons;
   - how each temporary yard is recovering.
3. **Erosion-risk screen:** bare ground in the ROW beyond what the neighbouring land shows, combined with slope (USGS 3DEP 10 m elevation) and water or wetland nearby.

It shares setup and helpers with `row_watch.py`; see `README.md` for installation.

## Keep client data off synced drives
Put project files, the imagery cache and outputs in a **local, non-synced folder** (e.g. `~/Merjent_local/<project>/`). The code folder stays free of client data.

## Inputs
A GeoPackage with three layers:

| Layer | Contents |
|---|---|
| `route` | Pipeline centerline, column `segment` (e.g. PL-1) |
| `mileposts` | Points with `mp` (number) and `segment` |
| `workspace` | Polygons with `name` and `kind`: `row` (restored workspace), `yard` (temporary yard or staging area), `facility` (permanent; not scored) |

Building it from KMZ files is a short script; keep that script with the project data, not in this repository.

## Running

```bash
~/.venvs/row_watch/bin/python reveg_watch.py \
  --project ~/Merjent_local/<project>/data/project.gpkg \
  --data ~/Merjent_local/<project>/data --out ~/Merjent_local/<project>/outputs \
  --title "Revegetation screening" \
  --start 2023-04-01 --end 2026-10-19 \
  --pre-years 2023,2024 --construction-year 2025 --current-from 2026-10-01
```

**Run time:** the first run on about 51 miles plus 6 yards takes roughly 20 minutes (about 2,000 Sentinel-2 scenes, corridor pixels only). Re-runs use the cache and take a few minutes. Use `--segments PL-2` to run one segment.

## Outputs

| File | What it is |
|---|---|
| `map.html` | Map with a side panel. The panel shows miles at 70%+ of reference, one bar per segment coloured by section, and four tabs: lowest sections, erosion screen, bare ground outside workspace, and yards. Click a section for year-by-year bars and before / construction / 2026 / latest greenness thumbnails. Map layers include **Sentinel-2 since 1 Oct** (the latest free imagery), **NAIP 2022–23** pre-construction aerials, and Esri imagery. |
| `sections.csv` / `.geojson` | One row per section: % of reference by year, latest, trend, land type before construction, reference used, erosion metrics. |
| `outside_workspace.geojson` | Candidate disturbance patches outside the approved polygons. |
| `yards.csv` | % of reference by year for each temporary yard. |
| `timeseries.csv` | Section × date: ROW and reference greenness and % of reference. |

## Reading the map
- **Side panel:**
  - **Header:** the title, the latest clear image date, and four summary cards: miles at 70%+, miles below 70%, erosion flags, outside workspace. Clicking a card filters the map to those sections or opens that list.
  - **Toolbar:** **Compare** (before/after slider), **Go to MP**, **Filter** (a badge shows how many filters are on), and a settings menu with the pixel-detail and colour-blind switches.
  - **Overview view:** the **milepost ladder**, one column per segment and one cell per half-mile section coloured by status, with ▲ for erosion flags and ● for bare ground outside the workspace. Click a cell, or focus it and press Enter, to open its details. The open section is outlined, and ↑ ↓ (or ← →) step along the pipeline. Below the ladder is the recovery-over-time chart.
  - **Lists view:** Lowest, Erosion, Outside ws and Yards.
  - **Footer:** **Data and method** and the screening caveat.
  - **Resizing:** drag the panel's right edge (or focus it and use ← →) between about 300 and 760 px; the width is remembered. Wider panels give bigger ladder cells, with each section's % shown from about 470 px. The ‹ button collapses the panel.
- **Details panel:** opens beside the ladder when the window is wide enough, leaving a strip of map showing where the section is.
- **On the map:** zoomed out, each section is a coloured line along the route. From zoom 14 the **approved ROW workspace itself** is filled in its section's colour, and milepost labels appear every mile from zoom 12.
- **Colours:** green = similar or establishing, orange = lagging, red = poor. The **Colour-blind-safe colours** switch in the panel changes these everywhere (map, strips, lists, charts, details, pixel layer) to blue, light blue, orange and red. The choice is remembered in that browser. Greenness images use brown (bare) to teal (vigorous), the same scale as the comparison view.
- **Filters:** chips under the strips show only certain land uses before construction (cropland, grass/open, cleared forest, wetland) and/or statuses. The map, strips and lists follow the filter, the line below shows the filtered miles, and ◀ ▶ in the details panel skips hidden sections. **Show all** resets.
- **Pixel detail (10 m):** a switch in the panel, also in the layer menu, colours each 10 m pixel inside the ROW by its peak-season greenness as % of that section's reference, in the same classes. It shows patterns inside a section, such as one bad corner, a trench line, or a crossing. Pixels at the workspace edge mix ROW and neighbouring land, so read the interior and treat single edge pixels with caution.
- **Section details:** click a section (line, footprint, list entry or bar) to open a large details panel on the right. The side panel tucks away and the map highlights the section in yellow in the remaining space. The panel shows:
  - the score and class;
  - land use before construction and the reference used;
  - trend, latest reading and the erosion screen;
  - a **timeline**: every clear image as a dot, a monthly-median line, labelled peak-season medians, the construction year and peak-season windows shaded, and 100% / 70% guides;
  - four maps of the section (before construction, each peak season since, latest), click to enlarge. A switch flips them between **Greenness** (brown = bare, teal = vigorous) and **% of reference (pixel detail)**. In pixel detail, each 10 m pixel inside the workspace is coloured by its greenness as % of that section's reference land for the same period, and land outside is grey. The choice stays as you step between sections.

  Use ◀ ▶ or the ← → keys to step along the pipeline, **Compare before / after imagery here** to open the slider, and Esc to close.

## Comparing before and after
Use **Compare before / after** in the side panel, the button in a section's details panel, the **⇄** button beside each entry in the Erosion, Outside ws and Yards tabs, or the button in an outside-workspace or yard popup. Each splits the map with a draggable divider, centred on that feature:
- **Before (left):** the NAIP aerial photo from before construction (0.6 m), or any clear Sentinel-2 date from the pre-construction years.
- **After (right):** any clear Sentinel-2 date from the construction year onward. ◀ ▶ steps through the dates.
- **Photo / Greenness:** true colour, or greenness (brown = bare, teal = vigorous).
- **Default pairing:** the latest clear image, paired with a pre-construction image from the same time of year so the seasons match.
- **Which dates are listed:** only dates at least 80% clear over that segment's ROW, April–November.
- **Yards** have their own entries in the Segment list, because several lie outside the pipeline segments' imagery areas. Their "before" dates come only from pre-construction years when the yard was still undisturbed: for example 2023 only for yards already in use in 2024.

The imagery is streamed live from Microsoft Planetary Computer, so it needs an internet connection; nothing is stored in the map file.

## How to read it
- **% of reference** is relative greenness (NDVI), not measured % cover. Before construction most sections sit near 100%. Field plots are needed to turn it into a calibrated % cover.
- **Choice of reference land:** each section is compared with land of the same type before construction: cropland with cropland, grass with grass, wetland with wetland. Cleared forest is compared with nearby open land, because the ROW is reseeded to grass.
- **Classes at peak season:** similar ≥90%, establishing 70–90%, lagging 50–70%, poor <50%. The 70% line matches common "70% of background" success language; adjust it to the governing plan.
- **Cropland:** the October reading and the in-season trend follow harvest and crop growth, so they are not shown. For cropland, the erosion screen uses peak-season bare ground.
- **Not scored:** built-up sections (roads, station pads) and permanent facilities.
- **"Was X% before":** sections that already differed from their neighbours before construction. Judge those by "vs. before", not by 100%.

## Limits
- **Resolution:** 10 m pixels. Only pixels fully inside the workspace are used, which leaves about 3 pixels across a 45 m ROW. Results are section-level; small bare patches, trench-line detail and erosion-control devices need PlanetScope (3 m), sub-meter imagery or drones.
- **Time of year:** October imagery is late season. Revegetation judgements rest on the peak season (15 Jun–15 Sep).
- **Bare ground outside the workspace** can be unrelated (other projects, new buildings, drained ponds). Always check against variances and the NAIP layer.
- **Rainfall:** the rainfall overlay (CHIRPS) is optional (`--no-rain`). The CHIRPS server may refuse requests after heavy use.
