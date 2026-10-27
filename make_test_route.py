#!/usr/bin/env python3
"""
make_test_route.py — build a straight test route through a point.

For validation only: when the real centerline isn't available, draw a straight
line through a known incident location along the pipeline's likely bearing.

  python make_test_route.py --lat 33.86608 --lon -97.96573 --bearing 62 \
      --km-before 7 --km-after 3 --out routes/montague_tx.geojson

Bearing is degrees clockwise from north (0 = N, 90 = E). The route runs from
`km-before` behind the point to `km-after` ahead of it along that bearing.
"""
import argparse
import math
import os

import geopandas as gpd
from shapely.geometry import LineString, Point


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--bearing", type=float, required=True, help="degrees clockwise from north")
    p.add_argument("--km-before", type=float, default=5)
    p.add_argument("--km-after", type=float, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    pt = gpd.GeoSeries([Point(a.lon, a.lat)], crs=4326)
    utm = pt.estimate_utm_crs()
    x, y = pt.to_crs(utm).iloc[0].coords[0]
    dx, dy = math.sin(math.radians(a.bearing)), math.cos(math.radians(a.bearing))
    line = LineString([(x - dx * a.km_before * 1000, y - dy * a.km_before * 1000),
                       (x + dx * a.km_after * 1000, y + dy * a.km_after * 1000)])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    gpd.GeoDataFrame({"name": ["test route"]}, geometry=[line], crs=utm).to_crs(4326).to_file(a.out, driver="GeoJSON")
    print(f"Wrote {a.out} ({line.length / 1000:.1f} km)")


if __name__ == "__main__":
    main()
