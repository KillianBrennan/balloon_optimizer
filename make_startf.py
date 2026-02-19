'''create startf for lagranto for balloon trajectory ensemble'''

import os
import numpy as np

LAT = 51.963517
LON = 7.615650
P = 900
TIMES = [-2,0,2]      # times in hours; caltra expects hh.mm but uses only minutes internally
LAT_STEP = 0.15
LON_STEP = LAT_STEP / np.cos(np.deg2rad(LAT))
# print(f"lon step: {LON_STEP:.3f}")


def main():
    # Write to ./lagranto/startf next to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "lagranto", "startf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lats = [LAT - LAT_STEP, LAT, LAT + LAT_STEP]
    lons = [LON - LON_STEP, LON, LON + LON_STEP]

    with open(out_path, "w", encoding="utf-8") as f:
        for t in TIMES:
            # caltra expects time in hh.mm format; here we just use t as given
            t_hhmm = float(f"{t:.2f}")
            for lat in lats:
                for lon in lons:
                    # time, lon, lat, p
                    f.write(f"{t_hhmm:.2f} {lon:.6f} {lat:.6f} {int(P):d}\n")

    n_points = len(TIMES) * len(lats) * len(lons)
    print(f"Wrote {n_points} start points to {out_path}")


if __name__ == "__main__":
    main()