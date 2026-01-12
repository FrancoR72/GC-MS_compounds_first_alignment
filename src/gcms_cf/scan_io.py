from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict

from .models import Peak
from .scan_models import Scan

def read_scan_centroid_csv_long(path: str | Path) -> List[Scan]:
    """
    Legge un CSV lungo con colonne:
      scan_id, rt_min, mz, intensity
    Ritorna una lista di Scan ordinata per rt_min.
    """
    path = Path(path)
    scans: Dict[int, Dict] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"scan_id", "rt_min", "mz", "intensity"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV deve contenere colonne {sorted(required)}. Trovate: {reader.fieldnames}")

        for row in reader:
            sid = int(float(row["scan_id"]))
            rt = float(row["rt_min"])
            mz = float(row["mz"])
            inten = float(row["intensity"])

            if sid not in scans:
                scans[sid] = {"rt": rt, "peaks": []}
            scans[sid]["peaks"].append(Peak(mz=mz, intensity=inten))

    out: List[Scan] = []
    for sid, v in scans.items():
        out.append(Scan(scan_id=sid, rt_min=float(v["rt"]), peaks=v["peaks"]))

    out.sort(key=lambda s: s.rt_min)
    return out
