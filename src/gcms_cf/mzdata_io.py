from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Tuple
import base64
import xml.etree.ElementTree as ET
import numpy as np

def _decode_array(data_elem) -> np.ndarray:
    if data_elem is None:
        return np.array([], dtype=np.float32)

    length = int(data_elem.attrib.get("length", "0") or "0")
    text = (data_elem.text or "").strip()
    if length == 0 or not text:
        return np.array([], dtype=np.float32)

    precision = int(data_elem.attrib.get("precision", "32") or "32")
    endian = (data_elem.attrib.get("endian", "little") or "little").lower()

    if precision == 64:
        dtype = np.dtype("<f8" if endian == "little" else ">f8")
    else:
        dtype = np.dtype("<f4" if endian == "little" else ">f4")

    raw = base64.b64decode(text)
    arr = np.frombuffer(raw, dtype=dtype)
    if length and arr.size != length:
        arr = arr[:length]
    return arr

def iter_mzdata_spectra(xml_path: str | Path) -> Iterator[Tuple[int, float, np.ndarray, np.ndarray]]:
    """
    Yields: (spectrum_id, rt_min, mz_array, intensity_array)
    """
    xml_path = Path(xml_path)

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "spectrum":
            continue

        sid = int(elem.attrib.get("id", "0") or "0")

        rt_min = None
        for cv in elem.findall(".//cvParam"):
            if cv.attrib.get("name") == "TimeInMinutes":
                rt_min = float(cv.attrib.get("value"))
                break
        if rt_min is None:
            rt_min = float("nan")

        mz_data = elem.find("./mzArrayBinary/data")
        it_data = elem.find("./intenArrayBinary/data")

        mz = _decode_array(mz_data)
        inten = _decode_array(it_data)

        yield sid, rt_min, mz, inten

        elem.clear()

def compute_tic_mzdata(
    xml_path: str | Path,
    *,
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
    max_scans: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ritorna (rt, tic, n_peaks) calcolati in streaming.
    """
    rts = []
    tics = []
    nps = []

    n = 0
    for sid, rt, mz, inten in iter_mzdata_spectra(xml_path):
        if not np.isfinite(rt):
            continue
        if rt_min is not None and rt < rt_min:
            continue
        if rt_max is not None and rt > rt_max:
            continue

        rts.append(rt)
        tics.append(float(np.sum(inten)) if inten.size else 0.0)
        nps.append(int(inten.size))

        n += 1
        if max_scans is not None and n >= max_scans:
            break

    return np.array(rts, dtype=float), np.array(tics, dtype=float), np.array(nps, dtype=int)
