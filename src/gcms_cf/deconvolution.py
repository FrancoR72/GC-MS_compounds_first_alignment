from __future__ import annotations
from typing import Dict, List, Optional

from .models import Peak, Compound

def toy_deconvolve_rt_peaks(
    peaks_by_rt: Dict[float, List[Peak]],
    *,
    sample_id: str,
    compound_prefix: str = "C",
    purity_default: float = 1.0,
    ri_default: Optional[float] = None,
    make_ids_unique: bool = True,
) -> List[Compound]:
    """
    Deconvoluzione 'baseline':
    - ogni RT diventa un Compound
    - area = somma intensità dei picchi a quella RT
    - purity = purity_default
    - ri = ri_default (None di default)

    compound_id:
    - se make_ids_unique=True: "{sample_id}_C000001" (univoco tra campioni)
    - altrimenti: "C000001" (riparte per campione)
    """
    compounds: List[Compound] = []
    counter = 1

    for rt in sorted(peaks_by_rt.keys()):
        spectrum = peaks_by_rt[rt]
        area = sum(p.intensity for p in spectrum)

        base_id = f"{compound_prefix}{counter:06d}"
        compound_id = f"{sample_id}_{base_id}" if make_ids_unique else base_id
        counter += 1

        compounds.append(
            Compound(
                compound_id=compound_id,
                sample_id=sample_id,
                rt=rt,          # minuti
                ri=ri_default,  # None per ora
                area=area,
                purity=purity_default,
                spectrum=spectrum,
            )
        )

    return compounds
