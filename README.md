# GC-MS compounds first alignment

Pipeline Python (Google Colab) per GC-MS TOF HR centroid con approccio compound-first:
- deconvoluzione → composti locali (spettro + RT/RI + area coerente)
- allineamento tra campioni: RT/RI + similarità spettrale + vincolo Δlog10(area)
- core compounds (prevalenza) con rescue attivo (matched-filter) e soglie adattive
- blank handling (filtro/ratio)
- NIST opzionale e posticipata
- output finale: .xlsx

## Quick start (Colab)
1) Clona il repo
2) Installa requirements.txt
3) Esegui il notebook in `notebooks/colab_runner.ipynb`
