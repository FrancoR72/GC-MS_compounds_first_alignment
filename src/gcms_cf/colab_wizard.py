# Colab UI Wizard for GC-MS TOF HR mzData.zip
# Uso:
#   import sys
#   sys.path.insert(0, "/content/GC-MS_compounds_first_alignment/src")
#   from gcms_cf.colab_wizard import launch
#   launch()

from __future__ import annotations

import os, sys, re, json, hashlib, pickle, shutil, zipfile, subprocess
from pathlib import Path
import importlib

import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm.auto import tqdm

try:
    from google.colab import files  # type: ignore
except Exception:
    files = None  # type: ignore


def launch(
    project_root: str = "/content/GC-MS_compounds_first_alignment",
    *,
    output_xlsx_name: str = "FEATURE_TABLE_UI.xlsx",
) -> None:
    if files is None:
        raise RuntimeError("Questo wizard richiede Google Colab (google.colab.files non disponibile).")

    BASE = Path(project_root)
    SRC  = BASE / "src"

    DATA_RAW_ZIP   = BASE / "data" / "raw" / "incoming_zip"
    DATA_EXTRACTED = BASE / "data" / "raw" / "extracted"
    DATA_OUTPUT    = BASE / "data" / "output"
    CACHE_DIR      = BASE / ".cache_pipeline" / "ui_cache"

    for d in [DATA_RAW_ZIP, DATA_EXTRACTED, DATA_OUTPUT, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    state = {
        "xml_files": [],
        "all_compounds": [],
        "last_excel": None,
    }

    def ensure_imports(out: widgets.Output) -> bool:
        with out:
            print("📦 Controllo import del progetto...")
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        importlib.invalidate_caches()
        try:
            import gcms_cf  # noqa
            with out:
                print("✅ Import gcms_cf OK.")
            return True
        except Exception:
            with out:
                print("ℹ️ Import non disponibile: provo pip install -e .")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(BASE)], check=True)
                importlib.invalidate_caches()
                import gcms_cf  # noqa
                with out:
                    print("✅ Install/editable + import OK.")
                return True
            except Exception as e:
                with out:
                    print("❌ Import fallito:", repr(e))
                return False

    def params_hash(d: dict) -> str:
        payload = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    def guess_sample_id(p: Path) -> str:
        name = p.name
        m = re.search(r"(AU\\d+)", name)
        if m:
            return m.group(1)
        stem = p.stem.replace(".mzdata", "")
        return stem

    def refresh_file_list():
        xmls = sorted(DATA_EXTRACTED.rglob("*.mzdata.xml"))
        file_select.options = [(p.name, str(p)) for p in xmls]
        state["xml_files"] = xmls
        btn_run_extract.disabled = (len(xmls) == 0)

    def get_selected_files():
        chosen = list(file_select.value)
        if chosen:
            return [Path(p) for p in chosen]
        return list(state["xml_files"])

    def make_param_row(slider, text_box, help_html):
        slider.layout = widgets.Layout(flex="3 1 auto", width="auto")
        text_box.layout = widgets.Layout(width="120px")
        help_html.layout = widgets.Layout(flex="4 1 auto", width="auto")
        row = widgets.HBox([slider, text_box, help_html],
                           layout=widgets.Layout(width="100%", justify_content="space-between", align_items="center"))
        widgets.link((slider, "value"), (text_box, "value"))
        return row

    def set_button_full(btn, minw=420):
        btn.layout = widgets.Layout(width="100%", min_width=f"{minw}px")
        btn.style.button_width = "auto"

    # ---------------- UI ----------------
    title = widgets.HTML(
        "<h3>GC-MS TOF HR Wizard (mzData.xml dentro ZIP) — Colab</h3>"
        "<div style='padding:8px 10px; background:#fff3cd; border:1px solid #ffeeba; border-radius:8px;'>"
        "<b>Run di Colab:</b> premi ▶️ (Run) sulla cella che chiama <code>launch()</code>. "
        "Poi usa i bottoni <b>RUN Step 1</b> e <b>RUN Step 2</b> qui sotto."
        "</div>"
    )

    out = widgets.Output(layout={"border":"1px solid #ddd","padding":"10px","max_height":"420px","overflow_y":"auto"})

    lbl_step1 = widgets.HTML(
        "<span style='display:inline-block; padding:6px 10px; background:#1a73e8; color:white; "
        "border-radius:6px; font-weight:600;'>1) Carica ZIP</span>"
    )

    uploader = widgets.FileUpload(accept=".zip", multiple=False, description="Scegli file", button_style="primary")
    uploader.layout = widgets.Layout(width="220px")

    chk_clean_extract = widgets.Checkbox(value=True, description="Pulisci 'extracted' prima dell’unzip")
    btn_refresh_files = widgets.Button(description="Aggiorna lista file", icon="refresh")
    btn_refresh_files.layout = widgets.Layout(width="220px")

    lbl_zip = widgets.HTML("<b>ZIP:</b> nessuno")

    file_select = widgets.SelectMultiple(options=[], description="File mzData", rows=7, layout=widgets.Layout(width="100%"))

    btn_run_extract = widgets.Button(description="RUN Step 1 — Estrai compound", button_style="success", icon="play")
    btn_run_align   = widgets.Button(description="RUN Step 2 — Allinea + Excel", button_style="warning", icon="cogs")
    btn_download    = widgets.Button(description="Scarica Excel", icon="download")
    for b in [btn_run_extract, btn_run_align, btn_download]:
        set_button_full(b, minw=420)

    btn_run_align.disabled = True
    btn_download.disabled = True
    btn_run_extract.disabled = True

    # ---------- Step 1 params ----------
    s_top_k = widgets.IntSlider(value=250, min=20, max=400, step=10, description="TIC windows", continuous_update=False)
    t_top_k = widgets.IntText(value=250)
    h_top_k = widgets.HTML("<small>Finestre TIC analizzate (↑ = più compound, ↑ tempo).</small>")

    s_half = widgets.FloatSlider(value=0.35, min=0.10, max=0.80, step=0.05, description="half_width", continuous_update=False)
    t_half = widgets.FloatText(value=0.35)
    h_half = widgets.HTML("<small>Finestra RT: apice ± half_width (min).</small>")

    s_ppm = widgets.FloatSlider(value=10.0, min=3.0, max=20.0, step=1.0, description="ppm_tol", continuous_update=False)
    t_ppm = widgets.FloatText(value=10.0)
    h_ppm = widgets.HTML("<small>Tolleranza ±ppm per EIC e match nello scan.</small>")

    s_seeds = widgets.IntSlider(value=300, min=60, max=600, step=20, description="top_seeds", continuous_update=False)
    t_seeds = widgets.IntText(value=300)
    h_seeds = widgets.HTML("<small>Seed m/z dallo scan apice.</small>")

    s_hitrt = widgets.FloatSlider(value=0.08, min=0.03, max=0.15, step=0.01, description="hit_rt_tol", continuous_update=False)
    t_hitrt = widgets.FloatText(value=0.08)
    h_hitrt = widgets.HTML("<small>Accetta hit se apex EIC entro ±hit_rt_tol dall’apice TIC.</small>")

    s_hitrel = widgets.FloatSlider(value=0.04, min=0.01, max=0.15, step=0.01, description="hit_rel_h", continuous_update=False)
    t_hitrel = widgets.FloatText(value=0.04)
    h_hitrel = widgets.HTML("<small>Soglia peak-picking EIC (↓ = più hits).</small>")

    s_maxhits = widgets.IntSlider(value=150, min=30, max=300, step=10, description="max_hits", continuous_update=False)
    t_maxhits = widgets.IntText(value=150)
    h_maxhits = widgets.HTML("<small>Max hit per finestra.</small>")

    s_corr = widgets.FloatSlider(value=0.78, min=0.60, max=0.95, step=0.01, description="min_corr", continuous_update=False)
    t_corr = widgets.FloatText(value=0.78)
    h_corr = widgets.HTML("<small>Correlazione EIC per cluster.</small>")

    s_cmpw = widgets.IntSlider(value=5, min=1, max=10, step=1, description="cmp/window", continuous_update=False)
    t_cmpw = widgets.IntText(value=5)
    h_cmpw = widgets.HTML("<small>Max compound per finestra.</small>")

    s_minions = widgets.IntSlider(value=5, min=3, max=12, step=1, description="min_ions", continuous_update=False)
    t_minions = widgets.IntText(value=5)
    h_minions = widgets.HTML("<small>Min ioni nel cluster.</small>")

    s_areaf = widgets.FloatSlider(value=0.05, min=0.01, max=0.50, step=0.01, description="area_frac", continuous_update=False)
    t_areaf = widgets.FloatText(value=0.05)
    h_areaf = widgets.HTML("<small>Cluster secondari: area ≥ area_frac × area_best.</small>")

    s_scanpur = widgets.FloatSlider(value=0.02, min=0.00, max=0.30, step=0.01, description="scan_pur", continuous_update=False)
    t_scanpur = widgets.FloatText(value=0.02)
    h_scanpur = widgets.HTML("<small>Min scan_purity (matched_sum/TICscan).</small>")

    s_minmatch = widgets.IntSlider(value=4, min=1, max=20, step=1, description="min_match", continuous_update=False)
    t_minmatch = widgets.IntText(value=4)
    h_minmatch = widgets.HTML("<small>Min picchi matchati nello scan.</small>")

    step1 = widgets.VBox([
        widgets.HTML("<b>Step 1 — Estrazione compound</b>"),
        make_param_row(s_top_k, t_top_k, h_top_k),
        make_param_row(s_half, t_half, h_half),
        make_param_row(s_ppm, t_ppm, h_ppm),
        make_param_row(s_seeds, t_seeds, h_seeds),
        make_param_row(s_hitrt, t_hitrt, h_hitrt),
        make_param_row(s_hitrel, t_hitrel, h_hitrel),
        make_param_row(s_maxhits, t_maxhits, h_maxhits),
        make_param_row(s_corr, t_corr, h_corr),
        make_param_row(s_cmpw, t_cmpw, h_cmpw),
        make_param_row(s_minions, t_minions, h_minions),
        make_param_row(s_areaf, t_areaf, h_areaf),
        make_param_row(s_scanpur, t_scanpur, h_scanpur),
        make_param_row(s_minmatch, t_minmatch, h_minmatch),
    ])

    # ---------- Step 2 params ----------
    s_mzppm = widgets.FloatSlider(value=10.0, min=3.0, max=20.0, step=1.0, description="mz_ppm", continuous_update=False)
    t_mzppm = widgets.FloatText(value=10.0)
    h_mzppm = widgets.HTML("<small>Matching m/z in ppm per allineamento.</small>")

    s_rtS = widgets.FloatSlider(value=0.35, min=0.05, max=1.00, step=0.05, description="RT strict", continuous_update=False)
    t_rtS = widgets.FloatText(value=0.35)
    h_rtS = widgets.HTML("<small>Tolleranza RT strict per match tra campioni.</small>")

    s_rtR = widgets.FloatSlider(value=0.55, min=0.10, max=1.50, step=0.05, description="RT relax", continuous_update=False)
    t_rtR = widgets.FloatText(value=0.55)
    h_rtR = widgets.HTML("<small>Tolleranza RT per rescue.</small>")

    s_cosS = widgets.FloatSlider(value=0.75, min=0.50, max=0.95, step=0.01, description="cos strict", continuous_update=False)
    t_cosS = widgets.FloatText(value=0.75)
    h_cosS = widgets.HTML("<small>Cosine minima strict.</small>")

    s_cosR = widgets.FloatSlider(value=0.70, min=0.40, max=0.95, step=0.01, description="cos relax", continuous_update=False)
    t_cosR = widgets.FloatText(value=0.70)
    h_cosR = widgets.HTML("<small>Cosine minima rescue.</small>")

    s_dlogS = widgets.FloatSlider(value=1.0, min=0.2, max=3.0, step=0.1, description="dlog strict", continuous_update=False)
    t_dlogS = widgets.FloatText(value=1.0)
    h_dlogS = widgets.HTML("<small>Δlog10(area) max strict (1.0≈10×).</small>")

    s_dlogR = widgets.FloatSlider(value=2.0, min=0.2, max=4.0, step=0.1, description="dlog relax", continuous_update=False)
    t_dlogR = widgets.FloatText(value=2.0)
    h_dlogR = widgets.HTML("<small>Δlog10(area) max rescue (2.0≈100×).</small>")

    s_core = widgets.FloatSlider(value=0.67, min=0.34, max=1.0, step=0.01, description="core_frac", continuous_update=False)
    t_core = widgets.FloatText(value=0.67)
    h_core = widgets.HTML("<small>Core: con 3 campioni, 0.67 ≈ presenti in ≥2/3.</small>")

    txt_maxscore = widgets.Text(value="1.2", description="max_score")
    txt_margin   = widgets.Text(value="0.15", description="min_margin")
    h_freni = widgets.HTML("<small>Freni rescue: svuota per disattivarli.</small>")

    step2 = widgets.VBox([
        widgets.HTML("<b>Step 2 — Allineamento + Core/Rescue + Excel</b>"),
        make_param_row(s_mzppm, t_mzppm, h_mzppm),
        make_param_row(s_rtS, t_rtS, h_rtS),
        make_param_row(s_rtR, t_rtR, h_rtR),
        make_param_row(s_cosS, t_cosS, h_cosS),
        make_param_row(s_cosR, t_cosR, h_cosR),
        make_param_row(s_dlogS, t_dlogS, h_dlogS),
        make_param_row(s_dlogR, t_dlogR, h_dlogR),
        make_param_row(s_core, t_core, h_core),
        widgets.HBox([txt_maxscore, txt_margin, h_freni], layout=widgets.Layout(width="100%", align_items="center"))
    ])

    accordion = widgets.Accordion(children=[step1, step2])
    accordion.set_title(0, "Parametri Step 1")
    accordion.set_title(1, "Parametri Step 2")
    accordion.selected_index = 0

    def on_zip_uploaded(change):
        if not change["new"]:
            return
        with out:
            clear_output()
            print("📥 ZIP ricevuto. Salvo e decomprimo...")

        value = uploader.value
        if not isinstance(value, dict) or len(value) == 0:
            with out:
                print("❌ Upload vuoto. Riprova.")
            return

        zip_name = list(value.keys())[0]
        content = value[zip_name]["content"]
        dst = DATA_RAW_ZIP / zip_name
        with open(dst, "wb") as f:
            f.write(content)

        lbl_zip.value = f"<b>ZIP:</b> {zip_name} ({dst.stat().st_size/1e6:.1f} MB)"

        if chk_clean_extract.value:
            shutil.rmtree(DATA_EXTRACTED, ignore_errors=True)
            DATA_EXTRACTED.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(dst, "r") as z:
                z.extractall(DATA_EXTRACTED)
        except Exception as e:
            with out:
                print("❌ Unzip fallito:", repr(e))
            return

        refresh_file_list()
        with out:
            print(f"✅ Unzip completato. File mzData trovati: {len(state['xml_files'])}")
            print("Ora premi: RUN Step 1 — Estrai compound")

    def run_extract(_):
        if not ensure_imports(out):
            return
        selected = get_selected_files()
        if not selected:
            with out:
                print("❌ Nessun file mzData trovato. Carica lo ZIP.")
            return

        step1_params = dict(
            top_k_windows=int(s_top_k.value),
            half_width_min=float(s_half.value),
            ppm_tol=float(s_ppm.value),
            top_n_seeds=int(s_seeds.value),
            hit_rt_tol=float(s_hitrt.value),
            hit_min_rel_height=float(s_hitrel.value),
            max_hits=int(s_maxhits.value),
            min_corr=float(s_corr.value),
            max_compounds_per_window=int(s_cmpw.value),
            min_cluster_ions=int(s_minions.value),
            min_cluster_area_frac=float(s_areaf.value),
            min_scan_purity=float(s_scanpur.value),
            min_matched_peaks=int(s_minmatch.value),
        )
        ph = params_hash(step1_params)

        from gcms_cf.hr_deconv_extract import extract_compounds_from_mzdata

        compounds_by_sample = {}

        with out:
            clear_output()
            print("🚀 STEP 1 — Estrazione compound")
            print("Parametri:", step1_params)
            print("Cache key:", ph)
            print("—")

        for xml_path in tqdm(selected, desc="Estrazione per file"):
            sample_id = guess_sample_id(xml_path)
            pkl_c = CACHE_DIR / f"{sample_id}_compounds_{ph}.pkl"

            if pkl_c.exists():
                with open(pkl_c, "rb") as f:
                    comps = pickle.load(f)
                with out:
                    print(f"[CACHE] {sample_id}: compounds={len(comps)}")
            else:
                comps, _summary = extract_compounds_from_mzdata(str(xml_path), sample_id=sample_id, **step1_params)
                with open(pkl_c, "wb") as f:
                    pickle.dump(comps, f)
                with out:
                    print(f"[RUN]   {sample_id}: compounds={len(comps)}")

            compounds_by_sample[sample_id] = comps

        state["all_compounds"] = [c for sid in compounds_by_sample for c in compounds_by_sample[sid]]

        with out:
            print("\\n✅ STEP 1 completato. Totale compounds:", len(state["all_compounds"]))
            print("➡️ Ora premi: RUN Step 2 — Allinea + Excel")

        btn_run_align.disabled = False
        accordion.selected_index = 1

    def to_float_or_none(s: str):
        try:
            ss = str(s).strip()
            if ss == "":
                return None
            return float(ss)
        except Exception:
            return None

    def run_align(_):
        if not ensure_imports(out):
            return
        all_compounds = state.get("all_compounds", [])
        if not all_compounds:
            with out:
                print("❌ Nessun compound in memoria. Esegui Step 1.")
            return

        step2_params = dict(
            mz_ppm=float(s_mzppm.value),
            rt_tol_strict=float(s_rtS.value),
            rt_tol_relax=float(s_rtR.value),
            min_cosine_strict=float(s_cosS.value),
            min_cosine_relax=float(s_cosR.value),
            max_dlog10_area_strict=float(s_dlogS.value),
            max_dlog10_area_relax=float(s_dlogR.value),
            core_frac=float(s_core.value),
            max_rescue_score=to_float_or_none(txt_maxscore.value),
            min_score_margin=to_float_or_none(txt_margin.value),
        )

        from gcms_cf.alignment import align_compounds_clusters, clusters_to_table
        from gcms_cf.core_compounds import add_core_flags_df
        from gcms_cf.rescue import rescue_core_missing
        from gcms_cf.export_xlsx import export_feature_table

        compounds_by_sample = {}
        for c in all_compounds:
            compounds_by_sample.setdefault(c.sample_id, []).append(c)
        sample_ids = sorted(compounds_by_sample.keys())

        with out:
            clear_output()
            print("🧩 STEP 2 — Allineamento + Core/Rescue + Excel")
            print("Sample IDs:", sample_ids)
            print("Parametri:", step2_params)
            print("—")

        clusters = align_compounds_clusters(
            all_compounds,
            rt_tol=step2_params["rt_tol_strict"],
            use_ri=False,
            ri_tol=20.0,
            area_agg="max",
            min_cosine=step2_params["min_cosine_strict"],
            mz_tol=0.01,
            mz_ppm=step2_params["mz_ppm"],
            max_dlog10_area=step2_params["max_dlog10_area_strict"],
        )

        df_strict = clusters_to_table(clusters, fill_missing=0.0)
        df_strict2, sample_cols = add_core_flags_df(df_strict, core_frac=step2_params["core_frac"])
        core_ids = list(df_strict2[df_strict2["is_core"] == True]["feature_id"].astype(str))
        with out:
            print(f"Feature totali (strict): {len(df_strict)}")
            print(f"Core (>= {step2_params['core_frac']:.2f}): {len(core_ids)}")

        rescued_map = rescue_core_missing(
            clusters,
            compounds_by_sample,
            core_feature_ids=core_ids,
            sample_ids=sample_ids,
            area_agg="max",
            rt_tol=step2_params["rt_tol_relax"],
            use_ri=False,
            ri_tol=20.0,
            min_cosine=step2_params["min_cosine_relax"],
            mz_tol=0.01,
            mz_ppm=step2_params["mz_ppm"],
            max_dlog10_area=step2_params["max_dlog10_area_relax"],
            max_rescue_score=step2_params["max_rescue_score"],
            min_score_margin=step2_params["min_score_margin"],
        )

        df_final = clusters_to_table(clusters, fill_missing=0.0)

        df_final = df_final.merge(
            df_strict2[["feature_id", "n_present", "presence_frac", "is_core"]],
            on="feature_id",
            how="left",
        ).rename(columns={"n_present":"n_present_strict", "presence_frac":"presence_frac_strict"})

        present_final = (df_final[sample_cols] > 0.0)
        df_final["n_present_final"] = present_final.sum(axis=1)
        df_final["presence_frac_final"] = df_final["n_present_final"] / float(len(sample_cols))

        df_final["n_rescued"] = df_final["feature_id"].map(lambda fid: len(rescued_map.get(fid, []))).fillna(0).astype(int)
        df_final["rescued_samples"] = df_final["feature_id"].map(lambda fid: ";".join(rescued_map.get(fid, [])) if fid in rescued_map else "")

        df_final = df_final.sort_values(
            ["is_core","presence_frac_final","rt_ref"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        out_xlsx = DATA_OUTPUT / output_xlsx_name
        export_feature_table(df_final, str(out_xlsx), sheet_name="FeatureTable")
        state["last_excel"] = str(out_xlsx)

        n_resc_feat = sum(1 for v in rescued_map.values() if v)
        with out:
            print(f"Feature con almeno 1 rescue: {n_resc_feat}")
            print(f"✅ Excel creato: {out_xlsx}")
            display(df_final.head(15))
            print("➡️ Ora premi: Scarica Excel")

        btn_download.disabled = False

    def download_excel(_):
        p = state.get("last_excel")
        if not p or not os.path.exists(p):
            with out:
                print("❌ Nessun Excel disponibile. Esegui prima lo Step 2.")
            return
        files.download(p)

    def refresh_files_btn(_):
        refresh_file_list()
        with out:
            print(f"🔎 File aggiornati: {len(state['xml_files'])} mzData trovati.")
        btn_run_extract.disabled = (len(state["xml_files"]) == 0)

    # Bind
    uploader.observe(on_zip_uploaded, names="value")
    btn_run_extract.on_click(run_extract)
    btn_run_align.on_click(run_align)
    btn_download.on_click(download_excel)
    btn_refresh_files.on_click(refresh_files_btn)

    refresh_file_list()

    top_row = widgets.HBox(
        [lbl_step1, uploader, chk_clean_extract, btn_refresh_files],
        layout=widgets.Layout(width="100%", justify_content="flex-start", align_items="center", column_gap="14px")
    )
    left_col = widgets.VBox([lbl_zip, file_select], layout=widgets.Layout(width="62%"))
    right_col = widgets.VBox([btn_run_extract, btn_run_align, btn_download], layout=widgets.Layout(width="38%"))
    mid_row = widgets.HBox([left_col, right_col], layout=widgets.Layout(width="100%", column_gap="16px"))

    display(title, top_row, mid_row, accordion, out)

    with out:
        print("👉 Flusso:")
        print("1) Carica ZIP con 'Scegli file' (in alto).")
        print("2) Premi 'RUN Step 1 — Estrai compound'.")
        print("3) Premi 'RUN Step 2 — Allinea + Excel' (si abilita dopo Step 1).")
        print("4) Premi 'Scarica Excel'.")
