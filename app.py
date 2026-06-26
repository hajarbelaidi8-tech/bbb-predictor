"""
BBB Predictor · v3.4 · mediane_train integrated
"""

import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, SaltRemover,
                        rdDepictor, rdMolDescriptors)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBB Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --cb:   #2563EB; --cb2: #EFF6FF; --cb3: rgba(37,99,235,.12);
  --cg:   #059669; --cg2: #ECFDF5;
  --cr:   #DC2626; --cr2: #FEF2F2;
  --ca:   #D97706; --ca2: #FFFBEB;
  --cv:   #7C3AED; --cv2: #F5F3FF;
  --ct1:  #111827; --ct2: #6B7280; --ct3: #9CA3AF;
  --bg:   #FFFFFF; --bg2: #F8F9FA; --bg3: #F1F3F5;
  --bdr:  #DEE2E6;
  --fd:   'Outfit',sans-serif;
  --fs:   'Inter',sans-serif;
  --fm:   'JetBrains Mono',monospace;
  --r:    10px; --rs: 8px;
  --sh:   0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);
}
.stApp { background: #fff !important; font-family: 'Inter',sans-serif; }
section[data-testid="stSidebar"] {
  background: #F8F9FA !important;
  border-right: 1px solid #DEE2E6 !important;
}
.block-container { padding: 1.8rem 2.2rem !important; max-width: 100% !important; }
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"] {
  display: none !important;
}
</style>
<script>
  document.documentElement.setAttribute('lang', 'en');
  document.documentElement.classList.add('notranslate');
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
_FP_COLS = [f"fp_{i}" for i in range(2048)]
_RDKIT_COLS = [
    'MaxAbsEStateIndex','MaxEStateIndex','MinAbsEStateIndex','MinEStateIndex',
    'qed','SPS','MolWt','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons',
    'NumRadicalElectrons','MaxPartialCharge','MinPartialCharge',
    'MaxAbsPartialCharge','MinAbsPartialCharge','FpDensityMorgan1',
    'FpDensityMorgan2','FpDensityMorgan3','BCUT2D_MWHI','BCUT2D_MWLOW',
    'BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW',
    'BCUT2D_MRHI','BCUT2D_MRLOW','AvgIpc','BalabanJ','BertzCT',
    'Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v',
    'Chi3n','Chi3v','Chi4n','Chi4v','HallKierAlpha','Ipc','Kappa1',
    'Kappa2','Kappa3','LabuteASA','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11',
    'PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3',
    'PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8',
    'PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4',
    'SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SlogP_VSA1',
    'SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3',
    'SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8',
    'TPSA','EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2',
    'EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7',
    'EState_VSA8','EState_VSA9','VSA_EState1','VSA_EState10','VSA_EState2',
    'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
    'VSA_EState8','VSA_EState9','FractionCSP3','HeavyAtomCount',
    'NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles',
    'NumAliphaticRings','NumAmideBonds','NumAromaticCarbocycles',
    'NumAromaticHeterocycles','NumAromaticRings','NumAtomStereoCenters',
    'NumBridgeheadAtoms','NumHAcceptors','NumHDonors','NumHeteroatoms',
    'NumHeterocycles','NumRotatableBonds','NumSaturatedCarbocycles',
    'NumSaturatedHeterocycles','NumSaturatedRings','NumSpiroAtoms',
    'NumUnspecifiedAtomStereoCenters','Phi','RingCount','MolLogP','MolMR',
    'fr_Al_COO','fr_Al_OH','fr_Al_OH_noTert','fr_ArN','fr_Ar_COO','fr_Ar_N',
    'fr_Ar_NH','fr_Ar_OH','fr_COO','fr_COO2','fr_C_O','fr_C_O_noCOO',
    'fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2','fr_N_O',
    'fr_Ndealkylation1','fr_Ndealkylation2','fr_Nhpyrrole','fr_SH',
    'fr_aldehyde','fr_alkyl_carbamate','fr_alkyl_halide','fr_allylic_oxid',
    'fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl','fr_azide',
    'fr_azo','fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic',
    'fr_dihydropyridine','fr_epoxide','fr_ester','fr_ether','fr_furan',
    'fr_guanido','fr_halogen','fr_hdrzine','fr_hdrzone','fr_imidazole',
    'fr_imide','fr_ketone','fr_ketone_Topliss','fr_lactam','fr_lactone',
    'fr_methoxy','fr_morpholine','fr_nitrile','fr_nitro','fr_nitro_arom',
    'fr_nitro_arom_nonortho','fr_nitroso','fr_oxazole','fr_oxime',
    'fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond',
    'fr_phos_acid','fr_phos_ester','fr_piperdine','fr_piperzine',
    'fr_priamide','fr_prisulfonamd','fr_pyridine','fr_quatN','fr_sulfide',
    'fr_sulfonamd','fr_sulfone','fr_term_acetylene','fr_tetrazole',
    'fr_thiazole','fr_thiophene','fr_unbrch_alkane','fr_urea',
]
_DESC_NAMES = [d[0] for d in Descriptors._descList]
LIPINSKI = {"MW": 500, "LogP": 5, "TPSA": 90, "HBD": 5, "HBA": 10, "RotB": 10}
_PROP_META = {
    "MW":    ("Mol. Weight (MW)",          "Da",  500),
    "LogP":  ("Lipophilicity (LogP)",      "",    5),
    "TPSA":  ("Polar Surface Area (TPSA)", "Å²",  90),
    "HBD":   ("H-Bond Donors (HBD)",       "",    5),
    "HBA":   ("H-Bond Acceptors (HBA)",    "",    10),
    "RotB":  ("Rotatable Bonds",           "",    10),
    "Rings": ("Ring Count",                "",    None),
    "QED":   ("Drug-Likeness (QED)",       "",    None),
}
_PB = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(family="Inter,sans-serif", color="#6B7280", size=11))
_M  = dict(l=10, r=10, t=30, b=10)


# ══════════════════════════════════════════════════════════════════
# CHEMISTRY
# ══════════════════════════════════════════════════════════════════
def _h(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]

def strip_salts(mol):
    rem = SaltRemover.SaltRemover()
    try:
        mol = rem.StripMol(mol, dontRemoveEverything=True)
    except:
        pass
    frags = Chem.GetMolFrags(mol, asMols=True)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms()) if frags else mol

def mol_to_svg(mol, w=440, h=320):
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.padding, o.bondLineWidth, o.backgroundColour, o.addStereoAnnotation = \
        0.12, 1.8, (1, 1, 1, 1), True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return svg[svg.index("<svg"):] if "<svg" in svg else svg

def smiles_from_pubchem(name):
    try:
        r = pcp.get_compounds(name.strip(), "name")
        if r:
            return r[0].smiles
    except Exception as e:
        st.error(f"Erreur PubChem : {e}")
    return None

def compute_features(smiles, colonnes, mediane_train):
    """
    Calcule les features et impute les NaN avec mediane_train
    (identique à ce qui a été fait pendant l'entraînement).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = strip_salts(mol)

    # Morgan Fingerprints (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    feat = {f"fp_{i}": int(fp[i]) for i in range(2048)}

    # Descripteurs RDKit
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
    feat.update(dict(zip(_DESC_NAMES, calc.CalcDescriptors(mol))))

    # Construction du DataFrame aligné sur les colonnes du train
    row = {c: feat.get(c, np.nan) for c in colonnes}
    df  = pd.DataFrame([row])[colonnes]

    # Remplacement inf par NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # ✅ Imputation avec la médiane du TRAIN (cohérent avec l'entraînement)
    # On n'impute que les colonnes présentes dans mediane_train
    for col in df.columns:
        if df[col].isna().any():
            if col in mediane_train.index:
                df[col] = df[col].fillna(mediane_train[col])
            else:
                df[col] = df[col].fillna(0.0)

    return df, mol

def physchem(mol):
    return {
        "MW":    round(Descriptors.MolWt(mol), 2),
        "LogP":  round(Descriptors.MolLogP(mol), 3),
        "TPSA":  round(Descriptors.TPSA(mol), 2),
        "HBD":   rdMolDescriptors.CalcNumHBD(mol),
        "HBA":   rdMolDescriptors.CalcNumHBA(mol),
        "RotB":  rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": Descriptors.RingCount(mol),
        "QED":   round(Descriptors.qed(mol), 3),
    }

def run_prediction(smiles, model, scaler, colonnes, mediane_train):
    """Prédit la perméabilité BBB avec imputation correcte."""
    X, mol = compute_features(smiles, colonnes, mediane_train)
    if X is None:
        return None, None, None, None
    Xs   = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    prob = model.predict_proba(Xs)[0]
    return pred, prob, physchem(mol), mol


# ══════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════
def _hdr(eyebrow, title, sub=""):
    st.markdown(
        f'<p style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.12em;color:var(--cb);margin-bottom:.2rem">{eyebrow}</p>'
        f'<h1 style="font-family:var(--fd);font-size:1.7rem;font-weight:800;'
        f'color:var(--ct1);margin:0 0 .3rem;line-height:1.2">{title}</h1>'
        + (f'<p style="font-size:.9rem;color:var(--ct2);margin-bottom:1.2rem">{sub}</p>' if sub else ""),
        unsafe_allow_html=True,
    )

def _sec(label):
    st.markdown(
        f'<p style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.12em;color:var(--cb);margin:.8rem 0 .5rem;'
        f'display:flex;align-items:center;gap:.5rem">'
        f'{label}<span style="flex:1;height:1px;background:#DEE2E6;display:inline-block"></span></p>',
        unsafe_allow_html=True,
    )

def _kpi(col, color, icon, val, lbl, sub):
    with col:
        st.markdown(
            f'<div style="background:#fff;border:1px solid #DEE2E6;border-top:3px solid {color};'
            f'border-radius:10px;padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(0,0,0,.07)">'
            f'<div style="font-size:1rem;margin-bottom:.5rem">{icon}</div>'
            f'<div style="font-family:var(--fd);font-size:1.85rem;font-weight:900;color:{color}">{val}</div>'
            f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#9CA3AF;margin-top:.3rem">{lbl}</div>'
            f'<div style="font-size:.77rem;color:#6B7280;margin-top:.15rem">{sub}</div></div>',
            unsafe_allow_html=True,
        )

def _prop_table(props):
    rows = ""
    for k, v in props.items():
        label, unit, lim = _PROP_META.get(k, (k, "", None))
        badge = ""
        if lim is not None:
            ok = v <= lim
            c, bg, bd = ("#059669","#ECFDF5","rgba(5,150,105,.2)") if ok \
                   else ("#DC2626","#FEF2F2","rgba(220,38,38,.2)")
            txt = "✓ OK" if ok else f"✗ &gt;{lim}"
            badge = (f'<span style="font-size:.65rem;font-weight:700;padding:2px 7px;'
                     f'border-radius:100px;background:{bg};color:{c};border:1px solid {bd};'
                     f'margin-left:.4rem">{txt}</span>')
        rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:.52rem .9rem;border-bottom:1px solid #F1F3F5;font-size:.83rem">'
            f'<span style="color:#6B7280;font-weight:500">{label}</span>'
            f'<span style="display:flex;align-items:center">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;color:#111827;font-weight:600">'
            f'{v} {unit}</span>{badge}</span></div>'
        )
    st.markdown(
        f'<div style="background:#fff;border:1px solid #DEE2E6;border-radius:10px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,.07);overflow:hidden">{rows}</div>',
        unsafe_allow_html=True,
    )

def _verdict(is_pos, prob_pos):
    bg  = "#ECFDF5" if is_pos else "#FEF2F2"
    bdc = "rgba(5,150,105,.25)" if is_pos else "rgba(220,38,38,.25)"
    col = "#059669" if is_pos else "#DC2626"
    em  = "✅" if is_pos else "❌"
    lbl = "BBB⁺ Perméable" if is_pos else "BBB⁻ Non perméable"
    msg = ("Devrait traverser la BBB (SNC accessible)."
           if is_pos else "Ne devrait pas pénétrer le SNC.")
    st.markdown(
        f'<div style="background:{bg};border:1px solid {bdc};border-radius:10px;'
        f'padding:1.5rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.07)">'
        f'<div style="font-size:2.4rem;line-height:1;margin-bottom:.4rem">{em}</div>'
        f'<div style="font-family:var(--fd);font-size:1.1rem;font-weight:800;color:{col}">{lbl}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.8rem;font-weight:700;'
        f'color:{col};margin-top:.35rem">{prob_pos*100:.1f}%</div>'
        f'<div style="font-size:.78rem;color:#6B7280;margin-top:.35rem;line-height:1.5">{msg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _mol_frame(mol, w=440, h=320):
    try:
        svg = mol_to_svg(mol, w, h)
        st.markdown(
            f'<div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:10px;'
            f'padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.07)">{svg}</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Rendu 2D indisponible : {e}")


# ══════════════════════════════════════════════════════════════════
# PLOTLY
# ══════════════════════════════════════════════════════════════════
def _gauge(prob_pos, is_pos):
    col = "#059669" if is_pos else "#DC2626"
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number", value=round(float(prob_pos)*100, 1),
        number=dict(suffix="%", font=dict(size=28, color=col, family="Inter")),
        title=dict(text="Probabilité BBB⁺", font=dict(size=12, color="#6B7280")),
        gauge=dict(
            axis=dict(range=[0,100], tickfont=dict(size=9, color="#9CA3AF")),
            bar=dict(color=col, thickness=0.22),
            bgcolor="#F8F9FA", borderwidth=1, bordercolor="#DEE2E6",
            steps=[dict(range=[0,50],   color="rgba(220,38,38,0.06)"),
                   dict(range=[50,100], color="rgba(5,150,105,0.06)")],
            threshold=dict(line=dict(color="#D97706",width=2), thickness=0.8, value=50),
        ),
    ))
    fig.update_layout(**_PB, margin=_M, height=210)
    return fig

def _radar(props, is_pos):
    keys = ["MW","LogP","TPSA","HBD","HBA","RotB"]
    norm = [min(float(props[k])/LIPINSKI[k], 1.6) for k in keys]
    kc   = keys+[keys[0]]; nc = norm+[norm[0]]; lc = [1.0]*(len(keys)+1)
    col  = "#059669" if is_pos else "#DC2626"
    fc   = "rgba(5,150,105,.10)" if is_pos else "rgba(220,38,38,.10)"
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(r=lc, theta=kc, fill="toself",
        fillcolor="rgba(107,114,128,.06)",
        line=dict(color="#9CA3AF",width=1.5,dash="dot"), name="Lipinski"))
    fig.add_trace(go.Scatterpolar(r=nc, theta=kc, fill="toself", fillcolor=fc,
        line=dict(color=col,width=2.5), marker=dict(size=6,color=col), name="Molécule"))
    fig.update_layout(**_PB, margin=dict(l=20,r=20,t=20,b=55), height=310,
        polar=dict(bgcolor="rgba(248,249,250,.5)",
            radialaxis=dict(visible=True, range=[0,1.6],
                tickvals=[0,.5,1.0,1.5], ticktext=["0","50%","Lim.",">Lim"],
                tickfont=dict(size=8,color="#9CA3AF"),
                gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.06)"),
            angularaxis=dict(tickfont=dict(size=11,color="#6B7280"),
                gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.06)")),
        legend=dict(font=dict(size=10,color="#6B7280"),
            bgcolor="rgba(255,255,255,.95)", bordercolor="#DEE2E6", borderwidth=1,
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5))
    return fig

def _perf_radar():
    cats = ["AUC-ROC","Accuracy","Précision","Rappel","F1"]
    vals = [0.9542,0.9093,0.9056,0.9840,0.9432]
    cc   = cats+[cats[0]]; vc = vals+[vals[0]]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.80]*6, theta=cc, fill="toself",
        fillcolor="rgba(107,114,128,.06)",
        line=dict(color="#9CA3AF",width=1.5,dash="dot"), name="Baseline"))
    fig.add_trace(go.Scatterpolar(r=vc, theta=cc, fill="toself",
        fillcolor="rgba(37,99,235,.10)",
        line=dict(color="#2563EB",width=2.5), marker=dict(size=7,color="#2563EB"),
        name="Modèle"))
    fig.update_layout(**_PB, margin=_M, height=290,
        polar=dict(bgcolor="rgba(248,249,250,.6)",
            radialaxis=dict(visible=True, range=[0.7,1.0],
                tickfont=dict(size=9,color="#9CA3AF"),
                gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.06)"),
            angularaxis=dict(tickfont=dict(size=10,color="#6B7280"),
                gridcolor="rgba(0,0,0,.06)", linecolor="rgba(0,0,0,.06)")),
        legend=dict(font=dict(size=10,color="#6B7280"),
            bgcolor="rgba(255,255,255,.95)", bordercolor="#DEE2E6", borderwidth=1))
    return fig


# ══════════════════════════════════════════════════════════════════
# MODEL — charge les 4 artefacts dont mediane_train
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        m  = joblib.load("model_bbb.pkl")
        s  = joblib.load("scaler_bbb.pkl")
        c  = list(joblib.load("colonnes_finales.pkl"))
        mt = joblib.load("mediane_train.pkl")   # ✅ chargé ici
        return m, s, c, mt
    except FileNotFoundError as e:
        st.warning(f"⚠️ Fichier introuvable : `{e.filename}`", icon="⚠️")
        return None, None, None, None


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding:1.5rem 1.1rem 1.1rem;border-bottom:1px solid #DEE2E6;margin-bottom:.4rem">'
        '<div style="display:flex;align-items:center;gap:.7rem">'
        '<div style="width:34px;height:34px;background:#2563EB;border-radius:8px;'
        'display:flex;align-items:center;justify-content:center;font-size:1rem;'
        'box-shadow:0 2px 8px rgba(37,99,235,.3)">🧬</div>'
        '<div><div style="font-family:\'Outfit\',sans-serif;font-size:.92rem;font-weight:800;'
        'color:#111827;line-height:1.1">BBB Predictor</div>'
        '<div style="font-size:.58rem;color:#9CA3AF;font-weight:600;'
        'letter-spacing:.08em;text-transform:uppercase">Drug Discovery</div>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Page",
        ["🏠 Dashboard", "🔬 Analyse Unique", "📂 Batch Screening", "📊 Propriétés"],
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    info_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:.32rem 0;'
        f'border-bottom:1px solid #F1F3F5;font-size:.76rem">'
        f'<span style="color:#6B7280">{k}</span>'
        f'<span style="color:{vc};font-weight:700;font-family:\'JetBrains Mono\',monospace">{vv}</span></div>'
        for k, vv, vc in [
            ("Modèle","Extra Trees","#DC2626"),
            ("AUC-ROC","0.954","#2563EB"),
            ("Stratégie","Hybride","#7C3AED"),
            ("Morgan bits","2 048","#059669"),
            ("Descripteurs","~200","#D97706"),
        ]
    )
    st.markdown(
        f'<div style="margin:0 .5rem;padding:.85rem;background:#fff;border:1px solid #DEE2E6;'
        f'border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.07)">'
        f'<div style="font-size:.63rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.1em;color:#9CA3AF;margin-bottom:.6rem">Informations modèle</div>'
        f'{info_rows}</div>',
        unsafe_allow_html=True,
    )

# ── Chargement des 4 artefacts ────────────────────────────────────
model, scaler, colonnes, mediane_train = load_artifacts()
if colonnes is None:
    colonnes = _FP_COLS + _RDKIT_COLS
if mediane_train is None:
    # Fallback : Series de zéros (ne devrait pas arriver si pkl présent)
    mediane_train = pd.Series(0.0, index=colonnes)


# ══════════════════════════════════════════════════════════════════
# PAGE ①  DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    _hdr("Vue d'ensemble", "Blood-Brain Barrier Permeability",
         "Prédiction hybride Morgan ECFP4 + RDKit pour le criblage CNS.")

    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1,"#2563EB","🎯","0.954","AUC-ROC","Performance globale")
    _kpi(c2,"#059669","🧩","2 048","Morgan Fingerprints","Radius 2 · ECFP4")
    _kpi(c3,"#7C3AED","📐","~200","Descripteurs RDKit","Physicochimiques")
    _kpi(c4,"#D97706","🧪","BBB+","Cible CNS","Perméabilité SNC")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([5,3], gap="large")

    with left:
        _sec("Contexte scientifique")
        st.markdown(
            '<div style="background:#EFF6FF;border:1px solid rgba(37,99,235,.15);'
            'border-left:3px solid #2563EB;border-radius:8px;'
            'padding:.9rem 1.1rem;font-size:.84rem;color:#6B7280;line-height:1.7;margin-bottom:1.1rem">'
            'La <strong style="color:#2563EB">Barrière Hémato-Encéphalique (BBB)</strong> '
            'régule le passage des xénobiotiques vers le SNC via jonctions serrées et '
            'transporteurs d\'efflux (P-gp, BCRP).<br><br>'
            'Notre pipeline combine <strong style="color:#059669">Morgan ECFP4</strong> '
            'et <strong style="color:#D97706">descripteurs RDKit</strong> '
            'pour capturer topologie moléculaire et propriétés ADMET.</div>',
            unsafe_allow_html=True,
        )
        _sec("Performances du modèle")
        st.plotly_chart(_perf_radar(), use_container_width=True, key="dash_radar")

    with right:
        _sec("Pipeline technique")
        for num, title, desc, col in [
            ("01","Entrée SMILES / Nom","PubChem API ou SMILES directe","#2563EB"),
            ("02","Salt Stripping","Plus grand fragment organique","#7C3AED"),
            ("03","Features Hybrides","2048 Morgan + ~200 RDKit","#D97706"),
            ("04","RobustScaler","Normalisation robuste","#059669"),
            ("05","Prédiction ML","Probabilité BBB+ et verdict","#DC2626"),
        ]:
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:.8rem;padding:.75rem .85rem;'
                f'background:#fff;border:1px solid #DEE2E6;border-radius:8px;margin-bottom:.35rem;'
                f'box-shadow:0 1px 3px rgba(0,0,0,.06)">'
                f'<div style="font-size:.6rem;font-weight:700;min-width:22px;height:22px;'
                f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                f'flex-shrink:0;background:{col}1a;color:{col}">{num}</div>'
                f'<div><div style="font-size:.85rem;font-weight:700;color:#111827">{title}</div>'
                f'<div style="font-size:.74rem;color:#6B7280;margin-top:1px">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        _sec("Critères BBB — Lipinski étendu")
        for prop, lim, col in [
            ("Mol. Weight (MW)","≤ 500 Da","#2563EB"),
            ("Lipophilicity (LogP)","≤ 5","#7C3AED"),
            ("TPSA","≤ 90 Å²","#D97706"),
            ("H-Bond Donors (HBD)","≤ 5","#059669"),
            ("H-Bond Acceptors (HBA)","≤ 10","#2563EB"),
        ]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:.48rem 0;border-bottom:1px solid #F1F3F5;font-size:.83rem">'
                f'<span style="color:#6B7280;font-weight:500">{prop}</span>'
                f'<span style="background:{col}15;color:{col};padding:2px 9px;'
                f'border-radius:100px;font-size:.67rem;font-weight:700">{lim}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# PAGE ②  ANALYSE UNIQUE
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Analyse Unique":
    _hdr("Analyse moléculaire", "Analyse Moléculaire Unique",
         "Prédisez la perméabilité BBB via nom PubChem ou SMILES.")

    m1, m2, _ = st.columns([1.6, 1.6, 6])
    with m1:
        if st.button("🔍 PubChem", key="au_btn_pc_mode",
                     type="primary" if st.session_state.get("_au_mode","pc")=="pc" else "secondary"):
            st.session_state["_au_mode"] = "pc"
            st.session_state.pop("_au_smi_val", None)
    with m2:
        if st.button("✏️ SMILES", key="au_btn_smi_mode",
                     type="primary" if st.session_state.get("_au_mode","pc")=="smi" else "secondary"):
            st.session_state["_au_mode"] = "smi"
            st.session_state.pop("_au_smi_val", None)

    mode = st.session_state.get("_au_mode", "pc")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    smiles_to_run = None

    if mode == "pc":
        cf, cb = st.columns([6,1])
        with cf:
            cname = st.text_input("Nom", placeholder="aspirin · caffeine · donepezil",
                                  key="_au_cname", label_visibility="collapsed")
        with cb:
            search_clicked = st.button("Chercher", key="_au_go_pc")

        if search_clicked and cname.strip():
            with st.spinner("PubChem…"):
                smi = smiles_from_pubchem(cname.strip())
            if smi:
                st.session_state["_au_smi_val"] = smi
                st.session_state.pop("_au_cache", None)
                st.success(f"SMILES · `{smi}`")
            else:
                st.error("Composé introuvable.")
                st.session_state.pop("_au_smi_val", None)

        smiles_to_run = st.session_state.get("_au_smi_val")

    else:
        cf2, cb2 = st.columns([6,1])
        with cf2:
            smi_in = st.text_input("SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O",
                                   key="_au_smi_input", label_visibility="collapsed")
        with cb2:
            analyse_clicked = st.button("Analyser", key="_au_go_smi")

        if analyse_clicked and smi_in.strip():
            st.session_state["_au_smi_val"] = smi_in.strip()
            st.session_state.pop("_au_cache", None)

        smiles_to_run = st.session_state.get("_au_smi_val")

    if smiles_to_run is None:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;margin-top:1rem;'
            'background:#F8F9FA;border:1.5px dashed #DEE2E6;border-radius:10px">'
            '<div style="font-size:2rem;margin-bottom:.6rem">🧪</div>'
            '<div style="font-size:.88rem;color:#6B7280;font-weight:500">'
            'Entrez un nom ou un SMILES pour lancer l\'analyse.</div></div>',
            unsafe_allow_html=True,
        )

    elif not model:
        st.warning("Modèle non chargé — vérifiez les `.pkl`.", icon="⚠️")

    else:
        cache = st.session_state.get("_au_cache", {})
        if cache.get("smiles") != smiles_to_run or "pred" not in cache:
            with st.spinner("Calcul en cours…"):
                pred, prob, props, mol = run_prediction(
                    smiles_to_run, model, scaler, colonnes, mediane_train  # ✅ passé ici
                )
            st.session_state["_au_cache"] = {
                "smiles": smiles_to_run,
                "pred": pred, "prob": prob, "props": props, "mol": mol,
            }
        else:
            pred, prob, props, mol = (cache["pred"], cache["prob"],
                                      cache["props"], cache["mol"])

        if pred is None:
            st.error("⛔ SMILES invalide.")
        else:
            prob_pos = float(prob[1]) if len(prob) > 1 else float(prob[0])
            is_pos   = pred == 1
            mk       = _h(smiles_to_run)

            st.divider()
            cl, cr = st.columns([5,6], gap="large")

            with cl:
                _sec("Structure 2D")
                _mol_frame(mol)
                st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
                _sec("Propriétés physicochimiques")
                _prop_table(props)

            with cr:
                _sec("Verdict")
                _verdict(is_pos, prob_pos)
                st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
                _sec("Confiance du modèle")
                st.plotly_chart(_gauge(prob_pos, is_pos),
                                use_container_width=True, key=f"au_g_{mk}")
                _sec("Radar Lipinski")
                st.plotly_chart(_radar(props, is_pos),
                                use_container_width=True, key=f"au_r_{mk}")


# ══════════════════════════════════════════════════════════════════
# PAGE ③  BATCH SCREENING
# ══════════════════════════════════════════════════════════════════
elif page == "📂 Batch Screening":
    _hdr("Criblage par lots", "Screening Multi-Molécules",
         "Fichier CSV/Excel avec colonne <code style='background:#F1F3F5;padding:1px 5px;"
         "border-radius:3px;color:#2563EB'>smiles</code> (et optionnellement "
         "<code style='background:#F1F3F5;padding:1px 5px;border-radius:3px;color:#2563EB'>name</code>).")

    st.markdown(
        '<div style="background:#EFF6FF;border:1px solid rgba(37,99,235,.15);'
        'border-left:3px solid #2563EB;border-radius:8px;'
        'padding:.85rem 1rem;font-size:.83rem;color:#6B7280;margin-bottom:1.1rem">'
        '<strong style="color:#2563EB">Format CSV :</strong> '
        '<code style="background:rgba(37,99,235,.08);padding:1px 5px;border-radius:3px">'
        'smiles,name</code> &nbsp;·&nbsp; '
        '<code style="background:rgba(37,99,235,.08);padding:1px 5px;border-radius:3px">'
        'CC(=O)Oc1ccccc1C(=O)O,Aspirin</code></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Fichier CSV ou Excel", type=["csv","xlsx"], key="b_up")

    if uploaded and model:
        try:
            df_in = (pd.read_excel(uploaded, engine="openpyxl")
                     if uploaded.name.lower().endswith(".xlsx")
                     else pd.read_csv(uploaded))
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            df_in = None

        if df_in is not None:
            if "smiles" not in df_in.columns:
                st.error("Colonne `smiles` introuvable.")
            else:
                st.info(f"**{len(df_in)}** molécules · colonnes : `{list(df_in.columns)}`")
                with st.expander("Aperçu", expanded=False):
                    st.dataframe(df_in.head(8), use_container_width=True)

                if st.button("🚀 Lancer le criblage", key="b_run"):
                    results = []
                    pb  = st.progress(0.0)
                    box = st.empty()
                    n   = len(df_in)
                    for i, row in df_in.iterrows():
                        smi  = str(row["smiles"]).strip()
                        name = str(row.get("name", f"mol_{i+1}"))
                        pb.progress((i+1)/n, text=f"{i+1}/{n} — {name}")
                        box.caption(f"⚙️ **{name}**")
                        try:
                            pred, prob, props, _ = run_prediction(
                                smi, model, scaler, colonnes, mediane_train  # ✅ passé ici
                            )
                            if pred is not None:
                                p = float(prob[1]) if len(prob)>1 else float(prob[0])
                                results.append({
                                    "Name": name, "SMILES": smi,
                                    "BBB": "BBB+" if pred==1 else "BBB−",
                                    "P(BBB+)": round(p,4),
                                    "MW": props["MW"], "LogP": props["LogP"],
                                    "TPSA": props["TPSA"], "HBD": props["HBD"],
                                    "HBA": props["HBA"], "QED": props["QED"],
                                })
                            else:
                                results.append({"Name":name,"SMILES":smi,"BBB":"ERREUR",
                                    "P(BBB+)":None,"MW":None,"LogP":None,"TPSA":None,
                                    "HBD":None,"HBA":None,"QED":None})
                        except:
                            results.append({"Name":name,"SMILES":smi,"BBB":"ERREUR",
                                "P(BBB+)":None,"MW":None,"LogP":None,"TPSA":None,
                                "HBD":None,"HBA":None,"QED":None})
                    pb.progress(1.0, text="✓ Terminé")
                    box.empty()
                    st.session_state["b_results"] = pd.DataFrame(results)

    elif uploaded and not model:
        st.warning("Modèle non chargé.", icon="⚠️")

    if "b_results" in st.session_state:
        df_out = st.session_state["b_results"]
        valid  = df_out[df_out["BBB"] != "ERREUR"]
        n_pos  = int((valid["BBB"]=="BBB+").sum())
        n_neg  = int((valid["BBB"]=="BBB−").sum())
        n_tot  = len(df_out)
        pct    = round(n_pos/max(n_pos+n_neg,1)*100, 1)

        st.divider()
        s1,s2,s3,s4 = st.columns(4)
        for col, val, lbl, color in [
            (s1,str(n_tot),"Traitées","#2563EB"),
            (s2,str(n_pos),"BBB+ Perméables","#059669"),
            (s3,str(n_neg),"BBB− Bloquées","#DC2626"),
            (s4,f"{pct}%","Taux BBB+","#D97706"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #DEE2E6;'
                    f'border-top:2px solid {color};border-radius:8px;'
                    f'padding:.85rem 1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.07)">'
                    f'<div style="font-family:\'Outfit\',sans-serif;font-size:1.65rem;'
                    f'font-weight:900;color:{color}">{val}</div>'
                    f'<div style="font-size:.7rem;color:#6B7280;margin-top:.15rem">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2, gap="medium")
        with bc1:
            _sec("Distribution BBB")
            fig_pie = go.Figure(go.Pie(
                labels=["BBB+","BBB−"], values=[n_pos,n_neg], hole=0.65,
                marker=dict(colors=["#059669","#DC2626"],line=dict(color="#fff",width=2)),
                textinfo="percent+label", textfont=dict(size=13,family="Inter")))
            fig_pie.add_annotation(text=f"<b>{n_tot}</b>", x=0.5, y=0.5,
                showarrow=False, font=dict(size=18,color="#111827",family="Outfit"))
            fig_pie.update_layout(**_PB, height=260,
                margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True, key="b_pie")

        with bc2:
            _sec("Distribution des probabilités")
            probs_v = valid["P(BBB+)"].dropna().tolist()
            counts, edges = np.histogram(probs_v, bins=20, range=(0.0,1.0))
            mids   = [(edges[i]+edges[i+1])/2 for i in range(len(counts))]
            colors = ["#059669" if m>=0.5 else "#D97706" if m>=0.35 else "#DC2626" for m in mids]
            fig_h = go.Figure(go.Bar(
                x=mids, y=counts.tolist(),
                width=[(edges[i+1]-edges[i])*.9 for i in range(len(counts))],
                marker=dict(color=colors,line=dict(color="#fff",width=.5))))
            fig_h.update_layout(**_PB, margin=_M, height=260,
                xaxis=dict(title="P(BBB+)",range=[0,1],gridcolor="rgba(0,0,0,.05)"),
                yaxis=dict(title="n",gridcolor="rgba(0,0,0,.05)"), bargap=0.05)
            st.plotly_chart(fig_h, use_container_width=True, key="b_hist")

        if len(valid) > 1:
            _sec("Espace chimique (MW vs LogP)")
            fig_sc = go.Figure()
            for sub, col2, nm in [
                (valid[valid["BBB"]=="BBB+"],"#059669","BBB+"),
                (valid[valid["BBB"]=="BBB−"],"#DC2626","BBB−"),
            ]:
                if len(sub):
                    fig_sc.add_trace(go.Scatter(
                        x=sub["MW"], y=sub["LogP"], mode="markers", name=nm,
                        marker=dict(color=col2,size=8,opacity=0.8,
                                    line=dict(color="#fff",width=1)),
                        hovertemplate="<b>%{customdata[0]}</b><br>MW:%{x:.1f}·LogP:%{y:.2f}"
                                      "<br>P:%{customdata[1]:.3f}<extra></extra>",
                        customdata=list(zip(sub["Name"],sub["P(BBB+)"]))))
            fig_sc.add_shape(type="rect",x0=0,y0=-10,x1=500,y1=5,
                fillcolor="rgba(37,99,235,.04)",
                line=dict(color="#2563EB",width=1,dash="dot"))
            fig_sc.update_layout(**_PB, margin=_M, height=310,
                xaxis=dict(title="MW (Da)",gridcolor="rgba(0,0,0,.05)"),
                yaxis=dict(title="LogP",gridcolor="rgba(0,0,0,.05)"),
                legend=dict(font=dict(size=11,color="#6B7280"),
                    bgcolor="rgba(255,255,255,.95)",bordercolor="#DEE2E6",borderwidth=1))
            st.plotly_chart(fig_sc, use_container_width=True, key="b_sc")

        _sec("Tableau de résultats")
        st.dataframe(
            df_out.style.apply(
                lambda col: ["color:#059669;font-weight:700;" if v=="BBB+"
                             else "color:#DC2626;font-weight:700;" if v=="BBB−"
                             else "color:#D97706;" for v in col]
                if col.name == "BBB" else [""]*len(col), axis=0),
            use_container_width=True, hide_index=True)
        st.download_button("⬇️ Télécharger CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="bbb_results.csv", mime="text/csv", key="b_dl")


# ══════════════════════════════════════════════════════════════════
# PAGE ④  PROPRIÉTÉS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Propriétés":
    _hdr("Descripteurs moléculaires", "Propriétés Physicochimiques",
         "Profil ADMET complet + règle des 5 de Lipinski.")

    smi_p = st.text_input("SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O",
                          key="prop_smi", label_visibility="collapsed")

    if not smi_p.strip():
        st.markdown(
            '<div style="text-align:center;padding:2.8rem 1rem;margin-top:1rem;'
            'background:#F8F9FA;border:1.5px dashed #DEE2E6;border-radius:10px">'
            '<div style="font-size:2rem;margin-bottom:.6rem">📊</div>'
            '<div style="font-size:.88rem;color:#6B7280;font-weight:500">'
            'Entrez un SMILES pour calculer les propriétés.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        mol = Chem.MolFromSmiles(smi_p.strip())
        if mol is None:
            st.error("⛔ SMILES invalide.")
        else:
            mol   = strip_salts(mol)
            props = physchem(mol)
            pk    = _h(smi_p.strip())

            st.divider()
            cm, cd = st.columns([4,5], gap="large")
            with cm:
                _sec("Structure 2D")
                _mol_frame(mol, 400, 300)
            with cd:
                _sec("Descripteurs clés")
                _prop_table(props)

            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            _sec("Radar Lipinski")
            st.plotly_chart(_radar(props, True),
                            use_container_width=True, key=f"prop_r_{pk}")

            _sec("Règle des 5 de Lipinski")
            rules = [
                ("MW ≤ 500 Da", props["MW"]<=500),
                ("LogP ≤ 5",    props["LogP"]<=5),
                ("HBD ≤ 5",     props["HBD"]<=5),
                ("HBA ≤ 10",    props["HBA"]<=10),
                ("RotB ≤ 10",   props["RotB"]<=10),
                ("TPSA ≤ 90",   props["TPSA"]<=90),
            ]
            n_ok = sum(1 for _, ok in rules if ok)
            for cr, (lbl, ok) in zip(st.columns(len(rules)), rules):
                with cr:
                    ic = "✅" if ok else "❌"
                    co = "#059669" if ok else "#DC2626"
                    bg = "#ECFDF5" if ok else "#FEF2F2"
                    bd = "rgba(5,150,105,.2)" if ok else "rgba(220,38,38,.2)"
                    st.markdown(
                        f'<div style="text-align:center;padding:.65rem .35rem;background:{bg};'
                        f'border:1px solid {bd};border-radius:8px">'
                        f'<div style="font-size:1.25rem">{ic}</div>'
                        f'<div style="font-size:.66rem;font-weight:700;color:{co};margin-top:.2rem">'
                        f'{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )
            qual  = "Excellent ✓" if n_ok==6 else "Acceptable" if n_ok>=4 else "À optimiser"
            color = "#059669" if n_ok==6 else "#D97706"
            bg    = "#ECFDF5" if n_ok==6 else "#FFFBEB"
            bd    = "rgba(5,150,105,.2)" if n_ok==6 else "rgba(217,119,6,.2)"
            st.markdown(
                f'<div style="margin-top:.7rem;padding:.75rem 1rem;background:{bg};'
                f'border:1px solid {bd};border-radius:8px;font-size:.84rem">'
                f'<strong style="color:{color}">{n_ok}/6 règles respectées</strong>'
                f'<span style="color:#6B7280;margin-left:.5rem">— {qual}</span></div>',
                unsafe_allow_html=True,
            )
