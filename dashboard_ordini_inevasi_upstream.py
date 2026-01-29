# =====================================================
# DASHBOARD ORDINI INEVASI — CODICE COMPLETO DEFINITIVO
# =====================================================

#!/usr/bin/env python
# coding: utf-8

# ========================= IMPORT =========================
import streamlit as st
import pandas as pd
import numpy as np
import chardet
import re
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Dashboard Ordini Inevasi",
    layout="wide"
)

st.title("Dashboard Ordini Inevasi — Upstream")
st.sidebar.subheader("📂 Caricamento dati")

uploaded_file = st.sidebar.file_uploader(
    "Carica file inevasi (CSV o Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("⬅️ Carica un file per iniziare")
    st.stop()

FILE_IS_EXCEL = uploaded_file.name.lower().endswith(".xlsx")


# ========================= UTILS =========================
def detect_encoding(path, n_bytes=20000):
    with open(path, 'rb') as f:
        return chardet.detect(f.read(n_bytes))['encoding']

def detect_separator(path, encoding):
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        sample = ''.join([f.readline() for _ in range(5)])
    if sample.count(';') > sample.count(','): return ';'
    if sample.count('\t') > sample.count(','): return '\t'
    return ','

def to_numeric(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('\xa0', '').strip()
    if s.count(',') and s.count('.'):
        s = s.replace('.', '').replace(',', '.')
    else:
        s = s.replace(',', '.')
    try: return float(s)
    except: return np.nan

def text_positions_for_bars(values, thresh_ratio=0.08):
    if len(values) == 0:
        return []
    maxv = max(values) if max(values) != 0 else 1
    positions = ['inside' if (v / maxv) >= thresh_ratio else 'outside' for v in values]
    return positions

# ========================= LOAD DATA =========================
@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes, filename):
    is_excel = filename.lower().endswith(".xlsx")

    if is_excel:
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        enc = chardet.detect(file_bytes[:20000])['encoding']
        df = pd.read_csv(
            BytesIO(file_bytes),
            sep=None,
            engine="python",
            encoding=enc,
            on_bad_lines="skip"
        )

    df.columns = df.columns.str.strip()
    return df

file_bytes = uploaded_file.getvalue()

try:
    df_raw = load_data_from_bytes(file_bytes, uploaded_file.name)
except Exception as e:
    st.error(f"Errore nel caricamento file: {e}")
    st.stop()


# ========================= PRE-PROCESSING AND CLEANING =========================
df = df_raw.copy()

for c in df.columns:
    if any(k in c for k in ['Valore']):
        df[c] = df[c].apply(to_numeric)

def preparazione_analisi_inevasi(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    # Maschera righe da eliminare
    mask = df["Descrizione Articolo"].astype(str).str.startswith("-")
    df_eliminate = df.loc[mask] 
    df = df.loc[~mask].reset_index(drop=True)

    df.rename(columns={"Bauzzar Qta Ordinata": "Bauzaar Qta Ordinata"}, inplace=True)
    df.rename(columns={"Bauzzar Qta Preparata": "Bauzaar Qta Preparata"}, inplace=True)
    df.rename(columns={"Bauzzar Qta Inevaso Comm.": "Bauzaar Qta Inevaso Comm."}, inplace=True)
    df.rename(columns={"Bauzzar Qta Inevaso Logist.": "Bauzaar Qta Inevaso Logist."}, inplace=True)
    df.rename(columns={"BAUZZAR QTA INEVASO ANAGR.": "Bauzaar Qta inevaso anagr."}, inplace=True)


    # Converte tutte le colonne numeriche coinvolte in numeric
    colonne_numeriche = [
        "Carico Qta Prev. Consegna", "Carico Qta Ricevuta", "Carico  Qta Inevasa",

        "Proprietà Qta Ordinata", "Proprietà Qta Preparata",
        "Proprietà Qta Inevaso Comm.", "Proprietà Qta Inevaso Logist.", "Proprietà Qta Inevaso Anagr.",

        "Affiliati Qta Ordinata", "Affiliati Qta Preparata",
        "Affiliati Qta Inevaso Comm.", "Affiliati Qta Inevaso Logist.", "Affiliati Qta Inevaso Anagr.",

        "Brico Qta Ordinata", "Brico Qta Preparata",
        "Brico Qta Inevaso Comm.", "Brico Qta Inevaso Logist.", "Brico Qta Inevaso Anagr.",

        "SPA Qta Ordinata", "SPA Qta Preparata",
        "SPA Qta Inevaso Comm.", "SPA Qta Inevaso Logist.", "SPA Qta Inevaso Anagr.",

        "Bauzaar Qta Ordinata", "Bauzaar Qta Preparata",
        "Bauzaar Qta Inevaso Comm.", "Bauzaar Qta Inevaso Logist.", "Bauzaar Qta inevaso anagr.",

        "Altri Qta Ordinata", "Altri Qta Preparata",
        "Altri Qta Inevaso Comm.", "Altri Qta Inevaso Logist.", "Altri Qta Inevaso Anagr."
    ]

    for col in colonne_numeriche:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": "0", "nan": "0", "None": "0"})
                .str.replace(".", "", regex=False)   # separatore migliaia
                .str.replace(",", ".", regex=False)  # decimale italiano
                .astype(float)
                .astype(int)
            )

    # ======================
    # UPSTREAM
    # ======================
    df["check_upstream"] = (df["Carico Qta Prev. Consegna"] - (df["Carico Qta Ricevuta"] + df["Carico  Qta Inevasa"]))
    
    df["Upstream"] = df["check_upstream"].apply(lambda x: "OK" if x <= 0 else "Attenzione")

    # ======================
    # Colonna 'Buyer' (normalizzazione)
    # ======================
    
    if "Buyer" not in df.columns:
        df["Buyer"] = "NON ASSEGNATO"
    else:
        df["Buyer"] = (
            df["Buyer"]
            .astype(str)
            .str.replace('\xa0', ' ', regex=False)  # spazio non-breaking
            .str.strip()                            # rimuove spazi
            .str.upper()                            # uniforma maiuscole
        )
    
    # normalizza valori vuoti / sporchi
    df.loc[df["Buyer"].isin(["", "NAN", "NONE"]), "Buyer"] = "NON ASSEGNATO"

    return df

df = preparazione_analisi_inevasi(df)
    
# =====================================================
# 5. COSTRUZIONE VARIABILI  
# =====================================================

# --- UPSTREAM ---

DS_QTA_ORD = 'Carico Qta Prev. Consegna'
DS_QTA_CONS = 'Carico Qta Ricevuta'
DS_QTA_INEV = 'Carico  Qta Inevasa'

VAL_CONS = 'Valore Consegnato Costo Premi'
VAL_INEV = 'Valore Inevaso Costo Premi'
VAL_UNIT = 'Valore Unitario Costo Premi Fine Periodo'

SUPPLIER = 'Descriz. Fornitore del Carico'

# Valore totale ordinato al fornitore (Upstream).
df['Val_Ordinato'] = df[VAL_CONS].fillna(0) + df[VAL_INEV].fillna(0)  # Valore Consegnato + Valore Inevaso

# =========================
# SIDEBAR FILTRI
# =========================
st.sidebar.header("Filtri")

def multiselect_filter(label, col):
    if col in df.columns:
        values = sorted(df[col].dropna().unique())
        return st.sidebar.multiselect(label, values)
    return []

provider_sel   = multiselect_filter("Fornitore", SUPPLIER)
sector_sel     = multiselect_filter("Settore", 'Settore')
reparto_sel    = multiselect_filter("Reparto", 'Reparto')
category_sel   = multiselect_filter("Categoria", 'Categoria')
marca_sel      = multiselect_filter("Marca", 'Marca')
stato_sel      = multiselect_filter("Stato Articolo Magazzino", 'Stato articolo Magazzino')
best_sel       = multiselect_filter("Best 1000", 'Best 1000')
buyer_sel      = multiselect_filter("Buyer", 'Buyer')
sku_search     = st.sidebar.text_input("Cerca SKU / Descrizione")

# =========================
# APPLY FILTERS
# =========================
def apply_filters(df_in):
    df_f = df_in.copy()

    # =========================
    # FILTRO UPSTREAM IMPLICITO
    # =========================
    if 'Upstream' in df_f.columns:
        df_f = df_f[df_f['Upstream'] == 'OK']

    filters = {
        SUPPLIER: provider_sel,
        'Settore': sector_sel,
        'Reparto': reparto_sel,
        'Categoria': category_sel,
        'Marca': marca_sel,
        'Stato articolo Magazzino': stato_sel,
        'Best 1000': best_sel,
        'Buyer': buyer_sel
    }

    for col, sel in filters.items():
        if sel and col in df_f.columns:
            df_f = df_f[df_f[col].isin(sel)]

    if sku_search:
        mask = False
        if 'Codice Articolo' in df_f.columns:
            mask |= df_f['Codice Articolo'].astype(str).str.contains(sku_search, case=False, na=False)
        if 'Descrizione Articolo' in df_f.columns:
            mask |= df_f['Descrizione Articolo'].astype(str).str.contains(sku_search, case=False, na=False)
        df_f = df_f[mask]

    return df_f
    
df_up   = apply_filters(df)

# ========================= DASHBOARD =========================
def render_upstream(df_sec):
    st.subheader('UPSTREAM — Analisi sui Fornitori')

    tot_ord = df_sec[DS_QTA_ORD].sum()
    tot_inev = df_sec[DS_QTA_INEV].sum()
    tot_val_ord = df_sec['Val_Ordinato'].sum() # valore totale ordinato al fornitore
    tot_val_inev = df_sec[VAL_INEV].sum() 

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Pz Ordinati', f"{tot_ord:,.0f}")
    c2.metric('Pz Inevasi', f"{tot_inev:,.0f}")
    c3.metric('% Inevaso Pz', f"{(tot_inev/tot_ord*100):.2f}%" if tot_ord else 'N/D')
    c4.metric('% Inevaso Valore', f"{(tot_val_inev/tot_val_ord*100):.2f}%" if tot_val_ord else 'N/D')
    c5.metric('Valore Inevaso Totale (Costo Premi)', f"{tot_val_inev:,.2f}")

    st.sidebar.markdown("Se nessun filtro è selezionato, la dashboard mostra tutti i dati")
    st.sidebar.markdown("- I best 1000 articoli coprono il 48.5% del fatturato annuo")
    st.markdown('---')








    
    # Top 20 articoli per valore inevaso 
    pareto = df_sec.groupby('Descrizione Articolo')[VAL_INEV].sum().reset_index().sort_values(VAL_INEV, ascending=False).head(20)
    fig = px.bar(pareto, x='Descrizione Articolo', y=VAL_INEV, text=pareto[VAL_INEV].map('{:,.0f}'.format), title='Top 20 referenze per valore di inevaso - UPSTREAM', color=VAL_INEV, color_continuous_scale='Viridis')
    fig.update_traces(textposition=text_positions_for_bars(pareto[VAL_INEV].tolist()))
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')

    # Top 20 articoli % valore inevaso
    top_pct = (df_sec.groupby('Descrizione Articolo').agg({VAL_INEV: 'sum',VAL_CONS: 'sum'}).reset_index())
    top_pct['VAL_ORD'] = top_pct[VAL_INEV] + top_pct[VAL_CONS]
    top_pct['Perc_Inevaso'] = (top_pct[VAL_INEV] / top_pct['VAL_ORD']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    top_pct = top_pct.sort_values('Perc_Inevaso', ascending=False).head(20)
    vals_pct = top_pct['Perc_Inevaso'].tolist()
    pos_pct = text_positions_for_bars(vals_pct, thresh_ratio=0.08)
    fig_pct = px.bar(top_pct, x='Descrizione Articolo', y='Perc_Inevaso', text=top_pct['Perc_Inevaso'].map('{:.2f}%'.format), title='Top 20 referenze per % di valore inevaso - UPSTREAM', color='Perc_Inevaso', color_continuous_scale='Viridis')
    fig_pct.update_traces(textposition=pos_pct)
    fig_pct.update_layout(xaxis_tickangle=-45,yaxis_title='% Valore Inevaso',margin=dict(t=100))
    st.plotly_chart(fig_pct, use_container_width=True)

    # Top 20 fornitori per valore inevaso 
    topf = df_sec.groupby('Descriz. Fornitore del Carico')[VAL_INEV].sum().reset_index().sort_values(VAL_INEV, ascending=False).head(20)
    fig2 = px.bar(topf, x='Descriz. Fornitore del Carico', y=VAL_INEV, text=topf[VAL_INEV].map('{:,.0f}'.format), title='Top 20 fornitori per valore di inevaso - UPSTREAM', color=VAL_INEV, color_continuous_scale='Viridis')
    fig2.update_traces(textposition=text_positions_for_bars(topf[VAL_INEV].tolist()))
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, width='stretch')

    # Top Fornitori - % quantità Inevaso
    agg_f = df_sec.groupby('Descriz. Fornitore del Carico').agg({DS_QTA_INEV: 'sum', DS_QTA_ORD: 'sum'}).reset_index()
    agg_f['Perc_Inevaso_Fornitore'] = (agg_f[DS_QTA_INEV] / agg_f[DS_QTA_ORD]).replace([np.inf, -np.inf], np.nan) * 100
    top_sup_pct = agg_f.sort_values('Perc_Inevaso_Fornitore', ascending=False).head(20)
    top_sup_pct['Text_Perc'] = top_sup_pct['Perc_Inevaso_Fornitore'].map('{:.2f}%'.format)
    vals_pct = top_sup_pct['Perc_Inevaso_Fornitore'].fillna(0).tolist()
    pos = text_positions_for_bars(vals_pct, thresh_ratio=0.08)
    fig_tp = px.bar(top_sup_pct, x='Descriz. Fornitore del Carico', y='Perc_Inevaso_Fornitore', text='Text_Perc', title=f'Top 20 fornitori per % di quantità inevasa - UPSTREAM', color='Perc_Inevaso_Fornitore', color_continuous_scale='Viridis')
    fig_tp.update_traces(textposition=pos)
    fig_tp.update_layout(xaxis_tickangle=-45, yaxis_title='% Quantità Inevasa', uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(t=100))
    st.plotly_chart(fig_tp, use_container_width=True)    

    # Treemap
    path = [c for c in ['Settore', 'Reparto', 'Categoria', 'Marca','Descrizione Articolo'] if c in df_sec.columns]
    fig3 = px.treemap(df_sec, path=path, values=VAL_INEV, color=VAL_INEV, color_continuous_scale='Viridis', title='Treemap gerarchica per valore inevaso: Settore > Reparto > Categoria > Marca > Articolo - UPSTREAM')
    st.plotly_chart(fig3, width='stretch')

    # Heatmap
    heat = df_sec.groupby(['Reparto', 'Categoria']).agg({DS_QTA_ORD:'sum', DS_QTA_INEV:'sum'}).reset_index()
    heat['Perc_Inevaso'] = (heat[DS_QTA_INEV] / heat[DS_QTA_ORD]).replace([np.inf, -np.inf], 0) * 100
    pivot = heat.pivot(index='Reparto', columns='Categoria', values='Perc_Inevaso').fillna(0)
    fig4 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Viridis', colorbar=dict(title=' % Quantità Inevasa')))
    fig4.update_layout(title=f'Heatmap per % di quantità inevasa: Reparto vs Categoria - UPSTREAM')
    st.plotly_chart(fig4, width='stretch')
    
    # Heatmap Valore Inevaso
    heat_val = df_sec.groupby(['Reparto', 'Categoria']).agg({VAL_INEV: 'sum'}).reset_index()
    pivot_val = heat_val.pivot(index='Reparto', columns='Categoria', values=VAL_INEV).fillna(0)
    fig_hv = go.Figure(data=go.Heatmap(z=pivot_val.values, x=pivot_val.columns.tolist(), y=pivot_val.index.tolist(), colorscale='Viridis', colorbar=dict(title='Valore Inevaso Costo Premi')))
    fig_hv.update_layout(title=f'Heatmap per valore inevaso: Reparto vs Categoria - UPSTREAM')
    st.plotly_chart(fig_hv, use_container_width=True)

    # Scatter % Inevaso vs Valore per Fornitore
    agg_sup = df_sec.groupby('Descriz. Fornitore del Carico').agg({DS_QTA_INEV:'sum', DS_QTA_ORD:'sum', VAL_INEV:'sum'}).reset_index()
    agg_sup['Perc_Inevaso_Fornitore'] = (agg_sup[DS_QTA_INEV] / agg_sup[DS_QTA_ORD]).replace([np.inf, -np.inf], np.nan) * 100
    y_col = VAL_INEV if VAL_INEV in agg_sup.columns else DS_QTA_INEV
    fig_sc = px.scatter(agg_sup, x='Perc_Inevaso_Fornitore', y=y_col, size=y_col, hover_name='Descriz. Fornitore del Carico', title=f'% quantità inevasa vs valore inevaso per fornitore - UPSTREAM', color=y_col, color_continuous_scale='Viridis', size_max=30)
    fig_sc.update_layout(xaxis_title='% Qta Inevasa', yaxis_title='Valore Inevaso')
    st.plotly_chart(fig_sc, use_container_width=True)

    # Scatter % Inevaso vs Valore per Articolo
    agg_art = df_sec.groupby('Descrizione Articolo').agg({DS_QTA_INEV:'sum', DS_QTA_ORD:'sum', VAL_INEV:'sum'}).reset_index()
    agg_art['Perc_Inevaso_Articolo'] = (agg_art[DS_QTA_INEV] / agg_art[DS_QTA_ORD]).replace([np.inf, -np.inf], np.nan) * 100
    y_col = VAL_INEV if VAL_INEV in agg_art.columns else DS_QTA_INEV
    fig_sc_art = px.scatter(agg_art, x='Perc_Inevaso_Articolo', y=y_col, size=y_col, hover_name='Descrizione Articolo', title=f'% quantità inevasa vs valore inevaso per referenza - UPSTREAM', color=y_col, color_continuous_scale='Viridis', size_max=30)
    fig_sc_art.update_layout(xaxis_title='% Qta Inevasa', yaxis_title='Valore Inevaso')
    st.plotly_chart(fig_sc_art, use_container_width=True)
                
    # % INEVASI PER BUYER (QTA e VALORE) 
    # Calcoli aggregati
    agg_buyer = df_sec.groupby('Buyer').agg({
        DS_QTA_INEV: 'sum',
        DS_QTA_ORD: 'sum',
        VAL_INEV: 'sum',
        VAL_CONS: 'sum'
    }).reset_index()
    # Creazione VAL_ORD come somma delle colonne aggregate
    agg_buyer['VAL_ORD'] = agg_buyer[VAL_INEV] + agg_buyer[VAL_CONS]
    # Percentuali
    agg_buyer['Perc_Inevaso_Qta'] = (agg_buyer[DS_QTA_INEV] / agg_buyer[DS_QTA_ORD]).replace([np.inf, -np.inf], 0).fillna(0) * 100
    agg_buyer['Perc_Inevaso_Val'] = (agg_buyer[VAL_INEV] / agg_buyer['VAL_ORD']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    # Ordine per valore % quantità
    agg_buyer = agg_buyer.sort_values("Perc_Inevaso_Qta", ascending=False)
    # Grafico quantità
    vals_q = agg_buyer['Perc_Inevaso_Qta'].tolist()
    pos_q = text_positions_for_bars(vals_q, thresh_ratio=0.08)
    fig_buyer_q = px.bar(agg_buyer, x='Buyer', y='Perc_Inevaso_Qta', text=agg_buyer['Perc_Inevaso_Qta'].map('{:.2f}%'.format), title=f'Quantità inevasa per buyer - UPSTREAM', color='Perc_Inevaso_Qta', color_continuous_scale='Viridis')
    fig_buyer_q.update_traces(textposition=pos_q)
    fig_buyer_q.update_layout(yaxis_title='% Quantità Inevasa', xaxis_tickangle=-45, margin=dict(t=100))
    st.plotly_chart(fig_buyer_q, use_container_width=True)
    # Grafico valore
    vals_v = agg_buyer['Perc_Inevaso_Val'].tolist()
    pos_v = text_positions_for_bars(vals_v, thresh_ratio=0.08)
    fig_buyer_v = px.bar(agg_buyer, x='Buyer', y='Perc_Inevaso_Val', text=agg_buyer['Perc_Inevaso_Val'].map('{:.2f}%'.format), title=f' Valore inevaso per buyer - UPSTREAM', color='Perc_Inevaso_Val', color_continuous_scale='Viridis')
    fig_buyer_v.update_traces(textposition=pos_v)
    fig_buyer_v.update_layout(yaxis_title='% Valore Inevaso', xaxis_tickangle=-45, margin=dict(t=100))
    st.plotly_chart(fig_buyer_v, use_container_width=True)

    # -----------------------
    # Preview tabellare
    # -----------------------
    st.markdown("**Anteprima dei dati filtrati per questa sezione**")
    cols_show = [c for c in df_sec.columns]
    st.dataframe(df_sec[cols_show].reset_index(drop=True), height=300)


t1 = st.tabs(['Upstream'])
with t1[0]:
    render_upstream(df_up)

# ========================= EXPORT =========================
def to_excel_bytes(df_up):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Scrive il DataFrame in Excel
        df_up.to_excel(writer, index=False, sheet_name='Upstream')
        # Ottimizza la larghezza delle colonne
        worksheet = writer.sheets['Upstream']
        for i, col in enumerate(df_up.columns[:40]):
            worksheet.set_column(i, i, max(12, min(50, len(str(col)) + 2)))
    return buffer.getvalue()

# Genera il file Excel in memoria
xlsx_bytes = to_excel_bytes(df_up)

# Mostra pulsante per download
st.subheader("📥 Esporta dati filtrati")
st.markdown("Scarica un file Excel: `Upstream`. Il foglio contiene i dati filtrati con i campi calcolati.")
st.download_button(
    label='Scarica Excel Upstream',
    data=xlsx_bytes,
    file_name='dati_inevasi_fornitori.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.markdown('---')
st.caption(f"Script eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

