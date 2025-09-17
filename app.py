# app.py
# ==========================================================
# Dashboard de Sentencias y Estados de Casos (Violencia Sexual)
# Ajustes:
#  - El análisis de resoluciones se basa SOLO en casos con Estado de caso == "Sentencia".
#  - Gráfico de estados simplificado.
#  - Selector de granularidad temporal (Año/Trimestre/Mes) para SENTENCIAS.
#  - Prosa enriquecida.
# Requisitos: streamlit, pandas, altair
#   pip install streamlit pandas altair
#   streamlit run app.py
# ==========================================================

import math
import unicodedata
from datetime import date
import pandas as pd
import streamlit as st
import altair as alt

# ================== CONFIG / TEMA ==================
st.set_page_config(page_title="Sentencias y estados de caso en registros de violencia sexual contra menores de edad, 2019-2022", page_icon="🟣", layout="wide")

COLORS = {
    "bg": "#F9F7FB",
    "text": "#2E2459",
    "muted": "#5E5A78",
    "primary": "#6A1B9A",
    "primary2": "#8E24AA",
    "accent": "#FDD835",
    "accent2": "#FFB300",
    "line": "#D1C4E9",
}

st.markdown(f"""
<style>
html, body, [class*="stApp"] {{
  background: {COLORS['bg']};
  color: {COLORS['text']};
}}
h1, h2, h3, h4 {{ color: {COLORS['primary']}; }}
hr {{ border: none; border-top: 1px solid {COLORS['line']}; margin: .8rem 0 1.2rem 0; }}
.stButton>button {{ background: {COLORS['primary']}; color: white; border-radius: 999px; border: none; }}
.stDownloadButton>button {{ background: {COLORS['accent']}; color: #3a3200; border-radius: 999px; border: none; }}
.metric-small > div > div:nth-child(1) {{ font-size: .9rem; color: {COLORS['muted']}; }}
.metric-small > div > div:nth-child(2) {{ font-size: 1.2rem; color: {COLORS['primary']}; }}
</style>
""", unsafe_allow_html=True)

# ================== DATOS ==================
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5TXNsUgBgMP6Cdv1Utkd01MkZi2Vmg7bbJBwkpbdL8jY-cwKP1WjtzHoaJDt4KcsPR2SpYPXBFUwo/pub?output=csv"

@st.cache_data(show_spinner=False)
def cargar_datos(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Normaliza nombres de columnas (tolerante a tildes/espacios)
    def norm(s):
        s = str(s).strip()
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        return s.lower()

    cols = {norm(c): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_fecha = pick("fecha denuncia", "fechadenuncia", "fecha", "fecha_hecho", "fechahecho")
    col_pais = pick("pais")
    col_depto = pick("departamento", "depto", "dpto")
    col_muni = pick("municipio", "muni")
    col_sexo = pick("sexo", "genero")
    col_edad = pick("edad")
    col_delito = pick("delito", "delitos", "tipo_delito", "tipo de delito")
    col_estado = pick("estado caso", "estadocaso", "estado", "estado_del_caso")
    col_resol = pick("resolucion", "resolución", "resultado", "sentencia")

    rename_map = {}
    if col_fecha: rename_map[col_fecha] = "fecha_denuncia"
    if col_pais: rename_map[col_pais] = "pais"
    if col_depto: rename_map[col_depto] = "departamento"
    if col_muni: rename_map[col_muni] = "municipio"
    if col_sexo: rename_map[col_sexo] = "sexo"
    if col_edad: rename_map[col_edad] = "edad"
    if col_delito: rename_map[col_delito] = "delito"
    if col_estado: rename_map[col_estado] = "estado_caso"
    if col_resol: rename_map[col_resol] = "resolucion"
    df = df.rename(columns=rename_map)

    # Tipos y derivados de tiempo
    if "fecha_denuncia" in df.columns:
        df["fecha_denuncia"] = pd.to_datetime(df["fecha_denuncia"], errors="coerce")
        df["anio"] = df["fecha_denuncia"].dt.year
        df["mes"] = df["fecha_denuncia"].dt.to_period("M").astype(str)  # YYYY-MM
        df["trimestre"] = df["fecha_denuncia"].dt.to_period("Q").astype(str)  # YYYYQn

    for c in ["departamento", "municipio", "pais", "sexo", "delito", "estado_caso", "resolucion"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "edad" in df.columns:
        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    return df

df = cargar_datos(GSHEET_CSV_URL)

# ================== HEADER ==================
st.title("🟣 Sentencias y estados de casos en denuncias de violencia sexual contra menores de edad en Guatemala 2017-2022")
st.caption("Análisis de violencia sexual por territorio, temporalidad, estados y delitos.")

# ================== SIDEBAR: FILTROS ==================
st.sidebar.header("Filtros")

nivel = st.sidebar.radio("Nivel de análisis", ["Departamento", "Municipio"], index=0)
territorio_col = "departamento" if nivel == "Departamento" else "municipio"

# País (si existe)
if "pais" in df.columns:
    paises = ["Todos"] + sorted([p for p in df["pais"].dropna().unique() if str(p).strip()])
    pais_sel = st.sidebar.selectbox("País", paises)
else:
    pais_sel = "Todos"

# Territorios
territorios_all = sorted([t for t in df.get(territorio_col, pd.Series(dtype=str)).dropna().unique() if str(t).strip()])
territorios_sel = st.sidebar.multiselect(f"{nivel}(s)", options=territorios_all, default=territorios_all[:15] if territorios_all else [])

# Sexo
sexos_all = sorted([s for s in df.get("sexo", pd.Series(dtype=str)).dropna().unique() if str(s).strip()])
sexos_sel = st.sidebar.multiselect("Sexo", options=sexos_all, default=sexos_all)

# Delito
delitos_all = sorted([d for d in df.get("delito", pd.Series(dtype=str)).dropna().unique() if str(d).strip()])
delitos_sel = st.sidebar.multiselect("Delito", options=delitos_all, default=delitos_all[:10])

# Estado del caso
estados_all = sorted([e for e in df.get("estado_caso", pd.Series(dtype=str)).dropna().unique() if str(e).strip()])
estados_sel = st.sidebar.multiselect("Estado del caso", options=estados_all, default=estados_all)

# Resolución (filtro general; recuerda que para análisis de sentencias, se aplicará SOLO sobre 'estado=sentencia')
res_all = sorted([r for r in df.get("resolucion", pd.Series(dtype=str)).dropna().unique() if str(r).strip()])
res_sel = st.sidebar.multiselect("Resolución (opcional)", options=res_all, default=res_all)

# Fechas
if "fecha_denuncia" in df.columns:
    fmin = pd.to_datetime(df["fecha_denuncia"], errors="coerce").min()
    fmax = pd.to_datetime(df["fecha_denuncia"], errors="coerce").max()
else:
    fmin = fmax = None
if fmin and fmax and pd.notna(fmin) and pd.notna(fmax):
    f_ini, f_fin = st.sidebar.date_input("Rango de fechas", value=[fmin.date(), fmax.date()],
                                         min_value=fmin.date(), max_value=fmax.date())
else:
    f_ini, f_fin = None, None

# Granularidad temporal (para sentencias)
gran = st.sidebar.radio("Granularidad (sentencias)", ["Año", "Trimestre", "Mes"], index=2)
periodo_col = "anio" if gran == "Año" else ("trimestre" if gran == "Trimestre" else "mes")

# ================== APLICAR FILTROS GENERALES ==================
df_f = df.copy()

if pais_sel != "Todos" and "pais" in df_f.columns:
    df_f = df_f[df_f["pais"] == pais_sel]
if territorios_sel:
    df_f = df_f[df_f[territorio_col].isin(territorios_sel)]
if sexos_sel and "sexo" in df_f.columns and len(sexos_sel) > 0:
    df_f = df_f[df_f["sexo"].isin(sexos_sel)]
if delitos_sel and "delito" in df_f.columns and len(delitos_sel) > 0:
    df_f = df_f[df_f["delito"].isin(delitos_sel)]
if estados_sel and "estado_caso" in df_f.columns and len(estados_sel) > 0:
    df_f = df_f[df_f["estado_caso"].isin(estados_sel)]
if res_sel and "resolucion" in df_f.columns and len(res_sel) > 0:
    df_f = df_f[df_f["resolucion"].isin(res_sel)]
if f_ini and f_fin and "fecha_denuncia" in df_f.columns:
    df_f = df_f[(df_f["fecha_denuncia"] >= pd.to_datetime(f_ini)) &
                (df_f["fecha_denuncia"] <= pd.to_datetime(f_fin))]

# ================== SUBCONJUNTO SOLO SENTENCIAS ==================
def _lower(s): 
    return str(s).strip().lower() if pd.notna(s) else s

df_sent = df_f.copy()
if "estado_caso" in df_sent.columns:
    df_sent = df_sent[df_sent["estado_caso"].map(_lower) == "sentencia"]
else:
    df_sent = df_sent.iloc[0:0]  # vacío si no existe la columna

# --- FIX CLAVE: mapeo robusto de Resolución -> {Condenatoria, Absolutoria} ---
def _norm(s):
    if pd.isna(s): 
        return ""
    s = str(s).strip().lower()
    # quitar tildes
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def map_res(v):
    """
    Detecta condena/absolución en textos variados:
    - condena / condenado / condenada / condenatoria
    - absolucion / absolutoria / absuelto / absuelta / 'se absuelve' / 'absuelve'
    """
    t = _norm(v)
    if not t:
        return None
    if "conden" in t:
        return "Condenatoria"
    if ("absolu" in t) or ("absuelt" in t) or ("absuelv" in t) or ("se absuelv" in t) or ("absuelve" in t):
        return "Absolutoria"
    return None
# -----------------------------------------------------------------

if "resolucion" in df_sent.columns:
    df_sent["resolucion_std"] = df_sent["resolucion"].map(map_res)

# ================== KPIs ==================
total_casos = len(df_f)
total_sent = len(df_sent)
base_res = df_sent.dropna(subset=["resolucion_std"]).shape[0] if "resolucion_std" in df_sent.columns else 0
n_condena = (df_sent["resolucion_std"] == "Condenatoria").sum() if "resolucion_std" in df_sent.columns else 0
n_absol = (df_sent["resolucion_std"] == "Absolutoria").sum() if "resolucion_std" in df_sent.columns else 0

def pct(a, b): 
    return (100.0 * a / b) if (b and b > 0) else 0.0
def fmt(x, nd=1):
    try: return f"{x:,.{nd}f}".replace(",", " ")
    except: return str(x)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Casos (filtros)", f"{total_casos:,}")
with c2:
    st.metric("Sentencias", f"{total_sent:,}")
with c3:
    st.metric("% condena (sobre sentencias con resolución)", f"{fmt(pct(n_condena, base_res))}%")
with c4:
    st.metric("% absolución (sobre sentencias con resolución)", f"{fmt(pct(n_absol, base_res))}%")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== ESTADOS DE CASO (GRÁFICO SIMPLE) ==================
st.subheader("Estados de caso (distribución simple)")
if "estado_caso" in df_f.columns and not df_f.empty:
    est = (df_f["estado_caso"].dropna().value_counts().reset_index())
    est.columns = ["estado_caso", "n"]
    chart_estado_simple = alt.Chart(est).mark_bar().encode(
        x=alt.X("n:Q", title="Cantidad"),
        y=alt.Y("estado_caso:N", sort='-x', title="Estado"),
        color=alt.value(COLORS["primary"]),
        tooltip=["estado_caso", "n"]
    ).properties(height=300)
    st.altair_chart(chart_estado_simple, use_container_width=True)
else:
    st.info("No hay datos de estados de caso para este filtro.")

# ================== RESOLUCIONES POR TERRITORIO (SOLO SENTENCIAS) ==================
st.subheader(f"Resoluciones por {nivel} (solo sentencias, distribución 100%)")
if not df_sent.empty and "resolucion_std" in df_sent.columns:
    base = (df_sent.dropna(subset=[territorio_col, "resolucion_std"])
                 .groupby([territorio_col, "resolucion_std"]).size().reset_index(name="n"))
    tot = base.groupby(territorio_col)["n"].transform("sum")
    base["pct"] = (base["n"] / tot) * 100

    # Orden por % condena
    orden = (base[base["resolucion_std"] == "Condenatoria"]
                .sort_values("pct", ascending=False)[territorio_col].tolist())
    if not orden:
        orden = base.groupby(territorio_col)["n"].sum().sort_values(ascending=False).index.tolist()

    chart_stack = alt.Chart(base).mark_bar().encode(
        x=alt.X(f"{territorio_col}:N", sort=orden, title=nivel),
        y=alt.Y("pct:Q", stack="normalize", title="Porcentaje"),
        color=alt.Color("resolucion_std:N", title="Resolución",
                        scale=alt.Scale(range=[COLORS["primary"], COLORS["accent2"], "#bbb"])),
        tooltip=[territorio_col, "resolucion_std", alt.Tooltip("n:Q", title="Cantidad"), alt.Tooltip("pct:Q", format=".1f", title="%")]
    ).properties(height=380)
    st.altair_chart(chart_stack, use_container_width=True)
else:
    st.info("No hay datos de sentencias (o resoluciones) para este filtro.")

# ================== TOP TERRITORIOS (CONDE/ABSUE) SOBRE SENTENCIAS ==================
st.subheader(f"¿Dónde se condena y se absuelve más? ({nivel}, sobre sentencias)")
if not df_sent.empty and "resolucion_std" in df_sent.columns:
    df_res = df_sent.dropna(subset=[territorio_col, "resolucion_std"])
    pivot = (df_res
             .groupby(territorio_col)["resolucion_std"].value_counts().unstack(fill_value=0)
             .rename(columns={"Condenatoria":"Condenatoria","Absolutoria":"Absolutoria"}))
    # asegurar columnas
    for col in ["Condenatoria","Absolutoria"]:
        if col not in pivot: pivot[col] = 0
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot[pivot["total"] > 0].copy()
    pivot["pct_condena"] = pivot["Condenatoria"] / pivot["total"] * 100.0
    pivot["pct_absol"] = pivot["Absolutoria"] / pivot["total"] * 100.0
    pivot = pivot.reset_index()

    top_condena = pivot.sort_values("pct_condena", ascending=False).head(10)
    top_absol = pivot.sort_values("pct_absol", ascending=False).head(10)

    cL, cR = st.columns(2)
    with cL:
        st.caption("▲ Mayor % de condena (sobre sentencias)")
        chart_c = alt.Chart(top_condena).mark_bar().encode(
            x=alt.X("pct_condena:Q", title="% condena", axis=alt.Axis(format=".1f")),
            y=alt.Y(f"{territorio_col}:N", sort='-x', title=nivel),
            color=alt.value(COLORS["primary"]),
            tooltip=[territorio_col, alt.Tooltip("total:Q", title="Sentencias"), alt.Tooltip("pct_condena:Q", title="% condena", format=".1f")]
        ).properties(height=340)
        st.altair_chart(chart_c, use_container_width=True)

    with cR:
        st.caption("▼ Mayor % de absolución (sobre sentencias)")
        chart_a = alt.Chart(top_absol).mark_bar().encode(
            x=alt.X("pct_absol:Q", title="% absolución", axis=alt.Axis(format=".1f")),
            y=alt.Y(f"{territorio_col}:N", sort='-x', title=nivel),
            color=alt.value(COLORS["accent2"]),
            tooltip=[territorio_col, alt.Tooltip("total:Q", title="Sentencias"), alt.Tooltip("pct_absol:Q", title="% absolución", format=".1f")]
        ).properties(height=340)
        st.altair_chart(chart_a, use_container_width=True)
else:
    st.info("No hay suficientes sentencias para calcular rankings por territorio.")

# ================== SERIE TEMPORAL (SOLO SENTENCIAS) ==================
st.subheader(f"Serie temporal de sentencias por {gran}")
if not df_sent.empty:
    if periodo_col not in df_sent.columns:
        st.info("No se puede construir la serie temporal con las columnas disponibles.")
    else:
        grp_cols = [periodo_col]
        total_t = df_sent.groupby(grp_cols).size().reset_index(name="n_sent")
        if "resolucion_std" in df_sent.columns:
            by_res = df_sent.groupby(grp_cols + ["resolucion_std"]).size().reset_index(name="n_res")

            line_total = alt.Chart(total_t).mark_line(point=True, stroke=COLORS["primary"]).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_sent:Q", title="Sentencias"),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), alt.Tooltip("n_sent:Q", title="Sentencias")]
            )

            line_res = alt.Chart(by_res).mark_line(point=True).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_res:Q", title="Sentencias"),
                color=alt.Color("resolucion_std:N", title="Resolución",
                                scale=alt.Scale(range=[COLORS["accent2"], COLORS["primary2"], "#777"])),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), "resolucion_std", alt.Tooltip("n_res:Q", title="Sentencias")]
            )
            st.altair_chart((line_total + line_res).properties(height=380), use_container_width=True)
        else:
            line_total = alt.Chart(total_t).mark_line(point=True, stroke=COLORS["primary"]).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_sent:Q", title="Sentencias"),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), alt.Tooltip("n_sent:Q", title="Sentencias")]
            ).properties(height=380)
            st.altair_chart(line_total, use_container_width=True)
else:
    st.info("No hay datos de sentencias para este filtro.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== DELITOS (GENERAL, NO SOLO SENTENCIAS) ==================
st.subheader("Delitos más frecuentes (en el universo filtrado)")
if "delito" in df_f.columns and not df_f.empty:
    topn = (df_f.dropna(subset=["delito"])
              .groupby("delito").size().reset_index(name="n")
              .sort_values("n", ascending=False).head(15))
    chart_delito = alt.Chart(topn).mark_bar().encode(
        x=alt.X("n:Q", title="Cantidad"),
        y=alt.Y("delito:N", sort='-x', title="Delito"),
        color=alt.value(COLORS["primary"]),
        tooltip=["delito", "n"]
    ).properties(height=380)
    st.altair_chart(chart_delito, use_container_width=True)
else:
    st.info("No hay datos de delitos para este filtro.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== PROSA ENRIQUECIDA ==================
st.subheader("Conclusiones")
def pct(a, b): 
    return (100.0 * a / b) if (b and b > 0) else 0.0

def conclusiones(df_all: pd.DataFrame, df_s: pd.DataFrame, nivel_txt: str, periodo_txt: str) -> str:
    if df_all.empty:
        return "Con los filtros actuales no hay registros. Ajusta territorio, fechas o categorías."
    frases = []

    # Rango temporal visible (si existe)
    if "fecha_denuncia" in df_all.columns and df_all["fecha_denuncia"].notna().any():
        fmin = pd.to_datetime(df_all["fecha_denuncia"], errors="coerce").min()
        fmax = pd.to_datetime(df_all["fecha_denuncia"], errors="coerce").max()
        if pd.notna(fmin) and pd.notna(fmax):
            frases.append(f"El universo filtrado abarca del **{fmin.date()}** al **{fmax.date()}**.")

    frases.append(f"Casos totales analizados: **{len(df_all):,}**. Sentencias: **{len(df_s):,}**.")

    # Resoluciones sobre sentencias
    if not df_s.empty and "resolucion_std" in df_s.columns:
        base = df_s.dropna(subset=["resolucion_std"])
        n_res = len(base)
        if n_res > 0:
            n_c = (base["resolucion_std"] == "Condenatoria").sum()
            n_a = (base["resolucion_std"] == "Absolutoria").sum()
            frases.append(
                f"Sobre sentencias con resolución (**{n_res:,}**), **{pct(n_c, n_res):.1f}%** fueron *condenatorias* "
                f"y **{pct(n_a, n_res):.1f}%** *absolutorias*."
            )
            # Top territorios (umbral mínimo para estabilidad)
            terr = (base.groupby(territorio_col)["resolucion_std"].value_counts().unstack(fill_value=0))
            for col in ["Condenatoria","Absolutoria"]:
                if col not in terr: terr[col] = 0
            terr["total"] = terr.sum(axis=1)
            terr = terr[terr["total"] >= max(5, round(0.01 * n_res))]  # al menos 5 o 1% del total
            if not terr.empty:
                terr["%Condena"] = terr["Condenatoria"]/terr["total"]*100
                terr["%Absolución"] = terr["Absolutoria"]/terr["total"]*100
                top_c = terr["%Condena"].sort_values(ascending=False).head(3)
                top_a = terr["%Absolución"].sort_values(ascending=False).head(3)
                if len(top_c) > 0:
                    frases.append("Mayor proporción de *condena* en " +
                                  ", ".join([f"**{idx}** ({val:.1f}%)" for idx, val in top_c.items()]) + ".")
                if len(top_a) > 0:
                    frases.append("Mayor proporción de *absolución* en " +
                                  ", ".join([f"**{idx}** ({val:.1f}%)" for idx, val in top_a.items()]) + ".")

    # Estados del caso (distribución general)
    if "estado_caso" in df_all.columns:
        vc = df_all["estado_caso"].dropna().value_counts(normalize=True)*100
        if not vc.empty:
            frases.append("Distribución de estados (top): " +
                          ", ".join([f"**{k}** ({v:.1f}%)" for k, v in vc.head(3).items()]) + ".")

    # Delitos
    if "delito" in df_all.columns:
        vd = df_all["delito"].dropna().value_counts().head(3)
        if not vd.empty:
            frases.append("Delitos más frecuentes: " +
                          ", ".join([f"**{k}** ({v})" for k, v in vd.items()]) + ".")

    # Tendencia reciente en sentencias
    if not df_s.empty and periodo_col in df_s.columns:
        serie = df_s.dropna(subset=[periodo_col]).groupby(periodo_col).size().sort_index()
        if len(serie) >= 4:
            q = max(1, len(serie)//4)
            ult = serie.tail(q).sum()
            ant = serie.iloc[:-q].tail(q).sum() if len(serie) >= 2*q else serie.iloc[:-q].sum()
            if ant > 0:
                var = (ult - ant) / ant * 100
                frases.append(f"En los {periodo_txt.lower()}s más recientes, las sentencias variaron **{var:.1f}%** vs. el periodo previo.")

    return " ".join(frases)

st.info(conclusiones(df_f, df_sent, nivel, gran))

# ================== TABLA ==================
st.subheader("Datos filtrados")
st.dataframe(df_f, use_container_width=True, hide_index=True)
