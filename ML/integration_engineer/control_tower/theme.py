"""DWELL design system: tokens, global CSS, Plotly defaults and provenance tags."""

from __future__ import annotations

import streamlit as st

INK = "#0A0A0A"
PLATE = "#121211"
PLATE_2 = "#1A1A18"
RULE = "#2B2B27"
MUTED = "#8E8E86"
SOFT = "#BDBDB4"
PAPER = "#F4F4EE"
LIME = "#C7F36B"
AMBER = "#F2A93B"
RED = "#FF5A47"
CLASS_COLOR = {"ON_TIME": LIME, "AT_RISK": AMBER, "CRITICAL_DELAY": RED}
CLASS_LABEL = {"ON_TIME": "On time", "AT_RISK": "At risk", "CRITICAL_DELAY": "Critical delay"}
CLASS_RANGE = {"ON_TIME": "≤ 65.1 h", "AT_RISK": "65.1–160.9 h", "CRITICAL_DELAY": "> 160.9 h"}

FONT_LINK = ("https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@500;700;800"
             "&family=Big+Shoulders+Stencil+Display:wght@700;800"
             "&family=Hanken+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap")

# Every panel declares what kind of truth it shows.
TAGS = {
    "observed": ("OBSERVED 2021", "Historical TPS event log. Not live."),
    "validation": ("VALIDATION", "Out-of-sample scores on development folds. The quarantined holdout is not scored."),
    "model": ("MODEL OUTPUT", "An individual prediction, not an observation."),
    "reference": ("REFERENCE", "Static directory data. Not a live feed."),
    "bound": ("INFORMATION BOUND", "Uses future information. Not achievable by a real model."),
    "lagged": ("OBSERVED · 7-DAY LAG", "PortWatch AIS estimates, usable only after about a week."),
    "dl_forecast": ("DL MODEL", "Neural network throughput forecast under 2-day operational latency."),
    "controlled_bench": ("BENCHMARK", "Evaluation across 2,065 global ports."),
    "anchorage_dwell": ("PORT DWELL", "Vessel dwell time model. Historical cutoff Dec 2022."),
}


def inject_css() -> None:
    st.markdown(f"""<link rel="stylesheet" href="{FONT_LINK}">
<style>
:root {{ --ink:{INK}; --plate:{PLATE}; --plate2:{PLATE_2}; --rule:{RULE}; --muted:{MUTED}; --soft:{SOFT}; --paper:{PAPER}; --lime:{LIME}; --amber:{AMBER}; --red:{RED}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background:var(--ink) !important; color:var(--paper); font-family:'Hanken Grotesk',sans-serif; }}
[data-testid="stHeader"] {{ background:var(--ink) !important; border-bottom:1px solid var(--rule); }}
[data-testid="stToolbar"] [data-testid="stToolbarActions"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
.block-container {{ padding:3.2rem 2rem 3rem !important; max-width:1440px !important; }}
h1,h2,h3,h4 {{ font-family:'Big Shoulders Display',sans-serif !important; letter-spacing:.01em; color:var(--paper) !important; }}
p, li, label, [data-testid="stMarkdownContainer"] {{ font-family:'Hanken Grotesk',sans-serif; }}
[data-testid="stHeader"] a, [data-testid="stNavSectionHeader"] {{ font-family:'IBM Plex Mono',monospace !important; }}
a[data-testid="stTopNavLink"], [data-testid="stTopNavLink"] {{ font-family:'IBM Plex Mono',monospace !important; font-size:.82rem !important; letter-spacing:.04em; text-transform:uppercase; border-radius:0 !important; padding:0.4rem 0.8rem !important; }}
[data-testid="stTopNavLink"] span {{ color:var(--muted) !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background:transparent !important; box-shadow:inset 0 -2px 0 var(--lime); }}
[data-testid="stTopNavLink"][aria-current="page"] span {{ color:var(--paper) !important; }}
[data-testid="stElementContainer"] iframe {{ display:block; }}
[data-testid="stVerticalBlock"] {{ gap:1.2rem; }}
/* Controls */
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input, .stTextInput input, .stDateInput input, .stTimeInput input {{ background:var(--plate) !important; border-radius:4px !important; border-color:var(--rule) !important; color:var(--paper) !important; font-family:'Hanken Grotesk',sans-serif !important; font-size:.92rem !important; padding:0.25rem 0.4rem !important; }}
label[data-testid="stWidgetLabel"] p {{ color:var(--soft) !important; font-family:'Hanken Grotesk',sans-serif !important; font-size:.86rem !important; font-weight:500 !important; letter-spacing:normal !important; text-transform:none !important; margin-bottom:0.2rem !important; }}
button[kind="primary"], [data-testid="stFormSubmitButton"] button {{ background:var(--lime) !important; color:#0A0A0A !important; border:0 !important; border-radius:4px !important; font-family:'Hanken Grotesk',sans-serif !important; font-weight:600 !important; font-size:.92rem !important; padding:0.45rem 1rem !important; }}
button[kind="secondary"] {{ background:var(--plate) !important; border:1px solid var(--rule) !important; border-radius:4px !important; color:var(--paper) !important; font-family:'Hanken Grotesk',sans-serif !important; font-size:.86rem !important; font-weight:500 !important; }}
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-pillsActive"] {{ background:var(--lime) !important; color:#0A0A0A !important; border-color:var(--lime) !important; }}
[data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pillsActive"] p {{ color:#0A0A0A !important; }}
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-pills"] {{ background:var(--plate) !important; border-color:var(--rule) !important; border-radius:4px !important; }}
[data-testid="stBaseButton-segmented_control"] p, [data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pills"] p, [data-testid="stBaseButton-pillsActive"] p {{ font-family:'Hanken Grotesk',sans-serif !important; font-size:.84rem !important; font-weight:500 !important; }}
:focus-visible {{ outline:2px solid var(--lime) !important; outline-offset:2px; }}
[data-testid="stForm"] {{ border:1px solid var(--rule) !important; border-radius:4px !important; background:var(--plate); padding:1.2rem !important; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--rule); border-radius:4px; }}
[data-testid="stCaptionContainer"] {{ color:var(--muted) !important; }}
/* Components */
.strip {{ display:flex; flex-wrap:wrap; gap:.4rem 1.6rem; align-items:baseline; border-bottom:1px solid var(--rule); padding-bottom:.6rem; font:400 .84rem/1.5 'Hanken Grotesk',sans-serif; color:var(--muted); }}
.strip b {{ color:var(--paper); font-weight:500; }} .strip .nolive {{ color:var(--amber); }}
.thesis {{ font:700 clamp(1.8rem,3vw,2.5rem)/1.15 'Big Shoulders Display',sans-serif; color:var(--paper); margin:.2rem 0 .4rem; text-transform:none; }}
.thesis em {{ font-style:normal; color:var(--red); }} .thesis i {{ font-style:normal; color:var(--lime); }}
.lede {{ color:var(--soft); font-size:.98rem; line-height:1.5; max-width:48rem; margin-bottom:.6rem; }}
.head {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; border-top:1px solid var(--rule); padding-top:.75rem; margin-top:.8rem; margin-bottom:.6rem; }}
.head h3 {{ font:700 1.35rem/1.1 'Big Shoulders Display',sans-serif !important; margin:0 !important; padding:0 !important; text-transform:none; letter-spacing:.01em; }}
.head .q {{ color:var(--muted); font-size:.88rem; line-height:1.4; margin-top:.2rem; }}
.tag {{ font:500 .72rem/1 'IBM Plex Mono',monospace; letter-spacing:.04em; color:var(--muted); border:1px solid var(--rule); padding:.3rem .5rem; border-radius:3px; white-space:nowrap; cursor:help; }}
.tag.validation,.tag.model,.tag.dl_forecast,.tag.controlled_bench {{ color:var(--lime); border-color:#4a5a26; }} .tag.bound,.tag.anchorage_dwell {{ color:var(--amber); border-color:#5c421a; }}
.bento {{ display:grid; grid-template-columns:repeat(6,1fr); gap:1px; background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden; }}
.cell {{ background:var(--ink); padding:1rem 1.1rem; min-width:0; }}
.cell.s2 {{ grid-column:span 2; }} .cell.s3 {{ grid-column:span 3; }} .cell.s4 {{ grid-column:span 4; }} .cell.s6 {{ grid-column:span 6; }}
.cell .k {{ font:500 .8rem/1.3 'Hanken Grotesk',sans-serif; color:var(--muted); display:flex; justify-content:space-between; gap:.4rem; }}
.cell .v {{ font:700 2rem/1.1 'Big Shoulders Display',sans-serif; color:var(--paper); margin-top:.35rem; font-variant-numeric:tabular-nums; }}
.cell .v small {{ font:400 .84rem 'Hanken Grotesk',sans-serif; color:var(--soft); margin-left:.35rem; }}
.cell .n {{ color:var(--soft); font-size:.84rem; line-height:1.4; margin-top:.4rem; }}
.mixbar {{ display:flex; height:10px; gap:2px; margin:.6rem 0 .4rem; }} .mixbar span {{ display:block; }}
.mono {{ font-family:'IBM Plex Mono',monospace; }}
.note {{ color:var(--soft); font-size:.88rem; line-height:1.5; }}
.note b {{ color:var(--paper); font-weight:600; }}
.verdict {{ border-left:4px solid var(--c); padding:.2rem 0 .2rem 1rem; margin:.2rem 0 .8rem; }}
.verdict .lab {{ font:800 2.4rem/1.05 'Big Shoulders Display',sans-serif; color:var(--c); }}
.verdict .sub {{ font:400 .86rem 'Hanken Grotesk',sans-serif; color:var(--soft); margin-top:.3rem; line-height:1.45; }}
.pbar {{ display:grid; grid-template-columns:8.5rem 1fr 3.5rem; align-items:center; gap:.75rem; font:400 .84rem 'Hanken Grotesk',sans-serif; color:var(--soft); margin:.35rem 0; }}
.pbar .t {{ height:10px; background:var(--plate2); border-radius:2px; overflow:hidden; }} .pbar .t span {{ display:block; height:100%; transition:width .45s cubic-bezier(.2,.7,.2,1); }} .pbar .p {{ text-align:right; color:var(--paper); font-weight:500; }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:.3rem 1.1rem; font:400 .84rem/1.5 'Hanken Grotesk',sans-serif; }} .kv dt {{ color:var(--muted); }} .kv dd {{ color:var(--paper); margin:0; }}
.shap {{ display:grid; grid-template-columns:minmax(11rem,15rem) 1fr 5.5rem; gap:.5rem; align-items:center; font:400 .84rem 'Hanken Grotesk',sans-serif; color:var(--soft); margin:.3rem 0; }}
.shap .bar {{ height:10px; position:relative; background:var(--plate2); border-radius:2px; }} .shap .bar span {{ position:absolute; top:0; bottom:0; border-radius:2px; }} .shap .mid {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--muted); }}
.statuses {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.5rem; }} .st {{ font:400 .8rem 'Hanken Grotesk',sans-serif; padding:.25rem .55rem; border:1px solid var(--rule); border-radius:3px; color:var(--muted); }} .st.on {{ color:var(--paper); border-color:var(--soft); background:var(--plate); }}
.unavailable {{ border:1px dashed var(--rule); padding:.9rem 1.1rem; color:var(--soft); font-size:.88rem; line-height:1.5; border-radius:4px; }} .unavailable b {{ color:var(--paper); font-weight:600; }}
@media (max-width: 900px) {{ .bento {{ grid-template-columns:repeat(2,1fr); }} .cell.s2,.cell.s3,.cell.s4,.cell.s6 {{ grid-column:span 2; }} .head {{ flex-direction:column; gap:.3rem; }} .block-container {{ padding:3rem 1rem 2rem !important; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>""", unsafe_allow_html=True)


def tag_html(kind: str) -> str:
    label, why = TAGS[kind]
    return f'<span class="tag {kind}" title="{why}">{label}</span>'


def head(title: str, question: str, kind: str) -> None:
    st.markdown(f'<div class="head"><div><h3>{title}</h3><div class="q">{question}</div></div>{tag_html(kind)}</div>',
                unsafe_allow_html=True)


def plotly_layout(fig, height: int = 320, **kw):
    base = dict(
        height=height, margin=dict(l=46, r=14, t=12, b=34), paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(family="IBM Plex Mono, monospace", size=11, color=SOFT),
        hoverlabel=dict(bgcolor=PLATE_2, bordercolor=RULE, font=dict(family="IBM Plex Mono, monospace", size=11, color=PAPER), align="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10.5)),
    )
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="#1c1c1a", zerolinecolor=RULE, linecolor=RULE, tickfont=dict(size=10, color=MUTED), title_font=dict(size=10.5, color=MUTED))
    fig.update_yaxes(gridcolor="#1c1c1a", zerolinecolor=RULE, linecolor=RULE, tickfont=dict(size=10, color=MUTED), title_font=dict(size=10.5, color=MUTED))
    return fig


PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
