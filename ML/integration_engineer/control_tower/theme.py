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
}


def inject_css() -> None:
    st.markdown(f"""<link rel="stylesheet" href="{FONT_LINK}">
<style>
:root {{ --ink:{INK}; --plate:{PLATE}; --plate2:{PLATE_2}; --rule:{RULE}; --muted:{MUTED}; --soft:{SOFT}; --paper:{PAPER}; --lime:{LIME}; --amber:{AMBER}; --red:{RED}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background:var(--ink) !important; color:var(--paper); font-family:'Hanken Grotesk',sans-serif; }}
[data-testid="stHeader"] {{ background:var(--ink) !important; border-bottom:1px solid var(--rule); }}
[data-testid="stToolbar"] [data-testid="stToolbarActions"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
.block-container {{ padding:4.5rem 1.6rem 3rem !important; max-width:1560px !important; }}
h1,h2,h3,h4 {{ font-family:'Big Shoulders Display',sans-serif !important; letter-spacing:.01em; color:var(--paper) !important; }}
p, li, label, [data-testid="stMarkdownContainer"] {{ font-family:'Hanken Grotesk',sans-serif; }}
[data-testid="stHeader"] a, [data-testid="stNavSectionHeader"] {{ font-family:'IBM Plex Mono',monospace !important; }}
a[data-testid="stTopNavLink"], [data-testid="stTopNavLink"] {{ font-family:'IBM Plex Mono',monospace !important; font-size:.76rem !important; letter-spacing:.06em; text-transform:uppercase; border-radius:0 !important; }}
[data-testid="stTopNavLink"] span {{ color:var(--muted) !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background:transparent !important; box-shadow:inset 0 -2px 0 var(--lime); }}
[data-testid="stTopNavLink"][aria-current="page"] span {{ color:var(--paper) !important; }}
[data-testid="stElementContainer"] iframe {{ display:block; }}
[data-testid="stVerticalBlock"] {{ gap:.75rem; }}
/* Controls */
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input, .stTextInput input, .stDateInput input {{ background:var(--plate) !important; border-radius:2px !important; border-color:var(--rule) !important; color:var(--paper) !important; font-family:'IBM Plex Mono',monospace !important; font-size:.84rem !important; }}
label[data-testid="stWidgetLabel"] p {{ color:var(--muted) !important; font-family:'IBM Plex Mono',monospace; font-size:.7rem !important; letter-spacing:.05em; text-transform:uppercase; }}
button[kind="primary"], [data-testid="stFormSubmitButton"] button {{ background:var(--lime) !important; color:#0A0A0A !important; border:0 !important; border-radius:2px !important; font-family:'IBM Plex Mono',monospace !important; font-weight:600 !important; }}
button[kind="secondary"] {{ background:var(--plate) !important; border:1px solid var(--rule) !important; border-radius:2px !important; color:var(--soft) !important; font-family:'IBM Plex Mono',monospace !important; font-size:.76rem !important; }}
[data-testid="stBaseButton-segmented_controlActive"], [data-testid="stBaseButton-pillsActive"] {{ background:var(--lime) !important; color:#0A0A0A !important; border-color:var(--lime) !important; }}
[data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pillsActive"] p {{ color:#0A0A0A !important; }}
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-pills"] {{ background:var(--plate) !important; border-color:var(--rule) !important; border-radius:2px !important; }}
[data-testid="stBaseButton-segmented_control"] p, [data-testid="stBaseButton-segmented_controlActive"] p, [data-testid="stBaseButton-pills"] p, [data-testid="stBaseButton-pillsActive"] p {{ font-family:'IBM Plex Mono',monospace !important; font-size:.74rem !important; }}
:focus-visible {{ outline:2px solid var(--lime) !important; outline-offset:2px; }}
[data-testid="stForm"] {{ border:1px solid var(--rule) !important; border-radius:2px !important; background:var(--plate); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--rule); }}
[data-testid="stCaptionContainer"] {{ color:var(--muted) !important; }}
/* Components */
.strip {{ display:flex; flex-wrap:wrap; gap:.3rem 1.4rem; align-items:baseline; border-bottom:1px solid var(--rule); padding-bottom:.5rem; font:400 .72rem/1.4 'IBM Plex Mono',monospace; color:var(--muted); letter-spacing:.03em; }}
.strip b {{ color:var(--paper); font-weight:500; }} .strip .nolive {{ color:var(--amber); }}
.thesis {{ font:800 clamp(2.1rem,3.6vw,3.5rem)/.98 'Big Shoulders Display',sans-serif; color:var(--paper); letter-spacing:.005em; margin:.2rem 0 .7rem; text-transform:uppercase; }}
.thesis em {{ font-style:normal; color:var(--red); }} .thesis i {{ font-style:normal; color:var(--lime); }}
.lede {{ color:var(--soft); font-size:.95rem; line-height:1.45; max-width:44rem; margin-bottom:.4rem; }}
.head {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; border-top:1px solid var(--rule); padding-top:.55rem; margin-top:.4rem; margin-bottom:.45rem; }}
.head h3 {{ font:700 1.42rem/1.05 'Big Shoulders Display',sans-serif !important; margin:0 !important; padding:0 !important; text-transform:uppercase; letter-spacing:.015em; }}
.head .q {{ color:var(--muted); font-size:.83rem; line-height:1.35; margin-top:.15rem; padding-bottom:.1rem; }}
.tag {{ font:500 .62rem/1 'IBM Plex Mono',monospace; letter-spacing:.08em; color:var(--muted); border:1px solid var(--rule); padding:.28rem .4rem; white-space:nowrap; cursor:help; }}
.tag.validation,.tag.model {{ color:var(--lime); border-color:#4a5a26; }} .tag.bound {{ color:var(--amber); border-color:#5c421a; }}
.bento {{ display:grid; grid-template-columns:repeat(6,1fr); gap:1px; background:var(--rule); border:1px solid var(--rule); }}
.cell {{ background:var(--ink); padding:.7rem .75rem .75rem; min-width:0; }}
.cell.s2 {{ grid-column:span 2; }} .cell.s3 {{ grid-column:span 3; }} .cell.s4 {{ grid-column:span 4; }} .cell.s6 {{ grid-column:span 6; }}
.cell .k {{ font:500 .62rem/1.3 'IBM Plex Mono',monospace; letter-spacing:.07em; color:var(--muted); text-transform:uppercase; display:flex; justify-content:space-between; gap:.4rem; }}
.cell .v {{ font:700 2.15rem/1.05 'Big Shoulders Display',sans-serif; color:var(--paper); margin-top:.28rem; font-variant-numeric:tabular-nums; }}
.cell .v small {{ font:500 .72rem 'IBM Plex Mono',monospace; color:var(--muted); margin-left:.2rem; }}
.cell .n {{ color:var(--muted); font-size:.77rem; line-height:1.32; margin-top:.25rem; }}
.mixbar {{ display:flex; height:9px; gap:2px; margin:.5rem 0 .3rem; }} .mixbar span {{ display:block; }}
.mono {{ font-family:'IBM Plex Mono',monospace; }}
.note {{ color:var(--muted); font-size:.8rem; line-height:1.42; }}
.note b {{ color:var(--soft); font-weight:500; }}
.verdict {{ border-left:3px solid var(--c); padding:.1rem 0 .1rem .8rem; margin:.1rem 0 .6rem; }}
.verdict .lab {{ font:800 2.7rem/1 'Big Shoulders Display',sans-serif; text-transform:uppercase; color:var(--c); }}
.verdict .sub {{ font:400 .72rem 'IBM Plex Mono',monospace; color:var(--muted); margin-top:.2rem; }}
.pbar {{ display:grid; grid-template-columns:7.5rem 1fr 3.2rem; align-items:center; gap:.5rem; font:400 .74rem 'IBM Plex Mono',monospace; color:var(--soft); margin:.26rem 0; }}
.pbar .t {{ height:10px; background:var(--plate2); }} .pbar .t span {{ display:block; height:100%; transition:width .45s cubic-bezier(.2,.7,.2,1); }} .pbar .p {{ text-align:right; color:var(--paper); }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:.2rem .9rem; font:400 .74rem/1.4 'IBM Plex Mono',monospace; }} .kv dt {{ color:var(--muted); }} .kv dd {{ color:var(--paper); margin:0; }}
.shap {{ display:grid; grid-template-columns:minmax(9rem,13rem) 1fr 5.5rem; gap:.35rem; align-items:center; font:400 .73rem 'IBM Plex Mono',monospace; color:var(--soft); margin:.2rem 0; }}
.shap .bar {{ height:9px; position:relative; background:var(--plate2); }} .shap .bar span {{ position:absolute; top:0; bottom:0; }} .shap .mid {{ position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--muted); }}
.statuses {{ display:flex; flex-wrap:wrap; gap:.3rem; }} .st {{ font:400 .69rem 'IBM Plex Mono',monospace; padding:.2rem .4rem; border:1px solid var(--rule); color:var(--muted); }} .st.on {{ color:var(--paper); border-color:var(--soft); }}
.unavailable {{ border:1px dashed var(--rule); padding:.7rem .8rem; color:var(--muted); font-size:.82rem; line-height:1.4; }} .unavailable b {{ color:var(--soft); font-weight:500; }}
@media (max-width: 900px) {{ .bento {{ grid-template-columns:repeat(2,1fr); }} .cell.s2,.cell.s3,.cell.s4,.cell.s6 {{ grid-column:span 2; }} .head {{ flex-direction:column; gap:.3rem; }} .block-container {{ padding:4.5rem 1rem 2rem !important; }} }}
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
