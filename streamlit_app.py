"""DWELL: maritime logistics intelligence for Terminal Petikemas Surabaya.

Streamlit entry point. Presentation lives in ML/integration_engineer/control_tower/;
every number comes from committed assets built by control_tower/build_assets.py.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ML.integration_engineer.control_tower import theme  # noqa: E402
from ML.integration_engineer.control_tower.views import model, ops_queue, overview, port, triage  # noqa: E402

st.set_page_config(page_title="DWELL · TPS Surabaya", page_icon=str(REPO_ROOT / "ML/integration_engineer/control_tower/assets/mark.svg"), layout="wide", initial_sidebar_state="collapsed")
st.logo(str(REPO_ROOT / "ML/integration_engineer/control_tower/assets/wordmark.svg"), size="large")
theme.inject_css()

pages = [
    st.Page(overview.render, title="Overview", url_path="overview", default=True),
    st.Page(port.render, title="Port intelligence", url_path="port"),
    st.Page(triage.render, title="Container triage", url_path="triage"),
    st.Page(model.render, title="Model intelligence", url_path="model"),
    st.Page(ops_queue.render, title="Operations queue", url_path="queue"),
]
st.navigation(pages, position="top").run()
