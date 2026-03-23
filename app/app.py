"""
app.py
======
Entry point. Run with:

    streamlit run app.py

All heavy resources (database connection, embedders, reranker, generative model)
are loaded once via @st.cache_resource in resources.py and reused across reruns.
"""

import streamlit as st

from config import CSS, TITLE
from modules.ux import SessionManager
from resources import get_library, get_retriever, get_genai


with st.spinner("Loading models..."):
    retriever = get_retriever()
    genai     = get_genai()
    library   = get_library()

app = SessionManager(CSS, TITLE, retriever, genai, library)
app.run()