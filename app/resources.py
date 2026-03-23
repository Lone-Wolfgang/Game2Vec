"""
resources.py
============
Cached resource loaders for all heavy objects.

Every function is decorated with @st.cache_resource so the object is
instantiated once per process and shared across all Streamlit reruns and users.

Load order: device → database → embedders → reranker → retriever → genai → library.
"""

import torch
import streamlit as st
import sentence_transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.db import Database
from modules.rag import Retriever, Generator, QueryEmbedder
from config import GENAI, BIENCODER, CONNECTION, RERANKER
import pandas as pd


@st.cache_resource
def get_device() -> str:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def get_database() -> Database:
    """Open and cache a persistent PostgreSQL connection."""
    return Database(**CONNECTION)


@st.cache_resource
def get_embedder(model_path: str) -> QueryEmbedder:
    """Load a bi-encoder and move it to the best available device."""
    return QueryEmbedder(model_path, device=get_device())


@st.cache_resource
def get_reranker() -> sentence_transformers.CrossEncoder:
    """Load the cross-encoder reranker."""
    return sentence_transformers.CrossEncoder(RERANKER).to(get_device())


@st.cache_resource
def get_genai() -> Generator:
    """Load the generative model for RAG pitches (bfloat16, device_map=auto)."""
    model = AutoModelForCausalLM.from_pretrained(
        GENAI,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(GENAI)
    return Generator(model, tokenizer)


@st.cache_resource
def get_retriever() -> Retriever:
    """
    Build the Retriever, which at init time:
    - loads usertag rankings from the database
    - builds the PMI co-occurrence graph over all app–tag pairs
    """
    return Retriever(
        database=get_database(),
        query_embedder=get_embedder(BIENCODER["base"]),
        small_query_embedder=get_embedder(BIENCODER["small"]),
        reranker=get_reranker(),
        rerank_tags=False,
    )


@st.cache_resource
def get_library() -> "pd.DataFrame":
    """
    Fetch the application catalogue (name, header image, description) and
    join it with grouped usertag lists for display in the results panel.
    """
    retriever = get_retriever()
    sql = """
        SELECT app_id, name, header_image, short_description
        FROM applications;
    """
    library = retriever.database.execute_query(sql)
    tags    = retriever.tag_ranker.group_usertag_rankings()
    return library.merge(tags, on="app_id", how="left")