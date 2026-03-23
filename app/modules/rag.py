"""
modules/rag.py
==============
Retrieval and generation pipeline.

Classes
-------
QueryEmbedder
    Wraps a SentenceTransformer bi-encoder; produces normalised query vectors.

Ranker (base)
    Shared rank-by-position and cross-encoder rerank utilities.

TitleRanker
    PostgreSQL trigram similarity (pg_trgm) over application names.
    Handles queries like "dark souls" where the user has a specific title in mind.

UsertagRanker
    Scores apps by user-tag match, boosted by a PMI co-occurrence graph.
    At init time, builds the full tag×app PMI graph from the database.
    Also exposes suggest_tags() for query-driven tag recommendation.

DescriptionRanker
    ANN (pgvector) + keyword union → cross-encoder rerank over short descriptions.

ReviewRanker
    ANN (pgvector) + keyword union → cross-encoder rerank over review text.
    Returns one row per app (highest-ranked review per game).

SentenceRanker
    Post-RRF sentence-level snippet retrieval for the selected top-N apps.
    Used to populate the "Reviewers have said..." panel and the RAG prompt.

Retriever
    Facade that owns all five rankers and exposes:
        rrf()               — Reciprocal Rank Fusion over the four main rankers
        suggest_tags()      — query → top-K matching user tags
        select_review_snippets() — per-app sentence retrieval

Generator
    Wraps a causal LM (via transformers) to produce game recommendation pitches
    from a query, game description, and review snippets.
"""

from typing import Iterable, List, Optional, Union, Dict, Tuple
from collections import defaultdict
from itertools import combinations
import logging

import numpy as np
import pandas as pd
import sentence_transformers
import transformers
import torch
import torch.nn.functional as F

from .db import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class QueryEmbedder:
    """Wraps a biencoder and produces normalized query embeddings."""

    def __init__(
        self,
        model_name: str,
        prefix: str = "search_query: ",
        device="cpu",
    ):
        self.nomic = model_name.split("/")[0] == "nomic-ai"
        if self.nomic:
            logger.info("Nomic model detected. Trusting remote code.")
            logger.info("Use kwarg 'matryoshka_dim' to specify the matryoshka dimension.")
        self.biencoder = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=self.nomic
        ).to(device)
        self.prefix = prefix

    def _postprocess(self, embeddings: torch.Tensor, **kwargs) -> np.ndarray:
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if self.nomic:
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[
                :, : kwargs.get("matryoshka_dim", self.biencoder.get_sentence_embedding_dimension())
            ]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.squeeze(0).cpu().numpy()
        return embeddings.squeeze(0).cpu().numpy()

    def embed(self, query_text: str, **kwargs) -> np.ndarray:
        embeddings = self.biencoder.encode(
            f"{self.prefix}{query_text}",
            convert_to_tensor=True,
            normalize_embeddings=not self.nomic,
        )
        return self._postprocess(embeddings, **kwargs)


# ---------------------------------------------------------------------------
# Base ranker
# ---------------------------------------------------------------------------

class Ranker:

    def __init__(
        self,
        database: Database,
        reranker: sentence_transformers.CrossEncoder,
        top_k: int = 50,
    ):
        self.database = database
        self.reranker = reranker
        self.top_k = top_k

    def rank_by_position(self, df: pd.DataFrame, id_column: str = "app_id") -> pd.DataFrame:
        """Assign rank by retrieval order, deduplicating on id_column."""
        df = df.copy()
        df = df.drop_duplicates(subset=id_column, keep="first")
        df["rank"] = np.arange(1, len(df) + 1, dtype=float)
        return df.reset_index(drop=True)

    def rerank(
        self,
        df: pd.DataFrame,
        query_text: str,
        text_column: str,
        id_column: str = "app_id",
        score_column: str = "rerank_score",
        ascending: bool = False,
    ) -> pd.DataFrame:
        texts = df[text_column].fillna("").astype(str).tolist()
        pairs = [(query_text, t) for t in texts]
        scores = self.reranker.predict(pairs)

        df = df.copy()
        df[score_column] = scores
        df["rank"] = df[score_column].rank(ascending=False, method="average")
        df = (
            df.sort_values(score_column, ascending=ascending)
            .drop_duplicates(subset=id_column, keep="first")
        )

        if self.top_k:
            df = df.head(self.top_k)

        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Title ranker  (keyword only, trigram similarity)
# ---------------------------------------------------------------------------

class TitleRanker(Ranker):
    """
    Keyword-only ranker over application titles using trigram similarity.

    Intended as a fourth RRF signal, not a replacement for DescriptionRanker.
    Handles queries like "dark souls" or "witcher" where the user has a
    specific title in mind. No reranker — trigram similarity is the score.
    """

    def rank_by_title(self, query_text: str) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            name,
            similarity(name, %s) AS trgm_score
        FROM applications
        WHERE name %% %s
        ORDER BY trgm_score DESC
        LIMIT %s
        """
        df = self.database.execute_query(sql, (query_text, query_text, self.top_k))

        if df.empty:
            return pd.DataFrame(columns=["app_id", "trgm_score", "rank"])

        df["rank"] = df["trgm_score"].rank(ascending=False, method="average")
        return df.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Usertag ranker  (ANN + PMI graph, no keyword search)
# ---------------------------------------------------------------------------

class UsertagRanker(Ranker):
    """
    Ranks apps by usertag match with silent PMI-based neighbor boosting.

    Flow
    ----
    1. At init, build a PMI co-occurrence graph from the full tag×app matrix.
    2. suggest_tags() picks the top-3 anchor tags from the query.
    3. rank_by_tags() scores every app as:

         score(app) = Σ anchor_score(t)                          for t in anchor_tags ∩ app_tags
                    + Σ pmi(t, n) * anchor_score(t) * decay      for t in anchor_tags,
                                                                      n in neighbors(t) ∩ app_tags

       One score column — RRF is untouched.
    """

    DEFAULT_NEIGHBOR_DECAY: float = 0.5

    def __init__(
        self,
        database: Database,
        reranker: sentence_transformers.CrossEncoder,
        rank_score_alpha: float = 0.3,
        rerank: bool = True,
        neighbor_decay: float = DEFAULT_NEIGHBOR_DECAY,
        pmi_min: float = 0.0,
        pmi_top_k: int = 20,
        **kwargs,
    ):
        super().__init__(database, reranker)
        self.alpha = rank_score_alpha
        self.rerank_enabled = rerank
        self.neighbor_decay = neighbor_decay
        self.pmi_min = pmi_min
        self.pmi_top_k = pmi_top_k

        self.usertag_rankings = self._rank_applications_usertags()
        self.pmi_graph: Dict[str, Dict[str, float]] = self._build_pmi_graph()

        logger.info(
            "PMI graph built: %d tags, mean degree %.1f",
            len(self.pmi_graph),
            np.mean([len(v) for v in self.pmi_graph.values()]) if self.pmi_graph else 0,
        )

    def _compute_rank_score(self, rank: int) -> float:
        return np.exp(-self.alpha * (rank - 1))

    def _rank_applications_usertags(self) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            usertag_id,
            usertag,
            RANK() OVER (
                PARTITION BY app_id
                ORDER BY votes DESC
            ) AS rank
        FROM applications_usertags
        LEFT JOIN usertags USING (usertag_id)
        """
        df = self.database.execute_query(sql)
        df["score"] = df["rank"].map(self._compute_rank_score)
        return df[["app_id", "usertag_id", "usertag", "score"]]

    def group_usertag_rankings(self) -> pd.DataFrame:
        return (
            self.usertag_rankings
            .groupby("app_id")
            .agg(tags=("usertag", list))
            .reset_index()
        )

    def _build_pmi_graph(self) -> Dict[str, Dict[str, float]]:
        grouped = self.group_usertag_rankings()
        n_apps = len(grouped)

        if n_apps == 0:
            logger.warning("No app-tag data found; PMI graph will be empty.")
            return {}

        tag_counts: Dict[str, int] = defaultdict(int)
        for tags in grouped["tags"]:
            for t in tags:
                tag_counts[t] += 1

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for tags in grouped["tags"]:
            for a, b in combinations(sorted(set(tags)), 2):
                pair_counts[(a, b)] += 1

        graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        log_n = np.log2(n_apps)

        for (a, b), co_count in pair_counts.items():
            if co_count == 0:
                continue
            pmi = (
                np.log2(co_count)
                - np.log2(tag_counts[a])
                - np.log2(tag_counts[b])
                + log_n
            )
            if pmi <= self.pmi_min:
                continue
            graph[a][b] = pmi
            graph[b][a] = pmi

        pruned: Dict[str, Dict[str, float]] = {}
        for tag, neighbors in graph.items():
            top = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
            pruned[tag] = dict(top[: self.pmi_top_k])

        return pruned

    def suggest_tags(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        final_top_k: int = 3,
    ) -> List[str]:
        sql = """
        SELECT
            usertag_id,
            usertag,
            usertag_description
        FROM public.usertags
        WHERE usertag_embedding IS NOT NULL
        ORDER BY usertag_embedding <-> %s::vector
        LIMIT %s
        """
        df = self.database.execute_query(sql, (query_embedding.tolist(), self.top_k))

        if self.rerank_enabled:
            df = self.rerank(df, query_text, text_column="usertag_description", id_column="usertag_id")
        else:
            df = self.rank_by_position(df, id_column="usertag_id")

        return df["usertag"].tolist()[:final_top_k]

    def rank_by_tags(self, tags: Iterable[str]) -> pd.DataFrame:
        anchor_tags = list(tags)
        rnk = self.usertag_rankings.copy()

        anchor_df = (
            rnk[rnk["usertag"].isin(anchor_tags)]
            .groupby("app_id", as_index=False)
            .agg(anchor_score=("score", "sum"))
        )

        neighbor_rows = []
        for anchor in anchor_tags:
            neighbors = self.pmi_graph.get(anchor, {})
            if not neighbors:
                continue

            anchor_scores_for_apps = rnk[rnk["usertag"] == anchor].set_index("app_id")["score"]

            for neighbor_tag, pmi_weight in neighbors.items():
                neighbor_apps = rnk[rnk["usertag"] == neighbor_tag][["app_id", "score"]]
                if neighbor_apps.empty:
                    continue

                neighbor_apps = neighbor_apps[
                    neighbor_apps["app_id"].isin(anchor_scores_for_apps.index)
                ].copy()
                if neighbor_apps.empty:
                    continue

                neighbor_apps["anchor_score"] = neighbor_apps["app_id"].map(anchor_scores_for_apps)
                neighbor_apps["neighbor_contribution"] = (
                    pmi_weight * neighbor_apps["anchor_score"] * self.neighbor_decay
                )
                neighbor_rows.append(neighbor_apps[["app_id", "neighbor_contribution"]])

        if neighbor_rows:
            neighbor_df = (
                pd.concat(neighbor_rows, ignore_index=True)
                .groupby("app_id", as_index=False)
                .agg(neighbor_score=("neighbor_contribution", "sum"))
            )
        else:
            neighbor_df = pd.DataFrame(columns=["app_id", "neighbor_score"])

        result = anchor_df.merge(neighbor_df, on="app_id", how="outer").fillna(0.0)
        result["score"] = result["anchor_score"] + result["neighbor_score"]
        result = result[["app_id", "score"]]
        result["rank"] = result["score"].rank(ascending=False, method="average")
        return result.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Description ranker  (ANN ∪ keyword → rerank)
# ---------------------------------------------------------------------------

class DescriptionRanker(Ranker):

    def __init__(
        self,
        database: Database,
        reranker: sentence_transformers.CrossEncoder,
        **kwargs,
    ):
        super().__init__(database, reranker)

    def _fetch_by_ann(self, query_embedding: np.ndarray) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            name,
            short_description,
            header_image
        FROM public.applications
        WHERE description_embedding IS NOT NULL
        ORDER BY description_embedding <-> %s::vector
        LIMIT %s
        """
        return self.database.execute_query(sql, (query_embedding.tolist(), self.top_k))

    def _fetch_by_keyword(self, query_text: str) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            name,
            short_description,
            header_image
        FROM public.applications
        WHERE search_fts @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(search_fts, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """
        return self.database.execute_query(sql, (query_text, query_text, self.top_k))

    def rank_by_description(
        self,
        query_text: str,
        query_embedding: np.ndarray,
    ) -> pd.DataFrame:
        ann_df = self._fetch_by_ann(query_embedding)
        kw_df = self._fetch_by_keyword(query_text)

        # Union: ANN results take priority on dedup (keep first = ANN row)
        df = (
            pd.concat([ann_df, kw_df], ignore_index=True)
            .drop_duplicates(subset="app_id", keep="first")
        )

        return self.rerank(df, query_text, text_column="short_description")


# ---------------------------------------------------------------------------
# Review ranker  (ANN ∪ keyword → rerank)
# ---------------------------------------------------------------------------

class ReviewRanker(Ranker):

    def __init__(
        self,
        database: Database,
        reranker: sentence_transformers.CrossEncoder,
        **kwargs,
    ):
        super().__init__(database, reranker)

    def _fetch_by_ann(self, query_embedding: np.ndarray) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            review_id,
            review
        FROM reviews
        ORDER BY review_embedding <-> %s::vector
        LIMIT %s
        """
        return self.database.execute_query(sql, (query_embedding.tolist(), self.top_k))

    def _fetch_by_keyword(self, query_text: str) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            review_id,
            review
        FROM reviews
        WHERE search_fts @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(search_fts, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """
        return self.database.execute_query(sql, (query_text, query_text, self.top_k))

    def rank_by_review(
        self,
        query_text: str,
        query_embedding: np.ndarray,
    ) -> pd.DataFrame:
        ann_df = self._fetch_by_ann(query_embedding)
        kw_df = self._fetch_by_keyword(query_text)

        df = (
            pd.concat([ann_df, kw_df], ignore_index=True)
            .drop_duplicates(subset="review_id", keep="first")
        )

        df = self.rerank(df, query_text, text_column="review", id_column="review_id")

        # One row per app — keep the highest-ranked review per game
        df = (
            df.sort_values("rank")
            .drop_duplicates(subset="app_id", keep="first")
            .reset_index(drop=True)
        )
        df["rank"] = np.arange(1, len(df) + 1, dtype=float)
        return df


# ---------------------------------------------------------------------------
# Sentence ranker  (ANN ∪ keyword → rerank, runs post-RRF)
# ---------------------------------------------------------------------------

class SentenceRanker(Ranker):
    """
    Retrieves and reranks sentence-level snippets for a set of candidate
    app_ids. Runs after RRF, not during it.
    """

    def __init__(
        self,
        database: Database,
        reranker: sentence_transformers.CrossEncoder,
        top_k: int = 32,
        **kwargs,
    ):
        super().__init__(database, reranker, top_k=top_k)

    def _fetch_by_ann(self, query_embedding: np.ndarray, app_id: int) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            sentence_id,
            sentence,
            sentence_embedding <-> %s::vector AS distance
        FROM sentences
        WHERE app_id = %s
        ORDER BY distance
        LIMIT %s
        """
        return self.database.execute_query(sql, (query_embedding.tolist(), app_id, self.top_k))

    def _fetch_by_keyword(self, query_text: str, app_id: int) -> pd.DataFrame:
        sql = """
        SELECT
            app_id,
            sentence_id,
            sentence,
            NULL AS distance
        FROM sentences
        WHERE app_id = %s
          AND search_fts @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(search_fts, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """
        return self.database.execute_query(sql, (app_id, query_text, query_text, self.top_k))

    def select_review_snippets(
        self,
        app_ids: Union[int, List[int]],
        query_text: str,
        query_embedding: np.ndarray,
    ) -> pd.DataFrame:
        if isinstance(app_ids, int):
            app_ids = [app_ids]

        results = []
        for aid in app_ids:
            ann_df = self._fetch_by_ann(query_embedding, aid)
            kw_df = self._fetch_by_keyword(query_text, aid)

            df = (
                pd.concat([ann_df, kw_df], ignore_index=True)
                .drop_duplicates(subset="sentence_id", keep="first")
            )

            if df.empty:
                continue

            result = self.rerank(df, query_text, text_column="sentence", id_column="sentence_id")
            result["app_id"] = aid
            results.append(result)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Retriever (facade)
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(
        self,
        database: Database,
        query_embedder: QueryEmbedder,
        small_query_embedder: QueryEmbedder,
        reranker: sentence_transformers.CrossEncoder,
        rerank_tags: bool = True,
        **kwargs,
    ):
        self.database = database
        self.embedder = query_embedder
        self.sm_embedder = small_query_embedder
        self.sentence_dim = kwargs.get(
            "sentence_dim",
            query_embedder.biencoder.get_sentence_embedding_dimension(),
        )

        self.title_ranker    = TitleRanker(database, reranker, **kwargs)
        self.desc_ranker     = DescriptionRanker(database, reranker, **kwargs)
        self.tag_ranker      = UsertagRanker(database, reranker, rerank=rerank_tags, **kwargs)
        self.review_ranker   = ReviewRanker(database, reranker, **kwargs)
        self.sentence_ranker = SentenceRanker(database, reranker, **kwargs)

    def suggest_tags(self, query_text: str) -> List[str]:
        embedding = self.embedder.embed(query_text)
        return self.tag_ranker.suggest_tags(query_text, embedding)

    def select_review_snippets(
        self,
        app_ids: Union[int, List[int]],
        query_text: str,
        **kwargs,
    ) -> pd.DataFrame:
        embedding = self.sm_embedder.embed(query_text, matryoshka_dim=self.sentence_dim)
        return self.sentence_ranker.select_review_snippets(app_ids, query_text, embedding)

    # Canonical ranker names — order matches the `ranked` list in rrf()
    RANKER_NAMES = ["Title", "Description", "Tags", "Reviews"]

    def rrf(
        self,
        query_text: str,
        tags: List[str],
        k: int = 10,
        active_rankers: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Reciprocal Rank Fusion over title, description, tag, and review rankings.

        Parameters
        ----------
        active_rankers : list of str or None
            Subset of RANKER_NAMES to include. None means all four are active.
        """
        # Empty selection → fall back to the last ranker that was passed in;
        # if nothing was passed at all, use all four.
        if active_rankers is not None and len(active_rankers) == 0:
            active = {self.RANKER_NAMES[-1]}   # last in canonical order
        else:
            active = set(active_rankers) if active_rankers else set(self.RANKER_NAMES)

        embedding = self.embedder.embed(query_text)

        ranked = [
            self.title_ranker.rank_by_title(query_text)
                if "Title"       in active else pd.DataFrame(),
            self.desc_ranker.rank_by_description(query_text, embedding)
                if "Description" in active else pd.DataFrame(),
            self.tag_ranker.rank_by_tags(tags)
                if "Tags"        in active else pd.DataFrame(),
            self.review_ranker.rank_by_review(query_text, embedding)
                if "Reviews"     in active else pd.DataFrame(),
        ]

        fused: Optional[pd.DataFrame] = None

        for i, df in enumerate(ranked):
            if df.empty:
                continue
            df = df[["app_id", "rank"]].copy()
            df[f"rrf_{i}"] = 1.0 / (k + df["rank"])
            df = df.drop(columns="rank")
            fused = df if fused is None else fused.merge(df, on="app_id", how="outer")

        if fused is None:
            return pd.DataFrame()

        fused = fused.fillna(0)
        rrf_cols = [c for c in fused.columns if c.startswith("rrf_")]
        fused["rrf_score"] = fused[rrf_cols].sum(axis=1)
        fused["rank"] = fused["rrf_score"].rank(ascending=False, method="average")

        return fused.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:

    def __init__(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _build_prompt(
        query_text: str,
        title: str,
        description: str,
        snippets: Optional[List[str]] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        prompt = template or (
            "Based on the users query, an excellent game has been selected. "
            "Your job is to concisely recommend the game, based on information "
            "stated in the description and the review snippets.\n\n"
            "You must take care to use the proper game title exactly as it is written.\n"
            "If possible, emphasize information from the description or review snippets "
            "that align well with the query.\n"
            "There is no need to attribute the source of your information. Avoid phrases "
            "like 'based on the review snippets', 'according to the description', or "
            "'aligns well with the query'.\n\n"
            "The query, title, game description, and review snippets are provided below.\n"
            "Context:\n```"
        )

        context_parts = [
            f"Query:\n{query_text}",
            f"Title:\n{title}",
            f"Game Description:\n{description}",
        ]

        if snippets:
            context_parts.append(f"Review Snippets:\n" + "\n".join(snippets))

        full_prompt = f"{prompt}\n" + "\n\n".join(context_parts) + "\n```"

        return [
            {
                "role": "system",
                "content": system_prompt
                or "You are a helpful assistant within a RAG pipeline for video game recommendation.",
            },
            {"role": "user", "content": full_prompt},
        ]

    def _generate_response(self, messages: List[dict], **kwargs) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        default_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": True,
            "temperature": 0.3,
            "min_p": 0.15,
            "repetition_penalty": 1.05,
            "max_new_tokens": 200,
        }

        output = self.model.generate(**{**default_kwargs, **kwargs})
        generated_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def rag(
        self,
        query_text: str,
        title: str,
        description: str,
        snippets: Optional[List[str]] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = self._build_prompt(
            query_text, title, description, snippets, template, system_prompt
        )
        return self._generate_response(messages, **kwargs)