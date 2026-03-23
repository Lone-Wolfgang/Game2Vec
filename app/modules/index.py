"""
modules/index.py
================
Offline preprocessing pipeline — not used at inference time.

Run once to populate the `sentences` and `reviews` tables before starting the app.

Classes
-------
SentenceExtractor
    Uses spaCy (tok2vec + parser) to segment review text into sentences,
    splitting first on newlines then on syntactic sentence boundaries.

Preprocessor
    Orchestrates the two-phase indexing pipeline:

    Phase 1 — extract_sentences()
        Filter short reviews → spaCy segmentation → filter short/long sentences
        → flat DataFrame of (sentence_id, review_id, app_id, sentence).

    Phase 2 — embed_sentences()
        Embed each sentence with a SentenceTransformer → adds sentence_embedding column.

    process_sentences() runs both phases in one call.
    process_reviews() embeds full review text (used for the Review ranker).
"""

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm.auto import tqdm


# ====================================================
# Sentence Extraction
# ====================================================
class SentenceExtractor:

    def __init__(
        self,
        nlp: spacy.language.Language,
        batch_size: int = 1000,
        n_process: int = 1,
    ):
        self.nlp = nlp
        self.batch_size = batch_size
        self.n_process = n_process

        self.nlp.select_pipes(enable=["tok2vec", "parser"])

    def extract_batch(self, reviews: List[str]) -> List[List[str]]:
        """
        Splits each review on newlines, applies spaCy senter to each paragraph,
        and flattens results back per review.

        Returns:
            List[List[str]] — one list of sentence strings per review.
        """
        review_paragraphs = [r.split("\n") for r in reviews]
        flat_paragraphs = [p for paras in review_paragraphs for p in paras if p.strip()]
        para_counts = [len([p for p in paras if p.strip()]) for paras in review_paragraphs]

        docs = self.nlp.pipe(
            flat_paragraphs,
            batch_size=self.batch_size,
            n_process=self.n_process,
        )

        flat_sentences = [
            [sent.text for sent in doc.sents]
            for doc in tqdm(docs, total=len(flat_paragraphs), desc="Extracting sentences", unit="para")
        ]

        sentences_per_review = []
        idx = 0
        for count in para_counts:
            review_sents = [s for para_sents in flat_sentences[idx : idx + count] for s in para_sents]
            sentences_per_review.append(review_sents)
            idx += count

        return sentences_per_review


# ====================================================
# Preprocessor
# ====================================================
class Preprocessor:

    def __init__(
        self,
        sentence_extractor: SentenceExtractor,
        model: SentenceTransformer,
        encode_kwargs: dict | None = None,
    ):
        """
        Args:
            sentence_extractor: SentenceExtractor instance for spaCy segmentation.
            model:              SentenceTransformer used for all embedding calls.
            encode_kwargs:      Extra kwargs forwarded to model.encode()
                                (e.g. {"batch_size": 256, "normalize_embeddings": True}).
        """
        self.sentence_extractor = sentence_extractor
        self.model = model
        self.encode_kwargs = encode_kwargs or {}

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _filter_short_reviews(
        df: pd.DataFrame,
        min_review_length: int = 32,
    ) -> pd.DataFrame:
        tqdm.pandas(desc="Filtering short reviews")
        mask = df.review.progress_apply(lambda x: len(x.split()) >= min_review_length)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _filter_sentences(
        sentences_batch: List[List[str]],
        min_words: int = 8,
        max_words: int = 32,
    ) -> List[List[str]]:
        return [
            [s for s in sentences if min_words <= len(s.split()) <= max_words]
            for sentences in sentences_batch
        ]

    def _embed(self, texts: List[str]) -> list:
        return self.model.encode(texts, **self.encode_kwargs).tolist()

    # --------------------------------------------------
    # process_reviews()
    # --------------------------------------------------
    def process_reviews(
        self,
        reviews: pd.DataFrame,
        min_review_length: int = 32,
    ) -> pd.DataFrame:
        """
        Embeds each review as a whole.

        Input:
            reviews : DataFrame with at least [review_id, review] columns,
                      plus any additional columns which are preserved.

        Output:
            DataFrame with all original columns plus [review_embedding].
        """
        df = reviews.copy()
        # df = self._filter_short_reviews(reviews.copy(), min_review_length)
        # if df.empty:
        #     print("No reviews remaining after filtering. Exiting.")
        #     return df

        df["review_embedding"] = self._embed(df.review.tolist())

        return df.reset_index(drop=True)

    # --------------------------------------------------
    # extract_sentences()   ← Phase 1: spaCy only
    # --------------------------------------------------
    def extract_sentences(
        self,
        reviews: pd.DataFrame,
        min_review_length: int = 32,
        min_sentence_words: int = 8,
        max_sentence_words: int = 32,
    ) -> pd.DataFrame:
        """
        Segments reviews into sentences using spaCy. No embeddings.

        Input:
            reviews : DataFrame with at least [review_id, app_id, review] columns.

        Output:
            Flat DataFrame with one row per sentence, no embeddings:
            [sentence_id, review_id, app_id, sentence]
        """
        df = self._filter_short_reviews(reviews.copy(), min_review_length)
        if df.empty:
            print("No reviews remaining after filtering. Exiting.")
            return df

        sentences_batch = self.sentence_extractor.extract_batch(df.review.tolist())
        filtered_sentences = self._filter_sentences(sentences_batch, min_sentence_words, max_sentence_words)

        df["_sentences"] = filtered_sentences

        has_sentences = df["_sentences"].apply(len) > 0
        n_dropped = (~has_sentences).sum()
        if n_dropped > 0:
            print(f"Dropping {n_dropped} review(s) with 0 sentences after length filtering.")
        df = df.loc[has_sentences].reset_index(drop=True)

        if df.empty:
            print("No sentences remaining after filtering. Exiting.")
            return df

        review_ids = df.review_id.tolist()
        app_ids = df.app_id.tolist()
        doc_sizes = df["_sentences"].apply(len).tolist()
        flat_sentences = [s for doc in df["_sentences"] for s in doc]

        rows = []
        flat_idx = 0

        for review_id, app_id, n_sents in tqdm(
            zip(review_ids, app_ids, doc_sizes),
            total=len(review_ids),
            desc="Building sentence table",
            unit="review",
        ):
            for sent_idx in range(n_sents):
                rows.append(
                    {
                        "sentence_id": f"{review_id}-{str(sent_idx).zfill(3)}",
                        "review_id": review_id,
                        "app_id": app_id,
                        "sentence": flat_sentences[flat_idx],
                    }
                )
                flat_idx += 1

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # embed_sentences()   ← Phase 2: embed only
    # --------------------------------------------------
    def embed_sentences(
        self,
        sentences: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Embeds a flat sentences DataFrame produced by extract_sentences().

        Input:
            sentences : DataFrame with at least [sentence_id, review_id, app_id, sentence] columns.

        Output:
            Same DataFrame with [sentence_embedding] column added.
        """
        df = sentences.copy()
        df["sentence_embedding"] = self._embed(df.sentence.tolist())
        return df

    # --------------------------------------------------
    # process_sentences()  ← convenience: Phase 1 + 2 combined
    # --------------------------------------------------
    def process_sentences(
        self,
        reviews: pd.DataFrame,
        min_review_length: int = 32,
        min_sentence_words: int = 8,
        max_sentence_words: int = 32,
    ) -> pd.DataFrame:
        """
        Convenience method: extract + embed in one call.

        Input:
            reviews : DataFrame with at least [review_id, app_id, review] columns.

        Output:
            Flat DataFrame with one row per sentence:
            [sentence_id, review_id, app_id, sentence, sentence_embedding]
        """
        extracted = self.extract_sentences(
            reviews,
            min_review_length=min_review_length,
            min_sentence_words=min_sentence_words,
            max_sentence_words=max_sentence_words,
        )
        if extracted.empty:
            return extracted
        return self.embed_sentences(extracted)