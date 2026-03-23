import html

import pandas as pd
import streamlit as st

from .pmi import render_pmi_explorer
from .rag import Generator, Retriever


class SessionManager:

    def __init__(
        self,
        css: str,
        title: str,
        retriever: Retriever,
        genai: Generator,
        library: pd.DataFrame,
        initial_state: dict = None,
    ):
        self.initial_state = initial_state or {
            "query": "",
            "suggested_tags": [],
            "selected_tags": [],
            "searched_tags": [],
            "cached_results": None,
            "_last_query": None,
            "pinned_app_id": None,
            "cached_reviews": {},
            "cached_response": {},      # app_id -> str, persists until reset/new query
            "_just_reset": False,
            "_initialized": True,
            "pmi_mode": False,
            "active_rankers": ["Title", "Description", "Tags", "Reviews"],
            "searched_rankers": ["Title", "Description", "Tags", "Reviews"],
        }

        self.retriever = retriever
        self.genai = genai
        self.library = library

        self._init_page(css, title)
        self.sidebar = st.sidebar

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_page(self, css: str, title: str) -> None:
        if not st.session_state.get("_initialized"):
            st.set_page_config(layout="wide")
            for k, v in self.initial_state.items():
                st.session_state[k] = v

        st.markdown(css, unsafe_allow_html=True)
        st.title(title)

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for k, v in self.initial_state.items():
            st.session_state[k] = v

    def search(self) -> None:
        with st.spinner("Searching..."):
            results = self.retriever.rrf(
                st.session_state.query,
                st.session_state.selected_tags,
                active_rankers=st.session_state.active_rankers,
            )

        st.session_state.cached_results = results
        st.session_state.searched_tags = st.session_state.selected_tags.copy()
        st.session_state.searched_rankers = st.session_state.active_rankers.copy()
        st.session_state.cached_reviews = {}
        st.session_state.cached_response = {}      # wipe all generations on new search

        top_app_ids = results["app_id"].iloc[:5].tolist()
        st.session_state.pinned_app_id = top_app_ids[0]

        self._fetch_all(top_app_ids)

    def pin_game(self, app_id: int) -> None:
        st.session_state.pinned_app_id = app_id

        games = st.session_state.cached_results
        if app_id != games.loc[0, "app_id"]:
            pinned = games[games["app_id"] == app_id]
            others = games[games["app_id"] != app_id]
            st.session_state.cached_results = pd.concat(
                [pinned, others], ignore_index=True
            )

    def generate(self) -> None:
        """Generate a pitch for the currently pinned game and cache it."""
        app_id = st.session_state.pinned_app_id

        game_row = (
            st.session_state.cached_results[st.session_state.cached_results["app_id"] == app_id]
            .iloc[[0]]
            .merge(self.library, on="app_id", how="left")
            .iloc[0]
        )
        snippets = st.session_state.cached_reviews.get(app_id)

        with st.spinner(f"Generating pitch for {game_row['name']}..."):
            st.session_state.cached_response[app_id] = self.genai.rag(
                st.session_state.query,
                title=game_row["name"],
                description=game_row["short_description"],
                snippets=snippets["sentence"].tolist()[:10] if snippets is not None and not snippets.empty else [],
            )

    def _fetch_all(self, app_ids: list) -> None:
        """Fetch snippets for all top results. Does not generate pitches."""
        with st.spinner("Retrieving snippets..."):
            all_snippets = self.retriever.select_review_snippets(
                app_ids,
                st.session_state.query,
            )

        for app_id in app_ids:
            snippets = all_snippets[all_snippets["app_id"] == app_id].reset_index(drop=True)
            st.session_state.cached_reviews[app_id] = snippets

    def input_query(self) -> None:
        query = st.session_state.query
        if not query or query == st.session_state._last_query:
            return
        with st.spinner("Suggesting tags..."):
            suggested = self.retriever.suggest_tags(query)
        st.session_state.suggested_tags = suggested
        st.session_state._last_query = query
        if st.session_state.pmi_mode:
            st.session_state.selected_tags = list(suggested)
        else:
            st.session_state.selected_tags = suggested.copy()
            self.search()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_control_panel(self) -> None:
        with self.sidebar:
            st.header("Controls")

            st.text_input(
                "Search",
                key="query",
                placeholder="e.g. cooperative survival crafting game",
                on_change=self.input_query,
            )

            st.multiselect(
                "User Tags",
                options=sorted(
                    self.retriever.tag_ranker.usertag_rankings["usertag"].unique()
                ),
                key="selected_tags",
            )

            pmi_label = "✕ Close Graph" if st.session_state.pmi_mode else "🕸️ Tag Graph"
            st.button(pmi_label, on_click=self._toggle_pmi)

            if st.session_state.pmi_mode:
                st.caption("Click nodes or neighbor rows to add tags to User Tags.")

            with st.expander("⚙️ Rankers", expanded=False):
                st.caption(
                    "If all rankers are disabled, search will fall back to "
                    "the most recently active one."
                )
                all_names = self.retriever.RANKER_NAMES
                current   = st.session_state.active_rankers
                for name in all_names:
                    toggled = st.toggle(
                        name,
                        value=name in current,
                        key=f"ranker_{name}",
                    )
                    if toggled and name not in current:
                        st.session_state.active_rankers = current + [name]
                    elif not toggled and name in current:
                        st.session_state.active_rankers = [r for r in current if r != name]

            # Show Search whenever query, tags, or ranker selection has changed
            if (
                st.session_state.query
                and not st.session_state.pmi_mode
                and (
                    st.session_state.selected_tags != st.session_state.searched_tags
                    or st.session_state.active_rankers != st.session_state.searched_rankers
                )
            ):
                st.button("Search", on_click=self.search)

            st.button("Reset", on_click=self.reset)

    def render_pinned_panel(self) -> None:
        results = st.session_state.cached_results
        if results is None or results.empty:
            return

        app_id = st.session_state.pinned_app_id
        top = (
            results[results["app_id"] == app_id]
            .iloc[[0]]
            .merge(self.library, on="app_id", how="left")
        )
        game = top.iloc[0]

        snippets = st.session_state.cached_reviews.get(app_id)[:5]
        response = st.session_state.cached_response.get(app_id)

        st.header("We recommend. . .")
        st.markdown("<div class='game-card'>", unsafe_allow_html=True)

        col_game, col_snippets, col_pitch = st.columns([2, 1, 1])

        with col_game:
            st.subheader(game["name"])
            st.image(game["header_image"], use_container_width=True)

            if game.get("tags"):
                pills = "".join(
                    f"<span class='pill'>{tag}</span>" for tag in game["tags"][:5]
                )
                st.markdown(
                    f"<div class='tags-container'>{pills}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(game["short_description"])
            st.link_button("🎮 View on Steam", f"https://store.steampowered.com/app/{int(app_id)}")

        with col_snippets:
            if snippets is not None and not snippets.empty:
                st.subheader("Reviewers have said...")
                for _, r in snippets.iterrows():
                    sentence = html.escape(str(r["sentence"])).replace("\n", " ")
                    st.markdown(
                        f"<div style='font-size:0.9rem; line-height:1.5; "
                        f"padding:0.5rem 0; border-bottom:1 px solid #333;'>"
                        f"{sentence}</div>",
                        unsafe_allow_html=True,
                    )

        with col_pitch:
            st.subheader("AI says...")
            if response:
                st.markdown(response)
                if st.button("⟳ Regenerate", key=f"regen_{app_id}"):
                    self.generate()
                    st.rerun()
            else:
                st.caption("No pitch generated yet.")
                if st.button("✨ Generate", key=f"gen_{app_id}"):
                    self.generate()
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

    def render_also_like(self) -> None:
        results = st.session_state.cached_results
        if results is None or results.empty:
            return

        pinned = st.session_state.pinned_app_id
        rest = (
            results[results["app_id"] != pinned]
            .iloc[:4]
            .merge(self.library, on="app_id", how="left")
        )
        if rest.empty:
            return

        st.header("You may also like. . .")

        cols = st.columns(len(rest))
        for col, (card_index, (_, row)) in zip(cols, enumerate(rest.iterrows())):
            app_id = row["app_id"]
            button_key = f"pin_{card_index + 1}_{st.session_state._last_query}"
            with col:
                st.image(row["header_image"], use_container_width=True)
                if st.button(row["name"], key=button_key):
                    self.pin_game(int(app_id))
                    st.rerun()
                if row.get("tags"):
                    pills = "".join(
                        f"<span class='pill'>{tag}</span>" for tag in row["tags"][:3]
                    )
                    st.markdown(
                        f"<div class='tags-container'>{pills}</div>",
                        unsafe_allow_html=True,
                    )

    def _toggle_pmi(self) -> None:
        st.session_state.pmi_mode = not st.session_state.pmi_mode

    def render_pmi_view(self) -> None:
        component_key = "pmi_explorer_graph"

        def _on_tags_change():
            result = st.session_state.get(component_key)
            tags = getattr(result, 'tags', None)
            if tags is not None and sorted(tags) != sorted(st.session_state.selected_tags):
                st.session_state.selected_tags = list(tags)

        render_pmi_explorer(
            pmi_graph      = self.retriever.tag_ranker.pmi_graph,
            init_anchors   = st.session_state.selected_tags,
            key            = component_key,
            on_tags_change = _on_tags_change,
        )


    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.render_control_panel()
        if st.session_state.pmi_mode:
            self.render_pmi_view()
        elif st.session_state.cached_results is not None:
            self.render_pinned_panel()
            self.render_also_like()

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def clean_sentence(text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = html.escape(text)
        text = text.replace("\n", " ").replace("\r", " ")
        return " ".join(text.split())