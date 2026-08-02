"""Intro feature screen."""

import streamlit as st


def render_intro(intro_text: str) -> None:
    st.markdown(intro_text)
