"""LDA topic modeling feature screen."""

import pandas as pd
import streamlit as st

from ui.renderers import display_topic_modeling


def render_topic_modeling(content: pd.Series) -> None:
    st.subheader("Тематичне моделювання (LDA)")
    st.markdown(
        "Latent Dirichlet Allocation — виявлення прихованих тем у корпусі статей. "
        "Потрібно щонайменше 3 тексти з достатньою кількістю слів."
    )
    display_topic_modeling(content)
