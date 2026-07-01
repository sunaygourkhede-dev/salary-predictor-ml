import numpy as np
import streamlit as st

# ── Load model params saved from notebook ─────────────────────────────
params = np.load('model_params.npy')
w, b, X_mean, X_std = params

# ── Prediction function ────────────────────────────────────────────────
def predict(years):
    years_norm = (years - X_mean) / X_std
    return w * years_norm + b

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────
st.title("💼 Salary Predictor")
st.markdown("Predicts salary based on years of experience using **Linear Regression from scratch** — no sklearn.")
st.divider()

# ── Slider ─────────────────────────────────────────────────────────────
years = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=15.0,
    value=3.0,
    step=0.5
)

# ── Prediction output ──────────────────────────────────────────────────
salary = predict(years)

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Years of Experience", value=f"{years} yrs")

with col2:
    st.metric(label="Predicted Salary", value=f"${salary:,.0f}")

st.divider()

# ── How it works section ───────────────────────────────────────────────
with st.expander("How does this work?"):
    st.markdown(f"""
    This model uses **Linear Regression implemented from scratch in NumPy**.

    **Model formula:**

    salary = w × years_normalised + b

    **Trained parameters:**
    - Weight (w): `{w:.4f}`
    - Bias (b): `{b:.4f}`
    - Training data mean: `{X_mean:.2f} years`
    - Training data std: `{X_std:.2f}`

    **Training:** Gradient descent over 3000 epochs with learning rate 0.01.

    No scikit-learn was used for the model — only NumPy.
    """)

# ── Footer ─────────────────────────────────────────────────────────────
st.caption("Built by Sunay Gourkhede · Linear Regression from scratch · NumPy")