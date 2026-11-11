from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "restaurant_rating_model.joblib"
METADATA_PATH = ROOT / "restaurant_rating_metadata.json"
FEATURE_COLUMNS = [
    "City",
    "Locality",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Cuisines",
]

st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #fffaf3 0%, #f5f7fb 55%, #eef4f3 100%); }
    .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.7rem 2rem; border-radius: 22px; background: #182b36; color: white; margin-bottom: 1.4rem; }
    .hero h1 { margin: 0 0 .4rem 0; font-size: 2.2rem; }
    .hero p { color: #d7e4e6; margin: 0; font-size: 1.05rem; }
    .note { color: #53636b; font-size: .92rem; }
    div[data-testid="stForm"] { background: white; padding: 1.4rem; border: 1px solid #e8ecec; border-radius: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Model files are missing. Run all cells in notebook.ipynb through the "
            "model-saving cell first."
        )
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return model, metadata


st.markdown(
    """
    <div class="hero">
      <h1>🍽️ Restaurant Rating Predictor</h1>
      <p>Explore an estimated restaurant rating based on its location, cost, services, and cuisine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, metadata = load_assets()
except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("About this estimate")
    st.write(
        "This model was trained on rated restaurants from the supplied dataset. "
        "It estimates the dataset's 0–5 aggregate-rating scale."
    )
    st.caption(
        "Predictions are estimates, not guarantees. They reflect historical data "
        "and may not generalize to every restaurant."
    )
    st.divider()
    st.caption(f"Training rows: {metadata['training_rows']:,}")
    st.caption(f"Held-out test MAE: {metadata['test_mae']:.2f} rating points")

st.subheader("Restaurant details")
st.markdown(
    '<p class="note">Choose the closest matching options. Fields are based on the dataset used to train the model.</p>',
    unsafe_allow_html=True,
)

with st.form("restaurant_details"):
    first, second = st.columns(2)
    with first:
        city = st.selectbox("City", metadata["cities"])
        locality = st.selectbox("Locality", metadata["localities"])
        cost = st.number_input(
            "Average cost for two",
            min_value=0.0,
            max_value=float(metadata["max_cost"]),
            value=float(metadata["median_cost"]),
            step=50.0,
        )
        currency = st.selectbox("Currency", metadata["currencies"])
    with second:
        table_booking = st.selectbox("Has table booking", metadata["booking_options"])
        online_delivery = st.selectbox(
            "Has online delivery", metadata["delivery_options"]
        )
        cuisine = st.selectbox("Cuisine", metadata["cuisines"])
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Predict rating", type="primary", use_container_width=True
        )

if submitted:
    inputs = pd.DataFrame(
        [
            {
                "City": city,
                "Locality": locality,
                "Average Cost for two": cost,
                "Currency": currency,
                "Has Table booking": table_booking,
                "Has Online delivery": online_delivery,
                "Cuisines": cuisine,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction = float(model.predict(inputs)[0])
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not generate a prediction: {exc}")
    else:
        prediction = min(5.0, max(0.0, prediction))
        st.success("Prediction ready")
        left, right = st.columns([1, 2])
        left.metric("Estimated aggregate rating", f"{prediction:.2f} / 5")
        right.progress(prediction / 5.0, text="Estimated rating")
        st.caption(
            "The model predicts a numeric rating. Actual ratings can differ from "
            "this estimate."
        )
