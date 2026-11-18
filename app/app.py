import streamlit as st
import pandas as pd
import joblib
import os

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Planetary Defense Analytics",
    page_icon="☄️",
    layout="wide"
)

# ==================================================
# LOAD MODELS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pha_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "pha_classifier.pkl"
    )
)

diameter_model = joblib.load(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "diameter_predictor.pkl"
    )
)

# ==================================================
# HEADER
# ==================================================

st.title("☄️ Planetary Defense Analytics")

st.caption(
    "Machine Learning Based Asteroid Hazard Detection System"
)

st.markdown("""
This application predicts:

✅ Potentially Hazardous Asteroid (PHA) Status

✅ Estimated Asteroid Diameter (km)

using Machine Learning models trained on NASA asteroid data.
""")

# ==================================================
# ABOUT PROJECT
# ==================================================

with st.expander("ℹ️ About This Project"):

    st.write("""
    This project uses two Machine Learning pipelines:

    **Regression Model**
    - Predicts asteroid diameter

    **Classification Model**
    - Predicts whether an asteroid is a
      Potentially Hazardous Asteroid (PHA)

    Models Used:
    - Gradient Boosting Regressor
    - Gradient Boosting Classifier
    """)

# ==================================================
# USER INPUTS
# ==================================================

st.header("🛰️ Enter Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:

    H = st.number_input(
        "Absolute Magnitude (H)",
        min_value=0.0,
        value=20.0
    )

    albedo = st.number_input(
        "Albedo",
        min_value=0.0,
        value=0.15
    )

    e = st.number_input(
        "Eccentricity (e)",
        min_value=0.0,
        value=0.10
    )

    a = st.number_input(
        "Semi-Major Axis (a)",
        min_value=0.0,
        value=2.50
    )

    q = st.number_input(
        "Perihelion Distance (q)",
        min_value=0.0,
        value=1.54
    )

with col2:

    i = st.number_input(
        "Inclination (i)",
        min_value=0.0,
        value=5.0
    )

    ad = st.number_input(
        "Aphelion Distance (ad)",
        min_value=0.0,
        value=3.05
    )

    n = st.number_input(
        "Mean Motion (n)",
        min_value=0.0,
        value=0.23
    )

    per = st.number_input(
        "Orbital Period (per)",
        min_value=0.0,
        value=1000.0
    )

# ==================================================
# INPUT SUMMARY
# ==================================================

st.subheader("📋 Input Summary")

summary_df = pd.DataFrame({

    "Parameter": [
        "H",
        "Albedo",
        "Eccentricity",
        "Semi Major Axis",
        "Perihelion Distance",
        "Inclination",
        "Aphelion Distance",
        "Mean Motion",
        "Orbital Period"
    ],

    "Value": [
        H,
        albedo,
        e,
        a,
        q,
        i,
        ad,
        n,
        per
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==================================================
# PREDICTION
# ==================================================

if st.button("🔍 Predict Asteroid Risk"):

    try:

        # --------------------------
        # Regression Input
        # --------------------------

        reg_input = pd.DataFrame({

            "H": [H],
            "albedo": [albedo],

            "e": [e],
            "a": [a],
            "q": [q],
            "i": [i],

            "ad": [ad],
            "n": [n],
            "per": [per],

            "om_sin": [0],
            "om_cos": [1],

            "w_sin": [0],
            "w_cos": [1],

            "ma_sin": [0],
            "ma_cos": [1]

        })

        # --------------------------
        # Diameter Prediction
        # --------------------------

        raw_diameter = diameter_model.predict(
            reg_input
        )[0]

        diameter_pred = max(
            0.01,
            float(raw_diameter)
        )

        # --------------------------
        # Classification Input
        # --------------------------

        class_input = pd.DataFrame({

            "H": [H],
            "diameter": [diameter_pred],
            "albedo": [albedo],

            "epoch": [0],
            "epoch_mjd": [0],
            "epoch_cal": ["2024"],

            "e": [e],
            "a": [a],
            "q": [q],
            "i": [i],

            "ad": [ad],
            "n": [n],

            "tp": [0],
            "tp_cal": ["2024"],

            "per": [per],
            "per_y": [per / 365],

            "rms": [0],

            "om_sin": [0],
            "om_cos": [1],

            "w_sin": [0],
            "w_cos": [1],

            "ma_sin": [0],
            "ma_cos": [1]

        })

        # --------------------------
        # PHA Prediction
        # --------------------------

        pha_pred = pha_model.predict(
            class_input
        )[0]

        # ==================================================
        # RESULTS
        # ==================================================

        st.divider()

        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Estimated Diameter (km)",
                value=f"{diameter_pred:.2f}"
            )

        with col2:

            st.metric(
                label="PHA Prediction",
                value=pha_pred
            )

        if pha_pred == "Y":

            st.error(
                "⚠️ HIGH RISK: Potentially Hazardous Asteroid"
            )

        else:

            st.success(
                "✅ LOW RISK: Non-Hazardous Asteroid"
            )

    except Exception as error:

        st.error(
            f"Prediction Error: {error}"
        )