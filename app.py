# -*- coding: utf-8 -*-
"""
streamlit_app.py
Deployment prototype for the BMDS2003 Data Science group assignment.
Dataset: Malaysian Condominium Prices ("houses.csv")

Layout mirrors a typical two-page trading/prediction dashboard:
    - "Market Overview"  : browse cleaned listing data and charts
    - "Price Predictor"  : single-listing prediction form, with an
                            "autofill from an existing listing" control
                            (same idea as picking a date to autofill a
                            single-day prediction form)

Run locally with:
    streamlit run streamlit_app.py

Requires (produced by ds_assignment.py):
    best_model.pkl, scaler.pkl, feature_columns.pkl,
    all_results.pkl, app_sample_listings.csv
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Condo.Price Predict", page_icon="🏢", layout="wide")

FACILITY_OPTIONS = [
    "Barbeque area", "Club house", "Gymnasium", "Jogging Track", "Lift",
    "Minimart", "Multipurpose hall", "Parking", "Playground", "Sauna",
    "Security", "Squash Court", "Swimming Pool", "Tennis Court",
]
NEARBY_OPTIONS = [
    "Mall", "Park", "School", "Hospital", "Bus_Stop", "Highway",
    "Railway_Station", "Nearby_School", "Nearby_Mall", "Nearby_Railway_Station",
]
PROPERTY_TYPES = ["Apartment", "Condominium", "Service Residence", "Studio",
                   "Duplex", "Flat", "Townhouse Condo", "Others"]
STATE_OPTIONS = ["Selangor", "Kuala Lumpur", "Johor", "Penang", "Melaka",
                  "Negeri Sembilan", "Sabah", "Sarawak", "Putrajaya", "Other"]


# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    try:
        with open("all_results.pkl", "rb") as f:
            results_df = pickle.load(f)
    except FileNotFoundError:
        results_df = None
    return model, scaler, feature_columns, results_df


@st.cache_data
def load_sample_listings():
    return pd.read_csv("app_sample_listings.csv")


model, scaler, feature_columns, results_df = load_model_artifacts()
sample_df = load_sample_listings()

# ---------------------------------------------------------------------
# Header / navigation (mirrors the reference app's top nav bar)
# ---------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color:#0f1b2d;padding:14px 20px;border-radius:6px;
                display:flex;align-items:center;justify-content:space-between;">
        <span style="color:white;font-size:22px;font-weight:600;">
            🏢 Condo.Price Predict
        </span>
        <span style="color:#9fb3c8;font-size:13px;">
            BMDS2003 Data Science &middot; Malaysian Condominium Prices
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

tab_overview, tab_predict = st.tabs(["📊 Market Overview", "🔮 Price Predictor"])

# =====================================================================
# PAGE 1 - MARKET OVERVIEW  (analogous to the "Trading Chart" page)
# =====================================================================
with tab_overview:
    st.subheader("Cleaned Listings Sample")
    st.caption(
        "A random sample of cleaned condominium listings used to train "
        "the model. Filter by state and property type to explore."
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        state_filter = st.multiselect("Filter by State", sorted(sample_df["State"].unique()))
    with col_f2:
        type_filter = st.multiselect("Filter by Property Type", sorted(sample_df["Property Type"].unique()))

    filtered = sample_df.copy()
    if state_filter:
        filtered = filtered[filtered["State"].isin(state_filter)]
    if type_filter:
        filtered = filtered[filtered["Property Type"].isin(type_filter)]

    st.dataframe(
        filtered[["Bedroom", "Bathroom", "Property Size", "Property Type",
                  "Tenure Type", "State", "price"]],
        use_container_width=True, height=280,
    )

    st.divider()
    st.subheader("Model Performance Comparison")
    if results_df is not None:
        st.caption("RMSE / MAE / R² for each of the models trained in ds_assignment.py.")
        st.dataframe(results_df.reset_index(drop=True), use_container_width=True)
        st.bar_chart(results_df.set_index("Model")[["RMSE"]])
    else:
        st.info("Run ds_assignment.py first to generate all_results.pkl.")

    st.divider()
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Average Price by State")
        avg_by_state = filtered.groupby("State")["price"].mean().sort_values(ascending=False)
        st.bar_chart(avg_by_state)
    with col_c2:
        st.subheader("Price Distribution")
        st.bar_chart(pd.cut(filtered["price"], bins=10).value_counts().sort_index())

# =====================================================================
# PAGE 2 - PRICE PREDICTOR (analogous to "Single Day Prediction" page)
# =====================================================================
with tab_predict:
    st.subheader("Condominium Price: Single Listing Prediction")
    st.caption(
        "Fill in the property details below, or use the *(Optional) Autofill "
        "from an Existing Listing* control to load a real sample listing, "
        "then click **Predict**."
    )

    # ---- Autofill control (same idea as the reference app's date-based autofill) ----
    with st.container(border=True):
        st.markdown("**(Optional) Autofill Data from an Existing Listing**")
        col_a1, col_a2 = st.columns([3, 1])
        with col_a1:
            listing_idx = st.selectbox(
                "Choose a sample listing (State - Property Type - Size)",
                options=list(sample_df.index),
                format_func=lambda i: (
                    f"#{i}: {sample_df.loc[i, 'State']} - "
                    f"{sample_df.loc[i, 'Property Type']} - "
                    f"{sample_df.loc[i, 'Property Size']:.0f} sqft"
                ),
            )
        with col_a2:
            st.write("")
            st.write("")
            autofill_clicked = st.button("Set", use_container_width=True)

    if autofill_clicked:
        st.session_state["autofill_row"] = sample_df.loc[listing_idx].to_dict()
        st.success("Input complete! Fields below have been autofilled.")

    prefill = st.session_state.get("autofill_row", {})

    def pf(col, default):
        val = prefill.get(col, default)
        return default if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    st.markdown("#### Property Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bedroom = st.number_input("Bedrooms", 0, 10, int(pf("Bedroom", 3)))
        completion_year = st.number_input("Completion Year", 1980, 2030, int(pf("Completion Year", 2015)))
    with col2:
        bathroom = st.number_input("Bathrooms", 0, 10, int(pf("Bathroom", 2)))
        floors = st.number_input("# of Floors (building)", 0, 100, int(pf("# of Floors", 20)))
    with col3:
        size = st.number_input("Property Size (sq.ft.)", 100, 5000, int(pf("Property Size", 1000)))
        total_units = st.number_input("Total Units", 0, 3000, int(pf("Total Units", 500)))
    with col4:
        parking = st.number_input("Parking Lots", 0, 10, int(pf("Parking Lot", 1)))
        tenure = st.selectbox("Tenure Type", ["Freehold", "Leasehold"],
                               index=["Freehold", "Leasehold"].index(pf("Tenure Type", "Freehold")))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        prop_options = PROPERTY_TYPES
        default_pt = pf("Property Type", "Condominium")
        pt_index = prop_options.index(default_pt) if default_pt in prop_options else 1
        property_type = st.selectbox("Property Type", prop_options, index=pt_index)
    with col6:
        state_options = STATE_OPTIONS
        default_state = pf("State", "Selangor")
        st_index = state_options.index(default_state) if default_state in state_options else 0
        state = st.selectbox("State", state_options, index=st_index)
    with col7:
        land_options = ["Non Bumi Lot", "Bumi Lot", "Malay Reserved"]
        default_land = pf("Land Title", "Non Bumi Lot")
        lt_index = land_options.index(default_land) if default_land in land_options else 0
        land_title = st.selectbox("Land Title", land_options, index=lt_index)
    with col8:
        floor_options = ["Low", "Medium", "High", "-"]
        default_floor = pf("Floor Range", "-")
        fr_index = floor_options.index(default_floor) if default_floor in floor_options else 3
        floor_range = st.selectbox("Floor Range", floor_options, index=fr_index)

    st.markdown("#### Facilities & Nearby Amenities")
    default_facilities = [f for f in FACILITY_OPTIONS if prefill.get(f"Facility_{f}") == 1]
    default_nearby = [n for n in NEARBY_OPTIONS if prefill.get(f"Has_{n}") == 1]

    col_fac, col_near = st.columns(2)
    with col_fac:
        facilities = st.multiselect("Facilities available", FACILITY_OPTIONS, default=default_facilities)
    with col_near:
        nearby = st.multiselect(
            "Nearby amenities",
            NEARBY_OPTIONS, default=default_nearby,
            format_func=lambda x: x.replace("_", " "),
        )

    col_predict, col_reset = st.columns([1, 1])
    predict_clicked = col_predict.button("🔍 Predict", type="primary", use_container_width=True)
    reset_clicked = col_reset.button("↺ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.pop("autofill_row", None)
        st.rerun()

    if predict_clicked:
        row = {col: 0 for col in feature_columns}
        row["Bedroom"] = bedroom
        row["Bathroom"] = bathroom
        row["Property Size"] = size
        row["# of Floors"] = floors
        row["Total Units"] = total_units
        row["Parking Lot"] = parking
        row["Completion Year"] = completion_year

        for f in facilities:
            key = f"Facility_{f}"
            if key in row:
                row[key] = 1
        for n in nearby:
            key = f"Has_{n}"
            if key in row:
                row[key] = 1
        for key in [
            f"Tenure Type_{tenure}", f"Property Type_{property_type}",
            f"Floor Range_{floor_range}", f"Land Title_{land_title}",
            f"State_{state}",
        ]:
            if key in row:
                row[key] = True

        X_input = pd.DataFrame([row])[feature_columns]
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        st.success(f"💰 Predicted price: **RM {prediction:,.0f}**")
        if "price" in prefill and prefill.get("price") is not None and not (
            isinstance(prefill.get("price"), float) and np.isnan(prefill.get("price"))
        ):
            st.caption(f"Actual price of the autofilled listing: RM {prefill['price']:,.0f}")
        st.caption(
            "This is an indicative estimate from the trained model and should "
            "not be used as the sole basis for financial decisions."
        )

st.divider()
st.caption(
    "Model: best of {KNN, Decision Tree, Random Forest, MLP Regressor} "
    "selected automatically by lowest test-set RMSE in ds_assignment.py."
)
