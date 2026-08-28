import random
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from api_client import (
    BackendError, get_meta, get_listings, get_listing, get_state_summary,
    get_model_comparison, get_model_diagnostics, predict, BACKEND_URL,
)

st.set_page_config(page_title="Condo.Price Predict", page_icon="🏢", layout="wide")

# ---------------------------------------------------------------------
# Prediction reveal: an animated, count-up price card. Rendered as a small
# HTML/JS component so the number visibly ticks up instead of just
# appearing - a bit of delight after the user fills in a whole form.
# ---------------------------------------------------------------------
def render_price_reveal(prediction: float, price_per_sqft: float | None, bracket: str | None):
    sqft_line = f"RM {price_per_sqft:,.0f} / sq.ft." if price_per_sqft is not None else ""
    bracket_badge = f'<div class="price-bracket">📊 {bracket} bracket</div>' if bracket else ""
    html = f"""
    <div class="price-card">
      <div class="price-label">Predicted Price</div>
      <div class="price-value" id="priceVal">RM 0</div>
      <div class="price-sqft">{sqft_line}</div>
      {bracket_badge}
    </div>
    <style>
      @keyframes priceCardIn {{
        0%   {{ transform: scale(0.88) translateY(6px); opacity: 0; }}
        65%  {{ transform: scale(1.03) translateY(0); opacity: 1; }}
        100% {{ transform: scale(1) translateY(0); }}
      }}
      .price-card {{
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        font-family: inherit;
        animation: priceCardIn 0.6s cubic-bezier(.22,1.2,.36,1) both;
      }}
      .price-label {{
        font-size: 13px;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .65;
        margin-bottom: 6px;
      }}
      .price-value {{
        font-size: 44px;
        font-weight: 750;
        color: #16a34a;
        line-height: 1.1;
      }}
      .price-sqft {{
        font-size: 13px;
        opacity: .65;
        margin-top: 6px;
      }}
      .price-bracket {{
        display: inline-block;
        margin-top: 10px;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 999px;
        background: rgba(34,197,94,0.16);
      }}
    </style>
    <script>
      const target = {prediction};
      const el = document.getElementById("priceVal");
      const duration = 900;
      let start = null;
      function step(ts) {{
        if (!start) start = ts;
        const progress = Math.min((ts - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const val = Math.round(eased * target);
        el.textContent = "RM " + val.toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
      }}
      requestAnimationFrame(step);
    </script>
    """
    components.html(html, height=185)


# ---------------------------------------------------------------------
# Colour ramp for the state map: cheap -> expensive goes blue -> cyan ->
# green -> yellow -> red, which reads a lot more vividly than a plain
# two-colour blend once it's sitting on top of an actual basemap.
# ---------------------------------------------------------------------
_HEAT_STOPS = [
    (0.00, (37, 99, 235)),    # blue
    (0.25, (14, 165, 233)),   # sky
    (0.50, (34, 197, 94)),    # green
    (0.75, (250, 204, 21)),   # yellow
    (1.00, (239, 68, 68)),    # red
]


def heat_to_rgb(heat: float) -> tuple[int, int, int]:
    heat = max(0.0, min(1.0, heat))
    for (p0, c0), (p1, c1) in zip(_HEAT_STOPS, _HEAT_STOPS[1:]):
        if p0 <= heat <= p1:
            t = (heat - p0) / (p1 - p0) if p1 > p0 else 0
            return tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
    return _HEAT_STOPS[-1][1]

# ---------------------------------------------------------------------
# Backend connectivity check
# ---------------------------------------------------------------------
try:
    META = get_meta()
except BackendError as e:
    st.error(
        f"Can't reach the prediction backend.\n\n{e}\n\n"
        "If you're running locally, start it first with:\n"
        "`uvicorn main:app --reload` (from the backend/ folder)\n\n"
        "If deployed, set the `BACKEND_URL` secret to your backend's public URL."
    )
    st.stop()

FACILITY_OPTIONS = META["facility_options"]
NEARBY_OPTIONS = META["nearby_options"]
PROPERTY_TYPES = META["property_types"]
TENURE_OPTIONS = META["tenure_options"]
LAND_OPTIONS = META["land_options"]
FLOOR_RANGE_OPTIONS = META["floor_range_options"]
STATE_OPTIONS = META["state_options"]
STATE_COORDS = META["state_coords"]  # {name: {lat, lon}}
CITY_OPTIONS = META.get("city_options", [])  # dynamic - see /meta on the backend

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div style="background-color:var(--secondary-background-color);
                padding:14px 20px;border-radius:6px;
                display:flex;align-items:center;justify-content:space-between;
                border:1px solid rgba(128,128,128,0.25);">
        <span style="color:var(--text-color);font-size:22px;font-weight:600;">
            🏢 Condo.Price Predict
        </span>
        <span style="color:var(--text-color);opacity:0.65;font-size:13px;">
            BMDS2003 Data Science &middot; Malaysian Condominium Prices
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Connected to backend: `{BACKEND_URL}`")
st.write("")

tab_overview, tab_compare, tab_predict = st.tabs(
    ["📊 Market Overview", "📈 Model Comparison", "🔮 Price Predictor"]
)

# =====================================================================
# PAGE 1 - MARKET OVERVIEW
# =====================================================================
with tab_overview:
    st.subheader("Cleaned Listings Sample")
    st.caption(
        "A random sample of cleaned condominium listings used to train "
        "the model. Filter by state and property type to explore."
    )

    all_listings = get_listings()
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        state_filter = st.multiselect("Filter by State", all_listings["available_states"])
    with col_f2:
        type_filter = st.multiselect("Filter by Property Type", all_listings["available_property_types"])

    result = get_listings(states=state_filter or None, property_types=type_filter or None)
    filtered = pd.DataFrame(result["listings"])

    st.dataframe(
        filtered.drop(columns=["index"]) if "index" in filtered.columns else filtered,
        width="stretch", height=280,
    )

    st.divider()
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Average Price by State")
        if len(filtered):
            avg_by_state = filtered.groupby("State")["price"].mean().sort_values(ascending=False)
            st.bar_chart(avg_by_state, height=260)
        else:
            st.info("No listings match the current filters.")
    with col_c2:
        st.subheader("Price Distribution")
        if len(filtered):
            price_bins = pd.cut(filtered["price"], bins=10)
            dist_counts = price_bins.value_counts().sort_index()
            dist_counts.index = dist_counts.index.astype(str)
            st.bar_chart(dist_counts, height=260)
        else:
            st.info("No listings match the current filters.")

# =====================================================================
# PAGE 2 - MODEL COMPARISON
# =====================================================================
with tab_compare:
    st.subheader("Model Performance Comparison")

    try:
        comp = get_model_comparison()
    except BackendError:
        comp = None

    if comp is None:
        st.info("Model comparison data isn't available from the backend right now.")
    else:
        results_sorted = pd.DataFrame(comp["results"])
        st.caption(
            "RMSE / MAE / R² for each of the models trained in ds_assignment.py "
            "(lower RMSE/MAE and higher R² is better)."
        )
        st.dataframe(results_sorted, width="stretch")

        best_name = comp["best_model_name"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Best model", best_name)
        m2.metric("Test RMSE", f"RM {comp['best_rmse']:,.0f}")
        m3.metric("Test R²", f"{comp['best_r2']:.3f}")

        st.markdown("#### RMSE / MAE / R² by Model")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.caption("RMSE (lower is better)")
            st.bar_chart(results_sorted.set_index("Model")[["RMSE"]], height=220)
        with col_m2:
            st.caption("MAE (lower is better)")
            st.bar_chart(results_sorted.set_index("Model")[["MAE"]], height=220)
        with col_m3:
            st.caption("R² (higher is better)")
            st.bar_chart(results_sorted.set_index("Model")[["R2"]], height=220)

    st.divider()

    try:
        diag = get_model_diagnostics()
    except BackendError:
        diag = None

    if diag is None:
        st.info(
            "Residual spread, prediction distributions, and the price-bracket "
            "accuracy/precision/recall view aren't available from the backend "
            "right now."
        )
    else:
        best_name = diag["best_model_name"]
        y_test = np.array(diag["y_test"])
        test_predictions = {k: np.array(v) for k, v in diag["test_predictions"].items()}

        resid_col, dist_col = st.columns(2)

        with resid_col:
            st.markdown("#### Prediction Error Spread (Residuals)")
            st.caption(
                "Residual = Actual price − Predicted price, on the held-out test "
                "set. A box centred near zero with a tight spread means a model "
                "is both unbiased and consistent."
            )
            resid_records = []
            for name, preds in test_predictions.items():
                residuals = y_test - preds
                resid_records.extend({"Model": name, "Residual (RM)": r} for r in residuals)
            resid_df = pd.DataFrame(resid_records)
            # Native chart (no matplotlib/seaborn needed in the frontend):
            # show median + IQR per model as a simple bar of spread.
            spread = resid_df.groupby("Model")["Residual (RM)"].agg(
                median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
            )
            spread["IQR"] = spread["q75"] - spread["q25"]
            st.dataframe(
                spread[["median", "IQR"]].round(0).rename(
                    columns={"median": "Median residual (RM)", "IQR": "IQR (RM)"}
                ),
                width="stretch",
            )
            st.caption("A boxplot version of this is available in the full report; shown here as summary stats to keep the frontend dependency-light.")

        with dist_col:
            st.markdown(f"#### Actual vs Predicted — {best_name}")
            st.caption(
                "How closely the best model's predicted prices track the "
                "actual test-set price distribution."
            )
            compare_df = pd.DataFrame({
                "Actual": pd.cut(y_test, bins=15).astype(str),
            })
            actual_counts = pd.Series(pd.cut(y_test, bins=15)).value_counts().sort_index()
            pred_counts = pd.Series(pd.cut(test_predictions[best_name], bins=actual_counts.index.categories if hasattr(actual_counts.index, "categories") else 15)).value_counts()
            chart_df = pd.DataFrame({"Actual": actual_counts.values}, index=actual_counts.index.astype(str))
            try:
                chart_df["Predicted"] = pred_counts.reindex(actual_counts.index).values
            except Exception:
                pass
            st.bar_chart(chart_df, height=260)

        st.divider()
        st.markdown("#### Price-Bracket Classification View")
        n_brackets = len(diag["price_bin_labels"])
        st.caption(
            "Price is a continuous value, so RMSE / MAE / R² above are the "
            "real regression metrics — accuracy, precision, recall, and a "
            "confusion matrix aren't natively defined for regression. To "
            f"still show them, test-set prices are grouped into {n_brackets} "
            f"brackets ({', '.join(diag['price_bin_labels'])}) and the "
            "question becomes: did the model predict the right price bracket? "
            "(Same bracket definition is used for every model below, so "
            "they're directly comparable.)"
        )

        if diag.get("bracket_results_all_models"):
            bracket_df = pd.DataFrame(diag["bracket_results_all_models"]).set_index("Model")
            st.markdown("**All models — bracket classification metrics**")
            st.dataframe(bracket_df.style.format("{:.1%}"), width="stretch")

        st.markdown(f"**Best model — {best_name}**")
        pa1, pa2, pa3, pa4 = st.columns(4)
        pa1.metric("Accuracy", f"{diag['bracket_accuracy']:.1%}")
        pa2.metric("Precision (macro)", f"{diag['bracket_precision']:.1%}")
        pa3.metric("Recall (macro)", f"{diag['bracket_recall']:.1%}")
        pa4.metric("F1 (macro)", f"{diag['bracket_f1']:.1%}")

        cm = np.array(diag["bracket_confusion_matrix"])
        labels = diag["price_bin_labels"]
        cm_df = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels],
                              columns=[f"Pred: {l}" for l in labels])
        st.markdown(f"Confusion Matrix — {best_name}")
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"), width="stretch")

# =====================================================================
# PAGE 3 - PRICE PREDICTOR (with map-based state picker)
# =====================================================================
with tab_predict:
    st.subheader("Condominium Price: Single Listing Prediction")
    st.caption(
        "Fill in the property details below, or use the *(Optional) Autofill "
        "from an Existing Listing* control to load a real sample listing, "
        "then click **Predict**."
    )

    # ---- Autofill control ----
    with st.container(border=True):
        st.markdown("**(Optional) Autofill Data from an Existing Listing**")
        listing_pool = get_listings()["listings"]
        col_a1, col_a2 = st.columns([3, 1])
        with col_a1:
            listing_idx = st.selectbox(
                "Choose a sample listing (State - Property Type - Size)",
                options=[row["index"] for row in listing_pool],
                format_func=lambda i: next(
                    (f"#{r['index']}: {r['State']} - {r['Property Type']} - "
                     f"{r['Property Size']:.0f} sqft" for r in listing_pool if r["index"] == i),
                    str(i),
                ),
            )
        with col_a2:
            st.write("")
            st.write("")
            autofill_clicked = st.button("Set", width="stretch")

    if autofill_clicked:
        loaded = get_listing(listing_idx)
        st.session_state["autofill_row"] = loaded
        # "state_dropdown" and "city_dropdown" are explicitly keyed (so the
        # map click handler can write to them), which means Streamlit will
        # otherwise keep showing whatever they last held and silently ignore
        # the new pf()-derived default below - same issue as the map-click
        # sync bug, just triggered from autofill instead. Setting (or
        # clearing) the keys directly here is what actually moves them.
        loaded_state = loaded.get("State")
        if loaded_state in STATE_OPTIONS:
            st.session_state["state_dropdown"] = loaded_state
            st.session_state["map_selected_state"] = loaded_state
        else:
            st.session_state.pop("state_dropdown", None)
            st.session_state.pop("map_selected_state", None)
        st.session_state.pop("state_pick_source", None)

        loaded_city = loaded.get("City")
        if CITY_OPTIONS and loaded_city in CITY_OPTIONS:
            st.session_state["city_dropdown"] = loaded_city
        else:
            st.session_state.pop("city_dropdown", None)
        st.success("Input complete! Fields below have been autofilled.")

    prefill = st.session_state.get("autofill_row", {})

    def pf(col, default):
        val = prefill.get(col, default)
        return default if (val is None or (isinstance(val, float) and np.isnan(val))) else val

    st.markdown("#### Property Details")
    st.caption("Drag the sliders to shape the listing - everything below reacts live.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bedroom = st.slider("🛏️ Bedrooms", 0, 10, int(pf("Bedroom", 3)))
        completion_year = st.slider("🗓️ Completion Year", 1980, 2030, int(pf("Completion Year", 2015)))
    with col2:
        bathroom = st.slider("🛁 Bathrooms", 0, 10, int(pf("Bathroom", 2)))
        floors = st.slider("🏗️ # of Floors (building)", 0, 100, int(pf("# of Floors", 20)))
    with col3:
        size = st.slider("📐 Property Size (sq.ft.)", 100, 5000, int(pf("Property Size", 1000)), step=50)
        total_units = st.slider("🏘️ Total Units", 0, 3000, int(pf("Total Units", 500)), step=10)
    with col4:
        parking = st.slider("🚗 Parking Lots", 0, 10, int(pf("Parking Lot", 1)))
        tenure = st.selectbox("Tenure Type", TENURE_OPTIONS,
                               index=TENURE_OPTIONS.index(pf("Tenure Type", "Freehold"))
                               if pf("Tenure Type", "Freehold") in TENURE_OPTIONS else 0)

    col5, col6, col7 = st.columns(3)
    with col5:
        default_pt = pf("Property Type", "Condominium")
        pt_index = PROPERTY_TYPES.index(default_pt) if default_pt in PROPERTY_TYPES else 1
        property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=pt_index)
    with col6:
        default_land = pf("Land Title", "Non Bumi Lot")
        lt_index = LAND_OPTIONS.index(default_land) if default_land in LAND_OPTIONS else 0
        land_title = st.selectbox("Land Title", LAND_OPTIONS, index=lt_index)
    with col7:
        default_floor = pf("Floor Range", "-")
        fr_index = FLOOR_RANGE_OPTIONS.index(default_floor) if default_floor in FLOOR_RANGE_OPTIONS else 3
        floor_range = st.selectbox("Floor Range", FLOOR_RANGE_OPTIONS, index=fr_index)

    # ---- State picker: map + dropdown, kept in sync ----
    st.markdown("#### State")
    st.caption(
        "Click a state marker on the map (bubble size = sample listing count, "
        "colour = average price, blue → red as price rises) or use the "
        "dropdown - they stay in sync. The map is locked to Malaysia. "
        "\"Other\" has no single location so it's dropdown-only."
    )

    try:
        state_summary = get_state_summary()
    except BackendError:
        state_summary = {}

    map_states = [s for s in STATE_OPTIONS if s in STATE_COORDS]
    max_avg = max((state_summary.get(s, {}).get("avg_price", 0) for s in map_states), default=1) or 1
    max_count = max((state_summary.get(s, {}).get("count", 0) for s in map_states), default=1) or 1

    map_rows = []
    for s in map_states:
        summ = state_summary.get(s, {"avg_price": 0, "count": 0})
        heat = summ["avg_price"] / max_avg if max_avg else 0
        r, g, b = heat_to_rgb(heat)
        map_rows.append({
            "state": s,
            "lat": STATE_COORDS[s]["lat"],
            "lon": STATE_COORDS[s]["lon"],
            "avg_price": summ["avg_price"],
            "count": summ["count"],
            "radius": 12000 + 28000 * (summ["count"] / max_count if max_count else 0),
            "r": r, "g": g, "b": b,
        })
    map_df = pd.DataFrame(map_rows)

    default_state = pf("State", "Selangor")
    if default_state not in STATE_OPTIONS:
        default_state = "Selangor"
    current_state = st.session_state.get("map_selected_state", default_state)

    map_col, picker_col = st.columns([2, 1])
    with map_col:
        layer = pdk.Layer(
            "ScatterplotLayer",
            id="state-layer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="[r, g, b, 210]",
            get_line_color="[255, 255, 255, 230]",
            line_width_min_pixels=2,
            stroked=True,
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            highlight_color="[255, 255, 255, 120]",
        )
        # Fixed on Malaysia: centred between the Peninsula and East Malaysia,
        # with zoom clamped so it can't be scrolled out to a world view or
        # zoomed past street level.
        view_state = pdk.ViewState(
            latitude=4.0, longitude=109.5, zoom=5.0, pitch=0,
            min_zoom=4, max_zoom=8,
        )
        # `controller` isn't a Deck-level argument in pydeck - it belongs on
        # the View, so the camera has to be frozen there instead.
        frozen_view = pdk.View(type="MapView", controller=False)
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            views=[frozen_view],
            map_style="dark",
            map_provider="carto",
            tooltip={
                "text": "{state}\nAvg price: RM {avg_price}\nListings: {count}",
                "style": {"backgroundColor": "#111827", "color": "white"},
            },
        )
        selection = st.pydeck_chart(
            deck, height=340, on_select="rerun", selection_mode="single-object",
            key="state_map",
        )
        clicked_state = None
        if selection and selection.selection and selection.selection.get("objects"):
            objs = selection.selection["objects"]
            picked = objs.get("state-layer") or next(iter(objs.values()), None)
            if picked:
                row = picked[0] if isinstance(picked, list) else picked
                clicked_state = row.get("state")
        if clicked_state and clicked_state != st.session_state.get("map_selected_state"):
            st.session_state["map_selected_state"] = clicked_state
            # The selectbox below is keyed "state_dropdown", and once a keyed
            # widget has been instantiated once, Streamlit shows whatever is
            # in session_state[key] on every later rerun and ignores the
            # `index=` argument we pass it. So to actually move the dropdown
            # from a map click, we have to set its own key directly here -
            # not just the plain "map_selected_state" value it reads from
            # on first run.
            st.session_state["state_dropdown"] = clicked_state
            st.session_state["state_pick_source"] = "map"
            st.rerun()

    with picker_col:
        st_index = STATE_OPTIONS.index(current_state) if current_state in STATE_OPTIONS else 0
        state = st.selectbox(
            "Selected state", STATE_OPTIONS, index=st_index, key="state_dropdown",
        )
        if state != st.session_state.get("map_selected_state"):
            st.session_state["map_selected_state"] = state
            st.session_state["state_pick_source"] = "dropdown"
        if state in state_summary:
            row = next((r for r in map_rows if r["state"] == state), None)
            if row is not None:
                swatch = f"rgb({row['r']},{row['g']},{row['b']})"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
                    f'<span style="width:14px;height:14px;border-radius:50%;'
                    f'background:{swatch};border:2px solid white;display:inline-block;"></span>'
                    f'<span style="opacity:.7;font-size:13px;">Matches the dot on the map</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.metric("Avg price here", f"RM {state_summary[state]['avg_price']:,.0f}")
            st.caption(f"{state_summary[state]['count']} sample listings")
            if st.session_state.get("state_pick_source") == "map":
                st.caption("📍 Picked by clicking the map")
        else:
            st.caption("No market summary for this state.")

    _NOT_SURE_CITY = "Not sure / Other"
    if CITY_OPTIONS:
        st.markdown("#### City")
        st.caption(
            "The trained model also picked up on city-level price differences "
            "(a bigger signal than State alone) - pick one if you know it."
        )
        default_city = pf("City", _NOT_SURE_CITY)
        city_list = [_NOT_SURE_CITY] + CITY_OPTIONS
        city_index = city_list.index(default_city) if default_city in city_list else 0
        city_choice = st.selectbox("City", city_list, index=city_index, key="city_dropdown")
        city = None if city_choice == _NOT_SURE_CITY else city_choice
    else:
        city = None

    st.markdown("#### Facilities & Nearby Amenities")
    st.caption("Tap to toggle - selected pills light up green.")
    default_facilities = [f for f in FACILITY_OPTIONS if prefill.get(f"Facility_{f}") == 1]
    default_nearby = [n for n in NEARBY_OPTIONS if prefill.get(f"Has_{n}") == 1]

    col_fac, col_near = st.columns(2)
    with col_fac:
        st.markdown("**Facilities available**")
        facilities = st.pills(
            "Facilities available", FACILITY_OPTIONS, default=default_facilities,
            selection_mode="multi", label_visibility="collapsed", key="facilities_pills",
        )
    with col_near:
        st.markdown("**Nearby amenities**")
        nearby = st.pills(
            "Nearby amenities", NEARBY_OPTIONS, default=default_nearby,
            selection_mode="multi", label_visibility="collapsed", key="nearby_pills",
            format_func=lambda x: x.replace("_", " "),
        )
    facilities = facilities or []
    nearby = nearby or []

    col_predict, col_reset = st.columns([1, 1])
    predict_clicked = col_predict.button("🔍 Predict", type="primary", width="stretch")
    reset_clicked = col_reset.button("↺ Reset", width="stretch")

    if reset_clicked:
        st.session_state.pop("autofill_row", None)
        st.session_state.pop("map_selected_state", None)
        st.session_state.pop("state_dropdown", None)
        st.session_state.pop("city_dropdown", None)
        st.session_state.pop("state_pick_source", None)
        st.rerun()

    if predict_clicked:
        payload = {
            "bedroom": bedroom, "bathroom": bathroom, "size": size,
            "floors": floors, "total_units": total_units, "parking": parking,
            "completion_year": completion_year, "tenure": tenure,
            "property_type": property_type, "state": state, "city": city,
            "land_title": land_title, "floor_range": floor_range,
            "facilities": facilities, "nearby": nearby,
        }
        spinner_messages = [
            "Crunching the numbers…", "Weighing bedrooms and bathrooms…",
            "Checking the neighbourhood…", "Consulting the model…",
        ]
        with st.spinner(random.choice(spinner_messages)):
            time.sleep(0.4)  # small pause so the spinner is actually visible
            try:
                res = predict(payload)
            except BackendError as e:
                st.error(str(e))
                res = None

        if res:
            prediction = res["prediction"]
            render_price_reveal(prediction, res.get("price_per_sqft"), res.get("bracket"))
            st.toast("Prediction ready! 🎉", icon="✅")
            st.balloons()

            rd1, rd2, rd3 = st.columns(3)
            rd1.metric("Price per sq.ft.", f"RM {res['price_per_sqft']:,.0f}")
            if res["best_rmse"] is not None:
                rd2.metric("Typical model error (RMSE)", f"± RM {res['best_rmse']:,.0f}")
                rd3.metric(
                    "Likely range",
                    f"RM {res['range_low']:,.0f} – RM {res['range_high']:,.0f}",
                )

            if res["similar_avg"] is not None:
                diff_pct = res["diff_pct"]
                direction = "above" if diff_pct >= 0 else "below"
                st.caption(
                    f"🏘️ Similar listings in **{state}** ({property_type}, n={res['similar_count']}) "
                    f"average **RM {res['similar_avg']:,.0f}** — this prediction is "
                    f"**{abs(diff_pct):.1f}% {direction}** that average."
                )

            if res["feature_importances"]:
                imp_df = pd.DataFrame(res["feature_importances"]).set_index("feature")
                with st.expander("🔍 What's driving this prediction? (top model features)"):
                    st.caption(
                        "Overall feature importance from the trained model "
                        "(not specific to this single prediction, but shows "
                        "what generally moves price for this model)."
                    )
                    st.bar_chart(imp_df, height=260)

            if prefill.get("price") is not None:
                st.caption(f"Actual price of the autofilled listing: RM {prefill['price']:,.0f}")
            st.caption(
                "This is an indicative estimate from the trained model and should "
                "not be used as the sole basis for financial decisions."
            )

st.divider()
st.caption(
    "Model: best of {KNN, Decision Tree, Random Forest, MLP Regressor} "
    "selected automatically by lowest test-set RMSE, served from a separate "
    "FastAPI backend."
)