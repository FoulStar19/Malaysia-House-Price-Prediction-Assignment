import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import backend


# ---------------------------------------------------------------------
# Colour palette (used across every Plotly chart + custom HTML widgets)
# ---------------------------------------------------------------------
PALETTE = ["#22C55E", "#3B82F6", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4"]
CARD_COLORS = ["#22C55E", "#3B82F6", "#F59E0B", "#EC4899"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12, color="#24304A"),
    xaxis=dict(gridcolor="rgba(109,74,255,.10)", zerolinecolor="rgba(109,74,255,.15)"),
    yaxis=dict(gridcolor="rgba(109,74,255,.10)", zerolinecolor="rgba(109,74,255,.15)"),
    margin=dict(l=10, r=10, t=45, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Condo Price Predict",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------
# Global theme: compact spacing + colourful accents
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2rem;
          max-width: 1200px;
      }
      hr { margin: 1.4rem 0 !important; }
      h3, h4, h5, h6 { margin-top: 1.1rem !important; margin-bottom: 0.6rem !important; }
      div[data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 0.9rem; }
      div[data-testid="stMarkdownContainer"] > p { margin-bottom: 0.5rem; }
      div[data-testid="column"] { padding: 0 0.5rem; }
      div[data-testid="stRadio"] { margin-bottom: 0.8rem; }
      div[data-testid="stExpander"] { margin: 0.7rem 0; }
      div[data-testid="stDataFrame"] { margin-top: 0.4rem; margin-bottom: 0.6rem; }
      div[data-testid="stPlotlyChart"] { margin-top: 0.3rem; margin-bottom: 0.6rem; }
      [data-testid="stAppViewContainer"] {
          background: radial-gradient(circle at 4% 0%, rgba(255,191,105,.25), transparent 28%),
                      radial-gradient(circle at 95% 10%, rgba(255,122,184,.20), transparent 26%),
                      radial-gradient(circle at 65% 70%, rgba(104,219,205,.16), transparent 34%),
                      #fff9f5;
      }
      div[data-testid="stNumberInput"] input,
      div[data-baseweb="select"] > div {
          border-radius: 14px !important;
          border-color: rgba(109,74,255,.20) !important;
          background: #ffffff !important;
          box-shadow: 0 8px 20px -18px rgba(45,34,89,.45) !important;
      }
      div[data-testid="stButton"] button {
          border-radius: 14px !important;
          min-height: 42px;
          font-weight: 700 !important;
          transition: transform .16s ease, box-shadow .16s ease !important;
      }
      div[data-testid="stButton"] button:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 28px -16px rgba(109,74,255,.75) !important;
      }

      /* Colourful tab bar */
      button[data-baseweb="tab"] {
          border-radius: 10px 10px 0 0 !important;
          font-weight: 600 !important;
          padding: 6px 16px !important;
      }
      div[data-baseweb="tab-highlight"] {
          background: linear-gradient(90deg,#22C55E,#3B82F6,#F59E0B,#EC4899) !important;
          height: 3px !important;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
          background: #f0ebff !important;
      }

      /* Pills / multiselect chips get a splash of colour */
      span[data-baseweb="tag"] {
          background: linear-gradient(135deg,#7c5cff,#ff72ad) !important;
      }
      [data-testid="stPills"] button {
          border-radius: 999px !important;
          border: 1px solid rgba(109,74,255,.20) !important;
          background: #ffffff !important;
          color: #4c3a85 !important;
          font-weight: 700 !important;
          box-shadow: 0 7px 14px -13px rgba(58,40,130,.65) !important;
      }
      [data-testid="stPills"] button[aria-pressed="true"] {
          border-color: transparent !important;
          background: linear-gradient(135deg,#6d4aff,#ff5ca8) !important;
          color: #ffffff !important;
          box-shadow: 0 10px 18px -12px rgba(109,74,255,.75) !important;
      }
      [data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
          background: rgba(255,255,255,.74);
          border: 1px solid rgba(109,74,255,.12);
          border-radius: 18px;
          padding: 8px;
          box-shadow: 0 18px 34px -30px rgba(45,34,89,.55);
      }

      /* Compact metric cards */
      .mini-card {
          border-radius: 14px;
          padding: 14px 14px;
          text-align: center;
          color: #24304a;
          box-shadow: 0 15px 28px -23px rgba(45,34,89,.5);
          margin-bottom: 0.9rem;
          margin-top: 0.2rem;
      }
      .mini-card .mc-label {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: .06em;
          opacity: .85;
          margin-bottom: 2px;
      }
      .mini-card .mc-value {
          font-size: 20px;
          font-weight: 800;
          line-height: 1.15;
      }
      .mini-card .mc-sub {
          font-size: 10.5px;
          opacity: .8;
          margin-top: 2px;
      }

      div[data-testid="stExpander"] {
          border-radius: 12px !important;
          border: 1px solid rgba(109,74,255,0.20) !important;
          background: rgba(255,255,255,.82) !important;
      }
      div[data-testid="stContainer"]:has(> div > div > div[data-testid="stMarkdownContainer"]) {
          border-radius: 14px;
      }
      .app-hero {
          position: relative; overflow:hidden; border-radius: 24px; padding: 24px 26px;
          margin: 6px 0 18px; border: 1px solid rgba(147,197,253,.28);
          background: linear-gradient(118deg,#fff5c7 0%,#ffe4f0 38%,#e4f6ff 70%,#dcfff0 100%);
          box-shadow: 0 25px 52px -33px rgba(109,74,255,.45);
      }
      .app-hero:after { content:""; position:absolute; width:220px; height:220px; border-radius:50%; right:-60px; top:-110px; background:rgba(255,255,255,.70); filter:blur(5px); }
      .hero-kicker { color:#7c3aed; font-weight:800; font-size:11px; letter-spacing:.16em; text-transform:uppercase; }
      .hero-title { font-size:clamp(27px,4vw,42px); font-weight:900; margin:6px 0; letter-spacing:-.04em; }
      .hero-copy { max-width:640px; opacity:.82; font-size:15px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def mini_metric(label, value, sub=None, color="#3B82F6"):
    """Render one compact, colourful metric card."""
    sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="mini-card" style="background:linear-gradient(135deg,{color}33,{color}11);
             border:1px solid {color}55;">
          <div class="mc-label" style="color:{color};">{label}</div>
          <div class="mc-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items):
    """items: list of (label, value, sub, color) tuples, rendered in equal columns."""
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        with col:
            mini_metric(label, value, sub, color)


FACILITY_OPTIONS = backend.FACILITY_OPTIONS
NEARBY_OPTIONS = backend.NEARBY_OPTIONS
PROPERTY_TYPES = backend.PROPERTY_TYPES
TENURE_OPTIONS = backend.TENURE_OPTIONS
LAND_OPTIONS = backend.LAND_OPTIONS
FLOOR_RANGE_OPTIONS = backend.FLOOR_RANGE_OPTIONS
STATE_OPTIONS = backend.STATE_OPTIONS
STATE_COORDS = backend.STATE_COORDS

# First load trains/tunes the models. backend.load_artifacts() is cached,
# so subsequent Streamlit reruns do not retrain the models.
try:
    ART = backend.load_artifacts()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.exception(exc)
    st.stop()

get_listings = backend.get_listings
get_listing = backend.get_listing
get_state_summary = backend.get_state_summary
get_model_comparison = backend.get_model_comparison
get_model_diagnostics = backend.get_model_diagnostics
get_tuning_results = backend.get_tuning_results
get_tuning_results = backend.get_tuning_results
get_data_quality = backend.get_data_quality
predict = backend.predict


# ---------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------
def money(value):
    if value is None or pd.isna(value):
        return "—"
    return f"RM {float(value):,.0f}"


def render_price_reveal(prediction, price_per_sqft=None, bracket=None):
    sqft_line = (
        f"RM {price_per_sqft:,.0f} / sq.ft."
        if price_per_sqft is not None
        else ""
    )
    bracket_line = (
        f'<div class="price-bracket">📊 {bracket}</div>'
        if bracket
        else ""
    )

    html = f"""
    <div class="price-card">
      <div class="price-label">Predicted Market Price</div>
      <div class="price-value" id="priceVal">RM 0</div>
      <div class="price-sqft">{sqft_line}</div>
      {bracket_line}
    </div>
    <style>
      .price-card {{
        background: linear-gradient(
          135deg,
          rgba(34,197,94,0.16),
          rgba(34,197,94,0.03)
        );
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin: 8px 0 18px 0;
      }}
      .price-label {{
        font-size: 13px;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .65;
        margin-bottom: 8px;
      }}
      .price-value {{
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
      }}
      .price-sqft {{
        margin-top: 8px;
        opacity: .65;
      }}
      .price-bracket {{
        display: inline-block;
        margin-top: 12px;
        padding: 5px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.14);
        font-size: 12px;
      }}
    </style>
    <script>
      const target = {float(prediction)};
      const el = document.getElementById("priceVal");
      const duration = 750;
      let start = null;
      function step(ts) {{
        if (!start) start = ts;
        const progress = Math.min((ts - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = "RM " + Math.round(eased * target).toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
      }}
      requestAnimationFrame(step);
    </script>
    """
    components.html(html, height=185)


def heat_to_rgb(heat):
    stops = [
        (0.00, (37, 99, 235)),
        (0.25, (14, 165, 233)),
        (0.50, (34, 197, 94)),
        (0.75, (250, 204, 21)),
        (1.00, (239, 68, 68)),
    ]
    heat = max(0.0, min(1.0, float(heat)))

    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= heat <= p1:
            t = (heat - p0) / (p1 - p0) if p1 > p0 else 0
            return tuple(
                int(c0[i] + (c1[i] - c0[i]) * t)
                for i in range(3)
            )

    return stops[-1][1]


def safe_int(value, default):
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def safe_float(value, default):
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-title">🏢 Condo Price Predict</div>
            <div class="app-subtitle">
                BMDS2003 Data Science · Malaysian Condominium Prices
            </div>
        </div>
    </div>

    <style>
      .app-header {
        background: #ffffff;
        border: 1px solid rgba(109,74,255,.16);
        border-radius: 20px;
        padding: 16px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
      }
      .app-title {
        font-size: 25px;
        font-weight: 900;
        background: linear-gradient(90deg,#6D4AFF,#FF5CA8,#FF9D36);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .app-subtitle { color:#64748b; font-size: 13px; margin-top: 2px; }
      .app-badge {
        font-size: 12px;
        font-weight: 600;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(139,92,246,.18);
        border: 1px solid rgba(139,92,246,.4);
        white-space: nowrap;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.divider()

tab_overview, tab_compare, tab_predict = st.tabs(
    ["📊 Market & Data Quality", "📈 Model Comparison", "🔮 Price Predictor"]
)

# =====================================================================
# PAGE 1 — MARKET OVERVIEW + DATA QUALITY
# =====================================================================
with tab_overview:
    st.markdown("##### 🗂️ Dataset Overview")

    quality = get_data_quality()
    original_rows = quality["original_rows"]
    usable_rows = quality["usable_rows"]

    metric_row([
        ("Raw rows", f"{original_rows:,}", None, "#3B82F6"),
        ("Usable rows", f"{usable_rows:,}", None, "#22C55E"),
        ("Rows removed", f"{original_rows - usable_rows:,}", None, "#F59E0B"),
        ("Target", "Property Price", None, "#EC4899"),
    ])

    st.caption(
        "Cleaning is documented below. Learned imputation and scaling are fitted "
        "inside the training pipeline to avoid test-data leakage."
    )

    flow = pd.DataFrame([
        {"Step": "Raw source", "Rows": original_rows, "Action": "Read houses.csv"},
        {"Step": "Duplicate check", "Rows": quality["rows_after_duplicates"],
         "Action": f"Removed {quality['duplicate_rows']:,} exact duplicate row(s)"},
        {"Step": "Validity filter", "Rows": usable_rows,
         "Action": (f"Removed {quality['removed_invalid_target']:,} invalid prices and "
                    f"{quality['removed_invalid_size']:,} invalid sizes")},
    ])
    with st.expander("🧹 Cleaning audit trail", expanded=True):
        st.dataframe(flow, width="stretch", hide_index=True)

    st.divider()

    st.markdown("##### 📋 Cleaned Listings Sample")

    all_listings = get_listings()

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        state_filter = st.multiselect(
            "Filter by State",
            all_listings["available_states"],
        )
    with col_f2:
        type_filter = st.multiselect(
            "Filter by Property Type",
            all_listings["available_property_types"],
        )

    result = get_listings(
        states=state_filter or None,
        property_types=type_filter or None,
    )

    filtered = pd.DataFrame(result["listings"])

    if len(filtered):
        display_df = filtered.drop(
            columns=["index"],
            errors="ignore",
        )
        st.dataframe(
            display_df,
            width="stretch",
            height=300,
        )
    else:
        st.info("No listings match the current filters.")

    st.divider()

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("##### 💰 Average Price by State")
        if len(filtered):
            avg_by_state = (filtered.groupby("State")["price"].agg(["mean", "count"])
                            .reset_index().sort_values("mean"))
            avg_by_state["label"] = avg_by_state.apply(
                lambda row: f"{row['State']} (n={int(row['count'])})", axis=1
            )
            fig = px.bar(
                avg_by_state,
                x="mean", y="label", orientation="h", text="mean",
                color_discrete_sequence=["#3B82F6"],
            )
            fig.update_traces(texttemplate="RM %{text:,.0f}", textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            fig.update_xaxes(title="Average price (RM)")
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data for the selected filters.")

    with col_c2:
        st.markdown("##### 📊 Price Distribution")
        if len(filtered):
            fig = px.histogram(filtered, x="price", nbins=20,
                               color_discrete_sequence=["#8B5CF6"])
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            fig.update_yaxes(title="Listings")
            fig.update_xaxes(title="Property price (RM)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data for the selected filters.")

    st.markdown("##### 🔗 Price and property-size relationship")
    if len(filtered):
        fig = px.scatter(
            filtered, x="Property Size", y="price", color="Property Type",
            hover_data=["State", "Bedroom", "Bathroom"],
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          xaxis_title="Property size (sq. ft.)",
                          yaxis_title="Property price (RM)")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("##### 🔍 Data Quality Checks")

    dq1, dq2 = st.columns(2)

    with dq1:
        st.markdown("###### ❓ Missing values before / after filtering")
        missing_df = pd.DataFrame(quality["raw_missing_summary"]).merge(
            pd.DataFrame(quality["missing_summary"]), on="Column", how="outer",
            suffixes=(" Before", " After"),
        ).fillna(0)
        if len(missing_df):
            missing_long = missing_df.melt(id_vars="Column", var_name="Stage", value_name="Missing Values")
            fig = px.bar(missing_long, x="Missing Values", y="Column", color="Stage",
                         barmode="group", orientation="h",
                         color_discrete_map={"Missing Values Before": "#F59E0B", "Missing Values After": "#3B82F6"})
            fig.update_layout(**PLOTLY_LAYOUT, height=420, yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No missing-value summary available.")

    with dq2:
        st.markdown("###### 📐 IQR Outlier Analysis")
        outlier_df = pd.DataFrame(quality["outlier_summary"])
        if len(outlier_df):
            st.dataframe(
                outlier_df.round(2),
                width="stretch",
                hide_index=True,
            )
            st.caption("IQR flags are reviewed, not automatically deleted: unusual high-value properties can be valid listings.")

    with st.expander("📍 Address-to-location validation sample"):
        st.caption("Derived State and City are checked against the original address; uncertain values remain explicit rather than guessed.")
        st.dataframe(pd.DataFrame(quality["location_validation"]), width="stretch", hide_index=True)


# =====================================================================
# PAGE 2 — MODEL COMPARISON
# =====================================================================
with tab_compare:
    st.markdown("##### 🤖 Machine Learning Model Comparison")

    comp = get_model_comparison()
    results_df = pd.DataFrame(comp["results"]).copy()

    best_name = comp["best_model_name"]
    baseline_name = comp["baseline_model_name"]

    diag = get_model_diagnostics()
    st.caption(
        "Four regression models compared on the same held-out test set. "
        "RMSE, MAE and R² are the decision metrics. Five-fold cross-validation "
        "is used for parameter tuning on training data only."
    )

    metric_row([
        ("Best Model", best_name, None, "#22C55E"),
        ("Baseline", baseline_name, None, "#3B82F6"),
        ("Test RMSE", money(comp["best_rmse"]), None, "#F59E0B"),
        ("Test R²", f"{comp['best_r2']:.3f}", None, "#EC4899"),
    ])

    with st.expander("📋 Final Test-Set Results table", expanded=False):
        display_cols = [
            "Model", "Role", "CV RMSE", "RMSE", "MAE", "R2"
        ]
        # Backend versions without Role are still supported.
        display_cols = [c for c in display_cols if c in results_df.columns]

        formatted = results_df[display_cols].copy()
        for col in ["CV RMSE", "RMSE", "MAE"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].map(lambda x: f"RM {x:,.0f}")
        if "R2" in formatted.columns:
            formatted["R2"] = formatted["R2"].map(lambda x: f"{x:.4f}")

        st.dataframe(formatted, width="stretch", hide_index=True)

    # These are intentionally separate from the regression table.  The
    # continuous predictions are binned into quartile price bands only as an
    # additional communication aid; they do not determine the winning model.
    bracket_df = pd.DataFrame(diag.get("bracket_results_all_models", []))
    with st.expander("🎯 Supplementary price-bracket metrics — all 4 models", expanded=True):
        st.caption(
            "For a familiar classification-style view, test predictions are converted "
            "to four data-derived price brackets. RMSE, MAE and R² remain the primary "
            "metrics because this is a regression problem."
        )
        if len(bracket_df):
            bracket_display = bracket_df[["Model", "Accuracy", "Precision", "Recall", "F1"]].copy()
            for metric in ["Accuracy", "Precision", "Recall", "F1"]:
                bracket_display[metric] = bracket_display[metric].map(lambda value: f"{value:.1%}")
            st.dataframe(bracket_display, width="stretch", hide_index=True)
        else:
            st.info("Supplementary bracket metrics are unavailable.")

    st.divider()

    # ================================================================
    # INTERACTIVE PLOTLY CHARTS — regression diagnostics only
    # ================================================================
    st.markdown("##### 📊 Visual Model Comparison")

    # Prepare test-set predictions and residuals for all four models.
    y_test = np.asarray(diag["y_test"], dtype=float)
    predictions = {
        name: np.asarray(values, dtype=float)
        for name, values in diag["test_predictions"].items()
    }

    residuals = {
        name: y_test - pred
        for name, pred in predictions.items()
    }

    model_order = [name for name in results_df["Model"] if name in predictions]
    model_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(model_order)}

    chart_options = [
        "⚖️ RMSE & MAE",
        "📈 R² Score",
        "📦 Residual boxplot",
        "🌊 Residual distribution",
        "🔀 Actual vs Predicted",
        "🧭 Residuals vs Predicted",
    ]
    if diag.get("feature_importances"):
        chart_options.append("🔑 Feature importance")

    chart_choice = st.radio(
        "Choose a chart",
        chart_options,
        horizontal=True,
        label_visibility="collapsed",
    )

    if chart_choice == "⚖️ RMSE & MAE":
        st.caption("Lower values indicate smaller prediction errors. Hover for exact numbers.")
        rmse_vals = [
            float(results_df.loc[results_df["Model"] == name, "RMSE"].iloc[0])
            for name in model_order
        ]
        mae_vals = [
            float(results_df.loc[results_df["Model"] == name, "MAE"].iloc[0])
            for name in model_order
        ]
        fig = go.Figure()
        fig.add_bar(name="RMSE", x=model_order, y=rmse_vals, marker_color="#3B82F6")
        fig.add_bar(name="MAE", x=model_order, y=mae_vals, marker_color="#EC4899")
        fig.update_layout(**PLOTLY_LAYOUT, height=420, barmode="group",
                           yaxis_title="Error (RM)", title="Regression Error Comparison")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "📈 R² Score":
        st.caption("Higher values indicate better explanatory performance.")
        r2_vals = [
            float(results_df.loc[results_df["Model"] == name, "R2"].iloc[0])
            for name in model_order
        ]
        fig = px.bar(
            x=model_order, y=r2_vals, text=[f"{v:.3f}" for v in r2_vals],
            color=model_order, color_discrete_map=model_colors,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            showlegend=False,
            xaxis_title=None,
            title="R² Score by Model",
        )
        fig.update_yaxes(title="R²", range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "📦 Residual boxplot":
        st.caption(
            "Residual = Actual − Predicted Price. A tighter box around zero "
            "indicates more consistent errors. Drag to zoom, hover for outliers."
        )
        fig = go.Figure()
        for name in model_order:
            fig.add_box(y=residuals[name], name=name, marker_color=model_colors[name])
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(45,34,89,0.45)")
        fig.update_layout(**PLOTLY_LAYOUT, height=460, showlegend=False,
                           yaxis_title="Residual (RM)", title="Residual Error Distribution")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🌊 Residual distribution":
        st.caption("How prediction errors are distributed for each model.")
        fig = go.Figure()
        for name in model_order:
            fig.add_histogram(x=residuals[name], name=name, opacity=0.55,
                               marker_color=model_colors[name], nbinsx=30)
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(45,34,89,0.45)")
        fig.update_layout(**PLOTLY_LAYOUT, height=460, barmode="overlay",
                           xaxis_title="Residual (RM)", yaxis_title="Frequency",
                           title="Residual Distribution")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🔀 Actual vs Predicted":
        st.caption(
            f"Best model: {best_name}. Points closer to the dashed 45° line "
            "indicate more accurate predictions."
        )
        best_predictions = predictions[best_name]
        low = float(min(y_test.min(), best_predictions.min()))
        high = float(max(y_test.max(), best_predictions.max()))
        fig = go.Figure()
        fig.add_scatter(x=y_test, y=best_predictions, mode="markers",
                         marker=dict(color="#22C55E", opacity=0.6, size=7),
                         name=best_name)
        fig.add_scatter(x=[low, high], y=[low, high], mode="lines",
                         line=dict(dash="dash", color="rgba(45,34,89,0.45)"),
                         name="Ideal (y = x)")
        fig.update_layout(**PLOTLY_LAYOUT, height=480,
                           xaxis_title="Actual Price (RM)", yaxis_title="Predicted Price (RM)",
                           title=f"Actual vs Predicted — {best_name}")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🧭 Residuals vs Predicted":
        st.caption(
            f"Best model: {best_name}. Checks whether errors change "
            "systematically as predicted prices increase."
        )
        best_predictions = predictions[best_name]
        fig = go.Figure()
        fig.add_scatter(x=best_predictions, y=residuals[best_name], mode="markers",
                         marker=dict(color="#8B5CF6", opacity=0.6, size=7))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(45,34,89,0.45)")
        fig.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=False,
                           xaxis_title="Predicted Price (RM)", yaxis_title="Residual (RM)",
                           title=f"Residuals vs Predicted — {best_name}")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🔑 Feature importance":
        importance_df = (
            pd.DataFrame(diag["feature_importances"])
            .sort_values("importance", ascending=True)
        )
        top = importance_df.tail(15)
        fig = px.bar(
            top, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#3B82F6", "#22C55E", "#F59E0B"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=480, coloraxis_showscale=False,
                           xaxis_title="Importance", yaxis_title=None,
                           title=f"Top 15 Feature Importances — {best_name}")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    with st.expander("⚙️ Parameter configuration"):
        tuning_df = pd.DataFrame(get_tuning_results())
        if len(tuning_df):
            st.dataframe(tuning_df, width="stretch", hide_index=True)

    st.caption(f"Baseline: {baseline_name}. The final model is selected by held-out test RMSE, supported by MAE and R².")


# PAGE 3 — PRICE PREDICTOR
# =====================================================================
with tab_predict:

    # -------------------------------------------------------------
    # Setup: 3D asset map + a single helper that pushes a bundle of
    # values into every widget's session_state key at once, so a
    # preset OR an autofilled listing updates *everything* together
    # (property type, sliders, dropdowns, pills, map, city) before
    # the tab reruns.
    # -------------------------------------------------------------
    STATIC_MODELS_DIR = Path(__file__).resolve().parent / "static" / "models"

    MODEL_ASSET_MAP = {
        "Condominium": "condominium.glb",
        "Apartment": "apartment.glb",
        "Flat": "flat.glb",
        "Penthouse": "penthouse.glb",
        "Townhouse": "townhouse.glb",
        "Service Residence": None,
        "Studio": None,
        "Others": None,
    }

    def get_model_asset_url(ptype):
        filename = MODEL_ASSET_MAP.get(ptype)
        if not filename:
            return None
        if not (STATIC_MODELS_DIR / filename).exists():
            return None
        # NOTE: Streamlit's static file server only resolves this at the
        # *relative* path "app/static/..." (no leading slash) — see
        # https://docs.streamlit.io/develop/concepts/configuration/serving-static-files
        return f"app/static/models/{filename}"

    FIELD_KEYS = {
        "bedroom": "pp_bedroom",
        "bathroom": "pp_bathroom",
        "parking": "pp_parking",
        "floors": "pp_floors",
        "size": "pp_size",
        "total_units": "pp_total_units",
        "completion_year": "pp_completion_year",
        "tenure": "pp_tenure",
        "land_title": "pp_land_title",
        "floor_range": "pp_floor_range",
        "facilities": "pp_facilities",
        "nearby": "pp_nearby",
    }

    # Values emitted by the animated HTML studio are returned via query
    # parameters, then applied to Streamlit state on the next rerun.
    studio_field = st.query_params.get("studio_field")
    studio_value = st.query_params.get("studio_value")
    studio_limits = {
        "bedroom": (0, 20), "bathroom": (0, 20), "parking": (0, 20),
        "size": (1, 200000),
    }
    if studio_field in studio_limits and studio_value is not None:
        try:
            number = int(float(studio_value))
            low, high = studio_limits[studio_field]
            st.session_state[FIELD_KEYS[studio_field]] = max(low, min(high, number))
        except (TypeError, ValueError):
            pass
        st.query_params.pop("studio_field", None)
        st.query_params.pop("studio_value", None)

    def apply_field_values(values):
        """Push a full bundle of values into every widget key at once."""
        ptype = values.get("property_type")
        if ptype in PROPERTY_TYPES:
            st.session_state["predict_ptype_idx"] = PROPERTY_TYPES.index(ptype)

        state_val = values.get("state")
        if state_val in STATE_OPTIONS:
            st.session_state["state_dropdown"] = state_val
            st.session_state["map_selected_state"] = state_val

        if "city" in values:
            city_val = values.get("city")
            st.session_state["city_dropdown"] = (
                city_val if city_val in ART.city_options else "Not sure / Other"
            )

        for field, key in FIELD_KEYS.items():
            if field in values:
                st.session_state[key] = values[field]
                if field in {"bedroom", "bathroom", "parking"}:
                    st.session_state[f"fp_editor_{field}"] = int(values[field])

        st.session_state["autofill_price"] = values.get("price")

    # -------------------------------------------------------------
    # Preset bundles — pick one and every field below updates together
    # -------------------------------------------------------------
    PRESETS = {
        "🏢 City Studio": dict(
            property_type="Studio", bedroom=1, bathroom=1, size=480, floors=28,
            total_units=450, parking=1, completion_year=2020, tenure="Leasehold",
            land_title="Non Bumi Lot", floor_range="High", state="Kuala Lumpur",
            facilities=["Security", "Lift", "Gymnasium"],
            nearby=["Bus Stop", "Mall", "Nearby Railway Station"],
        ),
        "👨‍👩‍👧 Family Condo": dict(
            property_type="Condominium", bedroom=4, bathroom=3, size=1400, floors=18,
            total_units=600, parking=2, completion_year=2016, tenure="Freehold",
            land_title="Non Bumi Lot", floor_range="Medium", state="Selangor",
            facilities=["Parking", "Security", "Swimming Pool", "Playground", "Gymnasium", "Clubhouse"],
            nearby=["School", "Park", "Mall", "Hospital"],
        ),
        "🌆 Luxury Penthouse": dict(
            property_type="Penthouse", bedroom=5, bathroom=5, size=3200, floors=45,
            total_units=200, parking=4, completion_year=2022, tenure="Freehold",
            land_title="Non Bumi Lot", floor_range="Top", state="Kuala Lumpur",
            facilities=["Security", "Swimming Pool", "Gymnasium", "Sauna", "Clubhouse", "Lift"],
            nearby=["Mall", "Highway", "Nearby Railway Station"],
        ),
        "🏘️ Suburban Townhouse": dict(
            property_type="Townhouse", bedroom=3, bathroom=2, size=1800, floors=3,
            total_units=80, parking=2, completion_year=2012, tenure="Freehold",
            land_title="Malay Reserved Land", floor_range="-", state="Penang",
            facilities=["Parking", "Security", "Playground", "Jogging Track"],
            nearby=["School", "Park", "Highway"],
        ),
        "🏬 Budget Flat": dict(
            property_type="Flat", bedroom=2, bathroom=1, size=750, floors=5,
            total_units=250, parking=1, completion_year=2005, tenure="Leasehold",
            land_title="Bumi Lot", floor_range="Low", state="Johor",
            facilities=["Parking", "Minimart"],
            nearby=["Bus Stop", "School"],
        ),
    }

    PTYPE_VISUALS = {
        "Condominium":        {"emoji": "🏙️", "floors": 9,  "grad": ("#22C55E", "#3B82F6")},
        "Apartment":          {"emoji": "🏢", "floors": 6,  "grad": ("#3B82F6", "#8B5CF6")},
        "Service Residence":  {"emoji": "🏨", "floors": 8,  "grad": ("#8B5CF6", "#EC4899")},
        "Studio":             {"emoji": "🏠", "floors": 3,  "grad": ("#F59E0B", "#EC4899")},
        "Flat":                {"emoji": "🏬", "floors": 5,  "grad": ("#06B6D4", "#3B82F6")},
        "Penthouse":          {"emoji": "🌆", "floors": 12, "grad": ("#EC4899", "#8B5CF6")},
        "Townhouse":          {"emoji": "🏘️", "floors": 3,  "grad": ("#22C55E", "#F59E0B")},
        "Others":              {"emoji": "🏗️", "floors": 4,  "grad": ("#64748B", "#3B82F6")},
    }

    st.markdown("##### 🔮 Condominium Price Prediction")
    st.caption(
        "Enter property characteristics and the tuned best-performing "
        f"model ({best_name}) will estimate the indicative market price."
    )

    # -------------------------------------------------------------
    # Scoped CSS for this tab only
    # -------------------------------------------------------------
    preset_button_css = "\n".join(
        f"""
          .pp-preset-row div[data-testid="column"]:nth-of-type({i + 1}) div[data-testid="stButton"] button {{
              border-radius: 14px !important;
              border: 1px solid {PTYPE_VISUALS.get(values["property_type"], PTYPE_VISUALS["Others"])["grad"][0]}70 !important;
              background: linear-gradient(135deg,
                  {PTYPE_VISUALS.get(values["property_type"], PTYPE_VISUALS["Others"])["grad"][0]}2A,
                  {PTYPE_VISUALS.get(values["property_type"], PTYPE_VISUALS["Others"])["grad"][1]}1A) !important;
              font-weight: 700 !important;
              font-size: 13px !important;
              padding: 10px 8px !important;
              transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
          }}
          .pp-preset-row div[data-testid="column"]:nth-of-type({i + 1}) div[data-testid="stButton"] button:hover {{
              transform: translateY(-3px) scale(1.015);
              box-shadow: 0 10px 20px -10px {PTYPE_VISUALS.get(values["property_type"], PTYPE_VISUALS["Others"])["grad"][0]}aa;
              border-color: {PTYPE_VISUALS.get(values["property_type"], PTYPE_VISUALS["Others"])["grad"][1]}cc !important;
          }}
        """
        for i, values in enumerate(PRESETS.values())
    )

    st.markdown(
        f"""
        <style>
          {preset_button_css}
          .floorplan-marker + div[data-testid="stVerticalBlockBorderWrapper"] {{
              background:
                repeating-linear-gradient(0deg, rgba(59,130,246,0.08) 0 1px, transparent 1px 26px),
                repeating-linear-gradient(90deg, rgba(59,130,246,0.08) 0 1px, transparent 1px 26px),
                linear-gradient(135deg, rgba(34,197,94,0.06), rgba(139,92,246,0.06));
              border: 1px solid rgba(59,130,246,0.45) !important;
              border-radius: 18px !important;
              box-shadow: 0 22px 34px -20px rgba(59,130,246,0.55);
          }}
          .ptype-cycle-btn button {{
              height: 46px !important;
              border-radius: 12px !important;
              font-size: 20px !important;
              font-weight: 800 !important;
              background: linear-gradient(135deg,#3B82F6,#8B5CF6) !important;
              border: none !important;
              color: white !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Optional autofill from a real listing (collapsed to stay compact)
    # -------------------------------------------------------------
    with st.expander("📋 Or autofill from an existing listing", expanded=False):
        listing_pool = get_listings()["listings"]

        if listing_pool:
            col_a1, col_a2 = st.columns([4, 1])

            with col_a1:
                listing_ids = [row["index"] for row in listing_pool]

                def listing_label(idx):
                    row = next(
                        (r for r in listing_pool if r["index"] == idx),
                        None,
                    )
                    if row is None:
                        return str(idx)
                    return (
                        f"#{idx}: {row['State']} · "
                        f"{row['Property Type']} · "
                        f"{row['Property Size']:.0f} sqft"
                    )

                selected_listing = st.selectbox(
                    "Choose a sample listing",
                    listing_ids,
                    format_func=listing_label,
                )

            with col_a2:
                st.write("")
                st.write("")
                autofill_clicked = st.button("Set", width="stretch")

            if autofill_clicked:
                loaded = get_listing(selected_listing)
                autofill_values = {
                    "property_type": loaded.get("Property Type"),
                    "bedroom": safe_int(loaded.get("Bedroom"), 3),
                    "bathroom": safe_int(loaded.get("Bathroom"), 2),
                    "parking": safe_int(loaded.get("Parking Lot"), 1),
                    "floors": max(1, safe_int(loaded.get("# of Floors"), 20)),
                    "size": max(1, safe_int(loaded.get("Property Size"), 900)),
                    "total_units": max(1, safe_int(loaded.get("Total Units"), 500)),
                    "completion_year": safe_int(loaded.get("Completion Year"), 2015),
                    "tenure": (
                        loaded.get("Tenure Type")
                        if loaded.get("Tenure Type") in TENURE_OPTIONS
                        else "Freehold"
                    ),
                    "land_title": (
                        loaded.get("Land Title")
                        if loaded.get("Land Title") in LAND_OPTIONS
                        else "Non Bumi Lot"
                    ),
                    "floor_range": (
                        loaded.get("Floor Range")
                        if loaded.get("Floor Range") in FLOOR_RANGE_OPTIONS
                        else "-"
                    ),
                    "state": loaded.get("State"),
                    "city": loaded.get("City"),
                    "facilities": [
                        f for f in FACILITY_OPTIONS
                        if loaded.get(f"Facility_{f}") == 1
                    ],
                    "nearby": [
                        n for n in NEARBY_OPTIONS
                        if loaded.get(f"Has_{n}") == 1
                    ],
                    "price": loaded.get("price"),
                }
                apply_field_values(autofill_values)
                st.success("Every field below has been autofilled from the selected listing.")
                st.rerun()
        else:
            st.info("No sample listings are available.")

    autofill_price = st.session_state.get("autofill_price")

    # -------------------------------------------------------------
    # Build your property in 3D — one unified studio that combines what
    # used to be three separate blocks (3D builder, property showcase,
    # floor-plan editor) into a single component so the model, the
    # property-type switcher, and the tappable stat editors all live
    # in one place instead of repeating the model / repeating the
    # bedroom-bathroom-parking inputs across sections.
    # -------------------------------------------------------------
    if "predict_ptype_idx" not in st.session_state:
        st.session_state["predict_ptype_idx"] = 0

    ptype_idx = st.session_state["predict_ptype_idx"] % len(PROPERTY_TYPES)
    property_type = PROPERTY_TYPES[ptype_idx]
    visual = PTYPE_VISUALS.get(property_type, PTYPE_VISUALS["Others"])
    color_a, color_b = visual["grad"]

    studio_model_url = get_model_asset_url(property_type)
    studio_values = {
        "bedroom": safe_int(st.session_state.get("pp_bedroom", 3), 3),
        "bathroom": safe_int(st.session_state.get("pp_bathroom", 2), 2),
        "parking": safe_int(st.session_state.get("pp_parking", 1), 1),
        "size": safe_int(st.session_state.get("pp_size", 900), 900),
    }
    studio_labels = {
        "bedroom": ("🛏️", "Bedrooms", "rooms"),
        "bathroom": ("🛁", "Bathrooms", "rooms"),
        "parking": ("🚗", "Parking", "lots"),
        "size": ("📐", "Property size", "sq.ft."),
    }
    studio_cards = "".join(
        f'<button class="studio-card card-{field}" onclick="openEditor(\'{field}\')">'
        f'<span>{icon}</span><b>{label}</b><em>{studio_values[field]:,} {unit}</em></button>'
        for field, (icon, label, unit) in studio_labels.items()
    )

    # Colourised, per-type animated building — used both as the "no .glb
    # bundled yet" fallback and, visually, as the same building shown in
    # the old Property Showcase block, so nothing is lost by merging.
    css_slats = "".join(
        f'<div class="bfloor" style="animation-delay:{i * 0.05}s;'
        f'background:linear-gradient(90deg,{color_a}cc,{color_b}cc);"></div>'
        for i in range(visual["floors"])
    )
    studio_fallback_building = f"""
      <div class="studio-building3d">
        <div class="sbface sbfront">{css_slats}</div>
        <div class="sbface sbside" style="background:linear-gradient(180deg,{color_a}66,{color_b}22);"></div>
        <div class="sbface sbtop" style="background:linear-gradient(90deg,{color_a}cc,{color_b}cc);"></div>
      </div>
    """
    studio_model = (
        f'<model-viewer src="{studio_model_url}" alt="3D {property_type}" camera-controls auto-rotate '
        'rotation-per-second="18deg" shadow-intensity="1" exposure="1.12"></model-viewer>'
        if studio_model_url else
        studio_fallback_building
    )
    studio_html = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
    <main class="studio-shell" id="studioShell" onclick="backdropClose(event)">
      <section class="studio-copy"><span>LIVE PROPERTY BUILDER</span><h2>Tap a detail to shape your home.</h2><p>The model responds as you build a property profile.</p></section>
      <section class="input-drawer" id="inputDrawer">
        <button class="drawer-close" onclick="closeEditor()">×</button>
        <div class="drawer-icon" id="drawerIcon">✨</div><p>PROPERTY DETAIL</p>
        <h3 id="drawerTitle">Choose a detail</h3><small id="drawerHint">Select a floating card around the model.</small>
        <input id="drawerInput" type="number" min="0" step="1" />
        <button class="drawer-save" onclick="saveEditor()">Apply to model ✨</button>
      </section>
      <section class="studio-stage" id="studioStage">
        <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
        <div class="studio-model">{studio_model}</div>
        <div class="studio-cards">{studio_cards}</div>
        <div class="model-badge">{visual['emoji']} {property_type} · 3D VIEW</div>
      </section>
    </main>
    <script>
      const values = {studio_values};
      const meta = {{bedroom:['🛏️','Bedrooms','rooms',0,20], bathroom:['🛁','Bathrooms','rooms',0,20], parking:['🚗','Parking lots','lots',0,20], size:['📐','Property size','sq.ft.',1,200000]}};
      let active = null;
      function openEditor(field) {{
        active = field; const item = meta[field];
        document.getElementById('studioShell').classList.add('editing');
        document.getElementById('drawerIcon').textContent = item[0];
        document.getElementById('drawerTitle').textContent = item[1];
        document.getElementById('drawerHint').textContent = 'Set the number of ' + item[1].toLowerCase() + '.';
        const input = document.getElementById('drawerInput'); input.value = values[field]; input.min = item[3]; input.max = item[4]; input.focus();
      }}
      function closeEditor() {{ active = null; document.getElementById('studioShell').classList.remove('editing'); }}
      function backdropClose(event) {{ if (event.target.id === 'studioShell') closeEditor(); }}
      function saveEditor() {{
        if (!active) return; const input = document.getElementById('drawerInput');
        const url = new URL(window.parent.location.href); url.searchParams.set('studio_field', active); url.searchParams.set('studio_value', input.value); window.parent.location.href = url.toString();
      }}
    </script>
    <style>
      * {{ box-sizing:border-box; }} body {{ margin:0; background:transparent; color:#283551; font-family:Inter,Arial,sans-serif; }}
      .studio-shell {{ min-height:425px; position:relative; overflow:hidden; border-radius:28px; padding:26px; background:linear-gradient(130deg,#fff7cf,#ffe7f3 45%,#dff7ff); border:1px solid rgba(109,74,255,.14); }}
      .studio-copy {{ position:relative; z-index:3; width:36%; }} .studio-copy span,.input-drawer p {{ color:#7759d7; font-size:10px; font-weight:900; letter-spacing:.14em; }} .studio-copy h2 {{ font-size:28px; line-height:1.02; margin:7px 0 10px; letter-spacing:-.04em; }} .studio-copy p {{ margin:0; font-size:13px; color:#65708a; max-width:210px; }}
      .studio-stage {{ position:absolute; inset:0 0 0 28%; transition:transform .62s cubic-bezier(.2,.9,.2,1); }} .editing .studio-stage {{ transform:translateX(19%); }}
      .studio-model {{ width:245px; height:280px; position:absolute; left:50%; top:58px; transform:translateX(-50%); z-index:2; display:flex; align-items:flex-end; justify-content:center; perspective:950px; }} model-viewer {{ width:100%; height:100%; --poster-color:transparent; }}
      .orbit {{ position:absolute; border:1px dashed rgba(109,74,255,.25); border-radius:50%; left:50%; top:55%; transform:translate(-50%,-50%); }} .orbit-one {{ width:340px; height:130px; }} .orbit-two {{ width:270px; height:100px; transform:translate(-50%,-50%) rotate(-19deg); }}
      .studio-cards {{ position:absolute; inset:0; z-index:4; }} .studio-card {{ position:absolute; display:flex; gap:7px; align-items:center; text-align:left; padding:9px 12px; border:1px solid rgba(109,74,255,.17); border-radius:16px; background:rgba(255,255,255,.88); box-shadow:0 16px 28px -19px rgba(67,47,143,.45); color:#293451; cursor:pointer; transition:transform .2s, box-shadow .2s; }} .studio-card:hover {{ transform:translateY(-4px) scale(1.03); box-shadow:0 18px 30px -15px rgba(109,74,255,.4); }} .studio-card span {{ font-size:21px; }} .studio-card b,.studio-card em {{ display:block; font-style:normal; }} .studio-card b {{ font-size:11px; }} .studio-card em {{ color:#7865b6; font-size:10px; margin-top:2px; }} .card-bedroom {{ left:10%; top:25%; }} .card-bathroom {{ right:8%; top:21%; }} .card-parking {{ right:10%; bottom:20%; }} .card-size {{ left:15%; bottom:18%; }}
      .model-badge {{ position:absolute; bottom:16px; left:50%; transform:translateX(-50%); z-index:4; color:#624fa4; font-size:10px; font-weight:900; padding:7px 10px; border-radius:999px; background:rgba(255,255,255,.8); }}
      .input-drawer {{ position:absolute; z-index:8; left:20px; top:50%; width:258px; padding:24px; border-radius:22px; background:rgba(255,255,255,.96); box-shadow:0 25px 45px -22px rgba(68,45,142,.45); opacity:0; transform:translate(-130%,-50%); transition:opacity .42s ease,transform .58s cubic-bezier(.2,.9,.2,1); pointer-events:none; }} .editing .input-drawer {{ opacity:1; transform:translate(0,-50%); pointer-events:auto; }} .drawer-close {{ position:absolute; right:12px; top:10px; border:0; background:#f1edff; border-radius:50%; width:26px; height:26px; font-size:20px; cursor:pointer; color:#684dd1; }} .drawer-icon {{ font-size:31px; }} .input-drawer h3 {{ margin:7px 0 5px; font-size:22px; }} .input-drawer small {{ color:#68738c; line-height:1.3; }} .input-drawer input {{ width:100%; margin:17px 0 10px; padding:12px; border:1px solid #dfd7ff; border-radius:13px; color:#303b59; font-size:18px; font-weight:800; outline-color:#7456ff; }} .drawer-save {{ width:100%; border:0; border-radius:13px; padding:12px; color:#fff; font-weight:800; background:linear-gradient(105deg,#6d4aff,#ff68a8); cursor:pointer; }}
      .studio-building3d {{ position:relative; width:100px; height:150px; transform-style:preserve-3d; animation: studioSpin3d 9s linear infinite; }}
      @keyframes studioSpin3d {{ from {{ transform: rotateY(0deg) rotateX(8deg); }} to {{ transform: rotateY(360deg) rotateX(8deg); }} }}
      @keyframes studioFloorGlow {{ 0%, 100% {{ opacity: 0.75; }} 50% {{ opacity: 1; }} }}
      .sbface {{ position:absolute; border:1px solid rgba(255,255,255,0.28); border-radius:4px; }}
      .sbfront {{ width:100px; height:150px; transform: translateZ(25px); display:flex; flex-direction:column-reverse; gap:3px; padding:5px; background: rgba(255,255,255,0.03); }}
      .sbside {{ width:50px; height:150px; transform: rotateY(90deg) translateZ(25px) translateX(25px); }}
      .sbtop {{ width:100px; height:50px; transform: rotateX(90deg) translateZ(25px) translateY(-50px); opacity: 0.85; }}
      .bfloor {{ flex:1; border-radius:2px; box-shadow: 0 0 8px rgba(255,255,255,0.18) inset; animation: studioFloorGlow 2.4s ease-in-out infinite; }}
      @media(max-width:650px) {{ .studio-copy {{ width:55%; }} .studio-copy h2 {{ font-size:22px; }} .studio-stage {{ left:8%; }} .studio-model {{ transform:translateX(-30%); }} .studio-card {{ padding:7px; }} .studio-card b {{ display:none; }} .card-bedroom {{ left:1%; }} .card-size {{ left:2%; }} .card-bathroom {{ right:1%; }} .card-parking {{ right:1%; }} .editing .studio-stage {{ transform:translateX(40%); }} }}
    </style>
    """
    st.markdown("#### 🧩 Build your property in 3D")
    st.caption(
        "Cycle property types with the arrows, tap a floating card to edit a stat, "
        "or pick a preset above — one model, always in sync."
    )

    studio_prev, studio_stage_col, studio_next = st.columns([1, 8, 1])

    with studio_prev:
        st.write("")
        st.markdown('<div class="ptype-cycle-btn">', unsafe_allow_html=True)
        if st.button("◀", key="ptype_prev", width="stretch", help="Previous property type"):
            st.session_state["predict_ptype_idx"] = (ptype_idx - 1) % len(PROPERTY_TYPES)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with studio_stage_col:
        components.html(studio_html, height=435)

    with studio_next:
        st.write("")
        st.markdown('<div class="ptype-cycle-btn">', unsafe_allow_html=True)
        if st.button("▶", key="ptype_next", width="stretch", help="Next property type"):
            st.session_state["predict_ptype_idx"] = (ptype_idx + 1) % len(PROPERTY_TYPES)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if not studio_model_url:
        st.caption(
            "🧩 No bundled 3D model for this type yet — showing the animated "
            "placeholder. Drop a `.glb` into `static/models/` (see README) to enable the real model."
        )

    # The floating cards' drawer owns bedroom/bathroom/parking/size — read
    # their current values back so the prediction payload stays synced.
    bedroom = safe_int(st.session_state.get("pp_bedroom", 3), 3)
    bathroom = safe_int(st.session_state.get("pp_bathroom", 2), 2)
    parking = safe_int(st.session_state.get("pp_parking", 1), 1)

    def visual_picker(label, options, key, default, format_func=None):
        """Chip-based input for a fast, visual alternative to sliders/selects."""
        current = st.session_state.get(key, default)
        values = list(options)
        if current not in values:
            values.append(current)
        selected = st.pills(
            label, values, default=current, selection_mode="single", key=key,
            format_func=format_func,
        )
        return selected if selected is not None else current

    st.markdown("#### ✨ Choose your property profile")
    st.caption("Tap visual chips to set the remaining details - no sliders required.")
    profile_1, profile_2 = st.columns(2)
    with profile_1:
        floors = safe_int(visual_picker("🏗️ Building floors", [3, 5, 10, 15, 20, 30, 45, 60], "pp_floors", 20), 20)
        size = safe_int(visual_picker("📐 Property size", [480, 650, 750, 900, 1200, 1400, 1800, 2500, 3200], "pp_size", 900,
                                       lambda value: f"{value:,} sq.ft."), 900)
        total_units = safe_int(visual_picker("🏘️ Total units", [50, 100, 200, 300, 450, 600, 800, 1200], "pp_total_units", 500), 500)
    with profile_2:
        completion_year = safe_int(visual_picker("🗓️ Completion year", [2000, 2005, 2010, 2012, 2015, 2016, 2020, 2022, 2025], "pp_completion_year", 2015), 2015)
        tenure = visual_picker("🔑 Tenure", TENURE_OPTIONS, "pp_tenure", "Freehold")
        land_title = visual_picker("📜 Land title", LAND_OPTIONS, "pp_land_title", "Non Bumi Lot")
        floor_range = visual_picker("⬆️ Floor range", FLOOR_RANGE_OPTIONS, "pp_floor_range", "-")

    # State map + dropdown
    # -------------------------------------------------------------
    st.divider()
    st.markdown("#### 📍 Location")

    state_summary = get_state_summary()

    map_states = [state for state in STATE_OPTIONS if state in STATE_COORDS]

    max_avg = max(
        [state_summary.get(state, {}).get("avg_price", 0) for state in map_states] or [1]
    ) or 1

    max_count = max(
        [state_summary.get(state, {}).get("count", 0) for state in map_states] or [1]
    ) or 1

    map_rows = []
    for state_name in map_states:
        summary = state_summary.get(state_name, {"avg_price": 0, "count": 0})
        heat = summary["avg_price"] / max_avg
        r, g, b = heat_to_rgb(heat)
        map_rows.append(
            {
                "state": state_name,
                "lat": STATE_COORDS[state_name]["lat"],
                "lon": STATE_COORDS[state_name]["lon"],
                "avg_price": summary["avg_price"],
                "count": summary["count"],
                "radius": 12000 + 28000 * (summary["count"] / max_count if max_count else 0),
                "r": r, "g": g, "b": b,
            }
        )

    map_df = pd.DataFrame(map_rows)

    current_state = st.session_state.get("map_selected_state", "Selangor")

    map_col, picker_col = st.columns([2, 1])

    with map_col:
        layer = pdk.Layer(
            "ScatterplotLayer",
            id="state-layer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="[r, g, b, 210]",
            get_line_color="[255, 255, 255, 220]",
            line_width_min_pixels=2,
            stroked=True,
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=4.0, longitude=109.5, zoom=5.0, pitch=0, min_zoom=4, max_zoom=8,
        )

        frozen_view = pdk.View(type="MapView", controller=False)

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            views=[frozen_view],
            map_style="light",
            map_provider="carto",
            tooltip={
                "text": (
                    "{state}\n"
                    "Avg price: RM {avg_price}\n"
                    "Listings: {count}"
                ),
            },
        )

        selection = st.pydeck_chart(
            deck, height=320, on_select="rerun", selection_mode="single-object", key="state_map",
        )

        clicked_state = None
        if selection and selection.selection:
            objects = selection.selection.get("objects", {})
            picked = objects.get("state-layer") or next(iter(objects.values()), None)
            if picked:
                row = picked[0] if isinstance(picked, list) else picked
                clicked_state = row.get("state")

        if clicked_state and clicked_state != st.session_state.get("map_selected_state"):
            st.session_state["map_selected_state"] = clicked_state
            st.session_state["state_dropdown"] = clicked_state
            st.rerun()

    with picker_col:
        st.markdown("##### Select a market")
        state = visual_picker("State", STATE_OPTIONS, "state_dropdown", current_state)

        if state != st.session_state.get("map_selected_state"):
            st.session_state["map_selected_state"] = state

        if state in state_summary:
            st.metric("Average price", money(state_summary[state]["avg_price"]))
            st.caption(f"{state_summary[state]['count']:,} sample listings")

    if ART.city_options:
        city_options = ["Not sure / Other"] + ART.city_options
        default_city = st.session_state.get("city_dropdown", "Not sure / Other")
        city_index = city_options.index(default_city) if default_city in city_options else 0

        city_choice = st.selectbox(
            "City", city_options, index=city_index, key="city_dropdown",
        )
        city = None if city_choice == "Not sure / Other" else city_choice
    else:
        city = None

    # -------------------------------------------------------------
    # Facilities + nearby amenities
    # -------------------------------------------------------------
    st.divider()
    st.markdown("#### 🏊 Facilities & Nearby Amenities")

    fac_col, nearby_col = st.columns(2)

    with fac_col:
        facilities = st.pills(
            "Facilities", FACILITY_OPTIONS,
            default=st.session_state.get("pp_facilities", []),
            selection_mode="multi", key="pp_facilities",
        ) or []

    with nearby_col:
        nearby = st.pills(
            "Nearby amenities", NEARBY_OPTIONS,
            default=st.session_state.get("pp_nearby", []),
            selection_mode="multi", key="pp_nearby",
            format_func=lambda x: x.replace("_", " "),
        ) or []

    # -------------------------------------------------------------
    # Predict / reset
    # -------------------------------------------------------------
    st.divider()
    pred_col, reset_col = st.columns(2)

    with pred_col:
        predict_clicked = st.button("🔍 Predict Price", type="primary", width="stretch")

    with reset_col:
        reset_clicked = st.button("↺ Reset", width="stretch")

    if reset_clicked:
        keys_to_clear = [
            "autofill_price", "map_selected_state", "state_dropdown",
            "city_dropdown", "state_map", "predict_ptype_idx",
        ] + list(FIELD_KEYS.values())
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

    if predict_clicked:
        payload = {
            "bedroom": bedroom,
            "bathroom": bathroom,
            "size": size,
            "floors": floors,
            "total_units": total_units,
            "parking": parking,
            "completion_year": completion_year,
            "tenure": tenure,
            "property_type": property_type,
            "state": state,
            "city": city,
            "land_title": land_title,
            "floor_range": floor_range,
            "facilities": facilities,
            "nearby": nearby,
        }

        spinner_messages = [
            "Crunching the numbers…",
            "Comparing property characteristics…",
            "Checking location effects…",
            "Consulting the tuned model…",
        ]

        with st.spinner(random.choice(spinner_messages)):
            time.sleep(0.25)
            result = predict(payload)

        render_price_reveal(
            result["prediction"], result.get("price_per_sqft"), result.get("bracket"),
        )

        st.success(f"Prediction generated using the tuned {best_name} model.")

        metric_row([
            ("Predicted Price", money(result["prediction"]), None, "#22C55E"),
            ("Test RMSE", money(result["best_rmse"]), None, "#3B82F6"),
            ("Price / sq.ft.", money(result["price_per_sqft"]), None, "#F59E0B"),
        ])

        if result["range_low"] is not None:
            st.info(
                "Indicative error band based on the best model's held-out "
                f"test RMSE: {money(result['range_low'])} – "
                f"{money(result['range_high'])}. "
                "This is not a formal confidence interval."
            )

        if result["similar_avg"] is not None:
            direction = "above" if result["diff_pct"] >= 0 else "below"
            st.caption(
                f"🏘️ Similar {property_type} listings in {state}: "
                f"average {money(result['similar_avg'])} "
                f"(n={result['similar_count']}). "
                f"The prediction is "
                f"{abs(result['diff_pct']):.1f}% {direction} that average."
            )

        if result.get("feature_importances"):
            importance_df = (
                pd.DataFrame(result["feature_importances"])
                .set_index("feature")
                .sort_values("importance", ascending=True)
            )
            with st.expander("🔍 What features generally drive the model?"):
                st.caption(
                    "These are overall model feature-importance values. "
                    "They should not be interpreted as causal effects."
                )
                st.bar_chart(importance_df, height=300)

        if autofill_price is not None:
            st.caption(f"Actual price of the autofilled listing: {money(autofill_price)}")

        st.caption(
            "This prediction is an indicative estimate for educational "
            "and analytical purposes and should not be used as the sole "
            "basis for a property or financial decision."
        )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()

st.caption(
    f"BMDS2003 · Four-model regression comparison · "
    f"Baseline: {baseline_name} · "
    f"Best held-out RMSE: {money(comp['best_rmse'])}"
)
