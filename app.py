import random
import time

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
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
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
          padding-top: 1.1rem;
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

      /* Pills / multiselect chips get a splash of colour */
      span[data-baseweb="tag"] {
          background: linear-gradient(135deg,#3B82F6,#8B5CF6) !important;
      }

      /* Compact metric cards */
      .mini-card {
          border-radius: 14px;
          padding: 14px 14px;
          text-align: center;
          color: white;
          box-shadow: 0 2px 10px rgba(0,0,0,0.15);
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
          border: 1px solid rgba(139,92,246,0.35) !important;
      }
      div[data-testid="stContainer"]:has(> div > div > div[data-testid="stMarkdownContainer"]) {
          border-radius: 14px;
      }
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
        background: linear-gradient(120deg, rgba(34,197,94,.16), rgba(59,130,246,.16) 45%, rgba(236,72,153,.16));
        border: 1px solid rgba(139,92,246,.35);
        border-radius: 16px;
        padding: 14px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
      }
      .app-title {
        font-size: 23px;
        font-weight: 800;
        background: linear-gradient(90deg,#22C55E,#3B82F6,#EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .app-subtitle { opacity: .7; font-size: 13px; margin-top: 2px; }
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
        "Cleaned listings from houses.csv — rows without a valid positive "
        "target or property size are excluded."
    )

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
            avg_by_state = (
                filtered.groupby("State")["price"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(
                avg_by_state,
                x="State",
                y="price",
                color="price",
                color_continuous_scale=["#3B82F6", "#22C55E", "#F59E0B", "#EC4899"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            fig.update_yaxes(title="Avg. Price (RM)")
            fig.update_xaxes(title=None)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data for the selected filters.")

    with col_c2:
        st.markdown("##### 📊 Price Distribution")
        if len(filtered):
            price_bins = pd.cut(filtered["price"], bins=10)
            dist_counts = price_bins.value_counts().sort_index()
            dist_df = pd.DataFrame({
                "range": dist_counts.index.astype(str),
                "count": dist_counts.values,
            })
            fig = px.bar(
                dist_df,
                x="range",
                y="count",
                color="count",
                color_continuous_scale=["#8B5CF6", "#EC4899", "#F59E0B"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            fig.update_yaxes(title="Listings")
            fig.update_xaxes(title=None, tickangle=30)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data for the selected filters.")

    st.divider()

    st.markdown("##### 🔍 Data Quality Checks")

    dq1, dq2 = st.columns(2)

    with dq1:
        st.markdown("###### ❓ Missing Values")
        missing_df = pd.DataFrame(quality["missing_summary"])
        if len(missing_df):
            missing_df = missing_df.sort_values(
                "Missing Values",
                ascending=False,
            )
            st.dataframe(
                missing_df,
                width="stretch",
                hide_index=True,
            )
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
    bracket_df = pd.DataFrame(diag["bracket_results_all_models"])

    # Merge the regression and supplementary classification metrics so that
    # every model appears in ONE complete comparison table.
    if len(bracket_df):
        results_df = results_df.merge(bracket_df, on="Model", how="left")

    st.caption(
        "Four regression models compared on the same held-out test set. "
        "RMSE/MAE/R² are the primary metrics; Accuracy/Precision/Recall/F1 "
        "are supplementary price-bracket metrics."
    )

    metric_row([
        ("Best Model", best_name, None, "#22C55E"),
        ("Baseline", baseline_name, None, "#3B82F6"),
        ("Test RMSE", money(comp["best_rmse"]), None, "#F59E0B"),
        ("Test R²", f"{comp['best_r2']:.3f}", None, "#EC4899"),
    ])

    with st.expander("📋 Final Test-Set Results table", expanded=False):
        display_cols = [
            "Model", "Role", "Accuracy", "Precision", "Recall", "F1",
            "RMSE", "MAE", "R2"
        ]
        # Backend versions without Role are still supported.
        display_cols = [c for c in display_cols if c in results_df.columns]

        formatted = results_df[display_cols].copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].map(lambda x: f"{x:.1%}")
        for col in ["RMSE", "MAE"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].map(lambda x: f"RM {x:,.0f}")
        if "R2" in formatted.columns:
            formatted["R2"] = formatted["R2"].map(lambda x: f"{x:.4f}")

        st.dataframe(formatted, width="stretch", hide_index=True)

    st.divider()

    # ================================================================
    # INTERACTIVE PLOTLY CHARTS — one selector instead of 8 stacked charts
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
        "🎯 Classification metrics",
        "🧩 Confusion matrix",
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
        fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False,
                           yaxis=dict(title="R²", range=[0, 1]), xaxis_title=None,
                           title="R² Score by Model")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🎯 Classification metrics":
        st.caption(
            "Supplementary metrics after converting the continuous price "
            "prediction into four price brackets."
        )
        if len(bracket_df):
            long_df = bracket_df.melt(
                id_vars="Model",
                value_vars=["Accuracy", "Precision", "Recall", "F1"],
                var_name="Metric", value_name="Score",
            )
            fig = px.bar(
                long_df, x="Model", y="Score", color="Metric", barmode="group",
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=420,
                               yaxis=dict(range=[0, 1]), xaxis_title=None,
                               title="Price-Bracket Classification Metrics")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No classification metrics available.")

    elif chart_choice == "🧩 Confusion matrix":
        cm = diag.get("bracket_confusion_matrix") or []
        labels = diag.get("price_bin_labels") or []
        st.caption(
            f"Best model: {best_name}. Rows = actual price bracket, "
            "columns = predicted price bracket. Darker cells on the "
            "diagonal mean more correct classifications."
        )
        if cm and labels:
            cm_arr = np.asarray(cm)
            short_labels = [
                lbl.replace("RM ", "").replace(",", "") for lbl in labels
            ]
            fig = px.imshow(
                cm_arr,
                x=short_labels,
                y=short_labels,
                color_continuous_scale=["#0f172a", "#3B82F6", "#22C55E"],
                text_auto=True,
                aspect="auto",
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, height=480,
                xaxis_title="Predicted bracket", yaxis_title="Actual bracket",
                title=f"Confusion Matrix — {best_name} (price brackets)",
                coloraxis_showscale=False,
            )
            fig.update_traces(textfont_size=14)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No confusion matrix available.")

    elif chart_choice == "📦 Residual boxplot":
        st.caption(
            "Residual = Actual − Predicted Price. A tighter box around zero "
            "indicates more consistent errors. Drag to zoom, hover for outliers."
        )
        fig = go.Figure()
        for name in model_order:
            fig.add_box(y=residuals[name], name=name, marker_color=model_colors[name])
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
        fig.update_layout(**PLOTLY_LAYOUT, height=460, showlegend=False,
                           yaxis_title="Residual (RM)", title="Residual Error Distribution")
        st.plotly_chart(fig, width="stretch")

    elif chart_choice == "🌊 Residual distribution":
        st.caption("How prediction errors are distributed for each model.")
        fig = go.Figure()
        for name in model_order:
            fig.add_histogram(x=residuals[name], name=name, opacity=0.55,
                               marker_color=model_colors[name], nbinsx=30)
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
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
                         line=dict(dash="dash", color="rgba(255,255,255,0.5)"),
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
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
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

    st.caption(
        f"Baseline: {baseline_name}. Classification metrics are supplementary; "
        "RMSE, MAE and R² remain the primary regression measures."
    )


# PAGE 3 — PRICE PREDICTOR
# =====================================================================
with tab_predict:
    st.markdown("##### 🔮 Condominium Price Prediction")

    st.caption(
        "Enter property characteristics and the tuned best-performing "
        f"model ({best_name}) will estimate the indicative market price."
    )

    # -------------------------------------------------------------
    # Optional autofill
    # -------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**Optional: Autofill from an Existing Listing**")

        listing_pool = get_listings()["listings"]

        if listing_pool:
            col_a1, col_a2 = st.columns([4, 1])

            with col_a1:
                listing_ids = [row["index"] for row in listing_pool]

                def listing_label(idx):
                    row = next(
                        (
                            r for r in listing_pool
                            if r["index"] == idx
                        ),
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
                autofill_clicked = st.button(
                    "Set",
                    width="stretch",
                )

            if autofill_clicked:
                loaded = get_listing(selected_listing)
                st.session_state["autofill_row"] = loaded

                loaded_state = loaded.get("State")
                if loaded_state in STATE_OPTIONS:
                    st.session_state["state_dropdown"] = loaded_state
                    st.session_state["map_selected_state"] = loaded_state

                loaded_city = loaded.get("City")
                if loaded_city in ART.city_options:
                    st.session_state["city_dropdown"] = loaded_city

                st.success(
                    "The property fields have been autofilled from the selected listing."
                )
        else:
            st.info("No sample listings are available.")

    prefill = st.session_state.get("autofill_row", {})

    def pf(column, default):
        value = prefill.get(column, default)
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return value

    st.markdown("#### 🏠 Property Details")

    row1 = st.columns(4)

    with row1[0]:
        bedroom = st.number_input(
            "🛏️ Bedrooms",
            min_value=0,
            max_value=20,
            value=safe_int(pf("Bedroom", 3), 3),
        )

    with row1[1]:
        bathroom = st.number_input(
            "🛁 Bathrooms",
            min_value=0,
            max_value=20,
            value=safe_int(pf("Bathroom", 2), 2),
        )

    with row1[2]:
        size = st.number_input(
            "📐 Property Size (sq.ft.)",
            min_value=1,
            max_value=200000,
            value=max(1, safe_int(pf("Property Size", 900), 900)),
            step=50,
        )

    with row1[3]:
        parking = st.number_input(
            "🚗 Parking Lots",
            min_value=0,
            max_value=100,
            value=safe_int(pf("Parking Lot", 1), 1),
        )

    row2 = st.columns(4)

    with row2[0]:
        completion_year = st.number_input(
            "🗓️ Completion Year",
            min_value=1900,
            max_value=2100,
            value=safe_int(pf("Completion Year", 2015), 2015),
        )

    with row2[1]:
        floors = st.number_input(
            "🏗️ Building Floors",
            min_value=1,
            max_value=1000,
            value=max(1, safe_int(pf("# of Floors", 20), 20)),
        )

    with row2[2]:
        total_units = st.number_input(
            "🏘️ Total Units",
            min_value=1,
            max_value=20000,
            value=max(1, safe_int(pf("Total Units", 500), 500)),
            step=10,
        )

    with row2[3]:
        default_tenure = pf("Tenure Type", "Freehold")
        tenure = st.selectbox(
            "Tenure Type",
            TENURE_OPTIONS,
            index=(
                TENURE_OPTIONS.index(default_tenure)
                if default_tenure in TENURE_OPTIONS
                else 0
            ),
        )

    row3 = st.columns(3)

    with row3[0]:
        default_pt = pf("Property Type", "Condominium")
        property_type = st.selectbox(
            "Property Type",
            PROPERTY_TYPES,
            index=(
                PROPERTY_TYPES.index(default_pt)
                if default_pt in PROPERTY_TYPES
                else 0
            ),
        )

    with row3[1]:
        default_land = pf("Land Title", "Non Bumi Lot")
        land_title = st.selectbox(
            "Land Title",
            LAND_OPTIONS,
            index=(
                LAND_OPTIONS.index(default_land)
                if default_land in LAND_OPTIONS
                else 0
            ),
        )

    with row3[2]:
        default_floor = pf("Floor Range", "-")
        floor_range = st.selectbox(
            "Floor Range",
            FLOOR_RANGE_OPTIONS,
            index=(
                FLOOR_RANGE_OPTIONS.index(default_floor)
                if default_floor in FLOOR_RANGE_OPTIONS
                else len(FLOOR_RANGE_OPTIONS) - 1
            ),
        )

    # -------------------------------------------------------------
    # State map + dropdown
    # -------------------------------------------------------------
    st.divider()
    st.markdown("#### 📍 Location")

    state_summary = get_state_summary()

    map_states = [
        state for state in STATE_OPTIONS
        if state in STATE_COORDS
    ]

    max_avg = max(
        [
            state_summary.get(state, {}).get("avg_price", 0)
            for state in map_states
        ] or [1]
    ) or 1

    max_count = max(
        [
            state_summary.get(state, {}).get("count", 0)
            for state in map_states
        ] or [1]
    ) or 1

    map_rows = []

    for state_name in map_states:
        summary = state_summary.get(
            state_name,
            {"avg_price": 0, "count": 0},
        )

        heat = summary["avg_price"] / max_avg
        r, g, b = heat_to_rgb(heat)

        map_rows.append(
            {
                "state": state_name,
                "lat": STATE_COORDS[state_name]["lat"],
                "lon": STATE_COORDS[state_name]["lon"],
                "avg_price": summary["avg_price"],
                "count": summary["count"],
                "radius": (
                    12000
                    + 28000 * (
                        summary["count"] / max_count
                        if max_count
                        else 0
                    )
                ),
                "r": r,
                "g": g,
                "b": b,
            }
        )

    map_df = pd.DataFrame(map_rows)

    default_state = pf("State", "Selangor")
    if default_state not in STATE_OPTIONS:
        default_state = "Selangor"

    current_state = st.session_state.get(
        "map_selected_state",
        default_state,
    )

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
            latitude=4.0,
            longitude=109.5,
            zoom=5.0,
            pitch=0,
            min_zoom=4,
            max_zoom=8,
        )

        frozen_view = pdk.View(
            type="MapView",
            controller=False,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            views=[frozen_view],
            map_style="dark",
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
            deck,
            height=340,
            on_select="rerun",
            selection_mode="single-object",
            key="state_map",
        )

        clicked_state = None

        if selection and selection.selection:
            objects = selection.selection.get("objects", {})
            picked = (
                objects.get("state-layer")
                or next(iter(objects.values()), None)
            )

            if picked:
                row = picked[0] if isinstance(picked, list) else picked
                clicked_state = row.get("state")

        if (
            clicked_state
            and clicked_state
            != st.session_state.get("map_selected_state")
        ):
            st.session_state["map_selected_state"] = clicked_state
            st.session_state["state_dropdown"] = clicked_state
            st.rerun()

    with picker_col:
        state_index = (
            STATE_OPTIONS.index(current_state)
            if current_state in STATE_OPTIONS
            else 0
        )

        state = st.selectbox(
            "Selected state",
            STATE_OPTIONS,
            index=state_index,
            key="state_dropdown",
        )

        if state != st.session_state.get("map_selected_state"):
            st.session_state["map_selected_state"] = state

        if state in state_summary:
            st.metric(
                "Average price",
                money(state_summary[state]["avg_price"]),
            )
            st.caption(
                f"{state_summary[state]['count']:,} sample listings"
            )

    if ART.city_options:
        default_city = pf("City", "Not sure / Other")
        city_options = ["Not sure / Other"] + ART.city_options

        city_index = (
            city_options.index(default_city)
            if default_city in city_options
            else 0
        )

        city_choice = st.selectbox(
            "City",
            city_options,
            index=city_index,
            key="city_dropdown",
        )

        city = (
            None
            if city_choice == "Not sure / Other"
            else city_choice
        )
    else:
        city = None

    # -------------------------------------------------------------
    # Facilities + nearby amenities
    # -------------------------------------------------------------
    st.divider()
    st.markdown("#### 🏊 Facilities & Nearby Amenities")

    default_facilities = [
        f
        for f in FACILITY_OPTIONS
        if prefill.get(f"Facility_{f}") == 1
    ]

    default_nearby = [
        n
        for n in NEARBY_OPTIONS
        if prefill.get(f"Has_{n}") == 1
    ]

    fac_col, nearby_col = st.columns(2)

    with fac_col:
        facilities = st.pills(
            "Facilities",
            FACILITY_OPTIONS,
            default=default_facilities,
            selection_mode="multi",
        ) or []

    with nearby_col:
        nearby = st.pills(
            "Nearby amenities",
            NEARBY_OPTIONS,
            default=default_nearby,
            selection_mode="multi",
            format_func=lambda x: x.replace("_", " "),
        ) or []

    # -------------------------------------------------------------
    # Predict / reset
    # -------------------------------------------------------------
    st.divider()
    pred_col, reset_col = st.columns(2)

    with pred_col:
        predict_clicked = st.button(
            "🔍 Predict Price",
            type="primary",
            width="stretch",
        )

    with reset_col:
        reset_clicked = st.button(
            "↺ Reset",
            width="stretch",
        )

    if reset_clicked:
        for key in [
            "autofill_row",
            "map_selected_state",
            "state_dropdown",
            "city_dropdown",
            "state_map",
        ]:
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
            result["prediction"],
            result.get("price_per_sqft"),
            result.get("bracket"),
        )

        st.success(
            f"Prediction generated using the tuned {best_name} model."
        )

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
            direction = (
                "above"
                if result["diff_pct"] >= 0
                else "below"
            )

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

            with st.expander(
                "🔍 What features generally drive the model?"
            ):
                st.caption(
                    "These are overall model feature-importance values. "
                    "They should not be interpreted as causal effects."
                )
                st.bar_chart(
                    importance_df,
                    height=300,
                )

        if prefill.get("price") is not None:
            st.caption(
                f"Actual price of the autofilled listing: "
                f"{money(prefill['price'])}"
            )

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