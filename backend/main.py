from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from constants import (
    FACILITY_OPTIONS, NEARBY_OPTIONS, PROPERTY_TYPES, TENURE_OPTIONS,
    LAND_OPTIONS, FLOOR_RANGE_OPTIONS, STATE_OPTIONS, STATE_COORDS,
)
from model_utils import load_artifacts, predict as run_predict

app = FastAPI(title="Condo.Price Predict API", version="1.0")

# CORS wide open: this API only ever returns non-sensitive, pre-trained
# model output, and the frontend URL isn't known ahead of deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean(records: list) -> list:
    """Replace NaN with None so it survives JSON serialisation."""
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in records
    ]


# ---------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------
class PredictRequest(BaseModel):
    bedroom: int = Field(3, ge=0, le=10)
    bathroom: int = Field(2, ge=0, le=10)
    size: float = Field(1000, ge=100, le=5000)
    floors: int = Field(20, ge=0, le=100)
    total_units: int = Field(500, ge=0, le=3000)
    parking: int = Field(1, ge=0, le=10)
    completion_year: int = Field(2015, ge=1980, le=2030)
    tenure: str = "Freehold"
    property_type: str = "Condominium"
    state: str = "Selangor"
    city: Optional[str] = None
    land_title: str = "Non Bumi Lot"
    floor_range: str = "-"
    facilities: List[str] = []
    nearby: List[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Everything the frontend needs to build its form + map, in one call."""
    # City categories aren't hardcoded like State/Tenure/etc. in constants.py -
    # ds_assignment.py buckets City into a variable top-N + "Other" at train
    # time, so feature_columns.pkl is the only source of truth for which
    # cities the deployed model actually recognises. /meta is otherwise
    # artifact-free by design (so the form can render even before the model
    # is available), so this degrades to an empty list rather than raising.
    try:
        city_options = load_artifacts().city_options
    except FileNotFoundError:
        city_options = []

    return {
        "facility_options": FACILITY_OPTIONS,
        "nearby_options": NEARBY_OPTIONS,
        "property_types": PROPERTY_TYPES,
        "tenure_options": TENURE_OPTIONS,
        "land_options": LAND_OPTIONS,
        "floor_range_options": FLOOR_RANGE_OPTIONS,
        "state_options": STATE_OPTIONS,
        "state_coords": {
            name: {"lat": lat, "lon": lon}
            for name, (lat, lon) in STATE_COORDS.items()
        },
        "city_options": city_options,
    }


@app.get("/listings")
def listings(
    state: Optional[List[str]] = Query(None),
    property_type: Optional[List[str]] = Query(None),
):
    """Sample listings, optionally filtered - powers the Market Overview table."""
    art = load_artifacts()
    df = art.sample_df.copy()
    if state:
        df = df[df["State"].isin(state)]
    if property_type:
        df = df[df["Property Type"].isin(property_type)]

    cols = ["Bedroom", "Bathroom", "Property Size", "Property Type",
            "Tenure Type", "State", "price"]
    records = df[cols].copy()
    records.insert(0, "index", records.index)
    return {
        "count": len(df),
        "listings": _clean(records.to_dict(orient="records")),
        "available_states": sorted(art.sample_df["State"].dropna().unique().tolist()),
        "available_property_types": sorted(art.sample_df["Property Type"].dropna().unique().tolist()),
    }


@app.get("/listings/{index}")
def listing_by_index(index: int):
    art = load_artifacts()
    if index not in art.sample_df.index:
        raise HTTPException(404, "Listing index not found")
    return _clean([art.sample_df.loc[index].to_dict()])[0]


@app.get("/market/state-summary")
def state_summary():
    """Avg price + listing count per model-recognised state, for the map."""
    art = load_artifacts()
    df = art.sample_df[art.sample_df["State"].isin(STATE_OPTIONS)]
    grouped = df.groupby("State")["price"].agg(["mean", "count"]).reset_index()
    out = {}
    for _, r in grouped.iterrows():
        out[r["State"]] = {"avg_price": float(r["mean"]), "count": int(r["count"])}
    return out


@app.get("/model/comparison")
def model_comparison():
    art = load_artifacts()
    if art.results_df is None:
        raise HTTPException(404, "all_results.pkl not available on the server")
    results_sorted = art.results_df.reset_index(drop=True).sort_values("RMSE")
    best_name = (
        art.extra["best_model_name"] if art.extra is not None
        else results_sorted.iloc[0]["Model"]
    )
    best_row = results_sorted[results_sorted["Model"] == best_name].iloc[0]
    return {
        "results": results_sorted.to_dict(orient="records"),
        "best_model_name": best_name,
        "best_rmse": float(best_row["RMSE"]),
        "best_r2": float(best_row["R2"]),
    }


@app.get("/model/diagnostics")
def model_diagnostics():
    """Everything needed for the residual/bracket-classification charts."""
    art = load_artifacts()
    if art.extra is None:
        raise HTTPException(404, "extra_artifacts.pkl not available on the server")
    e = art.extra

    def _finite(x):
        x = float(x)
        return None if np.isinf(x) else x

    return {
        "best_model_name": e["best_model_name"],
        "y_test": e["y_test"].tolist(),
        "test_predictions": {k: list(v) for k, v in e["test_predictions"].items()},
        "price_bin_edges": [_finite(x) for x in e["price_bin_edges"]],
        "price_bin_labels": list(e["price_bin_labels"]),
        "bracket_confusion_matrix": np.asarray(e["bracket_confusion_matrix"]).tolist(),
        "bracket_accuracy": float(e["bracket_accuracy"]),
        "bracket_precision": float(e["bracket_precision"]),
        "bracket_recall": float(e["bracket_recall"]),
        "bracket_f1": float(e["bracket_f1"]),
        "bracket_results_all_models": (
            e["bracket_results_all_models"].to_dict(orient="records")
            if "bracket_results_all_models" in e else None
        ),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    art = load_artifacts()
    if req.state not in STATE_OPTIONS and f"State_{req.state}" not in art.feature_columns:
        raise HTTPException(400, f"Unknown state: {req.state}")
    return run_predict(req.model_dump())