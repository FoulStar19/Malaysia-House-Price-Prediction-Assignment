from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "houses.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Practical 4 hard bounds for houses.csv (df_filtered step):
# (price_numeric > 50000) & (price_numeric < 2000000) & (size_numeric < 4000)
PRICE_LOWER_BOUND = 50_000
PRICE_UPPER_BOUND = 2_000_000
SIZE_UPPER_BOUND = 4_000
PROPERTY_REFERENCE_YEAR = 2026
MISSING_TOKENS = {"", "-", "--", "n/a", "na", "nan", "null", "none", "not available", "unknown"}

STATE_OPTIONS = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
    "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
    "Selangor", "Terengganu", "Kuala Lumpur", "Putrajaya", "Labuan", "Other",
]

STATE_COORDS = {
    "Johor": {"lat": 1.4927, "lon": 103.7414},
    "Kedah": {"lat": 6.1184, "lon": 100.3685},
    "Kelantan": {"lat": 6.1254, "lon": 102.2381},
    "Melaka": {"lat": 2.1896, "lon": 102.2501},
    "Negeri Sembilan": {"lat": 2.7258, "lon": 101.9424},
    "Pahang": {"lat": 3.8126, "lon": 103.3256},
    "Penang": {"lat": 5.4141, "lon": 100.3288},
    "Perak": {"lat": 4.5975, "lon": 101.0901},
    "Perlis": {"lat": 6.4449, "lon": 100.2048},
    "Sabah": {"lat": 5.9804, "lon": 116.0735},
    "Sarawak": {"lat": 1.5533, "lon": 110.3592},
    "Selangor": {"lat": 3.0738, "lon": 101.5183},
    "Terengganu": {"lat": 5.3117, "lon": 103.1324},
    "Kuala Lumpur": {"lat": 3.1390, "lon": 101.6869},
    "Putrajaya": {"lat": 2.9264, "lon": 101.6964},
    "Labuan": {"lat": 5.2831, "lon": 115.2308},
}

PROPERTY_TYPES = [
    "Condominium", "Apartment", "Service Residence", "Studio",
    "Flat", "Penthouse", "Townhouse", "Others",
]
TENURE_OPTIONS = ["Freehold", "Leasehold"]
LAND_OPTIONS = ["Non Bumi Lot", "Bumi Lot", "Malay Reserved Land"]
FLOOR_RANGE_OPTIONS = ["Low", "Medium", "High", "Top", "-"]

FACILITY_OPTIONS = [
    "Parking", "Security", "Swimming Pool", "Playground", "Barbeque area",
    "Jogging Track", "Gymnasium", "Minimart", "Lift", "Tennis Court",
    "Sauna", "Squash Court", "Clubhouse",
]

NEARBY_OPTIONS = [
    "Bus Stop", "Mall", "Park", "School", "Hospital",
    "Highway", "Nearby Railway Station", "Railway Station",
]

NUMERIC_FEATURES = [
    "Bedroom", "Bathroom", "Property Size", "# of Floors",
    "Total Units", "Parking Lot", "Completion Year",
    "Property Age", "Size per Bedroom", "Bathrooms per Bedroom", "Parking per Bedroom",
]

CATEGORICAL_FEATURES = [
    "Tenure Type", "Property Type", "Floor Range",
    "Land Title", "State", "City",
]

REQUIRED_COLUMNS = [
    "Bedroom", "Bathroom", "Property Size", "Completion Year",
    "# of Floors", "Total Units", "Parking Lot", "Floor Range",
    "Tenure Type", "Property Type", "Land Title", "Facilities",
    "Address", "price",
]

# The assignment requires at least 3 models and 4 models for a 4-member group.
# We deliberately train four models so the same project satisfies either group size.
MODEL_NAMES = ["Decision Tree", "KNN", "Random Forest", "MLP Regressor"]


@dataclass
class Artifacts:
    model: Pipeline
    models: dict[str, Pipeline]
    scaler: object
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    transformed_feature_names: list[str]
    results_df: pd.DataFrame
    tuning_df: pd.DataFrame
    extra: dict
    sample_df: pd.DataFrame
    city_options: list[str]
    numeric_limits: dict
    data_quality: dict


# --- Data cleaning helpers -------------------------------------------------
# The raw CSV stores several numeric fields as text and uses "-" as a
# missing-value marker.  We normalise these values to NaN first so that the
# sklearn pipeline can learn imputations from training folds only.
def _is_missing_token(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def _clean_numeric(value, remove_str=None):
    if _is_missing_token(value):
        return np.nan

    text = str(value).strip().upper()
    if remove_str:
        text = text.replace(remove_str.upper(), "")
    text = text.replace(",", "").replace(" ", "")

    # Support common shorthand such as 1.2M / 850K while preserving decimals.
    multiplier = 1.0
    if text.endswith("M"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text.endswith("K"):
        multiplier, text = 1_000.0, text[:-1]

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    return float(match.group()) * multiplier


def _clean_price(value):
    return _clean_numeric(value, "RM")


def _clean_property_size(value):
    return _clean_numeric(value, "SQ.FT.")


def _normalise_text(value):
    if _is_missing_token(value):
        return np.nan
    return str(value).strip()


def _coerce_numeric(df, columns, bounds=None):
    """Convert numeric columns and turn implausible values into NaN.

    We do not delete rows for missing predictor values: those are learned by
    the training pipeline's imputer.  Only impossible values are marked NaN.
    """
    bounds = bounds or {}
    for col in columns:
        df[col] = df[col].map(_clean_numeric)
        if col in bounds:
            low, high = bounds[col]
            bad = df[col].notna() & ((df[col] < low) | (df[col] > high))
            df.loc[bad, col] = np.nan
    return df


def _extract_state(address):
    if _is_missing_token(address):
        return "Other"

    text = str(address).strip()
    # Longest names first avoids partial matches.
    states = sorted(STATE_OPTIONS[:-1], key=len, reverse=True)
    for state in states:
        if re.search(rf"(?<![A-Za-z]){re.escape(state)}(?![A-Za-z])", text, re.I):
            return state

    # A few common address spellings.
    if re.search(r"Kuala\s+Lumpur|W.P.\s*Kuala\s+Lumpur", text, re.I):
        return "Kuala Lumpur"
    return "Other"

def _extract_city(address, state):
    if pd.isna(address):
        return "Not sure / Other"

    text = str(address).strip()
    if not text:
        return "Not sure / Other"

    known = [
        "Kuala Lumpur", "Petaling Jaya", "Shah Alam", "Subang Jaya",
        "Klang", "Kajang", "Cyberjaya", "Puchong", "Ampang",
        "Cheras", "Rawang", "Sepang", "Seremban", "Melaka City",
        "Johor Bahru", "Iskandar Puteri", "George Town", "Ipoh",
        "Kota Kinabalu", "Kuching", "Kuantan", "Kota Bharu",
        "Alor Setar", "Putrajaya",
    ]

    lower = text.lower()
    for city in known:
        if city.lower() in lower:
            return city

    if state != "Other":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for i, part in enumerate(parts):
            if state.lower() in part.lower() and i > 0:
                candidate = parts[i - 1]
                if 1 <= len(candidate) <= 40:
                    return candidate.title()

    return "Not sure / Other"


def _prepare_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find houses.csv beside app.py/backend.py: {DATA_FILE}"
        )

    raw = pd.read_csv(DATA_FILE)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(
            "houses.csv is missing required columns: " + ", ".join(missing_columns)
        )

    df = raw.copy()
    original_rows = len(df)

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    rows_after_duplicates = len(df)

    quality_columns = list(dict.fromkeys(REQUIRED_COLUMNS + [c for c in NUMERIC_FEATURES if c not in {"Property Age", "Size per Bedroom", "Bathrooms per Bedroom", "Parking per Bedroom"}] + ["price"]))
    raw_missing_summary = raw[quality_columns].isna().sum().sort_values(ascending=False)
    raw_placeholder_summary = {
        col: int(raw[col].map(_is_missing_token).sum())
        for col in raw.columns
        if raw[col].dtype == "object"
    }

    # Numeric cleaning.  Placeholder strings become NaN, and impossible
    # values are also converted to NaN rather than silently retained.
    numeric_bounds = {
        "Bedroom": (0, 20),
        "Bathroom": (0, 20),
        "# of Floors": (1, 200),
        "Total Units": (1, 20_000),
        "Parking Lot": (0, 20),
        "Completion Year": (1800, PROPERTY_REFERENCE_YEAR + 2),
    }
    plain_numeric_cols = ["Bedroom", "Bathroom", "# of Floors", "Total Units", "Parking Lot", "Completion Year"]
    df = _coerce_numeric(df, plain_numeric_cols, numeric_bounds)
    df["Property Size"] = df["Property Size"].map(_clean_property_size)
    df["price"] = df["price"].map(_clean_price)

    # Property size is constrained to a plausible range, while target and
    # size rows are removed only when they cannot be used for regression.
    df.loc[(df["Property Size"] <= 0) | (df["Property Size"] >= SIZE_UPPER_BOUND), "Property Size"] = np.nan
    df.loc[(df["price"] <= PRICE_LOWER_BOUND) | (df["price"] >= PRICE_UPPER_BOUND), "price"] = np.nan

    # Normalise categorical text and treat placeholders as true missingness.
    for col in ["Floor Range", "Tenure Type", "Property Type", "Land Title"]:
        df[col] = df[col].map(_normalise_text)

    df["State"] = df["Address"].map(_extract_state)
    df["City"] = [
        _extract_city(address, state)
        for address, state in zip(df["Address"], df["State"])
    ]

    invalid_target = df["price"].isna()
    invalid_size = df["Property Size"].isna()
    removed_invalid_target = int(invalid_target.sum())
    removed_invalid_size = int((~invalid_target & invalid_size).sum())
    df = df.loc[~invalid_target & ~invalid_size].copy()

    # Facility text is converted to binary presence/absence indicators.
    for facility in FACILITY_OPTIONS:
        pattern = re.escape(facility)
        df[f"Facility_{facility}"] = (
            df["Facilities"].fillna("").astype(str).str.contains(pattern, case=False, regex=True).astype(int)
        )

    nearby_source = {name: name for name in NEARBY_OPTIONS}
    for label, source_col in nearby_source.items():
        df[f"Has_{label}"] = df[source_col].notna().astype(int) if source_col in df.columns else 0

    # Derived features are calculated after cleaning.  They are also recreated
    # in build_feature_row() for live predictions, so training and inference match.
    df["Property Age"] = (PROPERTY_REFERENCE_YEAR - df["Completion Year"]).clip(lower=0)
    df["Size per Bedroom"] = df["Property Size"] / df["Bedroom"].replace(0, np.nan)
    df["Bathrooms per Bedroom"] = df["Bathroom"] / df["Bedroom"].replace(0, np.nan)
    df["Parking per Bedroom"] = df["Parking Lot"] / df["Bedroom"].replace(0, np.nan)

    # Report IQR outliers for transparency.  Legitimate expensive/large homes
    # are not automatically deleted merely because they are statistical outliers.
    outlier_summary = {}
    for col in ["Bedroom", "Bathroom", "Property Size", "# of Floors", "Total Units", "Parking Lot", "Completion Year", "Property Age", "Size per Bedroom", "Bathrooms per Bedroom", "Parking per Bedroom", "price"]:
        s = df[col].dropna()
        if len(s) == 0:
            outlier_summary[col] = {"lower_bound": None, "upper_bound": None, "count": 0, "percentage": 0.0}
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        outlier_summary[col] = {
            "lower_bound": lower,
            "upper_bound": upper,
            "count": int(mask.sum()),
            "percentage": float(mask.mean() * 100),
        }

    sample_cols = [
        "Bedroom", "Bathroom", "Property Size", "Property Type", "Tenure Type",
        "State", "City", "price", "Floor Range", "Land Title", "Parking Lot",
        "Completion Year", "# of Floors", "Total Units", "Facilities",
    ]
    sample_df = df[sample_cols].copy()

    location_validation = (
        df[["Address", "State", "City"]]
        .sample(n=min(5, len(df)), random_state=RANDOM_STATE)
        .rename(columns={"Address": "Original Address", "State": "Extracted State", "City": "Extracted City"})
    )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    binary_cols = [f"Facility_{f}" for f in FACILITY_OPTIONS] + [f"Has_{n}" for n in NEARBY_OPTIONS]
    feature_df = df[feature_columns + binary_cols].copy()
    for col in binary_cols:
        feature_df[col] = df[col].astype(int)

    y = df["price"].astype(float)
    missing_summary = df[quality_columns].isna().sum().sort_values(ascending=False)

    data_quality = {
        "original_rows": original_rows,
        "rows_after_duplicates": rows_after_duplicates,
        "usable_rows": int(len(df)),
        "duplicate_rows": duplicate_rows,
        "removed_invalid_target": removed_invalid_target,
        "removed_invalid_size": removed_invalid_size,
        "raw_missing_summary": raw_missing_summary.to_dict(),
        "raw_placeholder_summary": raw_placeholder_summary,
        "missing_summary": missing_summary.to_dict(),
        "outlier_summary": outlier_summary,
        "location_validation": location_validation.to_dict("records"),
        "cleaning_notes": [
            "Converted '-', blank, N/A, NA, null and None placeholders to NaN.",
            "Converted numeric text to numeric values and marked implausible numeric values as missing.",
            "Used median imputation for numeric predictors and a dedicated Missing category for categorical predictors inside the training pipeline.",
            "One-hot encoded categorical predictors with unknown categories ignored at prediction time.",
            "Kept IQR outliers for legitimate properties while reporting them separately.",
            "Extracted all Malaysian states rather than mapping most states to Other.",
        ],
    }
    return df, sample_df, feature_df, y, data_quality

def _make_brackets(y):
    """Create four quantile-based business price brackets."""
    q = np.quantile(y, [0, 0.25, 0.50, 0.75, 1.0])
    edges = np.unique(q)

    if len(edges) < 5:
        low, high = float(y.min()), float(y.max())
        if high <= low:
            high = low + 1
        edges = np.linspace(low, high, 5)

    labels = [
        f"RM {edges[i]:,.0f}–RM {edges[i + 1]:,.0f}"
        for i in range(len(edges) - 1)
    ]
    return edges, labels


def _build_preprocessor():
    binary_features = [f"Facility_{f}" for f in FACILITY_OPTIONS] + [f"Has_{n}" for n in NEARBY_OPTIONS]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("binary", "passthrough", binary_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def _model_searches(preprocessor):
    """Return four assignment-ready tuned models.

    Decision Tree is the explicitly designated baseline. Its grid is kept
    deliberately small and interpretable compared with the other models.
    """

    configs = {}

    configs["Decision Tree"] = (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "model__max_depth": [5, 10],
            "model__min_samples_leaf": [2, 5],
        },
    )

    configs["KNN"] = (
        KNeighborsRegressor(),
        {
            "model__n_neighbors": [5, 7],
            "model__weights": ["uniform", "distance"],
        },
    )

    configs["Random Forest"] = (
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "model__n_estimators": [100],
            "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 2],
        },
    )

    configs["MLP Regressor"] = (
        MLPRegressor(
            max_iter=180,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
        {
            "model__hidden_layer_sizes": [(32, 16), (64, 32)],
            "model__alpha": [0.001],
        },
    )

    searches = {}
    for name, (estimator, grid) in configs.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        searches[name] = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=CV_FOLDS,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )

    return searches


def _get_transformed_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


@lru_cache(maxsize=1)
def load_artifacts():
    df, sample_df, X, y, data_quality = _prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    searches = _model_searches(_build_preprocessor())

    models = {}
    results = []
    tuning_rows = []
    predictions = {}

    for name, search in searches.items():
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        models[name] = best_model

        pred = best_model.predict(X_test)
        predictions[name] = np.asarray(pred)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        results.append(
            {
                "Model": name,
                "Role": "Baseline" if name == "Decision Tree" else "Candidate",
                "CV RMSE": float(-search.best_score_),
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }
        )

        tuning_rows.append(
            {
                "Model": name,
                "Best Parameters": str(search.best_params_),
                "Best CV RMSE": float(-search.best_score_),
            }
        )

    results_df = pd.DataFrame(results).sort_values("CV RMSE").reset_index(drop=True)
    tuning_df = pd.DataFrame(tuning_rows)

    # Select the deployment model using cross-validation only. The held-out
    # test set remains an unbiased final evaluation set.
    best_name = str(results_df.iloc[0]["Model"])
    best_model = models[best_name]

    # Use the best model's fitted preprocessor for feature names.
    preprocessor = best_model.named_steps["preprocessor"]
    transformed_feature_names = _get_transformed_feature_names(preprocessor)

    # A scaler object is retained for backwards compatibility with the old app.
    scaler = (
        preprocessor.named_transformers_["numeric"]
        .named_steps["scaler"]
        if "numeric" in preprocessor.named_transformers_
        else None
    )

    edges, labels = _make_brackets(y)

    actual_bracket = pd.Series(
        pd.cut(
            y_test,
            bins=edges,
            labels=False,
            include_lowest=True,
        ),
        index=y_test.index,
    )

    bracket_results = []
    best_cm = None
    best_metrics = {}

    for name, pred in predictions.items():
        pred_bracket = pd.Series(
            pd.cut(
                pred,
                bins=edges,
                labels=False,
                include_lowest=True,
            ),
            index=y_test.index,
        )

        valid = actual_bracket.notna() & pred_bracket.notna()

        actual_b = actual_bracket.loc[valid].astype(int)
        pred_b = pred_bracket.loc[valid].astype(int)

        metrics = {
            "Model": name,
            "Accuracy": float(accuracy_score(actual_b, pred_b)),
            "Precision": float(
                precision_score(actual_b, pred_b, average="macro", zero_division=0)
            ),
            "Recall": float(
                recall_score(actual_b, pred_b, average="macro", zero_division=0)
            ),
            "F1": float(
                f1_score(actual_b, pred_b, average="macro", zero_division=0)
            ),
        }
        bracket_results.append(metrics)

        if name == best_name:
            best_cm = confusion_matrix(
                actual_b,
                pred_b,
                labels=list(range(len(labels))),
            )
            best_metrics = metrics

    # Feature importance for tree-based winners.
    feature_importances = []
    final_estimator = best_model.named_steps["model"]

    if hasattr(final_estimator, "feature_importances_") and transformed_feature_names:
        values = np.asarray(final_estimator.feature_importances_)
        pairs = sorted(
            zip(transformed_feature_names, values),
            key=lambda x: x[1],
            reverse=True,
        )[:12]
        feature_importances = [
            {"feature": str(name), "importance": float(value)}
            for name, value in pairs
        ]

    # Permutation importance is useful for non-tree winners, but only for a
    # small sample to keep Streamlit startup responsive.
    elif len(X_test) > 0 and transformed_feature_names:
        sample_n = min(500, len(X_test))
        X_perm = X_test.iloc[:sample_n]
        y_perm = y_test.iloc[:sample_n]
        try:
            perm = permutation_importance(
                best_model,
                X_perm,
                y_perm,
                scoring="neg_root_mean_squared_error",
                n_repeats=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            pairs = sorted(
                zip(X.columns, perm.importances_mean),
                key=lambda x: x[1],
                reverse=True,
            )[:12]
            feature_importances = [
                {"feature": str(name), "importance": float(value)}
                for name, value in pairs
            ]
        except Exception:
            feature_importances = []

    numeric_limits = {}
    for col in NUMERIC_FEATURES:
        s = df[col].dropna()
        if len(s):
            numeric_limits[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
            }

    extra = {
        "best_model_name": best_name,
        "baseline_model_name": "Decision Tree",
        "y_test": y_test.to_numpy(),
        "test_predictions": predictions,
        "price_bin_edges": edges,
        "price_bin_labels": labels,
        "bracket_confusion_matrix": best_cm,
        "bracket_accuracy": best_metrics.get("Accuracy", 0.0),
        "bracket_precision": best_metrics.get("Precision", 0.0),
        "bracket_recall": best_metrics.get("Recall", 0.0),
        "bracket_f1": best_metrics.get("F1", 0.0),
        "bracket_results_all_models": pd.DataFrame(bracket_results),
        "feature_importances": feature_importances,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "model_configs": tuning_df,
        "selection_metric": "CV RMSE",
        "data_cleaning": data_quality.get("cleaning_notes", []),
    }

    city_options = sorted(
        {
            c[len("City_"):]
            for c in transformed_feature_names
            if c.startswith("City_")
        }
    )

    return Artifacts(
        model=best_model,
        models=models,
        scaler=scaler,
        preprocessor=preprocessor,
        feature_columns=list(X.columns),
        transformed_feature_names=transformed_feature_names,
        results_df=results_df,
        tuning_df=tuning_df,
        extra=extra,
        sample_df=sample_df.reset_index(drop=True),
        city_options=city_options,
        numeric_limits=numeric_limits,
        data_quality=data_quality,
    )


def build_feature_row(payload):
    art = load_artifacts()

    row = {}

    values = {
        "Bedroom": payload.get("bedroom"),
        "Bathroom": payload.get("bathroom"),
        "Property Size": payload.get("size"),
        "# of Floors": payload.get("floors"),
        "Total Units": payload.get("total_units"),
        "Parking Lot": payload.get("parking"),
        "Completion Year": payload.get("completion_year"),
    }

    for key in NUMERIC_FEATURES:
        row[key] = values.get(key, np.nan)

    # Keep live prediction features identical to the training representation.
    bedroom = pd.to_numeric(pd.Series([row["Bedroom"]]), errors="coerce").iloc[0]
    size = pd.to_numeric(pd.Series([row["Property Size"]]), errors="coerce").iloc[0]
    bathroom = pd.to_numeric(pd.Series([row["Bathroom"]]), errors="coerce").iloc[0]
    parking = pd.to_numeric(pd.Series([row["Parking Lot"]]), errors="coerce").iloc[0]
    year = pd.to_numeric(pd.Series([row["Completion Year"]]), errors="coerce").iloc[0]
    row["Property Age"] = PROPERTY_REFERENCE_YEAR - year if pd.notna(year) else np.nan
    row["Size per Bedroom"] = size / bedroom if pd.notna(size) and pd.notna(bedroom) and bedroom > 0 else np.nan
    row["Bathrooms per Bedroom"] = bathroom / bedroom if pd.notna(bathroom) and pd.notna(bedroom) and bedroom > 0 else np.nan
    row["Parking per Bedroom"] = parking / bedroom if pd.notna(parking) and pd.notna(bedroom) and bedroom > 0 else np.nan

    for col in CATEGORICAL_FEATURES:
        row[col] = None

    row["Tenure Type"] = payload.get("tenure")
    row["Property Type"] = payload.get("property_type")
    row["Floor Range"] = payload.get("floor_range")
    row["Land Title"] = payload.get("land_title")
    row["State"] = payload.get("state")
    row["City"] = payload.get("city") or "Not sure / Other"

    for facility in FACILITY_OPTIONS:
        row[f"Facility_{facility}"] = int(
            facility in payload.get("facilities", [])
        )

    for nearby in NEARBY_OPTIONS:
        row[f"Has_{nearby}"] = int(
            nearby in payload.get("nearby", [])
        )

    return pd.DataFrame([row], columns=load_artifacts().feature_columns)


def predict(payload):
    art = load_artifacts()

    X = build_feature_row(payload)
    prediction = max(0.0, float(art.model.predict(X)[0]))

    result = {
        "prediction": prediction,
        "price_per_sqft": (
            prediction / float(payload["size"])
            if payload.get("size")
            else None
        ),
        "best_rmse": None,
        "range_low": None,
        "range_high": None,
        "bracket": None,
        "similar_avg": None,
        "similar_count": 0,
        "diff_pct": None,
        "feature_importances": art.extra.get("feature_importances", []),
    }

    if art.results_df is not None and len(art.results_df):
        best_row = art.results_df[
            art.results_df["Model"] == art.extra["best_model_name"]
        ].iloc[0]

        rmse = float(best_row["RMSE"])
        result.update(
            best_rmse=rmse,
            range_low=max(prediction - rmse, 0),
            range_high=prediction + rmse,
        )

    if art.extra:
        bracket = pd.cut(
            [prediction],
            bins=art.extra["price_bin_edges"],
            labels=art.extra["price_bin_labels"],
            include_lowest=True,
        )[0]
        result["bracket"] = str(bracket) if not pd.isna(bracket) else None

    state = payload.get("state")
    property_type = payload.get("property_type")

    similar = art.sample_df[
        (art.sample_df["State"] == state)
        & (art.sample_df["Property Type"] == property_type)
    ]

    if len(similar) >= 3:
        avg = float(similar["price"].mean())
        result.update(
            similar_avg=avg,
            similar_count=len(similar),
            diff_pct=((prediction - avg) / avg * 100) if avg else None,
        )

    return result


def get_listings(states=None, property_types=None):
    art = load_artifacts()
    df = art.sample_df.copy()

    if states:
        df = df[df["State"].isin(states)]
    if property_types:
        df = df[df["Property Type"].isin(property_types)]

    cols = [
        "Bedroom", "Bathroom", "Property Size", "Property Type",
        "Tenure Type", "State", "price",
    ]

    records = df[cols].copy()
    records.insert(0, "index", records.index)

    return {
        "count": len(df),
        "listings": records.where(pd.notna(records), None).to_dict("records"),
        "available_states": sorted(art.sample_df["State"].dropna().unique()),
        "available_property_types": sorted(
            art.sample_df["Property Type"].dropna().unique()
        ),
    }


def get_listing(index):
    art = load_artifacts()
    row = art.sample_df.loc[index]
    return row.where(pd.notna(row), None).to_dict()


def get_state_summary():
    art = load_artifacts()
    df = art.sample_df[art.sample_df["State"].isin(STATE_OPTIONS)]

    grouped = df.groupby("State")["price"].agg(["mean", "count"])

    return {
        state: {
            "avg_price": float(row["mean"]),
            "count": int(row["count"]),
        }
        for state, row in grouped.iterrows()
    }


def get_model_comparison():
    art = load_artifacts()
    results = art.results_df.reset_index(drop=True).sort_values("CV RMSE")
    best_name = art.extra["best_model_name"]
    best_row = results[results["Model"] == best_name].iloc[0]

    return {
        "results": results.to_dict("records"),
        "best_model_name": best_name,
        "baseline_model_name": art.extra["baseline_model_name"],
        "best_rmse": float(best_row["RMSE"]),
        "best_mae": float(best_row["MAE"]),
        "best_r2": float(best_row["R2"]),
        "cv_folds": CV_FOLDS,
    }


def get_model_diagnostics():
    art = load_artifacts()
    e = art.extra

    return {
        "best_model_name": e["best_model_name"],
        "baseline_model_name": e["baseline_model_name"],
        "y_test": np.asarray(e["y_test"]).tolist(),
        "test_predictions": {
            key: list(value) for key, value in e["test_predictions"].items()
        },
        "price_bin_edges": list(e["price_bin_edges"]),
        "price_bin_labels": list(e["price_bin_labels"]),
        "bracket_confusion_matrix": (
            np.asarray(e["bracket_confusion_matrix"]).tolist()
            if e["bracket_confusion_matrix"] is not None
            else []
        ),
        "bracket_accuracy": float(e["bracket_accuracy"]),
        "bracket_precision": float(e["bracket_precision"]),
        "bracket_recall": float(e["bracket_recall"]),
        "bracket_f1": float(e["bracket_f1"]),
        "bracket_results_all_models": e[
            "bracket_results_all_models"
        ].to_dict("records"),
        "feature_importances": e.get("feature_importances", []),
        "train_rows": e["train_rows"],
        "test_rows": e["test_rows"],
    }


def get_tuning_results():
    art = load_artifacts()
    return art.tuning_df.to_dict("records")


def get_data_quality():
    art = load_artifacts()

    missing = pd.DataFrame(
        [
            {"Column": col, "Missing Values": count}
            for col, count in art.data_quality["missing_summary"].items()
        ]
    )

    outliers = pd.DataFrame(
        [
            {
                "Column": col,
                "IQR Lower": info["lower_bound"],
                "IQR Upper": info["upper_bound"],
                "Outliers": info["count"],
                "Outlier %": info["percentage"],
            }
            for col, info in art.data_quality["outlier_summary"].items()
        ]
    )

    return {
        "original_rows": art.data_quality["original_rows"],
        "rows_after_duplicates": art.data_quality["rows_after_duplicates"],
        "usable_rows": art.data_quality["usable_rows"],
        "duplicate_rows": art.data_quality["duplicate_rows"],
        "removed_invalid_target": art.data_quality["removed_invalid_target"],
        "removed_invalid_size": art.data_quality["removed_invalid_size"],
        "raw_missing_summary": [
            {"Column": col, "Missing Values": count}
            for col, count in art.data_quality["raw_missing_summary"].items()
        ],
        "missing_summary": missing.to_dict("records"),
        "outlier_summary": outliers.to_dict("records"),
        "location_validation": art.data_quality["location_validation"],
        "cleaning_notes": art.data_quality.get("cleaning_notes", []),
    }
