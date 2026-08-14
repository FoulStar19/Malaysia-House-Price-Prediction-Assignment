# -*- coding: utf-8 -*-
"""
BMDS2003 Data Science - Group Assignment
Dataset: Malaysian Condominium Prices Data (houses.csv)
Problem type: REGRESSION - predict property "price" (RM)

This script follows the CRISP-DM framework and reuses techniques taught across
Practicals 1-6:
    - Practical 1 : core Python (functions, control flow)
    - Practical 2 : NumPy (arrays, vectorised operations, aggregation)
    - Practical 3 : Pandas (loading, indexing, groupby, describe, missing data)
    - Practical 4 : EDA, IQR/Z-score outlier detection, correlation heatmap,
                    matplotlib/seaborn visualisation
    - Practical 5a/5b : train_test_split, KNN, SVM, scaling, cross_val_score,
                    GridSearchCV, Decision Tree / Random Forest
    - Practical 6a : Naive Bayes / cross-validation style model comparison
    - Practical 6c : 3-way train/val/test split, MinMaxScaler (avoiding data
                    leakage - fit on train only), MLPRegressor, evaluation
                    metrics, simple deployment prototype
"""

import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set()
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42

# =====================================================================
# 1. BUSINESS UNDERSTANDING
# =====================================================================
# Business problem: property agents / buyers want to know a fair asking
# price for a Malaysian condominium given its physical attributes
# (size, bedrooms, bathrooms, facilities) and location. A reliable price
# prediction model supports pricing decisions, investment appraisal and
# helps flag listings that are under/over-priced relative to the market.
# This is framed as a SUPERVISED REGRESSION problem: target = price (RM).

# =====================================================================
# 2. DATA UNDERSTANDING (Practical 3: pandas load/inspect/describe)
# =====================================================================
DATA_PATH = "houses.csv" if os.path.exists("houses.csv") else "/content/sample_data/houses.csv"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head()
df.info()
df.describe(include="all").T  # summary statistics for report

# =====================================================================
# 3. DATA PREPARATION (Practical 2/3/4: cleaning, missing values, outliers)
# =====================================================================

# --- 3.1 Duplicates & essential missing rows -------------------------
df[df["Mall"].isna() == True]

print("Total duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

df = df[df["Ad List"].isna() == False]
df.info()

# --- 3.2 Multi-label facilities -> binary indicator columns ----------
# (Practical 2: array/vectorised style transform via sklearn MultiLabelBinarizer)
df["Facilities"].sample(10)

mlb = MultiLabelBinarizer()
df["Facilities_List"] = df["Facilities"].str.split(", ")
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Facilities_List')),
                           columns=[f"Facility_{c}" for c in mlb.classes_],
                           index=df.index))
df = df.drop(columns=["Facility_-", "Facility_10"])
df.head()

# --- 3.3 Location fields (Practical 1: string slicing/splitting) -----
for addr in df.loc[df["Address"].isna() == False, "Address"].sample(10):
    print(addr)

df["State"] = df["Address"].apply(lambda addr: addr.split(", ")[-1])
df["City"] = df["Address"].apply(lambda addr: addr.split(", ")[-2] if len(addr.split(", ")) > 1 else None)
df.head()

# --- 3.4 Numeric fields stored as text -> proper numeric dtype -------
df["Bedroom"].unique()
df["Bedroom"] = df["Bedroom"].replace("-", np.nan).astype("float64")

df["Bathroom"].unique()
df["Bathroom"] = df["Bathroom"].replace("More than 10", "10")
df["Bathroom"] = df["Bathroom"].replace("-", np.nan).astype('float64')

df["# of Floors"].unique()
df["# of Floors"] = df["# of Floors"].replace("-", np.nan).astype('float64')

df["Total Units"].unique()
df["Total Units"] = df["Total Units"].replace("-", np.nan).astype('float64')

df["Parking Lot"].unique()
df["Parking Lot"] = df["Parking Lot"].replace("-", 0).astype('float64')

df["Completion Year"].unique()
df["Completion Year"] = df["Completion Year"].replace("-", np.nan).astype('float64')

df["Property Size"].sample(10)
df["Property Size"] = df["Property Size"].apply(lambda s: s.split(" ")[0]).astype("float64")
df["Property Size"].sample(10)

df["price"].sample(10)
df["price"] = df["price"].apply(lambda s: "".join(s.split(" ")[1:])).astype('float64')
df["price"].sample(10)

# --- 3.5 Nearby-amenity text columns -> Has_<amenity> flags -----------
amenity_cols = ["Mall", "Park", "School", "Hospital", "Bus Stop", "Highway",
                 "Railway Station", "Nearby School", "Nearby Mall",
                 "Nearby Railway Station"]

for col in amenity_cols:
    df[f"Has_{col.replace(' ', '_')}"] = df[col].notna().astype(int)

df = df.drop(columns=amenity_cols)

df["Firm Type"] = df["Firm Type"].fillna("Unknown")
df = df.drop(columns=["Firm Number", "REN Number"])
df.head()
df.info()

# --- 3.6 Outlier detection & removal (Practical 4: IQR method) -------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="price", ax=axes[0])
axes[0].set_title("price - before outlier removal")
sns.boxplot(data=df, x="Property Size", ax=axes[1])
axes[1].set_title("Property Size - before outlier removal")
plt.tight_layout()
plt.savefig("boxplot_before_outliers.png", dpi=120)
plt.show()


def remove_outliers_iqr(data, column):
    """+-1.5*IQR rule, same method demonstrated in Practical 4."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


rows_before = len(df)
df = remove_outliers_iqr(df, "price")
df = remove_outliers_iqr(df, "Property Size")
print(f"Rows removed as price/size outliers: {rows_before - len(df)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="price", ax=axes[0])
axes[0].set_title("price - after outlier removal")
sns.boxplot(data=df, x="Property Size", ax=axes[1])
axes[1].set_title("Property Size - after outlier removal")
plt.tight_layout()
plt.savefig("boxplot_after_outliers.png", dpi=120)
plt.show()

df = df.reset_index(drop=True)
df.head()
df.info()
df["Property Size"].sample(10)

ax = sns.histplot(data=df, x="price", kde=True)
plt.title("Distribution of price after cleaning")
plt.savefig("price_distribution.png", dpi=120)
plt.show()

# =====================================================================
# 4. EXPLORATORY DATA ANALYSIS (Practical 3/4: groupby, correlation heatmap)
# =====================================================================

# --- 4.1 groupby summaries (Practical 3) ------------------------------
state_price = df.groupby("State")["price"].mean().sort_values(ascending=False)
print("Average price by State (top 10):")
print(state_price.head(10))

proptype_price = df.groupby("Property Type")["price"].agg(["mean", "median", "count"])
print("\nPrice by Property Type:")
print(proptype_price)

fig, ax = plt.subplots(figsize=(10, 5))
state_price.head(10).plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Average price (RM)")
ax.set_title("Top 10 States by Average Condominium Price")
plt.tight_layout()
plt.savefig("avg_price_by_state.png", dpi=120)
plt.show()

# --- 4.2 Correlation heatmap (Practical 4) ----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.show()

print("Features most correlated with price:")
print(corr["price"].sort_values(ascending=False).head(10))

# --- 4.3 Scatter relationship (Practical 4 style) ----------------------
plt.figure(figsize=(6, 5))
plt.scatter(df["Property Size"], df["price"], alpha=0.3)
plt.xlabel("Property Size (sq.ft.)")
plt.ylabel("price (RM)")
plt.title("Property Size vs price")
plt.tight_layout()
plt.savefig("size_vs_price.png", dpi=120)
plt.show()

# =====================================================================
# 5. FEATURE ENGINEERING / ENCODING
# =====================================================================
# Drop identifier / free-text / very-high-cardinality columns that add
# noise rather than predictive signal for a regression model.
drop_cols = ["description", "Ad List", "Building Name", "Developer",
             "Address", "City", "Category", "Facilities"]
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Group the long tail of States into "Other" to keep one-hot encoding compact
top_states = df_model["State"].value_counts().nlargest(10).index
df_model["State"] = df_model["State"].where(df_model["State"].isin(top_states), "Other")

categorical_cols = ["Tenure Type", "Property Type", "Floor Range",
                     "Land Title", "Firm Type", "State"]
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Impute remaining numeric missing values with the median (robust to skew)
numeric_feature_cols = df_model.select_dtypes(include=[np.number]).columns.drop("price")
for col in numeric_feature_cols:
    if df_model[col].isna().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())

df_model.info()

# =====================================================================
# 6. MODELLING (Practical 5a/5b/6c: split -> scale -> train -> tune)
# =====================================================================
X = df_model.drop(columns=["price"])
y = df_model["price"]

# 3-way split (Practical 6c): 60% train / 20% validation / 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE)  # 0.25*0.8=0.2

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Feature scaling - fit ONLY on training data to avoid data leakage
# (same principle emphasised in Practical 6c)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


def evaluate(name, model, X_te, y_te):
    """Common evaluation routine (Practical 6c metrics)."""
    preds = model.predict(X_te)
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    print(f"[{name}] RMSE={rmse:,.2f}  MAE={mae:,.2f}  R2={r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


results = []

# --- 6.1 Baseline model: K-Nearest Neighbours (Practical 5a) ----------
knn_baseline = KNeighborsRegressor(n_neighbors=5)
knn_baseline.fit(X_train_scaled, y_train)

cv_scores_knn = cross_val_score(knn_baseline, X_train_scaled, y_train, cv=5,
                                 scoring="r2")
print(f"KNN baseline 5-fold CV R2: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std():.4f})")
results.append(evaluate("KNN (baseline)", knn_baseline, X_test_scaled, y_test))

# --- 6.2 Decision Tree Regressor with GridSearchCV (Practical 5b) -----
param_grid_dt = {
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                        param_grid_dt, cv=5, scoring="r2", n_jobs=-1)
grid_dt.fit(X_train_scaled, y_train)
print("Best Decision Tree params:", grid_dt.best_params_)
best_dt = grid_dt.best_estimator_
results.append(evaluate("Decision Tree", best_dt, X_test_scaled, y_test))

# --- 6.3 Random Forest Regressor with GridSearchCV (Practical 5b) -----
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE),
                        param_grid_rf, cv=5, scoring="r2", n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)
print("Best Random Forest params:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_
results.append(evaluate("Random Forest", best_rf, X_test_scaled, y_test))

# --- 6.4 MLP Regressor (Practical 6c) ----------------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,
    random_state=RANDOM_STATE,
)
mlp.fit(X_train_scaled, y_train)
results.append(evaluate("MLP Regressor", mlp, X_test_scaled, y_test))

# Use validation split to sanity-check the chosen MLP before final test reporting
val_r2 = r2_score(y_val, mlp.predict(X_val_scaled))
print(f"MLP validation R2: {val_r2:.4f}")

# =====================================================================
# 7. EVALUATION - MODEL COMPARISON (Practical 5b/6c)
# =====================================================================
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nModel comparison (sorted by RMSE, lower is better):")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=results_df, x="Model", y="RMSE", hue="Model", ax=axes[0],
            palette="viridis", legend=False)
axes[0].set_title("RMSE by Model (lower is better)")
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(data=results_df, x="Model", y="R2", hue="Model", ax=axes[1],
            palette="viridis", legend=False)
axes[1].set_title("R2 Score by Model (higher is better)")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
plt.show()

# Pick the best model (lowest RMSE) to persist for the deployment prototype
best_row = results_df.iloc[0]
model_lookup = {
    "KNN (baseline)": knn_baseline,
    "Decision Tree": best_dt,
    "Random Forest": best_rf,
    "MLP Regressor": mlp,
}
best_model = model_lookup[best_row["Model"]]
print(f"\nBest model selected for deployment: {best_row['Model']}")

# Actual vs Predicted plot for the best model
best_preds = best_model.predict(X_test_scaled)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.3)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual price (RM)")
plt.ylabel("Predicted price (RM)")
plt.title(f"Actual vs Predicted price - {best_row['Model']}")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
plt.show()

# =====================================================================
# 8. DEPLOYMENT PREP - persist model + scaler + feature columns
#    (used by the accompanying Streamlit prototype, streamlit_app.py)
# =====================================================================
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open("all_results.pkl", "wb") as f:
    pickle.dump(results_df, f)

# Save a sample of cleaned (pre-encoding), human-readable rows so the
# Streamlit prototype can offer an "autofill from an existing listing"
# feature, the same idea as the reference prototype's "autofill from
# existing date" control.
readable_cols = [
    "Bedroom", "Bathroom", "Property Size", "# of Floors", "Total Units",
    "Parking Lot", "Completion Year", "Tenure Type", "Property Type",
    "Floor Range", "Land Title", "State", "price",
]
facility_cols = [c for c in df.columns if c.startswith("Facility_")]
has_cols = [c for c in df.columns if c.startswith("Has_")]
app_sample = df[readable_cols + facility_cols + has_cols].sample(
    n=min(300, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
app_sample.to_csv("app_sample_listings.csv", index=False)

print("\nSaved best_model.pkl, scaler.pkl, feature_columns.pkl, "
      "all_results.pkl, app_sample_listings.csv for deployment.")
