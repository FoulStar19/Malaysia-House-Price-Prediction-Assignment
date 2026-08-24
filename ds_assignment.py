import os
import pickle
from pathlib import Path

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
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)

sns.set()
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd() 

DATA_PATH = SCRIPT_DIR / "houses.csv"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
df.head()
df.info()
df.describe(include="all").T  # summary statistics for report

df[df["Mall"].isna() == True]

print("Total duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

df = df[df["Ad List"].isna() == False]
df.info()

df["Facilities"].sample(10)

mlb = MultiLabelBinarizer()
df["Facilities_List"] = df["Facilities"].str.split(", ")
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Facilities_List')),
                           columns=[f"Facility_{c}" for c in mlb.classes_],
                           index=df.index))
df = df.drop(columns=["Facility_-", "Facility_10"])
df.head()

for addr in df.loc[df["Address"].isna() == False, "Address"].sample(10):
    print(addr)

df["State"] = df["Address"].apply(lambda addr: addr.split(", ")[-1])
df["City"] = df["Address"].apply(lambda addr: addr.split(", ")[-2] if len(addr.split(", ")) > 1 else None)
df.head()

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

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(data=df, x="price", ax=axes[0])
axes[0].set_title("price - before outlier removal")
sns.boxplot(data=df, x="Property Size", ax=axes[1])
axes[1].set_title("Property Size - before outlier removal")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "boxplot_before_outliers.png", dpi=120)
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
plt.savefig(SCRIPT_DIR / "boxplot_after_outliers.png", dpi=120)
plt.show()

df = df.reset_index(drop=True)
df.head()
df.info()
df["Property Size"].sample(10)

ax = sns.histplot(data=df, x="price", kde=True)
plt.title("Distribution of price after cleaning")
plt.savefig(SCRIPT_DIR / "price_distribution.png", dpi=120)
plt.show()

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
plt.savefig(SCRIPT_DIR / "avg_price_by_state.png", dpi=120)
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "correlation_heatmap.png", dpi=120)
plt.show()

print("Features most correlated with price:")
print(corr["price"].sort_values(ascending=False).head(10))

plt.figure(figsize=(6, 5))
plt.scatter(df["Property Size"], df["price"], alpha=0.3)
plt.xlabel("Property Size (sq.ft.)")
plt.ylabel("price (RM)")
plt.title("Property Size vs price")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "size_vs_price.png", dpi=120)
plt.show()

drop_cols = ["description", "Ad List", "Building Name", "Developer",
             "Address", "City", "Category", "Facilities"]
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

top_states = df_model["State"].value_counts().nlargest(10).index
df_model["State"] = df_model["State"].where(df_model["State"].isin(top_states), "Other")

categorical_cols = ["Tenure Type", "Property Type", "Floor Range",
                     "Land Title", "Firm Type", "State"]
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

numeric_feature_cols = df_model.select_dtypes(include=[np.number]).columns.drop("price")
for col in numeric_feature_cols:
    if df_model[col].isna().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())

df_model.info()

X = df_model.drop(columns=["price"])
y = df_model["price"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE)  # 0.25*0.8=0.2

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


test_predictions = {}  # model name -> predicted values on X_test, filled in by evaluate()


def evaluate(name, model, X_te, y_te):
    """Common evaluation routine (Practical 6c metrics)."""
    preds = model.predict(X_te)
    test_predictions[name] = preds
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    print(f"[{name}] RMSE={rmse:,.2f}  MAE={mae:,.2f}  R2={r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


results = []

knn_baseline = KNeighborsRegressor(n_neighbors=5)
knn_baseline.fit(X_train_scaled, y_train)

cv_scores_knn = cross_val_score(knn_baseline, X_train_scaled, y_train, cv=5,
                                 scoring="r2")
print(f"KNN baseline 5-fold CV R2: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std():.4f})")
results.append(evaluate("KNN (baseline)", knn_baseline, X_test_scaled, y_test))

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

val_r2 = r2_score(y_val, mlp.predict(X_val_scaled))
print(f"MLP validation R2: {val_r2:.4f}")

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
plt.savefig(SCRIPT_DIR / "model_comparison.png", dpi=120)
plt.show()

best_row = results_df.iloc[0]
model_lookup = {
    "KNN (baseline)": knn_baseline,
    "Decision Tree": best_dt,
    "Random Forest": best_rf,
    "MLP Regressor": mlp,
}
best_model = model_lookup[best_row["Model"]]
print(f"\nBest model selected for deployment: {best_row['Model']}")

best_preds = best_model.predict(X_test_scaled)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.3)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual price (RM)")
plt.ylabel("Predicted price (RM)")
plt.title(f"Actual vs Predicted price - {best_row['Model']}")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "actual_vs_predicted.png", dpi=120)
plt.show()


# ---------------------------------------------------------------------
# Extra artifacts for the "Model Comparison" tab in the Streamlit app:
# residual boxplot data, actual-vs-predicted distributions, and a
# price-bracket classification view (accuracy / precision / recall /
# confusion matrix). Note: the underlying task is regression (price is
# continuous), so RMSE/MAE/R2 above are the real performance metrics.
# Accuracy/precision/recall/confusion matrix aren't natively defined for
# regression, so here they're computed by bucketing price into quartile
# brackets (Budget/Mid-range/High-end/Premium) and treating "did the
# model land in the right bracket?" as a classification problem — a
# common, easy-to-communicate way to show this alongside RMSE/MAE/R2.
# ---------------------------------------------------------------------
price_bin_edges = list(
    pd.qcut(y_train, q=4, retbins=True, duplicates="drop")[1]
)
price_bin_edges[0] = -np.inf
price_bin_edges[-1] = np.inf
price_bin_labels = ["Budget", "Mid-range", "High-end", "Premium"][: len(price_bin_edges) - 1]

y_test_bracket = pd.cut(y_test, bins=price_bin_edges, labels=price_bin_labels)
best_pred_bracket = pd.cut(best_preds, bins=price_bin_edges, labels=price_bin_labels)

bracket_cm = confusion_matrix(y_test_bracket, best_pred_bracket, labels=price_bin_labels)
bracket_accuracy = accuracy_score(y_test_bracket, best_pred_bracket)
bracket_precision = precision_score(y_test_bracket, best_pred_bracket,
                                     labels=price_bin_labels, average="macro", zero_division=0)
bracket_recall = recall_score(y_test_bracket, best_pred_bracket,
                               labels=price_bin_labels, average="macro", zero_division=0)
bracket_f1 = f1_score(y_test_bracket, best_pred_bracket,
                       labels=price_bin_labels, average="macro", zero_division=0)

print(f"\nPrice-bracket classification view for best model ({best_row['Model']}):")
print(f"Accuracy={bracket_accuracy:.4f}  Precision(macro)={bracket_precision:.4f}  "
      f"Recall(macro)={bracket_recall:.4f}  F1(macro)={bracket_f1:.4f}")

extra_artifacts = {
    "best_model_name": best_row["Model"],
    "y_test": y_test.reset_index(drop=True),
    "test_predictions": {name: np.asarray(preds) for name, preds in test_predictions.items()},
    "price_bin_edges": price_bin_edges,
    "price_bin_labels": price_bin_labels,
    "bracket_confusion_matrix": bracket_cm,
    "bracket_accuracy": bracket_accuracy,
    "bracket_precision": bracket_precision,
    "bracket_recall": bracket_recall,
    "bracket_f1": bracket_f1,
}
with open(SCRIPT_DIR / "extra_artifacts.pkl", "wb") as f:
    pickle.dump(extra_artifacts, f)

with open(SCRIPT_DIR / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(SCRIPT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(SCRIPT_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open(SCRIPT_DIR / "all_results.pkl", "wb") as f:
    pickle.dump(results_df, f)

readable_cols = [
    "Bedroom", "Bathroom", "Property Size", "# of Floors", "Total Units",
    "Parking Lot", "Completion Year", "Tenure Type", "Property Type",
    "Floor Range", "Land Title", "State", "price",
]
facility_cols = [c for c in df.columns if c.startswith("Facility_")]
has_cols = [c for c in df.columns if c.startswith("Has_")]
app_sample = df[readable_cols + facility_cols + has_cols].sample(
    n=min(300, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
app_sample.to_csv(SCRIPT_DIR / "app_sample_listings.csv", index=False)

print("\nSaved best_model.pkl, scaler.pkl, feature_columns.pkl, "
      "all_results.pkl, extra_artifacts.pkl, app_sample_listings.csv for deployment.")